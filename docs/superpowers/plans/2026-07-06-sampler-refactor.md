# Sampler Refactor + Review Fixes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Shrink `sampling/sampler.py` from 1,008 to ~350 lines by deleting dead code and deduplicating solver logic, fix the DDPM-sampled-with-bridge-SDE bug, and update stale paper references — all verified against `paper_v2/paper_v2.tex`.

**Architecture:** The live `Sampler` API (`sample`, `sample_batch`, `sample_ode`, `sample_batch_ode`, `sample_ode_torchdiffeq`, `ODESolver`) keeps its exact signatures; internals collapse onto two private helpers (`_solver_step`, `_batched`). Image I/O moves to free functions in a new `sampling/io.py`. `cli.py` gains a DDPM branch mirroring the existing Poisson branch.

**Tech Stack:** Python 3.13, PyTorch, torchdiffeq, pytest, uv.

## Global Constraints

- Work ONLY in the worktree `/Users/osianshelley/Projects/diffusion-bridge/.claude/worktrees/paper-v2-refactor` on branch `refactor/paper-v2-review`. Never touch the main checkout.
- Run everything via `uv run` from the worktree root (deps already synced with `--extra dev`).
- Baseline test state: 96 passed, 2 skipped. Every task must end with the full suite green: `uv run pytest tests/ -q`.
- Behavior-preserving except: (a) the DDPM CLI fix (Task 6), (b) the new Poisson assumption warning (Task 8), (c) `ValueError` on unknown fixed-step solver (Task 2 — previously fell through silently).
- Spec: `docs/superpowers/specs/2026-07-06-sampler-refactor-design.md`. paper_v2 anchors: Algorithms 1–4 (§8), Prop `prop:bridge_drift` (tex:999), Prop `cor:prob_flow_ode` (tex:1071).
- Commit messages: conventional prefix + `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>` trailer.

---

### Task 1: Commit the ODE characterization tests

The tests are ALREADY WRITTEN in the working tree (uncommitted edit to `tests/test_sampling.py`): a `TestSampleODE` class using the existing `_IdentityNet` (predicts ŷ = ξ, so from ξ = x₀ the ODE drift `½(ξ−x₀)/t + ½(ŷ−ξ)/(T−t)` is identically zero — a fixed point), parametrized over EULER/HEUN/RK4, plus determinism, trajectory-length, batch, and torchdiffeq-routing tests. The import line now reads `from bridge_diffusion.sampling import ODESolver, Sampler`.

**Files:**
- Modify (already applied): `tests/test_sampling.py`

**Interfaces:**
- Produces: the regression safety net for Tasks 2–5. Test names: `TestSampleODE::test_identity_net_is_fixed_point`, `test_determinism_with_seed`, `test_trajectory_length`, `test_batch_ode_fixed_step`, `test_batch_ode_torchdiffeq_routing`.

- [ ] **Step 1: Verify the edit is present**

Run: `git -C /Users/osianshelley/Projects/diffusion-bridge/.claude/worktrees/paper-v2-refactor diff --stat`
Expected: `tests/test_sampling.py` listed. If the diff is empty, re-apply the `TestSampleODE` class from the git stash or ask the orchestrator — do not improvise new tests.

- [ ] **Step 2: Run the new tests against the CURRENT (unrefactored) sampler**

Run: `uv run pytest tests/test_sampling.py -q`
Expected: all pass (they characterize existing behavior; ~9 new tests + 10 existing).

- [ ] **Step 3: Commit**

```bash
git add tests/test_sampling.py
git commit -m "test: characterize probability-flow ODE sampler before refactor"
```

---

### Task 2: Extract `_solver_step` and rewrite `sample_ode` on top of it

**Files:**
- Modify: `src/bridge_diffusion/sampling/sampler.py:216-261` (the solver dispatch inside `sample_ode`), adding `_solver_step` after `_ode_drift` (after line 168)
- Test: `tests/test_sampling.py` (existing, no changes)

**Interfaces:**
- Produces: `Sampler._solver_step(drift_fn: Callable[[torch.Tensor, float], torch.Tensor], xi: torch.Tensor, t: float, dt: float, solver: ODESolver) -> torch.Tensor`. `drift_fn(state, time)` returns the drift at `(state, time)`; `_solver_step` performs one Euler/Heun/RK4 update and raises `ValueError` for non-fixed-step solvers.

- [ ] **Step 1: Add `_solver_step` immediately after `_ode_drift`**

```python
    def _solver_step(
        self,
        drift_fn,
        xi: torch.Tensor,
        t: float,
        dt: float,
        solver: ODESolver,
    ) -> torch.Tensor:
        """Advance xi by one fixed-step ODE update.

        Args:
            drift_fn: Callable (state, time) -> drift tensor.
            xi: Current state of shape (batch, ...).
            t: Current time (scalar).
            dt: Step size.
            solver: Fixed-step solver (EULER, HEUN, or RK4).

        Returns:
            Updated state of shape (batch, ...).
        """
        if solver == ODESolver.EULER:
            return xi + drift_fn(xi, t) * dt
        if solver == ODESolver.HEUN:
            k1 = drift_fn(xi, t)
            k2 = drift_fn(xi + k1 * dt, t + dt)
            return xi + 0.5 * (k1 + k2) * dt
        if solver == ODESolver.RK4:
            k1 = drift_fn(xi, t)
            k2 = drift_fn(xi + 0.5 * k1 * dt, t + 0.5 * dt)
            k3 = drift_fn(xi + 0.5 * k2 * dt, t + 0.5 * dt)
            k4 = drift_fn(xi + k3 * dt, t + dt)
            return xi + (k1 + 2 * k2 + 2 * k3 + k4) * dt / 6
        raise ValueError(f"Not a fixed-step solver: {solver}")
```

- [ ] **Step 2: Replace the `if/elif` solver blocks inside `sample_ode` (currently lines 224-261)**

The loop body becomes:

```python
        def drift_fn(state: torch.Tensor, time: float) -> torch.Tensor:
            time_tensor = torch.full((num_samples,), time, device=self.device)
            return self._ode_drift(state, x0, time, time_tensor)

        for step in tqdm(
            range(num_steps),
            desc=f"Sampling (ODE {solver.value})",
            disable=not self.sampling_config.show_progress,
        ):
            t = eps + step * dt
            xi = self._solver_step(drift_fn, xi, t, dt, solver)

            if return_trajectory:
                trajectory.append(xi.clone())
```

(The `t_tensor = torch.full(...)` line that preceded the dispatch is deleted — `drift_fn` builds it per evaluation, exactly as each `k_i` evaluation did before.)

- [ ] **Step 3: Run the sampling tests**

Run: `uv run pytest tests/test_sampling.py -q`
Expected: all pass — identity fixed-point and determinism tests confirm numerical equivalence.

- [ ] **Step 4: Full suite, then commit**

Run: `uv run pytest tests/ -q` → 105 passed, 2 skipped (96 + 9 new).

```bash
git add src/bridge_diffusion/sampling/sampler.py
git commit -m "refactor(sampling): extract _solver_step; sample_ode uses it"
```

---

### Task 3: Delete the six dead sampler methods + `_compute_score`

Zero callers exist for these in src/, scripts/, tests/, and the untracked experiments/ (verified by grep before planning).

**Files:**
- Modify: `src/bridge_diffusion/sampling/sampler.py` — delete entire methods: `sample_with_guidance` (393-415), `_compute_score` (454-493), `sample_predictor_corrector` (495-618), `sample_batch_predictor_corrector` (620-664), `sample_hybrid` (666-765), `sample_batch_hybrid` (767-805), `sample_pc_sde` (807-916), `sample_batch_pc_sde` (918-956). (Line numbers are pre-Task-2; locate by method name.)

**Interfaces:**
- Consumes: nothing. Produces: a `Sampler` whose only public sampling methods are `sample`, `sample_batch`, `sample_ode`, `sample_batch_ode`, `sample_ode_torchdiffeq`.

- [ ] **Step 1: Delete the eight methods listed above** (method `def` line through the line before the next `def` at class level).

- [ ] **Step 2: Verify nothing references them**

Run: `grep -rn "sample_with_guidance\|predictor_corrector\|sample_hybrid\|pc_sde\|_compute_score" src/ tests/ scripts/`
Expected: no output.

- [ ] **Step 3: Full suite**

Run: `uv run pytest tests/ -q` → 105 passed, 2 skipped.

- [ ] **Step 4: Commit**

```bash
git add src/bridge_diffusion/sampling/sampler.py
git commit -m "refactor(sampling): delete dead sampler variants (PC, hybrid, PC-SDE, guidance stub)"
```

---

### Task 4: Extract the `_batched` helper

**Files:**
- Modify: `src/bridge_diffusion/sampling/sampler.py` — `sample_batch`, `sample_batch_ode`; add `_batched` below `_solver_step`.

**Interfaces:**
- Produces: `Sampler._batched(sample_fn: Callable[[int, int], torch.Tensor], total_samples: int, batch_size: int) -> torch.Tensor` where `sample_fn(start, n)` returns `n` samples for the chunk beginning at index `start`; results are moved to CPU and concatenated.

- [ ] **Step 1: Add `_batched`**

```python
    def _batched(
        self,
        sample_fn,
        total_samples: int,
        batch_size: int,
    ) -> torch.Tensor:
        """Generate total_samples in chunks of batch_size via sample_fn(start, n)."""
        all_samples = []
        start = 0
        while start < total_samples:
            n = min(batch_size, total_samples - start)
            all_samples.append(sample_fn(start, n).cpu())
            start += n
        return torch.cat(all_samples, dim=0)
```

- [ ] **Step 2: Rewrite `sample_batch` body** (keep signature and the existing `x0` size ValueError):

```python
        if x0 is not None and x0.shape[0] < total_samples:
            raise ValueError(f"x0 has {x0.shape[0]} samples but total_samples={total_samples}")

        def sample_fn(start: int, n: int) -> torch.Tensor:
            x0_batch = x0[start : start + n] if x0 is not None else None
            return self.sample(n, shape, x0=x0_batch, num_steps=num_steps)

        return self._batched(sample_fn, total_samples, batch_size)
```

- [ ] **Step 3: Rewrite `sample_batch_ode` body** (keep signature):

```python
        torchdiffeq_solvers = {ODESolver.DOPRI5, ODESolver.DOPRI8, ODESolver.ADAPTIVE_HEUN}

        def sample_fn(start: int, n: int) -> torch.Tensor:
            if solver in torchdiffeq_solvers:
                return self.sample_ode_torchdiffeq(
                    n, shape, num_steps=num_steps, solver=solver.value, rtol=rtol, atol=atol
                )
            return self.sample_ode(n, shape, num_steps=num_steps, solver=solver)

        return self._batched(sample_fn, total_samples, batch_size)
```

- [ ] **Step 4: Full suite** — `uv run pytest tests/ -q` → 105 passed, 2 skipped. The `TestSampleBatchX0` tests verify chunking semantics survived.

- [ ] **Step 5: Commit**

```bash
git add src/bridge_diffusion/sampling/sampler.py
git commit -m "refactor(sampling): single _batched helper for both batch methods"
```

---

### Task 5: Move image I/O to `sampling/io.py`

**Files:**
- Create: `src/bridge_diffusion/sampling/io.py`
- Modify: `src/bridge_diffusion/sampling/sampler.py` (delete `save_samples`, `save_grid` methods), `src/bridge_diffusion/sampling/__init__.py`, `src/bridge_diffusion/cli.py:251-252`
- Test: `tests/test_sampling.py` (unchanged — these methods had no tests)

**Interfaces:**
- Produces: `bridge_diffusion.sampling.save_samples(samples: torch.Tensor, output_dir: Path, prefix: str = "sample") -> None` and `bridge_diffusion.sampling.save_grid(samples: torch.Tensor, output_path: Path, nrow: int = 8) -> None`. Task 6 consumes both.

- [ ] **Step 1: Create `src/bridge_diffusion/sampling/io.py`** (bodies moved verbatim from the deleted methods; they never used `self`):

```python
"""Saving generated samples as images."""

import logging
from pathlib import Path

import torch

logger = logging.getLogger(__name__)


def save_samples(
    samples: torch.Tensor,
    output_dir: Path,
    prefix: str = "sample",
) -> None:
    """Save generated samples as individual images.

    Args:
        samples: Samples of shape (num_samples, channels, height, width) in [-1, 1].
        output_dir: Directory to save images.
        prefix: Prefix for filenames.
    """
    from torchvision.utils import save_image

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    samples = (samples + 1) / 2
    samples = torch.clamp(samples, 0, 1)

    for i, sample in enumerate(samples):
        path = output_dir / f"{prefix}_{i:04d}.png"
        save_image(sample, path)

    logger.info(f"Saved {len(samples)} samples to {output_dir}")


def save_grid(
    samples: torch.Tensor,
    output_path: Path,
    nrow: int = 8,
) -> None:
    """Save samples as a grid image.

    Args:
        samples: Samples of shape (num_samples, channels, height, width) in [-1, 1].
        output_path: Path for output image.
        nrow: Number of images per row.
    """
    from torchvision.utils import make_grid, save_image

    samples = (samples + 1) / 2
    samples = torch.clamp(samples, 0, 1)

    grid = make_grid(samples, nrow=nrow, padding=2, normalize=False)
    save_image(grid, output_path)

    logger.info(f"Saved sample grid to {output_path}")
```

- [ ] **Step 2: Delete `save_samples` and `save_grid` methods from `sampler.py`** (and the now-unused `from pathlib import Path` import if nothing else in the file uses `Path` — check with grep first).

- [ ] **Step 3: Re-export from `src/bridge_diffusion/sampling/__init__.py`** — add:

```python
from bridge_diffusion.sampling.io import save_grid, save_samples
```

and add `"save_grid", "save_samples"` to `__all__` if the file defines one.

- [ ] **Step 4: Update `cli.py`** — in `sample_main`'s else-branch replace:

```python
            sampler.save_samples(samples, output_dir)
        sampler.save_grid(samples[:64], output_dir.parent / f"{output_dir.name}_grid.png")
```

with:

```python
            save_samples(samples, output_dir)
        save_grid(samples[:64], output_dir.parent / f"{output_dir.name}_grid.png")
```

and change the import at `cli.py:12` to `from bridge_diffusion.sampling import Sampler, save_grid, save_samples`.

- [ ] **Step 5: Full suite + import smoke check**

Run: `uv run pytest tests/ -q` → 105 passed, 2 skipped.
Run: `uv run python -c "from bridge_diffusion.sampling import save_samples, save_grid, Sampler, ODESolver; print('ok')"` → `ok`.

- [ ] **Step 6: Commit**

```bash
git add src/bridge_diffusion/sampling/ src/bridge_diffusion/cli.py
git commit -m "refactor(sampling): move image saving out of Sampler into sampling.io"
```

---

### Task 6: Fix the DDPM sampling bug in the CLI

`sample_main` currently routes `method="ddpm"` through the bridge `Sampler` (drift assumes the net predicts `E[Y|ξ]`; DDPM predicts noise ε → garbage). `DDPMDiffusion.sample()` — the correct diffusers reverse process — is never called.

**Files:**
- Modify: `src/bridge_diffusion/cli.py` (new `_generate_ddpm_samples` function + `elif` branch in `sample_main`)
- Test: create `tests/test_cli.py`

**Interfaces:**
- Consumes: `save_samples`/`save_grid` from Task 5; `DDPMDiffusion.sample(num_samples, shape, device, num_inference_steps)` (exists, `models/ddpm.py:98`).
- Produces: `_generate_ddpm_samples(model, num_samples: int, shape: tuple[int, ...], batch_size: int, num_inference_steps: int, device: torch.device) -> torch.Tensor` in `cli.py`.

- [ ] **Step 1: Write the failing test** — `tests/test_cli.py`:

```python
"""Tests for CLI helpers."""

import torch

from bridge_diffusion.cli import _generate_ddpm_samples
from bridge_diffusion.config import BridgeConfig, ModelConfig
from bridge_diffusion.models import DDPMDiffusion, DiffusersUNetWrapper


def _tiny_ddpm() -> DDPMDiffusion:
    model_config = ModelConfig(
        in_channels=1,
        out_channels=1,
        sample_size=8,
        block_out_channels=(32, 64),
        layers_per_block=1,
        down_block_types=("DownBlock2D", "DownBlock2D"),
        up_block_types=("UpBlock2D", "UpBlock2D"),
    )
    network = DiffusersUNetWrapper(model_config)
    return DDPMDiffusion(network, BridgeConfig(), num_train_timesteps=10)


def test_generate_ddpm_samples_shape_and_batching() -> None:
    model = _tiny_ddpm()
    samples = _generate_ddpm_samples(
        model,
        num_samples=3,
        shape=(1, 8, 8),
        batch_size=2,
        num_inference_steps=4,
        device=torch.device("cpu"),
    )
    assert samples.shape == (3, 1, 8, 8)
    assert samples.device.type == "cpu"
```

- [ ] **Step 2: Run it — expect FAIL**

Run: `uv run pytest tests/test_cli.py -v`
Expected: `ImportError: cannot import name '_generate_ddpm_samples'`.

- [ ] **Step 3: Implement.** Add to `cli.py` after `_load_source_images`:

```python
def _generate_ddpm_samples(
    model: DDPMDiffusion,
    num_samples: int,
    shape: tuple[int, ...],
    batch_size: int,
    num_inference_steps: int,
    device: torch.device,
) -> torch.Tensor:
    """Generate samples with the DDPM reverse process, in batches."""
    model = model.to(device)
    model.eval()
    all_samples = []
    for start in range(0, num_samples, batch_size):
        n = min(batch_size, num_samples - start)
        samples = model.sample(
            n, shape, device=device, num_inference_steps=num_inference_steps
        )
        all_samples.append(samples.cpu())
    return torch.cat(all_samples, dim=0)
```

Then in `sample_main`, between the `poisson_bridge` branch and the `else`, insert:

```python
    elif config.method == "ddpm":
        if x0 is not None:
            logger.warning("ddpm does not support source images; ignoring source priors.")
        logger.info(
            f"Generating {args.num_samples} samples with DDPM reverse process "
            f"({args.num_steps} steps)..."
        )
        samples = _generate_ddpm_samples(
            model,
            num_samples=args.num_samples,
            shape=shape,
            batch_size=args.batch_size,
            num_inference_steps=args.num_steps,
            device=device,
        )
        output_dir = Path(args.output_dir)
        save_samples(samples, output_dir)
        save_grid(samples[:64], output_dir.parent / f"{output_dir.name}_grid.png")
```

- [ ] **Step 4: Run tests — expect PASS**

Run: `uv run pytest tests/test_cli.py -v` → PASS; then `uv run pytest tests/ -q` → 106 passed, 2 skipped.

- [ ] **Step 5: Commit**

```bash
git add src/bridge_diffusion/cli.py tests/test_cli.py
git commit -m "fix(cli): sample DDPM checkpoints with the DDPM reverse process, not the bridge SDE"
```

---

### Task 7: Deduplicate `cli.py` config-override block

**Files:**
- Modify: `src/bridge_diffusion/cli.py:100-103` and `:147-149`

**Interfaces:**
- Produces: `_apply_data_info(config: ExperimentConfig, data_info: dict) -> None` in `cli.py`.

- [ ] **Step 1: Add helper after `create_model`:**

```python
def _apply_data_info(config: ExperimentConfig, data_info: dict) -> None:
    """Override model config with the dataset's actual shape."""
    config.model.in_channels = data_info["num_channels"]
    config.model.out_channels = data_info["num_channels"]
    config.model.sample_size = data_info["image_size"]
```

Replace both three-line blocks (in `train_main` and `sample_main`) with `_apply_data_info(config, data_info)`.

- [ ] **Step 2: Full suite** — `uv run pytest tests/ -q` → 106 passed, 2 skipped.

- [ ] **Step 3: Commit**

```bash
git add src/bridge_diffusion/cli.py
git commit -m "refactor(cli): extract _apply_data_info helper"
```

---

### Task 8: Poisson `y ≥ x` assumption warning + dead-code deletion in models

**Files:**
- Modify: `src/bridge_diffusion/models/poisson_bridge.py` (warning + docstring), `src/bridge_diffusion/models/bridge.py` (delete `compute_training_target`, lines 111-138)
- Test: `tests/test_poisson_bridge.py` (append one test)

**Interfaces:**
- Consumes: nothing new. Produces: a module-level `logger` in `poisson_bridge.py`.

- [ ] **Step 1: Write the failing test** — append to `tests/test_poisson_bridge.py` (match its existing fixture style; construct the model the same way neighboring tests do):

```python
def test_warns_when_prior_exceeds_data(caplog) -> None:
    """The paper requires y >= x coordinatewise; violating it should warn."""
    import logging

    model = _make_model(prior="zeros")  # use the file's existing model factory/fixture
    x = torch.full((2, 1, 4, 4), 5.0)
    y = torch.zeros(2, 1, 4, 4)
    with caplog.at_level(logging.WARNING, logger="bridge_diffusion.models.poisson_bridge"):
        model.compute_training_loss(x, y)
    assert any("y >= x" in r.message for r in caplog.records)
```

(If `tests/test_poisson_bridge.py` has no factory named `_make_model`, construct `PoissonBridgeDiffusion` exactly as the file's other tests do — reuse their network stub and `PoissonBridgeConfig`.)

- [ ] **Step 2: Run — expect FAIL** (`assert any(...)` fails, no warning emitted).

Run: `uv run pytest tests/test_poisson_bridge.py -k warns -v`

- [ ] **Step 3: Implement.** In `poisson_bridge.py`: add after the imports:

```python
logger = logging.getLogger(__name__)
```

(plus `import logging` at the top). In `compute_training_loss`, before `xi_t = self.sample_bridge(x, y, t)`:

```python
        if (y < x).any():
            logger.warning(
                "Poisson bridge assumes y >= x coordinatewise (paper_v2 §5, "
                "y in x + N_0^n); %d coordinates violate this and will be "
                "clamped, so generation cannot reach them from above.",
                int((y < x).sum()),
            )
```

Also extend the class docstring's prior note: `prior="poisson"` is only valid when the data dominates the prior coordinatewise.

- [ ] **Step 4: Delete `BridgeDiffusion.compute_training_target`** (`bridge.py:111-138`; no callers — verify with `grep -rn "compute_training_target" src/ tests/ scripts/` → only the definition).

- [ ] **Step 5: Run — expect PASS, then full suite**

Run: `uv run pytest tests/test_poisson_bridge.py -q` then `uv run pytest tests/ -q` → 107 passed, 2 skipped.

- [ ] **Step 6: Commit**

```bash
git add src/bridge_diffusion/models/poisson_bridge.py src/bridge_diffusion/models/bridge.py tests/test_poisson_bridge.py
git commit -m "fix(models): warn when Poisson bridge y>=x assumption is violated; drop dead compute_training_target"
```

---

### Task 9: Update paper references to paper_v2 anchors

Docstrings cite the predecessor paper's numbering (Algorithm 2.2.1/2.2.2, Corollary 2.7/2.9, Proposition 4.1). Update to paper_v2 (`paper_v2/paper_v2.tex`) labels. Comment-only task; no behavior change.

**Files:**
- Modify: `src/bridge_diffusion/models/bridge.py`, `src/bridge_diffusion/sampling/sampler.py`, `src/bridge_diffusion/training/trainer.py`

**Interfaces:** none.

- [ ] **Step 1: Apply the mapping everywhere these strings appear** (grep for `2.2.1|2.2.2|Corollary 2.7|Corollary 2.9|Proposition 4.1` in src/):

| Old reference | New reference |
|---|---|
| Algorithm 2.2.1 (Training) | Algorithm 1, Gaussian Bridge Training (paper_v2 §8) |
| Algorithm 2.2.2 (Simulation) | Algorithm 2, Gaussian Bridge Simulation (paper_v2 §8) |
| Corollary 2.7 | the MSE-minimiser identity E[Y|ξ_t] (paper_v2 §8, eq. for L^f) |
| Corollary 2.9 | Algorithm 2 (paper_v2 §8) |
| Proposition 4.1 | Prop. prop:bridge_drift (paper_v2 §4) |
| "the ODE is:" comment in `_ode_drift` | add "Prop. cor:prob_flow_ode (paper_v2 §4)" |

- [ ] **Step 2: Verify no stale refs remain**

Run: `grep -rn "2\.2\.1\|2\.2\.2\|Corollary 2\.\|Proposition 4\.1" src/`
Expected: no output.

- [ ] **Step 3: Full suite** — `uv run pytest tests/ -q` → 107 passed, 2 skipped.

- [ ] **Step 4: Commit**

```bash
git add src/bridge_diffusion/models/bridge.py src/bridge_diffusion/sampling/sampler.py src/bridge_diffusion/training/trainer.py
git commit -m "docs(code): update paper citations to paper_v2 labels"
```

---

### Task 10: Small interface cleanups (DDPM config, trainer hint, LR metric)

**Files:**
- Modify: `src/bridge_diffusion/models/ddpm.py:21-35`, `src/bridge_diffusion/cli.py:29-35`, `src/bridge_diffusion/training/trainer.py:19,47,264`
- Test: `tests/test_cli.py` (Task 6's test already constructs `DDPMDiffusion` — update it), full suite for the rest.

**Interfaces:**
- Produces: `DDPMDiffusion.__init__(network: nn.Module, num_train_timesteps: int = 1000, beta_schedule: str = "linear")` — the unused `config` parameter is REMOVED. `Trainer.__init__` takes `model: nn.Module`.

- [ ] **Step 1: `ddpm.py`** — drop the `config: BridgeConfig` parameter and its `from bridge_diffusion.config import BridgeConfig` import; delete the "(used for compatibility, T not used)" docstring line.

- [ ] **Step 2: `cli.py` `create_model`** — the ddpm branch becomes:

```python
    elif config.method == "ddpm":
        return DDPMDiffusion(
            network,
            num_train_timesteps=config.ddpm.num_train_timesteps,
            beta_schedule=config.ddpm.beta_schedule,
        )
```

- [ ] **Step 3: `tests/test_cli.py`** — update `_tiny_ddpm`: `return DDPMDiffusion(network, num_train_timesteps=10)` and drop the now-unused `BridgeConfig` import.

- [ ] **Step 4: `trainer.py`** — change the hint at line 47 to `model: nn.Module,`; delete the `from bridge_diffusion.models import BridgeDiffusion` import (line 19) if nothing else uses it (`grep -n "BridgeDiffusion" src/bridge_diffusion/training/trainer.py` → only the import and hint); update the docstring line "model: Bridge diffusion model to train." → "model: Diffusion model exposing compute_training_loss (bridge, Poisson bridge, or DDPM)."; change line 264 to `current_lr = self.optimiser.param_groups[0]["lr"]`.

- [ ] **Step 5: Full suite** — `uv run pytest tests/ -q` → 107 passed, 2 skipped.

- [ ] **Step 6: Commit**

```bash
git add src/bridge_diffusion/models/ddpm.py src/bridge_diffusion/cli.py src/bridge_diffusion/training/trainer.py tests/test_cli.py
git commit -m "refactor: drop DDPM's unused config param; honest Trainer typing and LR metric"
```

---

### Task 11: Deduplicate `evaluation/metrics.py`

**Files:**
- Modify: `src/bridge_diffusion/evaluation/metrics.py`

**Interfaces:**
- Produces (module-private): `_require_torch_fidelity()` returning the imported module, `_to_uint8(images: torch.Tensor) -> torch.Tensor`.

- [ ] **Step 1: Add helpers after the module logger:**

```python
def _require_torch_fidelity():
    """Import torch_fidelity or fail with an actionable message."""
    try:
        import torch_fidelity
    except ImportError:
        logger.error("torch-fidelity not installed. Install with: pip install torch-fidelity")
        raise
    return torch_fidelity


def _to_uint8(images: torch.Tensor) -> torch.Tensor:
    """Convert [-1, 1] or [0, 1] float images to [0, 255] uint8 (NCHW)."""
    if images.min() < 0:
        images = (images + 1) / 2
    return (images * 255).clamp(0, 255).to(torch.uint8)
```

- [ ] **Step 2: Use them in all three `compute_fid*` functions** — replace each try/except import block with `torch_fidelity = _require_torch_fidelity()`, replace `compute_fid`'s inner `prepare_images` and `compute_fid_against_dataset`'s inline conversion (lines 124-126) with `_to_uint8(...)` calls.

- [ ] **Step 3: Full suite** — `uv run pytest tests/ -q` → 107 passed, 2 skipped (metrics has no tests; this guards imports/syntax).

- [ ] **Step 4: Commit**

```bash
git add src/bridge_diffusion/evaluation/metrics.py
git commit -m "refactor(metrics): shared torch-fidelity import and uint8 conversion helpers"
```

---

### Task 12: Final verification

**Files:** none (verification only).

- [ ] **Step 1: Line-count check**

Run: `wc -l src/bridge_diffusion/sampling/sampler.py src/bridge_diffusion/sampling/io.py`
Expected: sampler.py ≈ 330–400 lines (from 1,008), io.py ≈ 60.

- [ ] **Step 2: Full suite, verbose failures only**

Run: `uv run pytest tests/ -q` → 107 passed, 2 skipped.

- [ ] **Step 3: Public-API smoke check** (everything external callers import):

Run:
```bash
uv run python -c "
from bridge_diffusion.sampling import Sampler, ODESolver, save_samples, save_grid
from bridge_diffusion import BridgeDiffusion, DDPMDiffusion, PoissonBridgeDiffusion
import inspect
assert {'sample','sample_batch','sample_ode','sample_batch_ode','sample_ode_torchdiffeq'} <= {m for m,_ in inspect.getmembers(Sampler, inspect.isfunction)}
print('API ok')
"
```
Expected: `API ok`.

- [ ] **Step 4: Script imports still resolve** (scripts call `sample_batch_ode`/`sample_ode`/`ODESolver`):

Run: `uv run python -c "import ast,sys; [ast.parse(open(f).read()) for f in ['scripts/evaluate_fid.py','scripts/visualize_ode.py','scripts/visualize_sde.py']]; print('scripts parse ok')"`
Then: `grep -n "sampler\.\|ODESolver" scripts/evaluate_fid.py scripts/visualize_ode.py scripts/visualize_sde.py` and confirm every referenced method still exists in `sampler.py`.

- [ ] **Step 5: Report** — summarize commits and line deltas to the user; no push without an explicit ask.
