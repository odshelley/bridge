# Audit Fixes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task.

**Goal:** Fix the 3 critical and 7 important findings from the 2026-07-06 multi-agent audit of main, and add the missing automated E2E training test.

**Architecture:** All fixes are local and independent: DDPM model contract, config validation, FID normalisation, trainer sample-grid/RNG handling, sampler time-grid clamping, CLI/dataloader guards, plus two new test files. No public API breaks except `compute_fid` gaining an explicit `input_range` parameter.

**Tech Stack:** PyTorch, pytest, uv.

## Global Constraints

- Run everything via `uv` from the repo root: `uv run pytest -m "not slow"`, `uv run ruff check src/ tests/`, `uv run black --check src/ tests/`.
- Line length 100, Python 3.10+, Google-style docstrings with types, type hints required.
- The trainer's calling convention is fixed and shared by all methods: `compute_training_loss(x=prior, y=data)`. Do not change the trainer call site (trainer.py:247).
- TDD per task: failing test first, then fix. Pre-existing ruff/black debt exists; add no new violations.
- Do not refactor beyond what each task names.

---

### Task 1: DDPM wiring fixes

**Files:**
- Modify: `src/bridge_diffusion/models/ddpm.py` (compute_training_loss lines 43-75; add generate after sample)
- Modify: `src/bridge_diffusion/training/trainer.py` (docstring lines 54-58 only)
- Test: `tests/test_ddpm.py` (new file)

**Requirements:**
1. `DDPMDiffusion.compute_training_loss(x, y)`: `y` is the data tensor (noised via `self.scheduler.add_noise(y, noise, t)`); `x` is unused, kept for interface compatibility (prior). Update docstring accordingly.
2. Add:
```python
@torch.no_grad()
def generate(self, x: torch.Tensor, num_steps: int | None = None) -> torch.Tensor:
    """Generate samples from noise, matching the bridge models' interface.

    Args:
        x: Prior noise of shape (batch, channels, height, width); only its
            shape and device are used (DDPM sampling draws its own noise).
        num_steps: Number of inference steps (None = all training steps).

    Returns:
        Generated samples with the same shape as x.
    """
    return self.sample(
        num_samples=x.shape[0],
        shape=tuple(x.shape[1:]),
        device=x.device,
        num_inference_steps=num_steps,
    )
```
3. In `Trainer.__init__` docstring, replace the "DDPM training is not currently wired correctly — see evaluate/sample for DDPM inference" clause: the convention now holds for all three methods.
4. Tests (tiny network, e.g. the DiffusersUNetWrapper pattern from tests/test_trainer.py `_tiny_model`, or a 1-layer conv stub module with signature `net(x, t)`):
   - loss depends on `y`, not `x`: with `torch.manual_seed` fixed before each call, `compute_training_loss(x1, y)` == `compute_training_loss(x2, y)` for different x1/x2, and != for different y.
   - `generate(torch.randn(2, 1, 8, 8), num_steps=2)` returns shape (2, 1, 8, 8), finite.

**Commit:** `fix: DDPM trains on data not noise; add generate() for trainer sampling`

---

### Task 2: Config validation + stale config removal

**Files:**
- Delete: `configs/cifar10 2.yaml`, `configs/mnist 2.yaml`
- Modify: `src/bridge_diffusion/config/config.py` (`ExperimentConfig.__post_init__` lines 150-165; `from_yaml` lines 167-199)
- Test: `tests/test_config.py`

**Requirements:**
1. `__post_init__` additions (after the existing transport guard):
   - `method == "poisson_bridge"` and not `self.data.raw_pixels` → `ValueError("method='poisson_bridge' requires data.raw_pixels=true (integer pixel counts)")`
   - `method != "poisson_bridge"` and `self.data.raw_pixels` → `ValueError(f"data.raw_pixels=true requires method='poisson_bridge', got method='{self.method}'")`
2. `from_yaml`: define `known = {"name", "output_dir", "method", "mlflow_tracking_uri", "model", "training", "bridge", "poisson_bridge", "ddpm", "sampling", "data"}`; `unknown = set(data) - known`; if unknown, `raise ValueError(f"Unknown top-level config keys in {path}: {sorted(unknown)}")`.
3. Tests: both new validation errors (match=), unknown-key error, and a parametrized test that every `configs/*.yaml` loads via `ExperimentConfig.from_yaml` without error (glob the configs dir).
4. Check existing tests/configs for violations of the new rules (e.g. any test building poisson configs without raw_pixels) and update them to comply — they are now invalid configurations by design.

**Commit:** `fix: validate raw_pixels vs method, reject unknown config keys, drop stale configs`

---

### Task 3: Explicit FID input range

**Files:**
- Modify: `src/bridge_diffusion/evaluation/metrics.py` (`_to_uint8` lines 27-31, `compute_fid` lines 34-67, `compute_fid_against_dataset` if it uses `_to_uint8`)
- Test: `tests/test_metrics.py` (new file)

**Requirements:**
1. `_to_uint8(images: torch.Tensor, input_range: tuple[float, float]) -> torch.Tensor`: `lo, hi = input_range; images = (images - lo) / (hi - lo); return (images * 255).clamp(0, 255).to(torch.uint8)`. No min()-based guessing.
2. `compute_fid(..., input_range: tuple[float, float] = (-1.0, 1.0))` passes the same range to `_to_uint8` for BOTH real and generated tensors. Same for `compute_fid_against_dataset` if applicable (read the body; keep its dataset-side handling consistent).
3. Tests (no torch-fidelity import needed — test `_to_uint8` directly): a constant 0.6 tensor maps to the same uint8 value regardless of position/order; `input_range=(0, 255)` maps 255→255 and 0→0; `input_range=(-1, 1)` maps -1→0, 1→255.

**Commit:** `fix: explicit FID input_range instead of per-tensor guessing`

---

### Task 4: Trainer sample-grid normalisation + RNG checkpointing

**Files:**
- Modify: `src/bridge_diffusion/training/trainer.py` (`_log_samples` lines 136-169; `save_checkpoint` lines 287-307; `load_checkpoint` lines 309-324)
- Test: `tests/test_trainer.py`

**Requirements:**
1. Extract module-level:
```python
def _normalise_samples(samples: torch.Tensor, model: nn.Module) -> torch.Tensor:
    """Map generated samples to [0, 1] for image logging.

    Poisson models (exposing num_levels) output integer counts in
    [0, num_levels - 1]; others output [-1, 1] floats.
    """
    if hasattr(model, "num_levels"):
        return (samples / (model.num_levels - 1)).clamp(0, 1)
    return (samples.clamp(-1, 1) + 1) / 2
```
   `_log_samples` uses it (replacing lines 157-158), passing `model_to_sample`.
2. `save_checkpoint`: add `"rng_state": {"torch": torch.get_rng_state(), "numpy": np.random.get_state(), "python": random.getstate()}` plus `"cuda": torch.cuda.get_rng_state_all()` when `torch.cuda.is_available()`. Add the needed imports.
3. `load_checkpoint`: if `"rng_state"` present, restore all saved states (guard cuda restore with availability check).
4. Tests: `_normalise_samples` with a stub object exposing `num_levels=256` (128 → ~0.5) and with a plain module ([-1,1] path); RNG round-trip — build the minimal transport Trainer from the existing `_transport_experiment` helper, `save_checkpoint()`, record `torch.randn(3)`, then reseed differently, `load_checkpoint(path)`, and assert `torch.randn(3)` reproduces the recorded draw.

**Commit:** `fix: poisson-aware sample-grid normalisation; save/restore RNG state in checkpoints`

---

### Task 5: Sampler time clamp + CLI EMA warning + dataloader guard

**Files:**
- Modify: `src/bridge_diffusion/sampling/sampler.py` (drift_fn in sample_ode ~line 268; t_span in sample_ode_torchdiffeq ~line 329)
- Modify: `src/bridge_diffusion/cli.py` (EMA block lines 176-181)
- Modify: `src/bridge_diffusion/data/datasets.py` (`get_dataloader` lines 308-339)
- Test: `tests/test_sampling.py`, `tests/test_data.py`

**Requirements:**
1. In `sample_ode`'s `drift_fn`: `time = min(time, self.T - eps)` before building `time_tensor`/calling `_ode_drift` (eps is the local 1e-4). In `sample_ode_torchdiffeq`: `t_span = torch.linspace(eps, self.T - eps, num_steps + 1, ...)`.
2. `cli.py`: add `elif args.use_ema:` branch before the plain-weights fallback: `logger.warning("--use-ema requested but checkpoint has no EMA weights; using raw model weights")` (then still load raw weights).
3. `get_dataloader`: before constructing the DataLoader, `if train and len(dataset) < batch_size: raise ValueError(f"Training dataset has {len(dataset)} samples but batch_size={batch_size} with drop_last=True would yield zero batches")`.
4. Tests: HEUN and RK4 sampling with the existing tiny-model pattern from tests/test_sampling.py produce finite outputs (`torch.isfinite(out).all()`); dataloader guard raises on a tiny synthetic dataset (afhq_dir fixture has 12 train images — batch_size=64 triggers it) and existing loaders still work.

**Commit:** `fix: clamp ODE drift time away from T; warn on missing EMA; guard zero-batch dataloaders`

---

### Task 6: E2E training test + BridgeDiffusion unit tests

**Files:**
- Create: `tests/test_e2e.py`
- Create: `tests/test_bridge.py`
- Test-only task; no source changes. If a test exposes a real source bug, STOP and report BLOCKED with the failure — do not fix source yourself.

**Requirements:**
1. `tests/test_e2e.py`: parametrized over `method in ("bridge", "poisson_bridge", "ddpm")`. Build an `ExperimentConfig` in code (not YAML): 16px AFHQ (`afhq_dir` fixture), 2-block UNet (mirror `_tiny_model` in tests/test_trainer.py), `in_channels=3`, batch_size 2, `num_steps=3`, `checkpoint_every=2`, `log_every=1`, `use_ema=True`, `num_workers=0`, `mlflow_tracking_uri=f"sqlite:///{tmp_path}/mlflow.db"`, `output_dir=tmp_path`, poisson gets `raw_pixels=True` + `num_levels=256, prior="zeros"`. Run the real `Trainer(...).train()` (checkpoint_dir under tmp_path). Assert: training completes; a checkpoint file exists; reload it with `Trainer.load_checkpoint`; generate 2 samples via the model's `generate()` (2 steps) and assert shape + finiteness. Keep total runtime small (tiny model, 3 steps, 16px). NOTE: `_log_samples` generates 16 samples with num_steps=100 at each checkpoint — if that makes the test slow (>~60s per method), monkeypatch `Trainer._log_samples`' `num_samples` via `functools.partial`/lambda or patch `model.generate` to fewer steps, but the call itself MUST still execute (it is the regression target for the DDPM crash).
2. `tests/test_bridge.py`, direct `BridgeDiffusion` unit tests (construct with the tiny UNet or a stub net):
   - `compute_expectation(x, y, t)` equals `x + (y - x) * t / T` for hand-picked tensors.
   - `compute_variance(t)` equals `t * (T - t) / T`; zero at t=0 and t=T.
   - `sample_bridge` at t≈0 returns ≈x, at t≈T returns ≈y (tolerance for eps variance).
   - `compute_training_loss(x, y)` returns a finite scalar that backpropagates (loss.backward(); some param grad is non-None).
   - `generate(x, num_steps=4)` returns x's shape, finite.
   Read `src/bridge_diffusion/models/bridge.py` first and match the actual method names/signatures; if they differ from the above, adapt the tests to the real API.
3. Both files: no network access, no slow marker needed if total added runtime stays under ~90s.

**Commit:** `test: automated E2E training test for all methods; direct BridgeDiffusion unit tests`
