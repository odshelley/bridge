# Sampler refactor + review fixes — design

**Date:** 2026-07-06 · **Branch:** `refactor/paper-v2-review` · **Scope:** `src/bridge_diffusion`, `tests/`

## Context

A line-by-line verification of the implementation against `paper_v2/paper_v2.tex`
confirmed the algorithms are correct: the Brownian bridge statistics and training
loss match Prop `prop:bridge_stats` and Algorithm 1; the SDE simulation matches
Algorithm 2 (with σ = 1); the probability-flow ODE drift
`½(ξ−x₀)/t + ½(ŷ−ξ)/(T−t)` matches Prop `cor:prob_flow_ode`; the score
rearrangement matches Prop `prop:bridge_drift`; the Poisson Binomial-bridge
training and thinning simulation match Algorithms 3–4.

The review found one real bug, substantial dead/duplicated code, and stale
paper references. This design covers the cleanup.

## Findings driving the design

1. **BUG (high): DDPM sampled with the bridge SDE.** `cli.py sample_main`
   routes `method="ddpm"` through the generic `Sampler`, whose drift assumes
   the network predicts `E[Y|ξ_t]`; DDPM predicts noise ε. The correct
   `DDPMDiffusion.sample()` is never called anywhere.
2. **Dead code (~450 lines):** `Sampler.sample_with_guidance`,
   `sample_predictor_corrector`, `sample_hybrid`, `sample_pc_sde` and their
   three `sample_batch_*` wrappers have zero callers in src/, scripts/, tests/
   and the untracked experiments/. `BridgeDiffusion.compute_training_target`
   is also uncalled.
3. **Duplication:** the Euler/Heun/RK4 stepping block is copied 4×; the
   batch-loop wrapper is copied 4×; `cli.py` duplicates its config-override
   and save-images blocks; `metrics.py` triplicates torch-fidelity import
   boilerplate and uint8 conversion.
4. **Poisson prior assumption:** the paper requires `y ≥ x` coordinatewise
   (`y ∈ x + ℕ₀ⁿ`). `prior="poisson"` can violate this; the
   `(y−x).clamp(min=0)` silently masks it and generation can never move a
   coordinate downward.
5. **Stale references:** docstrings cite the predecessor paper's numbering
   (Algorithm 2.2.1/2.2.2, Corollary 2.7/2.9, Proposition 4.1). paper_v2
   anchors are Algorithms 1–4 (§8), `prop:bridge_drift`, `cor:prob_flow_ode`.
6. **Test gaps:** no tests for the ODE path, cli.py, metrics.py, ddpm.py.

## Approaches considered

- **A (chosen) — dedup + delete, single module.** Remove dead methods, extract
  shared helpers, move image I/O out of the sampler, fix the DDPM bug, update
  references. Public API of live methods unchanged.
- **B — package split keeping all 12 methods.** Rejected: organizes dead code
  instead of deleting it (YAGNI; git history preserves the experiments).
- **C — new unified `sample(method=...)` API.** Rejected for now: churns all
  callers (cli, evaluate_fid, visualize scripts, tests) for no behavior gain.
  Can be layered on later.

## Design

### sampling/sampler.py (~350 lines after)

- Keep: `ODESolver`, `Sampler.sample`, `sample_batch`, `sample_ode`,
  `sample_batch_ode`, `sample_ode_torchdiffeq`, `_ode_drift`.
- Delete: `sample_with_guidance`, `sample_predictor_corrector`,
  `sample_batch_predictor_corrector`, `sample_hybrid`, `sample_batch_hybrid`,
  `sample_pc_sde`, `sample_batch_pc_sde`, `save_samples`, `save_grid`, and
  `_compute_score` (its only consumer was the deleted Langevin corrector).
- New private helper `_solver_step(drift_fn, xi, t, dt, solver) -> xi` holding
  the Euler/Heun/RK4 logic once; `sample_ode` consumes it.
- New private helper `_batched(fn, total_samples, batch_size, **kw)` used by
  `sample_batch` and `sample_batch_ode`.
- Docstrings cite paper_v2 anchors.

### sampling/io.py (new, ~60 lines)

Free functions `save_samples(samples, output_dir, prefix)` and
`save_grid(samples, output_path, nrow)` — moved verbatim from `Sampler`
(they never used `self`). `cli.py` switches to these.

### cli.py

- **Fix BUG-1:** `sample_main` gets a `method == "ddpm"` branch calling
  `model.sample(...)` in batches (mirrors the existing poisson branch).
- Extract `_apply_data_info(config, data_info)` (used by train + sample).
- Use `sampling.io` save helpers.

### models/

- `bridge.py`: delete `compute_training_target`; update paper references.
- `poisson_bridge.py`: in `compute_training_loss`, warn once (module-level
  logger) if `(y < x).any()` — documents the `y ≥ x` model assumption without
  changing behavior; docstring notes that `prior="poisson"` is only valid when
  data dominates the prior. Update paper references.
- `ddpm.py`: remove the unused `config` parameter (callers updated).

### training/trainer.py

- Type hint `model: nn.Module` (duck-typed on `compute_training_loss`).
- Log `learning_rate` from `optimizer.param_groups[0]["lr"]`.

### evaluation/metrics.py

- Extract `_require_torch_fidelity()` and `_to_uint8(images)`.

### Out of scope

Exposing σ as a config knob, exposing ODE sampling in the CLI, new tests for
metrics/cli beyond the DDPM-branch smoke test, and any experiment code.

## Error handling

No new error paths; the Poisson guard is a warning, not an exception, to avoid
breaking existing training runs. `ValueError` on unknown solver moves into
`_solver_step` (previously unknown solvers fell through silently — minor
robustness gain).

## Testing

1. **Before refactor:** add characterization tests for `sample_ode` (all three
   solvers): output shape, determinism under fixed seed, identity-network
   invariant (net predicting ŷ = ξ from ξ = x₀ start gives zero ODE drift, so
   output == clamp(x₀)), and `sample_batch_ode` routing/shape.
2. Full `pytest` run before and after; all existing tests must stay green
   unmodified except imports of moved I/O helpers (none exist in tests).
3. DDPM CLI fix verified by a small end-to-end smoke test constructing a tiny
   DDPM model and asserting `sample_main`'s branch calls `model.sample`.
