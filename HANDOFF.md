# Gaussian Bridge Experiments: GPU Run Guide

Everything needed to run the five Gaussian-bridge experiments on a CUDA box
(A100/4090 class). All configs load, build, and forward-pass as of
2026-07-12 (post PR #4/#5 fixes); MNIST and CIFAR-10 were additionally
smoke-trained locally with healthy loss curves and samples.

## Setup (fresh machine)

```bash
git clone <repo-url> diffusion-bridge && cd diffusion-bridge
uv sync                       # CUDA-enabled torch resolves automatically on Linux
uv run python -c "import torch; print(torch.cuda.is_available())"   # expect True
```

Datasets:
- MNIST and CIFAR-10 auto-download on first run (needs outbound internet from
  the node; otherwise pre-copy a `data/` folder).
- AFHQ needs a one-time manual fetch **before** any `afhq_*` config:
  `bash scripts/download_afhq.sh`. The primary Dropbox URL can rot; fallback
  sources are listed in the script's comments.

Device selection is automatic (cuda > mps > cpu); there is no multi-GPU
support, so on a shared node set `CUDA_VISIBLE_DEVICES=<id>` per run.

## Checkpoint locations

Checkpoints default to `<output_dir>/checkpoints` (per-experiment, e.g.
`outputs/cifar10/checkpoints/`), so experiments never overwrite each other.
`--checkpoint-dir` overrides this if you need checkpoints on a different
filesystem.

## The five experiments

Run from the repo root. Runtimes are rough single-GPU estimates.

```bash
# 1. MNIST generation (~0.5-1h). Sanity anchor; digits should be clean.
uv run bridge-diffusion train --config configs/mnist.yaml

# 2. CIFAR-10 generation, Ho et al. recipe, 800k steps (~1.5-4 days).
#    The headline confirmatory FID number.
uv run bridge-diffusion train --config configs/cifar10.yaml

# 3. AFHQ all-classes generation at 64px, 150k steps (~1-2 days).
#    ~22GB of checkpoints at the default cadence; ensure disk.
uv run bridge-diffusion train --config configs/afhq_64.yaml

# 4. CIFAR cat->dog transport, 50k steps (~0.5-1 day).
uv run bridge-diffusion train --config configs/cifar_cat2dog.yaml

# 5. AFHQ cat->dog transport at 64px, 150k steps (~1-2 days).
#    The headline transport experiment.
uv run bridge-diffusion train --config configs/afhq_cat2dog_64.yaml
```

Resume after an interruption (restores model, EMA, optimizer, step, and RNG
state exactly):

```bash
uv run bridge-diffusion train --config <same config> \
    --resume <output_dir>/checkpoints/checkpoint_step_<N>.pt
```

Note: a resumed run appears as a *new* MLflow run (same name); the metric
timeline does not stitch onto the original.

## What gets tracked

Each experiment logs to its own MLflow sqlite store at
`<output_dir>/mlflow.db` (the mnist-family configs share
`outputs/mlflow.db`, separated by experiment name -- do not run two
mnist-family configs concurrently or sqlite will lock).

- Hyperparameters: logged once at run start.
- `train_loss` (100-step mean) and `learning_rate`: every 100 steps.
- Sample grid PNG (16 images, EMA weights, 100-step Euler-Maruyama): at every
  checkpoint interval, under Artifacts -> `samples/step_N/`.
- Checkpoints: every `checkpoint_every` steps + final, each containing model,
  EMA, optimizer, config, and RNG state.

View live: `uv run mlflow ui --backend-store-uri sqlite:///<output_dir>/mlflow.db`

**Do not judge early runs by the sample grids.** They are generated from EMA
weights (decay 0.9999), which remain dominated by the random initialisation
for the first ~20-50k steps; grids look far worse than the model actually is.
Loss curve first; for an honest early visual, sample the checkpoint without
`--use-ema`.

Two loss facts that look like bugs but are not: the loss has a large
irreducible floor (the bridge objective predicts data from mostly-noise at
small t), so it plateaus early while samples keep improving; and it will
never approach zero.

## Evaluation

Generation runs (1-3): sample, export a reference set, compute FID --

```bash
uv run bridge-diffusion sample --checkpoint outputs/cifar10/checkpoints/checkpoint_final.pt \
    --use-ema --num-samples 50000 --num-steps 100 --batch-size 256 \
    --output-dir outputs/cifar10_samples
uv run python scripts/export_val_images.py --dataset cifar10 \
    --image-size 32 --out data/fid_ref/cifar10
uv run bridge-diffusion evaluate --real-dir data/fid_ref/cifar10 \
    --generated-dir outputs/cifar10_samples
```

Transport runs (4-5): translate held-out source images (the CLI loads val
cats automatically for a transport checkpoint, or pass `--source-dir`), then
FID against the target class:

```bash
uv run bridge-diffusion sample --checkpoint outputs/afhq_cat2dog_64/checkpoints/checkpoint_final.pt \
    --use-ema --num-samples 500 --num-steps 100 --output-dir outputs/afhq_translations
uv run python scripts/export_val_images.py --dataset afhq --classes dog \
    --image-size 64 --out data/fid_ref/afhq_dog
uv run bridge-diffusion evaluate --real-dir data/fid_ref/afhq_dog \
    --generated-dir outputs/afhq_translations
```

`scripts/evaluate_fid.py` (the FID-vs-step-count sweep for the paper's
tables) handles both run types: for a transport checkpoint it automatically
translates source val images (tiling them when more samples than sources are
requested). Two caveats: transport + `--ode` is unsupported (the ODE batch
path has no source-image support and the script raises), and its FID
implementation (torchmetrics Inception + scipy) is not numerically identical
to `bridge-diffusion evaluate` (torch-fidelity) -- do not mix the two within
one comparison table.

## Performance notes

- Training is plain fp32 (no AMP); an A100/4090 will not be near peak. Known
  and accepted for correctness-first runs.
- `set_seed` enables `cudnn.deterministic`, costing roughly 10-30% throughput
  for bit-exact reproducibility.
- MNIST is dataloader-bound on a big GPU; low utilisation there is expected.
