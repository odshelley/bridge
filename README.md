# Bridge Diffusion

Implementation of Gaussian Random Bridge Diffusion Models for generative image synthesis.

## Installation

```bash
uv sync
```

## Usage

### Training

```bash
uv run bridge-diffusion train --config configs/mnist.yaml
```

### Sampling

```bash
uv run bridge-diffusion sample --checkpoint checkpoints/checkpoint_final.pt \
    --num-samples 1000 --num-steps 100 --batch-size 64 --use-ema --output-dir outputs/samples
```

### Evaluation

```bash
uv run bridge-diffusion evaluate --real-dir data/fid_ref/mnist --generated-dir outputs/samples
```

See [HANDOFF.md](HANDOFF.md) for a complete per-experiment run guide (GPU
sizing, checkpoint hygiene, metrics, and evaluation commands).

### Transport experiments (data -> data)

The bridge can transport between two image distributions instead of noise -> data.
Configure a `source_dataset` (and optional class filters) in the config's `data` block.

```bash
# CIFAR-10 cat -> dog (local-friendly)
uv run bridge-diffusion train --config configs/cifar_cat2dog.yaml

# AFHQ cat -> dog at 64x64 (needs a CUDA GPU)
bash scripts/download_afhq.sh
uv run bridge-diffusion train --config configs/afhq_cat2dog_64.yaml

# Translate held-out cats and evaluate FID against real dogs
uv run bridge-diffusion sample --checkpoint checkpoints/checkpoint_final.pt --use-ema \
    --num-samples 500 --num-steps 100 --output-dir outputs/afhq_translations
uv run python scripts/export_val_images.py --dataset afhq --classes dog \
    --image-size 64 --out data/fid_ref/afhq_dog
uv run bridge-diffusion evaluate --real-dir data/fid_ref/afhq_dog \
    --generated-dir outputs/afhq_translations
```

Use a fresh `--output-dir` (and a fresh `--out` for the exported reference images) for
each FID run: stale PNGs left over from a previous run will silently be counted by
`torch-fidelity` and distort the score.

Smoke-test configs (`*_smoke.yaml`) run the full pipeline in minutes on Apple Silicon.

## Development

```bash
uv sync --extra dev
uv run pytest
uv run ruff check src/
uv run black src/
```
