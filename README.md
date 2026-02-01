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
uv run bridge-diffusion sample --checkpoint outputs/checkpoints/model.pt --n-samples 1000 --steps 10
```

### Evaluation

```bash
uv run bridge-diffusion evaluate --samples-dir outputs/samples --dataset mnist
```

## Development

```bash
uv sync --extra dev
uv run pytest
uv run ruff check src/
uv run black src/
```
