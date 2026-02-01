# Bridge Diffusion

Implementation of Gaussian Random Bridge Diffusion Models for generative image synthesis.

## Installation

```bash
uv sync
```

## Usage

### Training

```bash
uv run bridge-train --config configs/mnist.yaml
```

### Sampling

```bash
uv run bridge-sample --checkpoint path/to/model.pt --n-samples 1000 --steps 10
```

### Evaluation

```bash
uv run bridge-evaluate --samples-dir path/to/samples --dataset mnist
```

## Development

```bash
uv sync --extra dev
uv run pytest
uv run ruff check src/
uv run black src/
```
