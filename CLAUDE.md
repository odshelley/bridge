# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**diffusion-bridge** is a Python library implementing diffusion bridge generative models -- stochastic and deterministic transport methods that move samples between two arbitrary probability distributions. See [THEORY.md](THEORY.md) for the full mathematical background.

## Development Commands

**Important**: This repository uses `uv` for package management. If you have a conda environment active, deactivate it first:
```bash
conda deactivate
```

All Python commands should be run via `uv run`:
```bash
uv sync                  # Install dependencies
uv sync --extra dev      # Install with dev dependencies
```

### Training
```bash
uv run bridge-diffusion train --config configs/mnist.yaml
uv run bridge-diffusion train --config configs/mnist.yaml --checkpoint-dir checkpoints --resume checkpoints/model.pt
```

### Sampling/Generation
```bash
uv run bridge-diffusion sample --checkpoint outputs/checkpoints/model.pt --n-samples 1000 --steps 10
uv run bridge-diffusion sample --checkpoint checkpoints/model.pt --num-samples 64 --num-steps 100 --batch-size 64 --use-ema
```

### Evaluation
```bash
uv run bridge-diffusion evaluate --real-dir data/mnist --generated-dir outputs/samples
```

### Testing & Linting
```bash
uv run pytest                                    # Run all tests
uv run pytest tests/test_sampling.py             # Run specific test file
uv run pytest tests/test_sampling.py::test_name  # Run specific test
uv run pytest -v --tb=short                      # Verbose with short tracebacks
uv run ruff check src/                           # Lint code
uv run black src/                                # Format code
```

## Architecture

### Module Structure

```
src/bridge_diffusion/
├── models/
│   ├── bridge.py           # BridgeDiffusion: Core Gaussian bridge process
│   ├── poisson_bridge.py   # Poisson (discrete jump) bridge
│   ├── ddpm.py             # DDPMDiffusion: DDPM baseline comparison
│   └── diffusers_unet.py   # DiffusersUNetWrapper: UNet from HuggingFace diffusers
├── training/
│   └── trainer.py          # Training loop with MLflow tracking, EMA, checkpointing
├── sampling/
│   └── sampler.py          # Sampling with Euler-Maruyama and ODE solvers
├── data/
│   └── datasets.py         # DataLoader utilities: MNIST/CIFAR-10/AFHQ, class filtering, PairedDataset (transport)
├── evaluation/
│   └── metrics.py          # FID computation
├── config/
│   └── config.py           # Pydantic configs: ExperimentConfig, BridgeConfig, etc.
└── cli.py                  # Command-line interface
```

### Core Bridge Diffusion Model (`models/bridge.py`)

The bridge process connects prior distribution (noise) at t=0 to data distribution at t=T:
- **Expectation**: `E_t = x + (y - x) * t / T` (linear interpolation)
- **Variance**: `V_t = t * (T - t) / T` (bridge variance)

**Training (Corollary 2.7)**: Network learns to predict data `y` directly from bridge sample `xi_t`
- Sample time `t ~ Uniform[0, T]`
- Sample from bridge: `xi_t ~ N(E_t, V_t)`
- Network predicts `y` from `xi_t`
- Loss: MSE between prediction and true data

**Sampling (Corollary 2.9)**: Generate by evolving from noise (t=0) towards data (t=T)
- Start from noise sample
- Use Euler-Maruyama discretization
- Drift: `(y_pred - xi_t) / (T - t)` where `y_pred` is network output
- Add noise at each step except the last

### Configuration System

Experiments are defined via YAML configs (see `configs/`). Key sections:
- **method**: `bridge`, `poisson`, or `ddpm` (for comparison)
- **model**: UNet architecture parameters (channels, blocks, attention)
- **bridge**: Bridge process parameters (`T`, `eps`)
- **training**: Batch size, steps, learning rate, EMA settings
- **data**: Dataset (`mnist`, `cifar10`, `afhq`), image size, data directory. Transport
  mode: set `source_dataset` (+ optional `classes`/`source_classes` filters) to train
  the bridge from a source image distribution instead of Gaussian noise. See
  `configs/cifar_cat2dog.yaml` and `configs/afhq_cat2dog_64.yaml`.

Config is loaded by `ExperimentConfig.from_yaml()` and passed throughout the system.

### Model Creation Pattern

The `create_model()` function in `cli.py` demonstrates the pattern:
1. Create network wrapper: `DiffusersUNetWrapper(config.model)`
2. Wrap in diffusion model: `BridgeDiffusion(network, config.bridge)` or `DDPMDiffusion(...)`
3. Network predicts data directly for bridge method

### Experiment Tracking

MLflow is used for experiment tracking:
- Training logs: loss, learning rate, samples
- Checkpoints contain: model weights, EMA weights, config, optimizer state
- Artifacts: Sample grids, training curves

### Sampling Methods

`Sampler` supports multiple approaches:
- **Euler-Maruyama** (stochastic): Default implementation from paper
- **ODE solvers** (deterministic): DOPRI5, RK4, Heun for probability flow ODE
- Configured via `SamplingConfig` (num_steps, clip_samples, etc.)

## Code Style

- Line length: 100 characters
- Target: Python 3.10+
- Format with Black
- Lint with Ruff (E, F, I, N, W, UP checks)
- Docstrings: Google style with types
- Type hints required

## Testing

Tests use pytest with fixtures in `conftest.py`:
- `tests/test_sampling.py`: Sampling algorithms
- `tests/test_config.py`: Configuration loading
- `tests/test_data.py`: Dataset utilities
- Test fixtures provide sample models, configs, and tensors
