# Gaussian and Poisson Generation on CIFAR-10 and AFHQ

**Date:** 2026-07-06
**Status:** Approved

## Goal

Enable noise-to-data generation training for both the Gaussian bridge and the
Poisson bridge on the two real datasets already supported by the library:
CIFAR-10 (32x32) and AFHQ (64x64, all three classes). Gaussian generation on
CIFAR-10 already exists (`configs/cifar10.yaml`); everything else is new.

Out of scope: Poisson data-to-data transport (cat->dog). The existing config
guard that restricts transport to `method: bridge` stays as is; Poisson
transport is deferred to a later design.

## Background

The Poisson bridge (`models/poisson_bridge.py`) operates on non-negative
integer pixel counts in [0, 255]. The data pipeline supports this via
`DataConfig.raw_pixels`, but the integer-pixel transform path is wired for
MNIST only — the CIFAR-10 and AFHQ branches of `_make_base_dataset` always
normalise to [-1, 1] floats. The Poisson model itself is per-pixel and
channel-agnostic, so no model changes are needed.

## Design

### 1. Data path (`src/bridge_diffusion/data/datasets.py`)

Two new transform builders, mirroring `get_mnist_int_transforms` (PIL-side
geometric ops preserve integer values; `PILToTensor` keeps [0, 255]):

- `get_cifar10_int_transforms(image_size)`:
  Resize -> RandomHorizontalFlip -> PILToTensor -> `_ToFloat`.
  Output: float32, integer-valued, [0, 255], shape (3, H, W).
- `get_afhq_int_transforms(image_size, train)`:
  Resize -> CenterCrop -> RandomHorizontalFlip (train only) -> PILToTensor ->
  `_ToFloat`.

`_make_base_dataset` selects the int transforms for the `cifar10` and `afhq`
branches when `config.raw_pixels` is true, exactly as the MNIST branch does.

### 2. Configs (4 new files in `configs/`)

- `afhq_64.yaml` — Gaussian generation, all classes (no `classes` filter),
  64px. Same UNet, training, and bridge sections as `afhq_cat2dog_64.yaml`
  but with no `source_dataset`/`source_classes`/`classes` keys.
- `afhq_poisson_64.yaml` — `method: poisson_bridge`, all classes, 64px,
  `raw_pixels: true`, `num_levels: 256`, `prior: zeros`, `clip_samples: false`.
  Same UNet as `afhq_64.yaml`.
- `cifar10_poisson.yaml` — `method: poisson_bridge`, 32px,
  `raw_pixels: true`, `num_levels: 256`, `prior: zeros`,
  `clip_samples: false`. Same UNet as `cifar10.yaml`.
- `cifar10_poisson_smoke.yaml` — 200-step, small-batch variant of
  `cifar10_poisson.yaml` for cheap end-to-end verification.

Poisson bridge hyperparameters (`T`, `eps`) follow `mnist_poisson.yaml`.

### 3. Model / trainer

No changes. The trainer already dispatches on `method: poisson_bridge`
(proven by the MNIST Poisson run) and the transport guard in
`DataConfig.__post_init__` already rejects Poisson transport.

### 4. Tests

- New transform builders: dtype float32, values in [0, 255], values
  integer-valued after resize, RGB shape (3, H, W) at the requested size.
- `_make_base_dataset` wiring: with `raw_pixels: true`, the cifar10 and afhq
  branches return int-pixel samples; with `raw_pixels: false` they keep the
  existing [-1, 1] behaviour. AFHQ tests use the existing synthetic
  directory-tree fixture.
- Config round-trip: the four new YAML files load via
  `ExperimentConfig.from_yaml()` without error.

## Error handling

No new failure modes. AFHQ missing-data and unknown-dataset errors are
unchanged. `raw_pixels` with `method: bridge` remains permitted (as today) —
it is a user configuration choice, not validated.

## Acceptance

- `uv run pytest` passes.
- `uv run bridge-diffusion train --config configs/cifar10_poisson_smoke.yaml`
  runs end-to-end on real CIFAR-10 data (already in `./data`) and produces a
  sample grid.
