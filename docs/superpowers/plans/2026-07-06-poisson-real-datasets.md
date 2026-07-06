# Gaussian and Poisson Generation on CIFAR-10 and AFHQ — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Enable noise-to-data generation training for the Gaussian bridge and the Poisson bridge on CIFAR-10 (32x32) and AFHQ (64x64, all classes).

**Architecture:** The Poisson bridge consumes integer pixel counts in [0, 255] via `DataConfig.raw_pixels`, currently wired for MNIST only. We add integer-pixel transform builders for CIFAR-10 and AFHQ (mirroring `get_mnist_int_transforms`), wire them into `_make_base_dataset`, and add four experiment configs. No model or trainer changes.

**Tech Stack:** PyTorch, torchvision, pydantic-style dataclass configs, pytest, uv.

**Spec:** `docs/superpowers/specs/2026-07-06-poisson-real-datasets-design.md`

## Global Constraints

- Run everything via `uv` from the repo root: `uv run pytest`, `uv run ruff check src/`, `uv run black src/ tests/`.
- Line length 100, Python 3.10+, Google-style docstrings with types, type hints required.
- Integer-pixel transforms MUST output float32 tensors, integer-valued, range [0, 255], shape (3, H, W). Geometric ops go on the PIL side (before `PILToTensor`) so values stay integral.
- No changes to `models/`, `training/`, or the transport guard in `config.py`.
- Tests that download real datasets go in the `@pytest.mark.slow`-marked `TestDataLoading` class in `tests/test_data.py`; everything else must run without network.

---

### Task 1: CIFAR-10 integer-pixel transforms + wiring

**Files:**
- Modify: `src/bridge_diffusion/data/datasets.py` (after `get_cifar10_eval_transforms`, ~line 92; and the `cifar10` branch of `_make_base_dataset`, ~line 141)
- Test: `tests/test_data.py`

**Interfaces:**
- Consumes: existing `_ToFloat` class and `DataConfig.raw_pixels` field (both already in the codebase).
- Produces: `get_cifar10_int_transforms(image_size: int = 32, train: bool = True) -> transforms.Compose`. `_make_base_dataset("cifar10", config, train)` honours `config.raw_pixels`.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_data.py` (after `TestDataInfo`, module-level imports at top of file):

```python
from PIL import Image

from bridge_diffusion.data.datasets import get_cifar10_int_transforms
```

```python
class TestCifar10IntTransforms:
    """Tests for the CIFAR-10 integer-pixel transform path (Poisson bridge)."""

    def test_output_is_integer_valued_float32_in_range(self) -> None:
        img = Image.new("RGB", (32, 32), color=(7, 100, 250))
        out = get_cifar10_int_transforms(image_size=32)(img)
        assert out.dtype == torch.float32
        assert out.shape == (3, 32, 32)
        assert out.min() >= 0.0
        assert out.max() <= 255.0
        assert torch.equal(out, out.round())

    def test_resize_preserves_integer_values(self) -> None:
        img = Image.new("RGB", (64, 64), color=(13, 37, 201))
        out = get_cifar10_int_transforms(image_size=32)(img)
        assert out.shape == (3, 32, 32)
        assert torch.equal(out, out.round())

    def test_eval_transform_has_no_flip(self) -> None:
        t = get_cifar10_int_transforms(image_size=32, train=False)
        names = [type(op).__name__ for op in t.transforms]
        assert "RandomHorizontalFlip" not in names
```

Also add to the existing `@pytest.mark.slow` `TestDataLoading` class:

```python
    def test_cifar10_raw_pixels_dataloader(self, tmp_path) -> None:
        """CIFAR-10 with raw_pixels=True yields integer pixels in [0, 255]."""
        config = DataConfig(
            dataset="cifar10",
            data_dir=tmp_path,
            image_size=32,
            raw_pixels=True,
        )
        loader = get_dataloader(config, batch_size=4, train=True)
        images, _ = next(iter(loader))
        assert images.dtype == torch.float32
        assert images.max() > 1.0
        assert torch.equal(images, images.round())
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_data.py::TestCifar10IntTransforms -v`
Expected: FAIL at import with `ImportError: cannot import name 'get_cifar10_int_transforms'`

- [ ] **Step 3: Implement the transform builder**

In `src/bridge_diffusion/data/datasets.py`, after `get_cifar10_eval_transforms`:

```python
def get_cifar10_int_transforms(image_size: int = 32, train: bool = True) -> transforms.Compose:
    """Get raw integer pixel transforms for CIFAR-10.

    Returns pixel values as float32 in the range [0, 255] (integers preserved),
    required by the Poisson Bridge which operates on non-negative count data.
    Geometric ops run on the PIL side so values stay integral.

    Args:
        image_size: Target image size.
        train: Whether to include training augmentation (horizontal flip).

    Returns:
        Composed transforms.
    """
    ops: list = [transforms.Resize(image_size)]
    if train:
        ops.append(transforms.RandomHorizontalFlip())
    ops += [
        transforms.PILToTensor(),  # uint8 tensor in [0, 255]
        _ToFloat(),                # cast to float32, keep range — picklable
    ]
    return transforms.Compose(ops)
```

- [ ] **Step 4: Wire into `_make_base_dataset`**

Replace the `cifar10` branch:

```python
    elif name.lower() == "cifar10":
        if config.raw_pixels:
            transform = get_cifar10_int_transforms(config.image_size, train=train)
        elif train:
            transform = get_cifar10_transforms(config.image_size)
        else:
            transform = get_cifar10_eval_transforms(config.image_size)
        return datasets.CIFAR10(
            root=config.data_dir,
            train=train,
            download=True,
            transform=transform,
        )
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_data.py -v -m "not slow"`
Expected: PASS (all fast data tests, including the 3 new ones)

- [ ] **Step 6: Lint and commit**

```bash
uv run ruff check src/ tests/ && uv run black --check src/ tests/
git add src/bridge_diffusion/data/datasets.py tests/test_data.py
git commit -m "feat: integer-pixel transform path for CIFAR-10 (Poisson bridge)"
```

---

### Task 2: AFHQ integer-pixel transforms + wiring

**Files:**
- Modify: `src/bridge_diffusion/data/datasets.py` (after `get_afhq_transforms`, ~line 112; and the `afhq` branch of `_make_base_dataset`, ~line 152)
- Test: `tests/test_data.py`

**Interfaces:**
- Consumes: `_ToFloat`, `DataConfig.raw_pixels`, the `afhq_dir` fixture in `tests/conftest.py` (synthetic AFHQ tree, 32x32 PNGs).
- Produces: `get_afhq_int_transforms(image_size: int = 64, train: bool = True) -> transforms.Compose`. `_make_base_dataset("afhq", config, train)` honours `config.raw_pixels`.

- [ ] **Step 1: Write the failing tests**

Add the import alongside the one from Task 1:

```python
from bridge_diffusion.data.datasets import get_afhq_int_transforms, get_cifar10_int_transforms
```

Add to `tests/test_data.py`:

```python
class TestAfhqIntTransforms:
    """Tests for the AFHQ integer-pixel transform path (Poisson bridge)."""

    def test_output_is_integer_valued_float32_in_range(self) -> None:
        img = Image.new("RGB", (512, 512), color=(7, 100, 250))
        out = get_afhq_int_transforms(image_size=64)(img)
        assert out.dtype == torch.float32
        assert out.shape == (3, 64, 64)
        assert out.min() >= 0.0
        assert out.max() <= 255.0
        assert torch.equal(out, out.round())

    def test_eval_transform_has_no_flip(self) -> None:
        t = get_afhq_int_transforms(image_size=64, train=False)
        names = [type(op).__name__ for op in t.transforms]
        assert "RandomHorizontalFlip" not in names

    def test_raw_pixels_dataset_returns_int_pixels(self, afhq_dir) -> None:
        config = DataConfig(
            dataset="afhq",
            data_dir=afhq_dir,
            image_size=16,
            raw_pixels=True,
        )
        ds = get_dataset(config, train=True)
        img, _ = ds[0]
        assert img.dtype == torch.float32
        assert img.max() > 1.0
        assert torch.equal(img, img.round())

    def test_normalised_path_unchanged(self, afhq_dir) -> None:
        config = DataConfig(dataset="afhq", data_dir=afhq_dir, image_size=16)
        ds = get_dataset(config, train=True)
        img, _ = ds[0]
        assert img.min() >= -1.0
        assert img.max() <= 1.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_data.py::TestAfhqIntTransforms -v`
Expected: FAIL at import with `ImportError: cannot import name 'get_afhq_int_transforms'`

- [ ] **Step 3: Implement the transform builder**

In `src/bridge_diffusion/data/datasets.py`, after `get_afhq_transforms`:

```python
def get_afhq_int_transforms(image_size: int = 64, train: bool = True) -> transforms.Compose:
    """Get raw integer pixel transforms for AFHQ (512x512 source images).

    Returns pixel values as float32 in the range [0, 255] (integers preserved),
    required by the Poisson Bridge which operates on non-negative count data.
    Geometric ops run on the PIL side so values stay integral.

    Args:
        image_size: Target image size.
        train: Whether to include training augmentation (horizontal flip).

    Returns:
        Composed transforms.
    """
    ops: list = [transforms.Resize(image_size), transforms.CenterCrop(image_size)]
    if train:
        ops.append(transforms.RandomHorizontalFlip())
    ops += [
        transforms.PILToTensor(),  # uint8 tensor in [0, 255]
        _ToFloat(),                # cast to float32, keep range — picklable
    ]
    return transforms.Compose(ops)
```

- [ ] **Step 4: Wire into `_make_base_dataset`**

In the `afhq` branch, replace the `ImageFolder` construction:

```python
        transform = (
            get_afhq_int_transforms(config.image_size, train=train)
            if config.raw_pixels
            else get_afhq_transforms(config.image_size, train=train)
        )
        return datasets.ImageFolder(root=str(root), transform=transform)
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_data.py -v -m "not slow"`
Expected: PASS

- [ ] **Step 6: Lint and commit**

```bash
uv run ruff check src/ tests/ && uv run black --check src/ tests/
git add src/bridge_diffusion/data/datasets.py tests/test_data.py
git commit -m "feat: integer-pixel transform path for AFHQ (Poisson bridge)"
```

---

### Task 3: Four experiment configs + load test

**Files:**
- Create: `configs/afhq_64.yaml`, `configs/afhq_poisson_64.yaml`, `configs/cifar10_poisson.yaml`, `configs/cifar10_poisson_smoke.yaml`
- Test: `tests/test_config.py`

**Interfaces:**
- Consumes: `ExperimentConfig.from_yaml(path)` (existing).
- Produces: the four YAML files, loadable without error.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_config.py` (imports `Path` and `pytest` already present; `ExperimentConfig` already imported):

```python
class TestNewDatasetConfigs:
    """The generation configs added for CIFAR-10/AFHQ ship loadable."""

    CONFIGS_DIR = Path(__file__).parent.parent / "configs"

    @pytest.mark.parametrize(
        "name,method,dataset,raw",
        [
            ("afhq_64", "bridge", "afhq", False),
            ("afhq_poisson_64", "poisson_bridge", "afhq", True),
            ("cifar10_poisson", "poisson_bridge", "cifar10", True),
            ("cifar10_poisson_smoke", "poisson_bridge", "cifar10", True),
        ],
    )
    def test_config_loads(self, name: str, method: str, dataset: str, raw: bool) -> None:
        config = ExperimentConfig.from_yaml(self.CONFIGS_DIR / f"{name}.yaml")
        assert config.method == method
        assert config.data.dataset == dataset
        assert config.data.raw_pixels is raw
        assert config.data.source_dataset is None  # generation, not transport
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_config.py::TestNewDatasetConfigs -v`
Expected: FAIL with `FileNotFoundError` for `afhq_64.yaml`

- [ ] **Step 3: Create `configs/afhq_64.yaml`**

```yaml
# AFHQ generation (noise -> data, all classes) at 64x64 with the Gaussian bridge.
# Same UNet as the cat->dog transport config for comparability.
# Requires: bash scripts/download_afhq.sh

name: bridge_afhq_64
output_dir: ./outputs/afhq_64
method: bridge

model:
  in_channels: 3
  out_channels: 3
  sample_size: 64
  block_out_channels: [128, 256, 384, 512]
  layers_per_block: 2
  down_block_types:
    - DownBlock2D
    - DownBlock2D
    - AttnDownBlock2D
    - AttnDownBlock2D
  up_block_types:
    - AttnUpBlock2D
    - AttnUpBlock2D
    - UpBlock2D
    - UpBlock2D
  attention_head_dim: 8
  dropout: 0.1

training:
  batch_size: 64
  num_steps: 150000
  learning_rate: 0.0001
  weight_decay: 0.0
  grad_clip_norm: 1.0
  checkpoint_every: 10000
  log_every: 100
  seed: 42
  use_ema: true
  ema_decay: 0.9999

bridge:
  T: 0.1
  eps: 0.0000001

sampling:
  num_steps: 100
  num_samples: 64
  show_progress: true
  clip_samples: true

data:
  dataset: afhq
  data_dir: ./data
  image_size: 64
  num_workers: 8
  pin_memory: true
```

- [ ] **Step 4: Create `configs/afhq_poisson_64.yaml`**

```yaml
# AFHQ generation (all classes) at 64x64 with the Poisson bridge.
# Operates on raw integer pixel counts [0, 255]; prior is a blank (zeros) image.
# Requires: bash scripts/download_afhq.sh

name: poisson_afhq_64
output_dir: ./outputs/afhq_poisson_64
method: poisson_bridge

model:
  in_channels: 3
  out_channels: 3
  sample_size: 64
  block_out_channels: [128, 256, 384, 512]
  layers_per_block: 2
  down_block_types:
    - DownBlock2D
    - DownBlock2D
    - AttnDownBlock2D
    - AttnDownBlock2D
  up_block_types:
    - AttnUpBlock2D
    - AttnUpBlock2D
    - UpBlock2D
    - UpBlock2D
  attention_head_dim: 8
  dropout: 0.1

training:
  batch_size: 64
  num_steps: 150000
  learning_rate: 0.0001
  weight_decay: 0.0
  grad_clip_norm: 1.0
  checkpoint_every: 10000
  log_every: 100
  seed: 42
  use_ema: true
  ema_decay: 0.9999

poisson_bridge:
  T: 1.0
  eps: 1.0e-7
  num_levels: 256        # raw 8-bit pixel values [0, 255]
  prior: zeros           # prior is a blank image (all zeros)
  prior_lambda: 1.0

bridge:
  T: 0.1
  eps: 1.0e-7

sampling:
  num_steps: 100
  num_samples: 64
  show_progress: true
  clip_samples: false    # Poisson output does not need clipping

data:
  dataset: afhq
  data_dir: ./data
  image_size: 64
  num_workers: 8
  pin_memory: true
  raw_pixels: true       # keep pixel values as integers in [0, 255]
```

- [ ] **Step 5: Create `configs/cifar10_poisson.yaml`**

```yaml
# CIFAR-10 generation with the Poisson bridge.
# Same UNet as configs/cifar10.yaml (Gaussian bridge) for fair comparison.
# Operates on raw integer pixel counts [0, 255]; prior is a blank (zeros) image.

name: poisson_cifar10
output_dir: ./outputs/cifar10_poisson
method: poisson_bridge

model:
  in_channels: 3
  out_channels: 3
  sample_size: 32
  block_out_channels: [128, 256, 256, 256]
  layers_per_block: 2
  down_block_types:
    - DownBlock2D
    - AttnDownBlock2D
    - AttnDownBlock2D
    - AttnDownBlock2D
  up_block_types:
    - AttnUpBlock2D
    - AttnUpBlock2D
    - AttnUpBlock2D
    - UpBlock2D
  attention_head_dim: 8
  dropout: 0.1

training:
  batch_size: 128
  num_steps: 800000  # Match configs/cifar10.yaml for fair comparison
  learning_rate: 0.0002
  weight_decay: 0.0
  grad_clip_norm: 1.0
  checkpoint_every: 50000
  log_every: 100
  seed: 42
  use_ema: true
  ema_decay: 0.9999

poisson_bridge:
  T: 1.0
  eps: 1.0e-7
  num_levels: 256        # raw 8-bit pixel values [0, 255]
  prior: zeros           # prior is a blank image (all zeros)
  prior_lambda: 1.0

bridge:
  T: 0.1
  eps: 1.0e-7

sampling:
  num_steps: 100
  num_samples: 64
  show_progress: true
  clip_samples: false    # Poisson output does not need clipping

data:
  dataset: cifar10
  data_dir: ./data
  image_size: 32
  num_workers: 4
  pin_memory: true
  raw_pixels: true       # keep pixel values as integers in [0, 255]
```

- [ ] **Step 6: Create `configs/cifar10_poisson_smoke.yaml`**

```yaml
# 200-step smoke test for the Poisson bridge on CIFAR-10.
# Tiny model, small batch — verifies the end-to-end pipeline, not sample quality.

name: poisson_cifar10_smoke
output_dir: ./outputs/cifar10_poisson_smoke
method: poisson_bridge

model:
  in_channels: 3
  out_channels: 3
  sample_size: 32
  block_out_channels: [32, 32]
  layers_per_block: 1
  down_block_types:
    - DownBlock2D
    - AttnDownBlock2D
  up_block_types:
    - AttnUpBlock2D
    - UpBlock2D
  attention_head_dim: 4
  dropout: 0.0

training:
  batch_size: 16
  num_steps: 200
  learning_rate: 0.0002
  weight_decay: 0.0
  grad_clip_norm: 1.0
  checkpoint_every: 200
  log_every: 50
  seed: 42
  use_ema: true
  ema_decay: 0.9999

poisson_bridge:
  T: 1.0
  eps: 1.0e-7
  num_levels: 256
  prior: zeros
  prior_lambda: 1.0

bridge:
  T: 0.1
  eps: 1.0e-7

sampling:
  num_steps: 20
  num_samples: 16
  show_progress: true
  clip_samples: false

data:
  dataset: cifar10
  data_dir: ./data
  image_size: 32
  num_workers: 0         # 0 = main process only (avoids pickle issues on macOS MPS)
  pin_memory: false      # pin_memory not supported on MPS
  raw_pixels: true
```

- [ ] **Step 7: Run test to verify it passes**

Run: `uv run pytest tests/test_config.py -v`
Expected: PASS (4 new parametrized cases + all existing config tests)

- [ ] **Step 8: Commit**

```bash
git add configs/afhq_64.yaml configs/afhq_poisson_64.yaml configs/cifar10_poisson.yaml configs/cifar10_poisson_smoke.yaml tests/test_config.py
git commit -m "feat: generation configs for CIFAR-10/AFHQ (Gaussian + Poisson)"
```

---

### Task 4: End-to-end smoke verification

**Files:**
- No source changes. Verification only.

**Interfaces:**
- Consumes: everything from Tasks 1–3; real CIFAR-10 data at `/Users/osianshelley/Projects/diffusion-bridge/data` (already downloaded).

- [ ] **Step 1: Run the full test suite**

Run: `uv run pytest -v --tb=short`
Expected: all tests pass (slow tests may be deselected/skipped as configured)

- [ ] **Step 2: Link the real data into the worktree**

The worktree has no `./data`; reuse the main checkout's download instead of re-fetching:

```bash
ln -sfn /Users/osianshelley/Projects/diffusion-bridge/data data
```

Expected: `ls data/cifar-10-batches-py` lists the CIFAR batch files.
(`data/` is gitignored via the anchored `/data/` rule, so the symlink cannot be committed.)

- [ ] **Step 3: Run the Poisson smoke train**

```bash
uv run bridge-diffusion train --config configs/cifar10_poisson_smoke.yaml
```

Expected: completes 200 steps without error; loss logged every 50 steps; checkpoint written under `outputs/cifar10_poisson_smoke/`; a sample grid image produced.

- [ ] **Step 4: Inspect the sample grid**

```bash
ls -R outputs/cifar10_poisson_smoke | head -30
```

Expected: a checkpoint `.pt` and a sample-grid `.png`. Open the grid; after 200 steps samples will be noisy blobs — the check is that images exist, are 3-channel, and are not all-black.

- [ ] **Step 5: Clean up run outputs (keep the tree clean, no commit needed)**

```bash
git status --short
```

Expected: no tracked changes (outputs/ and data are ignored). Nothing to commit for this task.

---

## Self-Review Notes

- Spec coverage: data path (Tasks 1–2), configs (Task 3), tests (Tasks 1–3), acceptance (Task 4). Model/trainer untouched per spec.
- Deviation from spec worth noting: the CIFAR int transform takes a `train` flag (flip only when training) to mirror the existing train/eval split in the normalised path — the spec listed the flip unconditionally.
- Type consistency: both builders return `transforms.Compose`; both wiring sites key off `config.raw_pixels`.
