# Dataset Transport (CIFAR cat→dog, AFHQ cat→dog) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let the Gaussian bridge train as a data-to-data transport between two image distributions, with runnable CIFAR-10 cat→dog and AFHQ cat→dog experiments.

**Architecture:** Transport mode is switched on by a new `source_dataset` field on `DataConfig`. The data layer builds a `PairedDataset` (independent coupling) when a source is configured; the trainer takes the bridge prior `x` from the batch instead of Gaussian noise; the sampler/CLI accept source images as `x0`. Everything is backwards-compatible: no source configured means today's behaviour, bit for bit.

**Tech Stack:** Python 3.10+, PyTorch + torchvision, HuggingFace diffusers UNet, MLflow, pytest, uv.

**Spec:** `docs/superpowers/specs/2026-07-05-dataset-transport-design.md`

## Global Constraints

- All commands run via `uv run` (deactivate conda first if active).
- Line length 100; format with Black; lint with Ruff (E, F, I, N, W, UP).
- Google-style docstrings with types; type hints required.
- Dataset-downloading tests must be marked `@pytest.mark.slow` (skipped unless `--run-slow`). All tests below use tmpdir fakes and must NOT download anything.
- The three new `DataConfig` fields must default to `None` so every existing YAML config keeps working unchanged.
- AFHQ layout on disk: `{data_dir}/afhq/{train,val}/{cat,dog,wild}/*.png|jpg`.

---

### Task 1: DataConfig transport fields

**Files:**
- Modify: `src/bridge_diffusion/config/config.py:108-117` (DataConfig)
- Test: `tests/test_config.py`

**Interfaces:**
- Produces: `DataConfig.classes: list[str] | None`, `DataConfig.source_dataset: Literal["mnist","cifar10","afhq"] | None`, `DataConfig.source_classes: list[str] | None`. `DataConfig.__post_init__` raises `ValueError` if `source_classes` is set without `source_dataset`. All later tasks key transport mode off `config.data.source_dataset is not None`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_config.py`:

```python
class TestTransportConfig:
    """Tests for transport-mode data configuration."""

    def test_transport_fields_default_to_none(self) -> None:
        config = DataConfig()
        assert config.classes is None
        assert config.source_dataset is None
        assert config.source_classes is None

    def test_transport_fields_set(self) -> None:
        config = DataConfig(
            dataset="afhq",
            classes=["dog"],
            source_dataset="afhq",
            source_classes=["cat"],
        )
        assert config.classes == ["dog"]
        assert config.source_dataset == "afhq"
        assert config.source_classes == ["cat"]

    def test_source_classes_without_source_dataset_raises(self) -> None:
        with pytest.raises(ValueError, match="source_classes requires source_dataset"):
            DataConfig(source_classes=["cat"])

    def test_yaml_round_trip_with_transport_fields(self, tmp_path) -> None:
        config = ExperimentConfig(
            data=DataConfig(
                dataset="afhq",
                classes=["dog"],
                source_dataset="afhq",
                source_classes=["cat"],
            )
        )
        path = tmp_path / "config.yaml"
        config.to_yaml(path)
        loaded = ExperimentConfig.from_yaml(path)
        assert loaded.data.classes == ["dog"]
        assert loaded.data.source_dataset == "afhq"
        assert loaded.data.source_classes == ["cat"]
```

Make sure `tests/test_config.py` imports include `pytest`, `DataConfig`, and `ExperimentConfig` (add whichever are missing to the existing import block).

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_config.py::TestTransportConfig -v`
Expected: 4 failures — `TypeError: __init__() got an unexpected keyword argument 'classes'`.

- [ ] **Step 3: Implement the config change**

In `src/bridge_diffusion/config/config.py`, replace the `DataConfig` dataclass with:

```python
@dataclass
class DataConfig:
    """Configuration for data loading.

    When ``source_dataset`` is set, training runs in transport mode: the bridge
    prior x is drawn from the source dataset instead of Gaussian noise, and the
    dataloader yields (x_source, y_target) pairs (independent coupling).
    """

    dataset: Literal["mnist", "cifar10", "afhq"] = "mnist"
    data_dir: Path = field(default_factory=lambda: Path("./data"))
    image_size: int = 32  # Resize images to this size
    num_workers: int = 4
    pin_memory: bool = True
    raw_pixels: bool = False  # If True, return integer pixel values [0, 255] (for Poisson bridge)
    classes: list[str] | None = None  # Filter target dataset to these class names
    source_dataset: Literal["mnist", "cifar10", "afhq"] | None = None
    source_classes: list[str] | None = None  # Filter source dataset to these class names

    def __post_init__(self) -> None:
        """Validate transport-mode field combinations."""
        if self.source_classes is not None and self.source_dataset is None:
            raise ValueError("source_classes requires source_dataset to be set")
```

No change needed in `from_yaml`/`to_yaml`: the fields are flat and pass through `DataConfig(**data.get("data", {}))` and `asdict` already.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_config.py -v`
Expected: all pass (including pre-existing config tests).

- [ ] **Step 5: Commit**

```bash
git add src/bridge_diffusion/config/config.py tests/test_config.py
git commit -m "Add transport-mode fields to DataConfig"
```

---

### Task 2: Fake-AFHQ fixture and filter_classes helper

**Files:**
- Modify: `tests/conftest.py`
- Modify: `src/bridge_diffusion/data/datasets.py`
- Modify: `src/bridge_diffusion/data/__init__.py` (export `filter_classes`)
- Test: `tests/test_data.py`

**Interfaces:**
- Consumes: nothing new.
- Produces: `filter_classes(dataset, class_names: list[str]) -> torch.utils.data.Subset` — works on any dataset exposing `class_to_idx: dict[str, int]` and `targets: Sequence[int]` (both `CIFAR10` and `ImageFolder` do). Pytest fixture `afhq_dir(tmp_path) -> Path` creating a fake AFHQ tree (4 images per class per split, 32×32 RGB PNGs).

- [ ] **Step 1: Add the shared fixture**

Append to `tests/conftest.py`:

```python
@pytest.fixture
def afhq_dir(tmp_path):
    """Create a fake AFHQ directory tree with 4 tiny images per class per split."""
    from PIL import Image

    for split in ("train", "val"):
        for cls in ("cat", "dog", "wild"):
            d = tmp_path / "afhq" / split / cls
            d.mkdir(parents=True)
            for i in range(4):
                Image.new("RGB", (32, 32), color=(i * 20, 100, 150)).save(d / f"{cls}_{i}.png")
    return tmp_path
```

- [ ] **Step 2: Write the failing tests**

Append to `tests/test_data.py`:

```python
from torchvision import datasets as tv_datasets

from bridge_diffusion.data import filter_classes


class TestFilterClasses:
    """Tests for class filtering."""

    def test_filters_imagefolder_to_named_classes(self, afhq_dir) -> None:
        ds = tv_datasets.ImageFolder(root=str(afhq_dir / "afhq" / "train"))
        subset = filter_classes(ds, ["cat"])
        assert len(subset) == 4
        cat_idx = ds.class_to_idx["cat"]
        assert all(ds.targets[i] == cat_idx for i in subset.indices)

    def test_filters_multiple_classes(self, afhq_dir) -> None:
        ds = tv_datasets.ImageFolder(root=str(afhq_dir / "afhq" / "train"))
        subset = filter_classes(ds, ["cat", "dog"])
        assert len(subset) == 8

    def test_unknown_class_raises_with_valid_names(self, afhq_dir) -> None:
        ds = tv_datasets.ImageFolder(root=str(afhq_dir / "afhq" / "train"))
        with pytest.raises(ValueError, match="wolf"):
            filter_classes(ds, ["wolf"])
```

Ensure `pytest` is imported at the top of `tests/test_data.py` (it already is).

- [ ] **Step 3: Run tests to verify they fail**

Run: `uv run pytest tests/test_data.py::TestFilterClasses -v`
Expected: `ImportError: cannot import name 'filter_classes'`.

- [ ] **Step 4: Implement filter_classes**

In `src/bridge_diffusion/data/datasets.py`, add after the imports:

```python
from torch.utils.data import DataLoader, Dataset, Subset
```

(replacing the existing `from torch.utils.data import DataLoader` line), then add:

```python
def filter_classes(
    dataset: torch.utils.data.Dataset,
    class_names: list[str],
) -> Subset:
    """Restrict a dataset to the given class names.

    Works with any torchvision dataset exposing ``class_to_idx`` and
    ``targets`` (CIFAR10 and ImageFolder both do).

    Args:
        dataset: Dataset to filter.
        class_names: Class names to keep (e.g. ["cat", "dog"]).

    Returns:
        Subset containing only items of the requested classes.

    Raises:
        ValueError: If any name is not a class of the dataset.
    """
    valid = list(dataset.class_to_idx)
    unknown = [name for name in class_names if name not in valid]
    if unknown:
        raise ValueError(f"Unknown class(es) {unknown}; valid classes: {valid}")

    keep = {dataset.class_to_idx[name] for name in class_names}
    indices = [i for i, target in enumerate(dataset.targets) if int(target) in keep]
    return Subset(dataset, indices)
```

In `src/bridge_diffusion/data/__init__.py`, add `filter_classes` to the imports from `.datasets` and to `__all__` (match the file's existing style).

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_data.py::TestFilterClasses -v`
Expected: 3 passed.

- [ ] **Step 6: Commit**

```bash
git add tests/conftest.py tests/test_data.py src/bridge_diffusion/data/datasets.py src/bridge_diffusion/data/__init__.py
git commit -m "Add filter_classes helper and fake-AFHQ test fixture"
```

---

### Task 3: AFHQ dataset entry

**Files:**
- Modify: `src/bridge_diffusion/data/datasets.py` (transforms, `get_dataset`, `get_data_info`)
- Test: `tests/test_data.py`

**Interfaces:**
- Consumes: `filter_classes` (Task 2), `DataConfig.classes` (Task 1).
- Produces: `get_dataset` accepts `dataset="afhq"` and applies `config.classes` filtering for every dataset; `get_afhq_transforms(image_size: int, train: bool) -> transforms.Compose`; `get_data_info` returns AFHQ info and class-filtered sizes. Internal helper `_make_base_dataset(name: str, config: DataConfig, train: bool) -> Dataset` used again in Task 4.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_data.py`:

```python
class TestAFHQ:
    """Tests for the AFHQ dataset entry."""

    def test_loads_train_split(self, afhq_dir) -> None:
        config = DataConfig(dataset="afhq", data_dir=afhq_dir, image_size=16)
        ds = get_dataset(config, train=True)
        assert len(ds) == 12  # 3 classes x 4 images
        img, label = ds[0]
        assert img.shape == (3, 16, 16)
        assert img.min() >= -1.0 and img.max() <= 1.0

    def test_classes_filter_applied(self, afhq_dir) -> None:
        config = DataConfig(dataset="afhq", data_dir=afhq_dir, image_size=16, classes=["dog"])
        ds = get_dataset(config, train=True)
        assert len(ds) == 4

    def test_missing_dir_raises_actionable_error(self, tmp_path) -> None:
        config = DataConfig(dataset="afhq", data_dir=tmp_path / "nowhere")
        with pytest.raises(FileNotFoundError, match="download_afhq"):
            get_dataset(config, train=True)

    def test_data_info(self) -> None:
        config = DataConfig(dataset="afhq", image_size=64)
        info = get_data_info(config)
        assert info["num_channels"] == 3
        assert info["num_classes"] == 3
        assert info["train_size"] == 5153 + 4739 + 4738

    def test_data_info_with_classes(self) -> None:
        config = DataConfig(dataset="afhq", image_size=64, classes=["cat", "dog"])
        info = get_data_info(config)
        assert info["num_classes"] == 2
        assert info["train_size"] == 5153 + 4739
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_data.py::TestAFHQ -v`
Expected: failures with `ValueError: Unknown dataset: afhq`.

- [ ] **Step 3: Implement the AFHQ entry**

In `src/bridge_diffusion/data/datasets.py`:

Add near the top (after the existing imports):

```python
from pathlib import Path
```

Add module-level constants:

```python
AFHQ_TRAIN_COUNTS = {"cat": 5153, "dog": 4739, "wild": 4738}
AFHQ_VAL_COUNTS = {"cat": 500, "dog": 500, "wild": 500}
```

Add the transform function (after `get_cifar10_eval_transforms`):

```python
def get_afhq_transforms(image_size: int = 64, train: bool = True) -> transforms.Compose:
    """Get transforms for AFHQ (512x512 source images).

    Args:
        image_size: Target image size.
        train: Whether to include training augmentation (horizontal flip).

    Returns:
        Composed transforms producing tensors in [-1, 1].
    """
    ops: list = [transforms.Resize(image_size), transforms.CenterCrop(image_size)]
    if train:
        ops.append(transforms.RandomHorizontalFlip())
    ops += [
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # Scale to [-1, 1]
    ]
    return transforms.Compose(ops)
```

Refactor `get_dataset`: rename the existing if/elif body into a helper and apply class filtering generically:

```python
def _make_base_dataset(name: str, config: DataConfig, train: bool) -> torch.utils.data.Dataset:
    """Construct an unfiltered dataset by name."""
    if name.lower() == "mnist":
        transform = (
            get_mnist_int_transforms(config.image_size)
            if config.raw_pixels
            else get_mnist_transforms(config.image_size)
        )
        return datasets.MNIST(
            root=config.data_dir,
            train=train,
            download=True,
            transform=transform,
        )
    elif name.lower() == "cifar10":
        if train:
            transform = get_cifar10_transforms(config.image_size)
        else:
            transform = get_cifar10_eval_transforms(config.image_size)
        return datasets.CIFAR10(
            root=config.data_dir,
            train=train,
            download=True,
            transform=transform,
        )
    elif name.lower() == "afhq":
        split = "train" if train else "val"
        root = Path(config.data_dir) / "afhq" / split
        if not root.is_dir():
            raise FileNotFoundError(
                f"AFHQ not found at {root}. Download it first: bash scripts/download_afhq.sh"
            )
        return datasets.ImageFolder(
            root=str(root),
            transform=get_afhq_transforms(config.image_size, train=train),
        )
    else:
        raise ValueError(f"Unknown dataset: {name}")


def get_dataset(
    config: DataConfig,
    train: bool = True,
) -> torch.utils.data.Dataset:
    """Get dataset based on configuration.

    Args:
        config: Data configuration.
        train: Whether to load training or test set.

    Returns:
        PyTorch dataset (a PairedDataset in transport mode).
    """
    dataset = _make_base_dataset(config.dataset, config, train)
    if config.classes:
        dataset = filter_classes(dataset, config.classes)
    return dataset
```

(The transport branch is added in Task 4; keep `get_dataset` exactly as above for now.)

Extend `get_data_info` with an AFHQ branch and class-aware sizes:

```python
def get_data_info(config: DataConfig) -> dict:
    """Get information about the dataset.

    Args:
        config: Data configuration.

    Returns:
        Dictionary with dataset information.
    """
    if config.dataset.lower() == "mnist":
        return {
            "num_channels": 1,
            "image_size": config.image_size,
            "num_classes": 10,
            "train_size": 60000,
            "test_size": 10000,
        }
    elif config.dataset.lower() == "cifar10":
        num_classes = len(config.classes) if config.classes else 10
        return {
            "num_channels": 3,
            "image_size": config.image_size,
            "num_classes": num_classes,
            "train_size": 5000 * num_classes,
            "test_size": 1000 * num_classes,
        }
    elif config.dataset.lower() == "afhq":
        class_names = config.classes or list(AFHQ_TRAIN_COUNTS)
        return {
            "num_channels": 3,
            "image_size": config.image_size,
            "num_classes": len(class_names),
            "train_size": sum(AFHQ_TRAIN_COUNTS[c] for c in class_names),
            "test_size": sum(AFHQ_VAL_COUNTS[c] for c in class_names),
        }
    else:
        raise ValueError(f"Unknown dataset: {config.dataset}")
```

Note: `AFHQ_TRAIN_COUNTS[c]` will KeyError on an unknown class name here; that is acceptable because `get_dataset` validates first in real runs, but guard anyway by validating against the dict:

```python
        unknown = [c for c in class_names if c not in AFHQ_TRAIN_COUNTS]
        if unknown:
            raise ValueError(f"Unknown class(es) {unknown}; valid classes: {list(AFHQ_TRAIN_COUNTS)}")
```

(insert immediately after `class_names = ...`).

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_data.py -v`
Expected: all pass, including pre-existing tests (the refactor must not change MNIST/CIFAR behaviour).

- [ ] **Step 5: Commit**

```bash
git add src/bridge_diffusion/data/datasets.py tests/test_data.py
git commit -m "Add AFHQ dataset entry with class filtering"
```

---

### Task 4: PairedDataset and transport dataloader

**Files:**
- Modify: `src/bridge_diffusion/data/datasets.py`
- Modify: `src/bridge_diffusion/data/__init__.py` (export `PairedDataset`)
- Test: `tests/test_data.py`

**Interfaces:**
- Consumes: `_make_base_dataset`, `filter_classes` (Tasks 2-3); `DataConfig.source_dataset` / `source_classes` (Task 1).
- Produces: `PairedDataset(source: Dataset, target: Dataset)` yielding `(x_source: Tensor, y_target: Tensor)` with labels dropped, `len == len(target)`. `get_dataset` returns a `PairedDataset` when `config.source_dataset` is set. Task 5's trainer relies on batches being 2-tuples of image tensors in transport mode.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_data.py`:

```python
from bridge_diffusion.data import PairedDataset


class TestPairedDataset:
    """Tests for the transport-mode paired dataset."""

    def _transport_config(self, afhq_dir) -> DataConfig:
        return DataConfig(
            dataset="afhq",
            data_dir=afhq_dir,
            image_size=16,
            classes=["dog"],
            source_dataset="afhq",
            source_classes=["cat"],
        )

    def test_get_dataset_returns_paired_dataset(self, afhq_dir) -> None:
        ds = get_dataset(self._transport_config(afhq_dir), train=True)
        assert isinstance(ds, PairedDataset)
        assert len(ds) == 4  # len(target dogs)

    def test_items_are_image_pairs_without_labels(self, afhq_dir) -> None:
        ds = get_dataset(self._transport_config(afhq_dir), train=True)
        item = ds[0]
        assert isinstance(item, tuple) and len(item) == 2
        x, y = item
        assert x.shape == (3, 16, 16)
        assert y.shape == (3, 16, 16)

    def test_dataloader_batches_pairs(self, afhq_dir) -> None:
        config = self._transport_config(afhq_dir)
        loader = get_dataloader(config, batch_size=2, train=True, num_workers=0)
        batch = next(iter(loader))
        assert batch[0].shape == (2, 3, 16, 16)  # x_source
        assert batch[1].shape == (2, 3, 16, 16)  # y_target
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_data.py::TestPairedDataset -v`
Expected: `ImportError: cannot import name 'PairedDataset'`.

- [ ] **Step 3: Implement PairedDataset and wire it in**

In `src/bridge_diffusion/data/datasets.py` add:

```python
class PairedDataset(Dataset):
    """Independent coupling of a source and a target dataset.

    __getitem__(i) returns (x_source, y_target) where y_target is target item i
    and x_source is drawn uniformly at random from the source dataset
    (torch.randint, so per-worker seeding behaves under num_workers > 0).
    Labels from both datasets are dropped.
    """

    def __init__(self, source: Dataset, target: Dataset):
        self.source = source
        self.target = target

    def __len__(self) -> int:
        return len(self.target)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        y = self.target[index][0]
        source_index = int(torch.randint(len(self.source), (1,)).item())
        x = self.source[source_index][0]
        return x, y
```

Extend `get_dataset` (from Task 3) with the transport branch:

```python
def get_dataset(
    config: DataConfig,
    train: bool = True,
) -> torch.utils.data.Dataset:
    """Get dataset based on configuration.

    Args:
        config: Data configuration.
        train: Whether to load training or test set.

    Returns:
        PyTorch dataset (a PairedDataset in transport mode).
    """
    dataset = _make_base_dataset(config.dataset, config, train)
    if config.classes:
        dataset = filter_classes(dataset, config.classes)

    if config.source_dataset is not None:
        source = _make_base_dataset(config.source_dataset, config, train)
        if config.source_classes:
            source = filter_classes(source, config.source_classes)
        dataset = PairedDataset(source, dataset)

    return dataset
```

Export `PairedDataset` from `src/bridge_diffusion/data/__init__.py`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_data.py -v`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add src/bridge_diffusion/data/datasets.py src/bridge_diffusion/data/__init__.py tests/test_data.py
git commit -m "Add PairedDataset and transport-mode dataloader wiring"
```

---

### Task 5: Trainer transport branch

**Files:**
- Modify: `src/bridge_diffusion/training/trainer.py`
- Create: `tests/test_trainer.py`

**Interfaces:**
- Consumes: transport batches `(x_source, y_target)` from Task 4; `config.data.source_dataset` from Task 1.
- Produces: `Trainer.transport: bool`; `Trainer._prepare_batch(batch) -> tuple[Tensor, Tensor]` returning `(x, y)` on the trainer's device; `Trainer._sample_sources: Tensor | None` (16 fixed val-split source images used by `_log_samples` in transport mode). The train loop and `_log_samples` use these; nothing else changes.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_trainer.py`:

```python
"""Tests for the training loop's transport mode."""

import torch

from bridge_diffusion.config import (
    BridgeConfig,
    DataConfig,
    ExperimentConfig,
    ModelConfig,
    TrainingConfig,
)
from bridge_diffusion.data import get_dataloader
from bridge_diffusion.models import BridgeDiffusion, DiffusersUNetWrapper
from bridge_diffusion.training import Trainer


def _tiny_model(image_size: int = 16) -> BridgeDiffusion:
    model_config = ModelConfig(
        in_channels=3,
        out_channels=3,
        sample_size=image_size,
        block_out_channels=(32, 64),
        layers_per_block=1,
        down_block_types=("DownBlock2D", "DownBlock2D"),
        up_block_types=("UpBlock2D", "UpBlock2D"),
    )
    network = DiffusersUNetWrapper(model_config)
    return BridgeDiffusion(network, BridgeConfig())


def _transport_experiment(afhq_dir, tmp_path) -> ExperimentConfig:
    return ExperimentConfig(
        name="test_transport",
        output_dir=tmp_path / "outputs",
        method="bridge",
        model=ModelConfig(in_channels=3, out_channels=3, sample_size=16),
        training=TrainingConfig(batch_size=2, num_steps=1, use_ema=False),
        data=DataConfig(
            dataset="afhq",
            data_dir=afhq_dir,
            image_size=16,
            classes=["dog"],
            source_dataset="afhq",
            source_classes=["cat"],
            num_workers=0,
        ),
    )


class TestTrainerTransport:
    def test_prepare_batch_uses_source_as_prior(self, afhq_dir, tmp_path) -> None:
        config = _transport_experiment(afhq_dir, tmp_path)
        loader = get_dataloader(config.data, batch_size=2, train=True, num_workers=0)
        trainer = Trainer(
            model=_tiny_model(),
            train_loader=loader,
            config=config,
            device=torch.device("cpu"),
            checkpoint_dir=tmp_path / "ckpt",
        )
        assert trainer.transport is True

        batch = next(iter(loader))
        x, y = trainer._prepare_batch(batch)
        assert x.shape == (2, 3, 16, 16)
        assert y.shape == (2, 3, 16, 16)
        # x must be the batch's source images, not fresh Gaussian noise
        assert torch.equal(x, batch[0])

    def test_prepare_batch_noise_prior_without_source(self, afhq_dir, tmp_path) -> None:
        config = _transport_experiment(afhq_dir, tmp_path)
        config.data.source_dataset = None
        config.data.source_classes = None
        loader = get_dataloader(config.data, batch_size=2, train=True, num_workers=0)
        trainer = Trainer(
            model=_tiny_model(),
            train_loader=loader,
            config=config,
            device=torch.device("cpu"),
            checkpoint_dir=tmp_path / "ckpt",
        )
        assert trainer.transport is False

        batch = next(iter(loader))
        x, y = trainer._prepare_batch(batch)
        assert x.shape == y.shape
        assert not torch.equal(x, batch[0])  # x is noise, not the images

    def test_transport_trainer_holds_val_source_images(self, afhq_dir, tmp_path) -> None:
        config = _transport_experiment(afhq_dir, tmp_path)
        loader = get_dataloader(config.data, batch_size=2, train=True, num_workers=0)
        trainer = Trainer(
            model=_tiny_model(),
            train_loader=loader,
            config=config,
            device=torch.device("cpu"),
            checkpoint_dir=tmp_path / "ckpt",
        )
        assert trainer._sample_sources is not None
        assert trainer._sample_sources.shape[1:] == (3, 16, 16)
        assert trainer._sample_sources.shape[0] <= 16
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_trainer.py -v`
Expected: `AttributeError: 'Trainer' object has no attribute 'transport'` (or `_prepare_batch`).

- [ ] **Step 3: Implement the trainer changes**

In `src/bridge_diffusion/training/trainer.py`:

At the end of `Trainer.__init__` (after `self.best_loss = float("inf")`), add:

```python
        # Transport mode: bridge prior x comes from the source dataset, not noise
        self.transport = config.data.source_dataset is not None
        self._sample_sources: Optional[torch.Tensor] = None
        if self.transport:
            self._sample_sources = self._load_sample_sources()
```

Add the two new methods to `Trainer`:

```python
    def _load_sample_sources(self, num_samples: int = 16) -> torch.Tensor:
        """Load a fixed batch of held-out source (val) images for sample logging."""
        import dataclasses

        from bridge_diffusion.data import get_dataset

        source_config = dataclasses.replace(
            self.config.data,
            dataset=self.config.data.source_dataset,
            classes=self.config.data.source_classes,
            source_dataset=None,
            source_classes=None,
        )
        val_dataset = get_dataset(source_config, train=False)
        count = min(num_samples, len(val_dataset))
        indices = torch.linspace(0, len(val_dataset) - 1, steps=count).long().tolist()
        return torch.stack([val_dataset[i][0] for i in indices])

    def _prepare_batch(self, batch) -> tuple[torch.Tensor, torch.Tensor]:
        """Extract (x, y) for the bridge loss from a dataloader batch.

        Transport mode: x is the source image from the paired batch.
        Otherwise: x is a prior sample (Gaussian noise, or the model's own prior).
        """
        if self.transport:
            x = batch[0].to(self.device)
            y = batch[1].to(self.device)
        else:
            y = batch[0].to(self.device)
            x = _sample_prior(self.model, y)
        return x, y
```

In `train()`, replace the two lines

```python
                # Paper notation: x = noise (prior), y = data (target)
                y = batch[0].to(self.device)
                x = _sample_prior(self.model, y)
```

with

```python
                # Paper notation: x = prior (noise or source image), y = data (target)
                x, y = self._prepare_batch(batch)
```

In `train()`'s `mlflow.log_params({...})` dict, add three entries:

```python
                "classes": str(self.config.data.classes),
                "source_dataset": str(self.config.data.source_dataset),
                "source_classes": str(self.config.data.source_classes),
```

In `_log_samples`, replace

```python
            x = _sample_prior(model_to_sample, torch.zeros(shape, device=self.device))
```

with

```python
            if self.transport and self._sample_sources is not None:
                x = self._sample_sources.to(self.device)
            else:
                x = _sample_prior(model_to_sample, torch.zeros(shape, device=self.device))
```

(the `shape` line above it stays; it is still needed for the non-transport branch).

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_trainer.py -v`
Expected: 3 passed.

- [ ] **Step 5: Run the full suite to catch regressions**

Run: `uv run pytest -v --tb=short`
Expected: all pass (dataset-download tests auto-skip without `--run-slow`).

- [ ] **Step 6: Commit**

```bash
git add src/bridge_diffusion/training/trainer.py tests/test_trainer.py
git commit -m "Use source images as bridge prior in transport mode"
```

---

### Task 6: Sampler x0 batching and CLI --source-dir

**Files:**
- Modify: `src/bridge_diffusion/sampling/sampler.py:418-445` (`sample_batch`)
- Modify: `src/bridge_diffusion/cli.py` (`sample_main`, argparser)
- Test: `tests/test_sampling.py`

**Interfaces:**
- Consumes: `Sampler.sample(num_samples, shape, x0=..., num_steps=...)` (already exists).
- Produces: `Sampler.sample_batch(total_samples, shape, batch_size=64, num_steps=None, x0: Optional[torch.Tensor] = None)` — chunks `x0` alongside the batches. CLI: `sample --source-dir PATH` loads images from a folder as `x0`; with no `--source-dir` but a transport-configured checkpoint, falls back to the source val split; generated files keep source file stems (`{stem}_translated.png`).

- [ ] **Step 1: Write the failing test**

Append to `tests/test_sampling.py`:

```python
from bridge_diffusion.config import BridgeConfig, SamplingConfig
from bridge_diffusion.sampling import Sampler


class _IdentityNet(torch.nn.Module):
    """Predicts E[Y|xi] = xi, so the SDE drift is zero."""

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return x


class TestSampleBatchX0:
    def test_x0_is_chunked_and_used(self) -> None:
        sampler = Sampler(
            model=_IdentityNet(),
            bridge_config=BridgeConfig(),
            sampling_config=SamplingConfig(num_steps=1, show_progress=False),
            device=torch.device("cpu"),
        )
        # With num_steps=1 and zero drift, output == clamp(x0)
        x0 = torch.full((5, 1, 8, 8), 0.5)
        out = sampler.sample_batch(total_samples=5, shape=(1, 8, 8), batch_size=2, x0=x0)
        assert out.shape == (5, 1, 8, 8)
        assert torch.allclose(out, x0)

    def test_x0_none_still_works(self) -> None:
        sampler = Sampler(
            model=_IdentityNet(),
            bridge_config=BridgeConfig(),
            sampling_config=SamplingConfig(num_steps=1, show_progress=False),
            device=torch.device("cpu"),
        )
        out = sampler.sample_batch(total_samples=3, shape=(1, 8, 8), batch_size=2)
        assert out.shape == (3, 1, 8, 8)
```

If `tests/test_sampling.py` does not already import `torch`, add it.

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_sampling.py::TestSampleBatchX0 -v`
Expected: `TypeError: sample_batch() got an unexpected keyword argument 'x0'`.

- [ ] **Step 3: Extend sample_batch**

Replace `Sampler.sample_batch` in `src/bridge_diffusion/sampling/sampler.py` with:

```python
    def sample_batch(
        self,
        total_samples: int,
        shape: tuple[int, ...],
        batch_size: int = 64,
        num_steps: Optional[int] = None,
        x0: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Generate samples in batches to manage memory.

        Args:
            total_samples: Total number of samples to generate.
            shape: Shape of each sample.
            batch_size: Batch size for generation.
            num_steps: Number of discretisation steps.
            x0: Optional prior samples of shape (total_samples, *shape). Chunked
                alongside the batches; if None, each batch starts from N(0, I).

        Returns:
            Generated samples of shape (total_samples, *shape).
        """
        if x0 is not None and x0.shape[0] < total_samples:
            raise ValueError(
                f"x0 has {x0.shape[0]} samples but total_samples={total_samples}"
            )

        all_samples = []
        start = 0

        while start < total_samples:
            current_batch = min(batch_size, total_samples - start)
            x0_batch = x0[start : start + current_batch] if x0 is not None else None
            samples = self.sample(current_batch, shape, x0=x0_batch, num_steps=num_steps)
            all_samples.append(samples.cpu())
            start += current_batch

        return torch.cat(all_samples, dim=0)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_sampling.py -v`
Expected: all pass.

- [ ] **Step 5: Wire --source-dir into the CLI**

In `src/bridge_diffusion/cli.py`:

Add a helper above `sample_main`:

```python
def _load_source_images(
    source_dir: Path,
    image_size: int,
    num_channels: int,
    num_samples: int,
) -> tuple[torch.Tensor, list[str]]:
    """Load up to num_samples images from a folder as prior samples x0.

    Returns:
        (images, stems): tensor of shape (n, C, H, W) in [-1, 1], and the
        source file stems for naming outputs.
    """
    from PIL import Image
    from torchvision import transforms

    extensions = {".png", ".jpg", ".jpeg"}
    paths = sorted(
        p for p in Path(source_dir).iterdir() if p.suffix.lower() in extensions
    )[:num_samples]
    if not paths:
        raise ValueError(f"No images found in {source_dir}")

    mode = "RGB" if num_channels == 3 else "L"
    mean = (0.5,) * num_channels
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean, mean),
    ])
    images = torch.stack([transform(Image.open(p).convert(mode)) for p in paths])
    return images, [p.stem for p in paths]
```

In `sample_main`, after the `sampler = Sampler(...)` block and before the `if config.method == "poisson_bridge":` branch, add:

```python
    # Transport mode: start the bridge from source images instead of noise
    x0 = None
    source_stems: list[str] | None = None
    if args.source_dir:
        x0, source_stems = _load_source_images(
            Path(args.source_dir),
            image_size=data_info["image_size"],
            num_channels=data_info["num_channels"],
            num_samples=args.num_samples,
        )
        logger.info(f"Loaded {x0.shape[0]} source images from {args.source_dir}")
    elif config.data.source_dataset is not None:
        import dataclasses

        from bridge_diffusion.data import get_dataset

        source_config = dataclasses.replace(
            config.data,
            dataset=config.data.source_dataset,
            classes=config.data.source_classes,
            source_dataset=None,
            source_classes=None,
        )
        val_dataset = get_dataset(source_config, train=False)
        count = min(args.num_samples, len(val_dataset))
        x0 = torch.stack([val_dataset[i][0] for i in range(count)])
        source_stems = [f"{i:04d}" for i in range(count)]
        logger.info(f"Using {count} val images from source dataset as priors")

    if x0 is not None and config.data.source_dataset is None:
        logger.warning(
            "Checkpoint was not trained in transport mode; sampling from "
            "--source-dir images anyway."
        )
    num_to_sample = x0.shape[0] if x0 is not None else args.num_samples
```

Then in the non-Poisson `else:` branch, replace the `sampler.sample_batch(...)` call and saving with:

```python
        samples = sampler.sample_batch(
            total_samples=num_to_sample,
            shape=shape,
            batch_size=args.batch_size,
            num_steps=args.num_steps,
            x0=x0,
        )

        output_dir = Path(args.output_dir)
        if source_stems is not None:
            from torchvision.utils import save_image

            output_dir.mkdir(parents=True, exist_ok=True)
            for stem, sample in zip(source_stems, samples):
                save_image((sample.clamp(-1, 1) + 1) / 2, output_dir / f"{stem}_translated.png")
            sampler.save_grid(samples[:64], output_dir / "grid.png")
        else:
            sampler.save_samples(samples, output_dir)
            sampler.save_grid(samples[:64], output_dir / "grid.png")
```

Add the argparse option to the sample subparser (after `--seed`):

```python
    sample_parser.add_argument(
        "--source-dir",
        type=str,
        default=None,
        help="Folder of images to use as bridge priors x0 (transport mode)",
    )
```

- [ ] **Step 6: Smoke-check the CLI wiring**

Run: `uv run python -c "from bridge_diffusion.cli import main; import sys; sys.argv=['x','sample','--help']; main()" | grep source-dir`
Expected: the `--source-dir` help line prints (the command exits 0 because `--help` on the subparser prints and exits).

Note: `--help` calls `sys.exit(0)`; if the grep pipeline exits non-zero, run without the grep and eyeball the output instead.

- [ ] **Step 7: Run the full suite**

Run: `uv run pytest -v --tb=short`
Expected: all pass.

- [ ] **Step 8: Commit**

```bash
git add src/bridge_diffusion/sampling/sampler.py src/bridge_diffusion/cli.py tests/test_sampling.py
git commit -m "Support source images as sampling priors (transport mode)"
```

---

### Task 7: export_val_images.py helper script

**Files:**
- Create: `scripts/export_val_images.py`
- Test: `tests/test_data.py`

**Interfaces:**
- Consumes: `get_dataset`, `DataConfig` (Tasks 1-4).
- Produces: `scripts/export_val_images.py` with `export_val_images(dataset: str, classes: list[str] | None, out: Path, image_size: int, data_dir: Path) -> int` (returns count written) plus an argparse `main()`. Used to build FID reference folders.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_data.py`:

```python
class TestExportValImages:
    def test_exports_pngs(self, afhq_dir, tmp_path) -> None:
        import importlib.util
        from pathlib import Path as _Path

        script = _Path(__file__).parent.parent / "scripts" / "export_val_images.py"
        spec = importlib.util.spec_from_file_location("export_val_images", script)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        out = tmp_path / "fid_ref"
        count = module.export_val_images(
            dataset="afhq",
            classes=["dog"],
            out=out,
            image_size=16,
            data_dir=afhq_dir,
        )
        assert count == 4
        pngs = sorted(out.glob("*.png"))
        assert len(pngs) == 4
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_data.py::TestExportValImages -v`
Expected: FAIL (file not found when loading the module spec).

- [ ] **Step 3: Write the script**

Create `scripts/export_val_images.py`:

```python
"""Export a dataset's val split (optionally class-filtered) as PNG files.

Used to build the reference folder for FID evaluation, e.g.:

    uv run python scripts/export_val_images.py \
        --dataset afhq --classes dog --out data/fid_ref/afhq_dog
"""

import argparse
from pathlib import Path

from torchvision.utils import save_image

from bridge_diffusion.config import DataConfig
from bridge_diffusion.data import get_dataset


def export_val_images(
    dataset: str,
    classes: list[str] | None,
    out: Path,
    image_size: int = 64,
    data_dir: Path = Path("./data"),
) -> int:
    """Write the val split of a dataset to out/ as PNGs in [0, 1].

    Args:
        dataset: Dataset name (mnist, cifar10, afhq).
        classes: Optional class names to keep.
        out: Output directory.
        image_size: Image size (must match the generated samples for FID).
        data_dir: Dataset root directory.

    Returns:
        Number of images written.
    """
    config = DataConfig(
        dataset=dataset,
        classes=classes,
        image_size=image_size,
        data_dir=data_dir,
    )
    val_dataset = get_dataset(config, train=False)

    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    for i in range(len(val_dataset)):
        image = val_dataset[i][0]
        save_image((image.clamp(-1, 1) + 1) / 2, out / f"{i:05d}.png")
    return len(val_dataset)


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=["mnist", "cifar10", "afhq"])
    parser.add_argument("--classes", nargs="*", default=None, help="Class names to keep")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--data-dir", type=Path, default=Path("./data"))
    args = parser.parse_args()

    count = export_val_images(
        dataset=args.dataset,
        classes=args.classes,
        out=args.out,
        image_size=args.image_size,
        data_dir=args.data_dir,
    )
    print(f"Wrote {count} images to {args.out}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_data.py::TestExportValImages -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/export_val_images.py tests/test_data.py
git commit -m "Add script to export val images for FID reference"
```

---

### Task 8: AFHQ download script

**Files:**
- Create: `scripts/download_afhq.sh`

**Interfaces:**
- Produces: `scripts/download_afhq.sh [DATA_DIR]` (default `data`) creating `{DATA_DIR}/afhq/{train,val}/{cat,dog,wild}`.

No unit test (network download); verification is manual via the steps below.

- [ ] **Step 1: Write the script**

Create `scripts/download_afhq.sh`:

```bash
#!/usr/bin/env bash
# Download AFHQ (Animal Faces-HQ) as released with StarGAN v2 (Choi et al. 2020).
# Usage: bash scripts/download_afhq.sh [DATA_DIR]   (default: data)
#
# If the Dropbox URL below has rotted, alternatives:
#   - StarGAN v2 repo: https://github.com/clovaai/stargan-v2 (download.sh, target afhq-dataset)
#   - HuggingFace: https://huggingface.co/datasets/huggan/AFHQ
#   - Kaggle: search "afhq"
set -euo pipefail

DATA_DIR="${1:-data}"
URL="https://www.dropbox.com/s/t9l9o3vsx2jai3z/afhq.zip?dl=1"
ZIP_PATH="${DATA_DIR}/afhq.zip"

if [ -d "${DATA_DIR}/afhq/train" ]; then
    echo "AFHQ already present at ${DATA_DIR}/afhq — nothing to do."
    exit 0
fi

mkdir -p "${DATA_DIR}"
echo "Downloading AFHQ (~500 MB)..."
curl -L --fail -o "${ZIP_PATH}" "${URL}"

echo "Unzipping..."
unzip -q "${ZIP_PATH}" -d "${DATA_DIR}"
rm "${ZIP_PATH}"

echo "Image counts:"
for split in train val; do
    for cls in cat dog wild; do
        count=$(find "${DATA_DIR}/afhq/${split}/${cls}" -type f | wc -l | tr -d ' ')
        echo "  ${split}/${cls}: ${count}"
    done
done
echo "Done. AFHQ is at ${DATA_DIR}/afhq"
```

- [ ] **Step 2: Verify syntax and permissions**

Run: `bash -n scripts/download_afhq.sh && chmod +x scripts/download_afhq.sh && echo OK`
Expected: `OK`.

Do NOT run the actual download as part of this plan (500 MB). The executing engineer should verify the Dropbox URL resolves with:
`curl -sIL "https://www.dropbox.com/s/t9l9o3vsx2jai3z/afhq.zip?dl=1" | grep -i "content-type\|HTTP/"` — if it 404s, substitute the StarGAN v2 repo's current URL (see comments in the script) before committing.

- [ ] **Step 3: Commit**

```bash
git add scripts/download_afhq.sh
git commit -m "Add AFHQ download script"
```

---

### Task 9: Experiment configs

**Files:**
- Create: `configs/cifar_cat2dog.yaml`
- Create: `configs/cifar_cat2dog_smoke.yaml`
- Create: `configs/afhq_cat2dog_64.yaml`
- Create: `configs/afhq_cat2dog_smoke.yaml`
- Test: `tests/test_config.py`

**Interfaces:**
- Consumes: transport fields (Task 1).
- Produces: four launchable YAML configs.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_config.py` (inside `TestTransportConfig` or as a new class):

```python
class TestTransportYamlConfigs:
    """The shipped transport configs must load and be transport-mode."""

    @pytest.mark.parametrize(
        "name",
        [
            "cifar_cat2dog.yaml",
            "cifar_cat2dog_smoke.yaml",
            "afhq_cat2dog_64.yaml",
            "afhq_cat2dog_smoke.yaml",
        ],
    )
    def test_config_loads_and_is_transport(self, name) -> None:
        from pathlib import Path as _Path

        path = _Path(__file__).parent.parent / "configs" / name
        config = ExperimentConfig.from_yaml(path)
        assert config.method == "bridge"
        assert config.data.source_dataset is not None
        assert config.data.classes == ["dog"]
        assert config.data.source_classes == ["cat"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_config.py::TestTransportYamlConfigs -v`
Expected: 4 failures — `FileNotFoundError`.

- [ ] **Step 3: Write the four configs**

`configs/cifar_cat2dog.yaml`:

```yaml
# CIFAR-10 cat -> dog transport with the Gaussian bridge.
# Stepping-stone experiment before AFHQ; runs overnight on Apple Silicon
# or in a few hours on a single CUDA GPU.

name: bridge_cifar_cat2dog
output_dir: ./outputs/cifar_cat2dog
method: bridge

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
  num_steps: 50000
  learning_rate: 0.0002
  weight_decay: 0.0
  grad_clip_norm: 1.0
  checkpoint_every: 10000
  log_every: 100
  seed: 42

bridge:
  T: 0.1
  eps: 0.0000001

sampling:
  num_steps: 100
  num_samples: 64
  show_progress: true
  clip_samples: true

data:
  dataset: cifar10
  classes: [dog]
  source_dataset: cifar10
  source_classes: [cat]
  data_dir: ./data
  image_size: 32
  num_workers: 4
  pin_memory: true
```

`configs/cifar_cat2dog_smoke.yaml`:

```yaml
# Smoke test for CIFAR cat -> dog transport: minutes on MPS/CPU.
# Verifies the full pipeline (paired loading, transport training, checkpointing),
# not sample quality.

name: bridge_cifar_cat2dog_smoke
output_dir: ./outputs/cifar_cat2dog_smoke
method: bridge

model:
  in_channels: 3
  out_channels: 3
  sample_size: 32
  block_out_channels: [64, 128]
  layers_per_block: 1
  down_block_types:
    - DownBlock2D
    - AttnDownBlock2D
  up_block_types:
    - AttnUpBlock2D
    - UpBlock2D
  attention_head_dim: 8
  dropout: 0.0

training:
  batch_size: 16
  num_steps: 200
  learning_rate: 0.0002
  weight_decay: 0.0
  grad_clip_norm: 1.0
  checkpoint_every: 100
  log_every: 50
  seed: 42
  use_ema: false

bridge:
  T: 0.1
  eps: 0.0000001

sampling:
  num_steps: 20
  num_samples: 16
  show_progress: true
  clip_samples: true

data:
  dataset: cifar10
  classes: [dog]
  source_dataset: cifar10
  source_classes: [cat]
  data_dir: ./data
  image_size: 32
  num_workers: 2
  pin_memory: false
```

`configs/afhq_cat2dog_64.yaml`:

```yaml
# AFHQ cat -> dog transport at 64x64 with the Gaussian bridge.
# Headline transport experiment (cf. Rectified Flow / DSBM / DDBM benchmarks).
# Sized for a single modern CUDA GPU (A100/4090: roughly 1-2 days at 150k steps).
# Requires: bash scripts/download_afhq.sh

name: bridge_afhq_cat2dog_64
output_dir: ./outputs/afhq_cat2dog_64
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
  classes: [dog]
  source_dataset: afhq
  source_classes: [cat]
  data_dir: ./data
  image_size: 64
  num_workers: 8
  pin_memory: true
```

`configs/afhq_cat2dog_smoke.yaml`:

```yaml
# Smoke test for AFHQ cat -> dog transport at 32x32: minutes on MPS/CPU.
# Requires: bash scripts/download_afhq.sh

name: bridge_afhq_cat2dog_smoke
output_dir: ./outputs/afhq_cat2dog_smoke
method: bridge

model:
  in_channels: 3
  out_channels: 3
  sample_size: 32
  block_out_channels: [64, 128]
  layers_per_block: 1
  down_block_types:
    - DownBlock2D
    - AttnDownBlock2D
  up_block_types:
    - AttnUpBlock2D
    - UpBlock2D
  attention_head_dim: 8
  dropout: 0.0

training:
  batch_size: 16
  num_steps: 200
  learning_rate: 0.0002
  weight_decay: 0.0
  grad_clip_norm: 1.0
  checkpoint_every: 100
  log_every: 50
  seed: 42
  use_ema: false

bridge:
  T: 0.1
  eps: 0.0000001

sampling:
  num_steps: 20
  num_samples: 16
  show_progress: true
  clip_samples: true

data:
  dataset: afhq
  classes: [dog]
  source_dataset: afhq
  source_classes: [cat]
  data_dir: ./data
  image_size: 32
  num_workers: 2
  pin_memory: false
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_config.py -v`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add configs/cifar_cat2dog.yaml configs/cifar_cat2dog_smoke.yaml configs/afhq_cat2dog_64.yaml configs/afhq_cat2dog_smoke.yaml tests/test_config.py
git commit -m "Add CIFAR and AFHQ cat->dog transport configs"
```

---

### Task 10: Documentation and final verification

**Files:**
- Modify: `README.md`
- Modify: `CLAUDE.md`

**Interfaces:** none (docs only).

- [ ] **Step 1: Add a Transport section to README.md**

Insert after the `### Evaluation` section of `README.md`:

```markdown
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
    --num-steps 100 --output-dir outputs/afhq_translations
uv run python scripts/export_val_images.py --dataset afhq --classes dog \
    --image-size 64 --out data/fid_ref/afhq_dog
uv run bridge-diffusion evaluate --real-dir data/fid_ref/afhq_dog \
    --generated-dir outputs/afhq_translations
```

Smoke-test configs (`*_smoke.yaml`) run the full pipeline in minutes on Apple Silicon.
```

- [ ] **Step 2: Update CLAUDE.md**

In `CLAUDE.md`, under `### Configuration System`, extend the `data` bullet:

```markdown
- **data**: Dataset (`mnist`, `cifar10`, `afhq`), image size, data directory. Transport
  mode: set `source_dataset` (+ optional `classes`/`source_classes` filters) to train
  the bridge from a source image distribution instead of Gaussian noise. See
  `configs/cifar_cat2dog.yaml` and `configs/afhq_cat2dog_64.yaml`.
```

And in the Module Structure listing, update the `datasets.py` line to:

```
│   └── datasets.py         # DataLoader utilities: MNIST/CIFAR-10/AFHQ, class filtering, PairedDataset (transport)
```

- [ ] **Step 3: Final verification**

```bash
uv run pytest -v --tb=short
uv run ruff check src/ scripts/
uv run black --check src/ scripts/ tests/
```

Expected: tests all pass; ruff/black clean (run `uv run black src/ scripts/ tests/` to fix formatting if needed and re-check).

- [ ] **Step 4: Commit**

```bash
git add README.md CLAUDE.md
git commit -m "Document transport experiments"
```

---

## Post-plan manual steps (for Osian, not the executing agent)

1. `uv run bridge-diffusion train --config configs/cifar_cat2dog_smoke.yaml` — end-to-end smoke on MPS (~minutes; downloads CIFAR on first run).
2. `bash scripts/download_afhq.sh` then the AFHQ smoke config.
3. Real runs: `cifar_cat2dog.yaml` locally; `afhq_cat2dog_64.yaml` on a CUDA GPU (1-2 days).
4. Bridge `T` is an open experimental knob for transport — start at 0.1, sweep if translations look washed out.
