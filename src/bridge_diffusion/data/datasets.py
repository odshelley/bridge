"""Data loading utilities for Bridge Diffusion."""

import dataclasses
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms

from bridge_diffusion.config import DataConfig

AFHQ_TRAIN_COUNTS = {"cat": 5153, "dog": 4739, "wild": 4738}
AFHQ_VAL_COUNTS = {"cat": 500, "dog": 500, "wild": 500}


def get_mnist_transforms(image_size: int = 32) -> transforms.Compose:
    """Get transforms for MNIST dataset.

    Normalises pixel values to [-1, 1] floats (for Gaussian bridge / DDPM).

    Args:
        image_size: Target image size.

    Returns:
        Composed transforms.
    """
    return transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),  # Scale to [-1, 1]
    ])


class _ToFloat:
    """Picklable transform that casts a tensor to float32."""

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return x.float()


def get_mnist_int_transforms(image_size: int = 32) -> transforms.Compose:
    """Get raw integer pixel transforms for MNIST dataset.

    Returns pixel values as float32 in the range [0, 255] (integers preserved),
    required by the Poisson Bridge which operates on non-negative count data.

    Args:
        image_size: Target image size.

    Returns:
        Composed transforms.
    """
    return transforms.Compose([
        transforms.Resize(image_size),
        transforms.PILToTensor(),  # uint8 tensor in [0, 255]
        _ToFloat(),                # cast to float32, keep range — picklable
    ])


def get_cifar10_transforms(image_size: int = 32) -> transforms.Compose:
    """Get transforms for CIFAR-10 dataset.

    Args:
        image_size: Target image size.

    Returns:
        Composed transforms.
    """
    return transforms.Compose([
        transforms.Resize(image_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # Scale to [-1, 1]
    ])


def get_cifar10_eval_transforms(image_size: int = 32) -> transforms.Compose:
    """Get evaluation transforms for CIFAR-10 (no augmentation).

    Args:
        image_size: Target image size.

    Returns:
        Composed transforms.
    """
    return transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])


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


def _make_base_dataset(name: str, config: DataConfig, train: bool) -> torch.utils.data.Dataset:
    """Construct an unfiltered dataset by name.

    Args:
        name: Dataset name (case-insensitive): "mnist", "cifar10", or "afhq".
        config: Data configuration.
        train: Whether to load training or test/val set.

    Returns:
        PyTorch dataset.

    Raises:
        FileNotFoundError: If AFHQ data is not present at the expected path.
        ValueError: If `name` is not a recognised dataset.
    """
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
    elif name.lower() == "afhq":
        split = "train" if train else "val"
        root = Path(config.data_dir) / "afhq" / split
        if not root.is_dir():
            raise FileNotFoundError(
                f"AFHQ not found at {root}. Download it first: bash scripts/download_afhq.sh"
            )
        transform = (
            get_afhq_int_transforms(config.image_size, train=train)
            if config.raw_pixels
            else get_afhq_transforms(config.image_size, train=train)
        )
        return datasets.ImageFolder(root=str(root), transform=transform)
    else:
        raise ValueError(f"Unknown dataset: {name}")


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


def load_source_val_images(config: DataConfig, n: int, spread: bool = False) -> torch.Tensor:
    """Load up to n images from the val split of a config's source dataset.

    Builds a plain (non-transport) view of the source side via dataclasses.replace,
    so the returned dataset yields (image, label) items.

    Args:
        config: A transport-mode data config (source_dataset must be set).
        n: Maximum number of images to load.
        spread: If True, take evenly spaced indices across the split
            (deterministic sample grid); otherwise take the first n.

    Returns:
        Tensor of shape (min(n, len(val)), C, H, W).

    Raises:
        ValueError: If config.source_dataset is None or the val split is empty.
    """
    if config.source_dataset is None:
        raise ValueError("load_source_val_images requires config.source_dataset to be set")

    source_config = dataclasses.replace(
        config,
        dataset=config.source_dataset,
        classes=config.source_classes,
        source_dataset=None,
        source_classes=None,
    )
    val = get_dataset(source_config, train=False)
    if len(val) == 0:
        raise ValueError("Source val split is empty")

    count = min(n, len(val))
    if spread:
        indices = torch.linspace(0, len(val) - 1, steps=count).long().tolist()
    else:
        indices = range(count)
    return torch.stack([val[i][0] for i in indices])


def get_dataloader(
    config: DataConfig,
    batch_size: int,
    train: bool = True,
    num_workers: Optional[int] = None,
) -> DataLoader:
    """Get dataloader based on configuration.

    Args:
        config: Data configuration.
        batch_size: Batch size.
        train: Whether to load training or test set.
        num_workers: Number of workers for data loading.

    Returns:
        PyTorch DataLoader.
    """
    dataset = get_dataset(config, train=train)

    if train and len(dataset) < batch_size:
        raise ValueError(
            f"Training dataset has {len(dataset)} samples but batch_size={batch_size} "
            "with drop_last=True would yield zero batches"
        )

    if num_workers is None:
        num_workers = config.num_workers

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train,
        num_workers=num_workers,
        pin_memory=config.pin_memory,
        drop_last=train,
    )

    return dataloader


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
        unknown = [c for c in class_names if c not in AFHQ_TRAIN_COUNTS]
        if unknown:
            raise ValueError(
                f"Unknown class(es) {unknown}; valid classes: {list(AFHQ_TRAIN_COUNTS)}"
            )
        return {
            "num_channels": 3,
            "image_size": config.image_size,
            "num_classes": len(class_names),
            "train_size": sum(AFHQ_TRAIN_COUNTS[c] for c in class_names),
            "test_size": sum(AFHQ_VAL_COUNTS[c] for c in class_names),
        }
    else:
        raise ValueError(f"Unknown dataset: {config.dataset}")


def filter_classes(
    dataset: Dataset,
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
