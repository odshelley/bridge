"""Data loading utilities for Bridge Diffusion."""

from typing import Optional

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from bridge_diffusion.config import DataConfig



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


def get_dataset(
    config: DataConfig,
    train: bool = True,
) -> torch.utils.data.Dataset:
    """Get dataset based on configuration.

    Args:
        config: Data configuration.
        train: Whether to load training or test set.

    Returns:
        PyTorch dataset.
    """
    if config.dataset.lower() == "mnist":
        transform = (
            get_mnist_int_transforms(config.image_size)
            if config.raw_pixels
            else get_mnist_transforms(config.image_size)
        )
        dataset = datasets.MNIST(
            root=config.data_dir,
            train=train,
            download=True,
            transform=transform,
        )
    elif config.dataset.lower() == "cifar10":
        if train:
            transform = get_cifar10_transforms(config.image_size)
        else:
            transform = get_cifar10_eval_transforms(config.image_size)
        dataset = datasets.CIFAR10(
            root=config.data_dir,
            train=train,
            download=True,
            transform=transform,
        )
    else:
        raise ValueError(f"Unknown dataset: {config.dataset}")

    return dataset


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
        return {
            "num_channels": 3,
            "image_size": config.image_size,
            "num_classes": 10,
            "train_size": 50000,
            "test_size": 10000,
        }
    else:
        raise ValueError(f"Unknown dataset: {config.dataset}")
