"""Tests for data loading module."""

import pytest
import torch

from bridge_diffusion.config import DataConfig
from bridge_diffusion.data import get_data_info, get_dataloader, get_dataset


class TestDataInfo:
    """Tests for dataset information."""

    def test_mnist_info(self) -> None:
        """Test MNIST dataset info."""
        config = DataConfig(dataset="mnist")
        info = get_data_info(config)

        assert info["num_channels"] == 1
        assert info["num_classes"] == 10
        assert info["train_size"] == 60000
        assert info["test_size"] == 10000

    def test_cifar10_info(self) -> None:
        """Test CIFAR-10 dataset info."""
        config = DataConfig(dataset="cifar10")
        info = get_data_info(config)

        assert info["num_channels"] == 3
        assert info["num_classes"] == 10
        assert info["train_size"] == 50000
        assert info["test_size"] == 10000

    def test_unknown_dataset_raises(self) -> None:
        """Test that unknown dataset raises error."""
        # Create config with invalid dataset through a workaround
        config = DataConfig()
        # Override the field directly for testing
        object.__setattr__(config, "dataset", "unknown")
        with pytest.raises(ValueError, match="Unknown dataset"):
            get_data_info(config)


class TestDataConfig:
    """Tests for DataConfig."""

    def test_default_values(self) -> None:
        """Test default data configuration."""
        config = DataConfig()
        assert config.dataset == "mnist"
        assert config.image_size == 32
        assert config.num_workers == 4

    def test_custom_values(self) -> None:
        """Test custom data configuration."""
        config = DataConfig(
            dataset="cifar10",
            image_size=64,
            num_workers=8,
        )
        assert config.dataset == "cifar10"
        assert config.image_size == 64
        assert config.num_workers == 8


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line("markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')")


# Note: These tests require downloading datasets, so they're marked as slow
# and can be skipped in quick test runs


@pytest.mark.slow
class TestDataLoading:
    """Tests for actual data loading (requires dataset download)."""

    def test_mnist_dataloader(self, tmp_path) -> None:
        """Test MNIST dataloader."""
        config = DataConfig(
            dataset="mnist",
            data_dir=tmp_path,
            image_size=32,
        )

        loader = get_dataloader(config, batch_size=4, train=True)
        batch = next(iter(loader))

        images, labels = batch
        assert images.shape == (4, 1, 32, 32)
        assert labels.shape == (4,)

    def test_cifar10_dataloader(self, tmp_path) -> None:
        """Test CIFAR-10 dataloader."""
        config = DataConfig(
            dataset="cifar10",
            data_dir=tmp_path,
            image_size=32,
        )

        loader = get_dataloader(config, batch_size=4, train=True)
        batch = next(iter(loader))

        images, labels = batch
        assert images.shape == (4, 3, 32, 32)
        assert labels.shape == (4,)
