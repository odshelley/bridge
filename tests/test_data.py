"""Tests for data loading module."""

import pytest
import torch
from PIL import Image
from torchvision import datasets as tv_datasets

from bridge_diffusion.config import DataConfig
from bridge_diffusion.data import (
    PairedDataset,
    filter_classes,
    get_data_info,
    get_dataloader,
    get_dataset,
    load_source_val_images,
)
from bridge_diffusion.data.datasets import get_cifar10_int_transforms


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


class TestLoadSourceValImages:
    """Tests for the shared source-val-image loading helper."""

    def _transport_config(self, afhq_dir) -> DataConfig:
        return DataConfig(
            dataset="afhq",
            data_dir=afhq_dir,
            image_size=16,
            classes=["dog"],
            source_dataset="afhq",
            source_classes=["cat"],
        )

    def test_returns_correct_shape_and_count(self, afhq_dir) -> None:
        config = self._transport_config(afhq_dir)
        images = load_source_val_images(config, n=3, spread=False)
        assert images.shape == (3, 3, 16, 16)

    def test_caps_count_at_split_size(self, afhq_dir) -> None:
        config = self._transport_config(afhq_dir)
        images = load_source_val_images(config, n=1000, spread=False)
        assert images.shape[0] == 4  # only 4 cat images in the fake val split

    def test_spread_is_deterministic(self, afhq_dir) -> None:
        config = self._transport_config(afhq_dir)
        first = load_source_val_images(config, n=3, spread=True)
        second = load_source_val_images(config, n=3, spread=True)
        assert torch.equal(first, second)

    def test_raises_when_source_dataset_is_none(self) -> None:
        config = DataConfig(dataset="afhq")
        with pytest.raises(ValueError, match="source_dataset"):
            load_source_val_images(config, n=3)


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
