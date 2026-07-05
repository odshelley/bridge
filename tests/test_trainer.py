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
