"""Tests for the training loop's transport mode."""

import logging

import pytest
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
from bridge_diffusion.training.trainer import _normalise_samples


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


class _StubPoissonModel:
    """Minimal stand-in exposing num_levels, for _normalise_samples tests."""

    num_levels = 256


class TestNormaliseSamples:
    def test_poisson_model_maps_to_unit_interval(self) -> None:
        model = _StubPoissonModel()
        samples = torch.tensor([0.0, 128.0, 255.0])
        normalised = _normalise_samples(samples, model)
        assert torch.allclose(normalised, torch.tensor([0.0, 128.0 / 255.0, 1.0]))

    def test_poisson_model_clamps_out_of_range(self) -> None:
        model = _StubPoissonModel()
        samples = torch.tensor([-10.0, 300.0])
        normalised = _normalise_samples(samples, model)
        assert torch.allclose(normalised, torch.tensor([0.0, 1.0]))

    def test_plain_model_maps_from_signed_unit_interval(self) -> None:
        model = _tiny_model()
        samples = torch.tensor([-1.0, 0.0, 1.0])
        normalised = _normalise_samples(samples, model)
        assert torch.allclose(normalised, torch.tensor([0.0, 0.5, 1.0]))

    def test_plain_model_clamps_out_of_range(self) -> None:
        model = _tiny_model()
        samples = torch.tensor([-5.0, 5.0])
        normalised = _normalise_samples(samples, model)
        assert torch.allclose(normalised, torch.tensor([0.0, 1.0]))


class TestCheckpointDirDefault:
    def test_defaults_to_output_dir_checkpoints(self, afhq_dir, tmp_path) -> None:
        """Omitting checkpoint_dir must namespace checkpoints under the
        experiment's own output_dir, not a shared cwd-relative 'checkpoints/'
        that back-to-back experiments would silently overwrite."""
        config = _transport_experiment(afhq_dir, tmp_path)
        loader = get_dataloader(config.data, batch_size=2, train=True, num_workers=0)
        trainer = Trainer(
            model=_tiny_model(),
            train_loader=loader,
            config=config,
            device=torch.device("cpu"),
        )
        assert trainer.checkpoint_dir == tmp_path / "outputs" / "checkpoints"
        assert trainer.checkpoint_dir.is_dir()


class TestCheckpointRngState:
    def test_load_checkpoint_restores_rng_state(self, afhq_dir, tmp_path) -> None:
        config = _transport_experiment(afhq_dir, tmp_path)
        loader = get_dataloader(config.data, batch_size=2, train=True, num_workers=0)
        trainer = Trainer(
            model=_tiny_model(),
            train_loader=loader,
            config=config,
            device=torch.device("cpu"),
            checkpoint_dir=tmp_path / "ckpt",
        )

        torch.manual_seed(0)
        checkpoint_path = trainer.checkpoint_dir / "checkpoint_step_0.pt"
        trainer.save_checkpoint()
        expected = torch.randn(3)

        # Perturb the RNG state so a naive resume would replay different draws.
        torch.manual_seed(999)
        torch.randn(50)

        trainer.load_checkpoint(checkpoint_path)
        actual = torch.randn(3)

        assert torch.equal(actual, expected)


class TestCheckpointRngStateOnDeviceResume:
    """Regression test: resuming on a non-CPU device must not crash.

    torch.load(..., map_location=self.device) moves the saved RNG ByteTensor
    onto that device, but torch.set_rng_state requires a CPU ByteTensor. This
    test builds and resumes the trainer on MPS (available on this machine) so
    it structurally cannot pass without restoring the RNG tensors to CPU
    before calling torch.set_rng_state / torch.cuda.set_rng_state_all.
    """

    @pytest.mark.skipif(not torch.backends.mps.is_available(), reason="requires MPS device")
    def test_load_checkpoint_on_mps_does_not_raise_and_restores_rng(
        self, afhq_dir, tmp_path
    ) -> None:
        config = _transport_experiment(afhq_dir, tmp_path)
        loader = get_dataloader(config.data, batch_size=2, train=True, num_workers=0)
        trainer = Trainer(
            model=_tiny_model(),
            train_loader=loader,
            config=config,
            device=torch.device("mps"),
            checkpoint_dir=tmp_path / "ckpt",
        )

        torch.manual_seed(0)
        checkpoint_path = trainer.checkpoint_dir / "checkpoint_step_0.pt"
        trainer.save_checkpoint()
        expected = torch.randn(3)

        # Perturb the RNG state so a naive resume would replay different draws.
        torch.manual_seed(999)
        torch.randn(50)

        trainer.load_checkpoint(checkpoint_path)  # must not raise TypeError
        actual = torch.randn(3)

        assert torch.equal(actual, expected)


class TestCheckpointCudaDeviceCountMismatch:
    """A checkpoint saved with N CUDA devices' RNG state may later be resumed
    on a machine with a different device count. torch.cuda.set_rng_state_all
    must not receive more states than there are current devices; restore
    what's usable, warn about the mismatch, and don't crash.
    """

    def test_resume_with_fewer_devices_clamps_and_warns(
        self, afhq_dir, tmp_path, monkeypatch, caplog
    ) -> None:
        fake_states_saved = [torch.ByteTensor([i]) for i in range(3)]  # saved with 3 "GPUs"
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(torch.cuda, "get_rng_state_all", lambda: fake_states_saved)

        config = _transport_experiment(afhq_dir, tmp_path)
        loader = get_dataloader(config.data, batch_size=2, train=True, num_workers=0)
        trainer = Trainer(
            model=_tiny_model(),
            train_loader=loader,
            config=config,
            device=torch.device("cpu"),
            checkpoint_dir=tmp_path / "ckpt",
        )
        checkpoint_path = trainer.checkpoint_dir / "checkpoint_step_0.pt"
        trainer.save_checkpoint()

        # Resume on a machine with only 1 CUDA device available.
        monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
        restore_calls = []
        monkeypatch.setattr(
            torch.cuda, "set_rng_state_all", lambda states: restore_calls.append(states)
        )

        caplog.set_level(logging.WARNING)
        trainer.load_checkpoint(checkpoint_path)  # must not raise

        assert len(restore_calls) == 1
        assert len(restore_calls[0]) == 1
        assert "3 device(s)" in caplog.text
        assert "1" in caplog.text
