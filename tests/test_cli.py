"""Tests for CLI helpers."""

import logging

import pytest
import torch

from bridge_diffusion.cli import _generate_ddpm_samples, _load_sampling_weights
from bridge_diffusion.config import ModelConfig
from bridge_diffusion.models import DDPMDiffusion, DiffusersUNetWrapper


def _tiny_ddpm() -> DDPMDiffusion:
    model_config = ModelConfig(
        in_channels=1,
        out_channels=1,
        sample_size=8,
        block_out_channels=(32, 64),
        layers_per_block=1,
        down_block_types=("DownBlock2D", "DownBlock2D"),
        up_block_types=("UpBlock2D", "UpBlock2D"),
    )
    network = DiffusersUNetWrapper(model_config)
    return DDPMDiffusion(network, num_train_timesteps=10)


def test_generate_ddpm_samples_shape_and_batching() -> None:
    model = _tiny_ddpm()
    samples = _generate_ddpm_samples(
        model,
        num_samples=3,
        shape=(1, 8, 8),
        batch_size=2,
        num_inference_steps=4,
        device=torch.device("cpu"),
    )
    assert samples.shape == (3, 1, 8, 8)
    assert samples.device.type == "cpu"


class TestLoadSamplingWeights:
    """Tests for the --use-ema checkpoint-loading helper."""

    def test_uses_ema_weights_when_present(self) -> None:
        ema_source = _tiny_ddpm()
        checkpoint = {
            "model_state_dict": _tiny_ddpm().state_dict(),
            "ema_model_state_dict": ema_source.state_dict(),
        }
        target = _tiny_ddpm()

        _load_sampling_weights(target, checkpoint, use_ema=True)

        for key, value in target.state_dict().items():
            assert torch.equal(value, ema_source.state_dict()[key])

    def test_warns_and_falls_back_when_ema_requested_but_missing(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        raw_source = _tiny_ddpm()
        checkpoint = {"model_state_dict": raw_source.state_dict()}
        target = _tiny_ddpm()

        with caplog.at_level(logging.WARNING):
            _load_sampling_weights(target, checkpoint, use_ema=True)

        assert "use-ema" in caplog.text
        assert "no EMA weights" in caplog.text
        for key, value in target.state_dict().items():
            assert torch.equal(value, raw_source.state_dict()[key])

    def test_no_warning_when_ema_not_requested(self, caplog: pytest.LogCaptureFixture) -> None:
        raw_source = _tiny_ddpm()
        checkpoint = {"model_state_dict": raw_source.state_dict()}
        target = _tiny_ddpm()

        with caplog.at_level(logging.WARNING):
            _load_sampling_weights(target, checkpoint, use_ema=False)

        assert caplog.text == ""
