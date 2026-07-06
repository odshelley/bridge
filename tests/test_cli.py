"""Tests for CLI helpers."""

import torch

from bridge_diffusion.cli import _generate_ddpm_samples
from bridge_diffusion.config import BridgeConfig, ModelConfig
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
    return DDPMDiffusion(network, BridgeConfig(), num_train_timesteps=10)


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
