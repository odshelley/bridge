"""Tests for DDPMDiffusion training loss wiring and generate()."""

import torch

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


class TestComputeTrainingLoss:
    def test_loss_depends_on_y_not_x(self) -> None:
        """compute_training_loss(x=prior, y=data) must noise y, not x."""
        model = _tiny_ddpm()
        y = torch.randn(2, 1, 8, 8)
        x1 = torch.randn(2, 1, 8, 8)
        x2 = torch.randn(2, 1, 8, 8)
        y2 = torch.randn(2, 1, 8, 8)

        torch.manual_seed(0)
        loss_x1 = model.compute_training_loss(x1, y)
        torch.manual_seed(0)
        loss_x2 = model.compute_training_loss(x2, y)
        assert torch.equal(loss_x1, loss_x2), "loss must be invariant to x (prior)"

        torch.manual_seed(0)
        loss_y2 = model.compute_training_loss(x1, y2)
        assert not torch.equal(loss_x1, loss_y2), "loss must change when y (data) changes"


class TestGenerate:
    def test_generate_returns_correct_shape_and_is_finite(self) -> None:
        model = _tiny_ddpm()
        x = torch.randn(2, 1, 8, 8)
        out = model.generate(x, num_steps=2)
        assert out.shape == (2, 1, 8, 8)
        assert torch.isfinite(out).all()
