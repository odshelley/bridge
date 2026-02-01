"""Tests for diffusers UNet wrapper."""

import pytest
import torch

from bridge_diffusion.config import ModelConfig
from bridge_diffusion.models import DiffusersUNetWrapper


class TestDiffusersUNetWrapper:
    """Tests for the diffusers UNet wrapper."""

    def test_output_shape_mnist(self) -> None:
        """Test that wrapper produces correct output shape for MNIST."""
        config = ModelConfig(
            in_channels=1,
            out_channels=1,
            sample_size=32,
            block_out_channels=(32, 64),
            layers_per_block=1,
            down_block_types=("DownBlock2D", "DownBlock2D"),
            up_block_types=("UpBlock2D", "UpBlock2D"),
        )
        
        model = DiffusersUNetWrapper(config)
        
        batch_size = 2
        x = torch.randn(batch_size, 1, 32, 32)
        t = torch.randint(0, 1000, (batch_size,)).float()
        
        output = model(x, t)
        
        assert output.shape == x.shape

    def test_output_shape_cifar10(self) -> None:
        """Test that wrapper produces correct output shape for CIFAR-10."""
        config = ModelConfig(
            in_channels=3,
            out_channels=3,
            sample_size=32,
            block_out_channels=(32, 64),
            layers_per_block=1,
            down_block_types=("DownBlock2D", "DownBlock2D"),
            up_block_types=("UpBlock2D", "UpBlock2D"),
        )
        
        model = DiffusersUNetWrapper(config)
        
        batch_size = 2
        x = torch.randn(batch_size, 3, 32, 32)
        t = torch.randint(0, 1000, (batch_size,)).float()
        
        output = model(x, t)
        
        assert output.shape == x.shape

    def test_different_timesteps_different_outputs(self) -> None:
        """Test that different timesteps produce different outputs."""
        config = ModelConfig(
            in_channels=1,
            out_channels=1,
            sample_size=32,
            block_out_channels=(32, 64),
            layers_per_block=1,
            down_block_types=("DownBlock2D", "DownBlock2D"),
            up_block_types=("UpBlock2D", "UpBlock2D"),
        )
        
        model = DiffusersUNetWrapper(config)
        
        x = torch.randn(1, 1, 32, 32)
        t1 = torch.tensor([100.0])
        t2 = torch.tensor([900.0])
        
        output1 = model(x, t1)
        output2 = model(x, t2)
        
        assert not torch.allclose(output1, output2)

    def test_gradient_flow(self) -> None:
        """Test that gradients flow through the model."""
        config = ModelConfig(
            in_channels=1,
            out_channels=1,
            sample_size=32,
            block_out_channels=(32, 64),
            layers_per_block=1,
            down_block_types=("DownBlock2D", "DownBlock2D"),
            up_block_types=("UpBlock2D", "UpBlock2D"),
        )
        
        model = DiffusersUNetWrapper(config)
        
        x = torch.randn(2, 1, 32, 32)
        t = torch.randint(0, 1000, (2,)).float()
        
        output = model(x, t)
        loss = output.mean()
        loss.backward()
        
        # Check that at least some parameters have gradients
        has_grad = False
        for param in model.parameters():
            if param.grad is not None and param.grad.abs().sum() > 0:
                has_grad = True
                break
        
        assert has_grad
