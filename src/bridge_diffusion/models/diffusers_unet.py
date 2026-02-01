"""Wrapper for diffusers UNet2DModel.

This module provides a thin wrapper around the HuggingFace diffusers UNet2DModel
for use with the Bridge Diffusion training objective. Using the standard diffusers
architecture ensures fair comparison with published DDPM/DDIM results.
"""

import torch
import torch.nn as nn
from diffusers import UNet2DModel

from bridge_diffusion.config import ModelConfig


class DiffusersUNetWrapper(nn.Module):
    """Wrapper to make diffusers UNet2DModel compatible with our training interface.

    The diffusers UNet2DModel returns a dataclass with `.sample` attribute.
    This wrapper extracts the sample tensor directly.
    """

    def __init__(self, config: ModelConfig):
        """Initialise wrapper.

        Args:
            config: Model configuration.
        """
        super().__init__()
        self.unet = UNet2DModel(
            sample_size=config.sample_size,
            in_channels=config.in_channels,
            out_channels=config.out_channels,
            block_out_channels=config.block_out_channels,
            layers_per_block=config.layers_per_block,
            down_block_types=config.down_block_types,
            up_block_types=config.up_block_types,
            attention_head_dim=config.attention_head_dim,
            dropout=config.dropout,
            num_class_embeds=config.num_class_embeds,
        )

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        class_labels: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input images of shape (batch, channels, height, width).
            t: Time values of shape (batch,) in range [0, T].
            class_labels: Optional class labels for conditional generation.

        Returns:
            Predicted output of shape (batch, channels, height, width).
        """
        # diffusers expects timesteps as integers in [0, 1000] by default
        # We'll scale our t (in [0, T]) to this range
        # For now, we pass t directly and let the model handle it
        output = self.unet(x, t, class_labels=class_labels)
        return output.sample


# Standard configurations for benchmarks
CIFAR10_CONFIG = ModelConfig(
    in_channels=3,
    out_channels=3,
    sample_size=32,
    block_out_channels=(128, 256, 256, 256),
    layers_per_block=2,
    down_block_types=(
        "DownBlock2D",
        "AttnDownBlock2D",
        "AttnDownBlock2D",
        "AttnDownBlock2D",
    ),
    up_block_types=(
        "AttnUpBlock2D",
        "AttnUpBlock2D",
        "AttnUpBlock2D",
        "UpBlock2D",
    ),
    attention_head_dim=8,
    dropout=0.0,
)

MNIST_CONFIG = ModelConfig(
    in_channels=1,
    out_channels=1,
    sample_size=32,
    block_out_channels=(64, 128, 256, 256),
    layers_per_block=2,
    down_block_types=(
        "DownBlock2D",
        "AttnDownBlock2D",
        "AttnDownBlock2D",
        "DownBlock2D",
    ),
    up_block_types=(
        "UpBlock2D",
        "AttnUpBlock2D",
        "AttnUpBlock2D",
        "UpBlock2D",
    ),
    attention_head_dim=8,
    dropout=0.0,
)
