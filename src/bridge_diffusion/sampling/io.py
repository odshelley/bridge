"""Saving generated samples as images."""

import logging
from pathlib import Path

import torch

logger = logging.getLogger(__name__)


def save_samples(
    samples: torch.Tensor,
    output_dir: Path,
    prefix: str = "sample",
) -> None:
    """Save generated samples as individual images.

    Args:
        samples: Samples of shape (num_samples, channels, height, width) in [-1, 1].
        output_dir: Directory to save images.
        prefix: Prefix for filenames.
    """
    from torchvision.utils import save_image

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    samples = (samples + 1) / 2
    samples = torch.clamp(samples, 0, 1)

    for i, sample in enumerate(samples):
        path = output_dir / f"{prefix}_{i:04d}.png"
        save_image(sample, path)

    logger.info(f"Saved {len(samples)} samples to {output_dir}")


def save_grid(
    samples: torch.Tensor,
    output_path: Path,
    nrow: int = 8,
) -> None:
    """Save samples as a grid image.

    Args:
        samples: Samples of shape (num_samples, channels, height, width) in [-1, 1].
        output_path: Path for output image.
        nrow: Number of images per row.
    """
    from torchvision.utils import make_grid, save_image

    samples = (samples + 1) / 2
    samples = torch.clamp(samples, 0, 1)

    grid = make_grid(samples, nrow=nrow, padding=2, normalize=False)
    save_image(grid, output_path)

    logger.info(f"Saved sample grid to {output_path}")
