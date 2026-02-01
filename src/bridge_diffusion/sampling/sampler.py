"""Sampling module for Bridge Diffusion.

Implements Algorithm 2.2.2 (Simulation) using Euler-Maruyama discretisation.
"""

import logging
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from tqdm import tqdm

from bridge_diffusion.config import BridgeConfig, SamplingConfig

logger = logging.getLogger(__name__)


class Sampler:
    """Sampler for Bridge Diffusion models.

    Implements the reverse-time simulation (Algorithm 2.2.2) using
    Euler-Maruyama discretisation to generate samples from the data distribution.
    """

    def __init__(
        self,
        model: nn.Module,
        bridge_config: BridgeConfig,
        sampling_config: SamplingConfig,
        device: torch.device,
    ):
        """Initialise Sampler.

        Args:
            model: Trained bridge diffusion model (or just the network).
            bridge_config: Bridge configuration.
            sampling_config: Sampling configuration.
            device: Device for sampling.
        """
        self.model = model.to(device)
        self.model.eval()
        self.bridge_config = bridge_config
        self.sampling_config = sampling_config
        self.device = device
        self.T = bridge_config.T

    @torch.no_grad()
    def sample(
        self,
        num_samples: int,
        shape: tuple[int, ...],
        y: Optional[torch.Tensor] = None,
        num_steps: Optional[int] = None,
        return_trajectory: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
        """Generate samples using Euler-Maruyama discretisation.

        Algorithm 2.2.2 (Simulation):
        1. Start with y ~ N(0, I) (prior)
        2. Simulate reverse-time SDE: dX_t = b(X_t, T-t) dt
        3. Use learned drift b_theta

        For the bridge diffusion without noise in the reverse process:
        X_{t+dt} = X_t + b_theta(X_t, T-t) * dt

        Args:
            num_samples: Number of samples to generate.
            shape: Shape of each sample (channels, height, width).
            y: Optional prior samples. If None, sample from N(0, I).
            num_steps: Number of discretisation steps. If None, use config.
            return_trajectory: Whether to return full trajectory.

        Returns:
            Generated samples of shape (num_samples, *shape).
            If return_trajectory, also returns list of intermediate samples.
        """
        if num_steps is None:
            num_steps = self.sampling_config.num_steps

        if y is None:
            x = torch.randn(num_samples, *shape, device=self.device)
        else:
            x = y.to(self.device)

        dt = self.T / num_steps
        trajectory = [x.clone()] if return_trajectory else []

        # Euler-Maruyama: start at prior and evolve to data distribution
        for step in tqdm(
            range(num_steps),
            desc="Sampling",
            disable=not self.sampling_config.show_progress,
        ):
            t = step * dt
            t_tensor = torch.full((num_samples,), t, device=self.device)

            drift = self.model(x, t_tensor)
            x = x + drift * dt

            if return_trajectory:
                trajectory.append(x.clone())

        if self.sampling_config.clip_samples:
            x = torch.clamp(x, -1.0, 1.0)

        if return_trajectory:
            return x, trajectory
        return x

    @torch.no_grad()
    def sample_with_guidance(
        self,
        num_samples: int,
        shape: tuple[int, ...],
        y: Optional[torch.Tensor] = None,
        num_steps: Optional[int] = None,
        guidance_scale: float = 1.0,
    ) -> torch.Tensor:
        """Generate samples with classifier-free guidance (if supported).

        Args:
            num_samples: Number of samples to generate.
            shape: Shape of each sample.
            y: Optional prior samples.
            num_steps: Number of discretisation steps.
            guidance_scale: Scale for classifier-free guidance.

        Returns:
            Generated samples.
        """
        # Placeholder for classifier-free guidance extension
        return self.sample(num_samples, shape, y, num_steps)

    @torch.no_grad()
    def sample_batch(
        self,
        total_samples: int,
        shape: tuple[int, ...],
        batch_size: int = 64,
        num_steps: Optional[int] = None,
    ) -> torch.Tensor:
        """Generate samples in batches to manage memory.

        Args:
            total_samples: Total number of samples to generate.
            shape: Shape of each sample.
            batch_size: Batch size for generation.
            num_steps: Number of discretisation steps.

        Returns:
            Generated samples of shape (total_samples, *shape).
        """
        all_samples = []
        remaining = total_samples

        while remaining > 0:
            current_batch = min(batch_size, remaining)
            samples = self.sample(current_batch, shape, num_steps=num_steps)
            all_samples.append(samples.cpu())
            remaining -= current_batch

        return torch.cat(all_samples, dim=0)

    def save_samples(
        self,
        samples: torch.Tensor,
        output_dir: Path,
        prefix: str = "sample",
    ) -> None:
        """Save generated samples as images.

        Args:
            samples: Samples of shape (num_samples, channels, height, width).
            output_dir: Directory to save images.
            prefix: Prefix for filenames.
        """
        from torchvision.utils import save_image

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Denormalise from [-1, 1] to [0, 1]
        samples = (samples + 1) / 2
        samples = torch.clamp(samples, 0, 1)

        for i, sample in enumerate(samples):
            path = output_dir / f"{prefix}_{i:04d}.png"
            save_image(sample, path)

        logger.info(f"Saved {len(samples)} samples to {output_dir}")

    def save_grid(
        self,
        samples: torch.Tensor,
        output_path: Path,
        nrow: int = 8,
    ) -> None:
        """Save samples as a grid image.

        Args:
            samples: Samples of shape (num_samples, channels, height, width).
            output_path: Path for output image.
            nrow: Number of images per row.
        """
        from torchvision.utils import make_grid, save_image

        # Denormalise from [-1, 1] to [0, 1]
        samples = (samples + 1) / 2
        samples = torch.clamp(samples, 0, 1)

        grid = make_grid(samples, nrow=nrow, padding=2, normalize=False)
        save_image(grid, output_path)

        logger.info(f"Saved sample grid to {output_path}")
