"""DDPM baseline model using diffusers scheduler.

Implements the standard DDPM (Ho et al. 2020) for comparison with Bridge diffusion.
Uses diffusers.DDPMScheduler for the noise schedule and sampling logic.
"""

import torch
import torch.nn as nn
from diffusers import DDPMScheduler

from bridge_diffusion.config import BridgeConfig


class DDPMDiffusion(nn.Module):
    """Standard DDPM diffusion model using diffusers scheduler.

    This wraps a network to predict noise ε, using the standard DDPM objective:
    L = E[||ε - ε_θ(√ᾱₜx₀ + √(1-ᾱₜ)ε, t)||²]
    """

    def __init__(
        self,
        network: nn.Module,
        config: BridgeConfig,
        num_train_timesteps: int = 1000,
        beta_schedule: str = "linear",
    ):
        """Initialise DDPMDiffusion.

        Args:
            network: Neural network that predicts noise.
            config: Bridge config (used for compatibility, T not used).
            num_train_timesteps: Number of diffusion steps.
            beta_schedule: Type of beta schedule ("linear", "cosine", "squaredcos_cap_v2").
        """
        super().__init__()
        self.network = network
        self.num_train_timesteps = num_train_timesteps

        self.scheduler = DDPMScheduler(
            num_train_timesteps=num_train_timesteps,
            beta_schedule=beta_schedule,
            prediction_type="epsilon",
            clip_sample=False,
        )

    def compute_training_loss(
        self,
        x: torch.Tensor,
        y: torch.Tensor,  # unused, for interface compatibility with Bridge
    ) -> torch.Tensor:
        """Compute DDPM training loss.

        Args:
            x: Data samples of shape (batch, channels, height, width).
            y: Unused (kept for interface compatibility with Bridge).

        Returns:
            Scalar loss value.
        """
        batch_size = x.shape[0]
        device = x.device

        # Sample random timesteps
        t = torch.randint(0, self.num_train_timesteps, (batch_size,), device=device)

        # Sample noise
        noise = torch.randn_like(x)

        # Add noise to images using scheduler
        noisy_images = self.scheduler.add_noise(x, noise, t)

        # Predict noise
        noise_pred = self.network(noisy_images, t)

        # MSE loss
        loss = torch.mean((noise - noise_pred) ** 2)

        return loss

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass - predict noise.

        Args:
            x: Noisy input of shape (batch, channels, height, width).
            t: Timesteps of shape (batch,).

        Returns:
            Predicted noise.
        """
        return self.network(x, t)

    @torch.no_grad()
    def sample(
        self,
        num_samples: int,
        shape: tuple[int, ...],
        device: torch.device,
        num_inference_steps: int | None = None,
    ) -> torch.Tensor:
        """Generate samples using DDPM reverse process.

        Args:
            num_samples: Number of samples to generate.
            shape: Shape of each sample (channels, height, width).
            device: Device to generate on.
            num_inference_steps: Steps for sampling (None = use all training steps).

        Returns:
            Generated samples.
        """
        if num_inference_steps is not None:
            self.scheduler.set_timesteps(num_inference_steps, device=device)
        else:
            self.scheduler.set_timesteps(self.num_train_timesteps, device=device)

        # Start from pure noise
        sample = torch.randn(num_samples, *shape, device=device)

        for t in self.scheduler.timesteps:
            t_batch = t.expand(num_samples).to(device)
            noise_pred = self.network(sample, t_batch)
            sample = self.scheduler.step(noise_pred, t, sample).prev_sample

        return sample
