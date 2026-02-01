"""Bridge Diffusion model.

Implements the Gaussian Random Bridge diffusion process for generative modelling,
following Algorithm 2.2.1 (Training) and Algorithm 2.2.2 (Simulation) from the paper.
"""

import torch
import torch.nn as nn

from bridge_diffusion.config import BridgeConfig


class BridgeDiffusion(nn.Module):
    """Gaussian Random Bridge diffusion model.

    This implements the bridge process that connects samples from a data distribution
    to samples from a prior distribution (typically Gaussian noise).

    The bridge process has:
    - Expectation: E_t = ((T-t)/T) * x + (t/T) * y
    - Variance: V_t = (t * (T-t)) / T

    Where:
    - x is a sample from the data distribution
    - y is a sample from the prior distribution
    - T is the terminal time
    - t is the current time in [0, T]
    """

    def __init__(self, network: nn.Module, config: BridgeConfig):
        """Initialise BridgeDiffusion.

        Args:
            network: Neural network that predicts the score/drift.
            config: Bridge configuration parameters.
        """
        super().__init__()
        self.network = network
        self.T = config.T
        self.eps = config.eps

    def compute_expectation(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the bridge expectation E_t.

        E_t = ((T-t)/T) * x + (t/T) * y

        Args:
            x: Data samples of shape (batch, ...).
            y: Prior samples of shape (batch, ...).
            t: Time values of shape (batch,).

        Returns:
            Expectation of shape (batch, ...).
        """
        # Reshape t for broadcasting
        t_shape = [t.shape[0]] + [1] * (x.ndim - 1)
        t = t.view(*t_shape)

        return ((self.T - t) / self.T) * x + (t / self.T) * y

    def compute_variance(self, t: torch.Tensor) -> torch.Tensor:
        """Compute the bridge variance V_t.

        V_t = (t * (T-t)) / T

        Args:
            t: Time values of shape (batch,).

        Returns:
            Variance of shape (batch,).
        """
        return (t * (self.T - t)) / self.T

    def sample_bridge(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """Sample from the bridge distribution at time t.

        xi_t ~ N(E_t, V_t * I)

        Args:
            x: Data samples of shape (batch, channels, height, width).
            y: Prior samples of shape (batch, channels, height, width).
            t: Time values of shape (batch,).

        Returns:
            Bridge samples of shape (batch, channels, height, width).
        """
        E_t = self.compute_expectation(x, y, t)
        V_t = self.compute_variance(t)

        # Reshape V_t for broadcasting
        V_t_shape = [V_t.shape[0]] + [1] * (x.ndim - 1)
        V_t = V_t.view(*V_t_shape)

        # Sample from N(E_t, V_t * I)
        noise = torch.randn_like(x)
        return E_t + torch.sqrt(V_t) * noise

    def compute_training_target(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        xi_t: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the training target for the network.

        The target is the drift of the bridge process:
        b(xi_t, t) = (y - xi_t) / (T - t)

        Args:
            x: Data samples (unused but kept for interface consistency).
            y: Prior samples of shape (batch, channels, height, width).
            xi_t: Bridge samples of shape (batch, channels, height, width).
            t: Time values of shape (batch,).

        Returns:
            Target of shape (batch, channels, height, width).
        """
        # Reshape t for broadcasting
        t_shape = [t.shape[0]] + [1] * (xi_t.ndim - 1)
        t = t.view(*t_shape)

        # Clamp to avoid division by zero near T
        denominator = torch.clamp(self.T - t, min=self.eps)
        return (y - xi_t) / denominator

    def compute_training_loss(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the training loss following Algorithm 2.2.1.

        Steps:
        1. Sample t uniformly from [eps, T - eps]
        2. Sample xi_t from the bridge distribution
        3. Compute target drift b(xi_t, t)
        4. Return MSE loss between network prediction and target

        Args:
            x: Data samples of shape (batch, channels, height, width).
            y: Prior samples of shape (batch, channels, height, width).

        Returns:
            Scalar loss value.
        """
        batch_size = x.shape[0]
        device = x.device

        # Sample time uniformly in [eps, T - eps]
        t = torch.rand(batch_size, device=device) * (self.T - 2 * self.eps) + self.eps

        xi_t = self.sample_bridge(x, y, t)
        target = self.compute_training_target(x, y, xi_t, t)
        prediction = self.network(xi_t, t)
        loss = torch.mean((prediction - target) ** 2)

        return loss

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass through the network.

        Args:
            x: Input samples of shape (batch, channels, height, width).
            t: Time values of shape (batch,).

        Returns:
            Network output of shape (batch, channels, height, width).
        """
        return self.network(x, t)
