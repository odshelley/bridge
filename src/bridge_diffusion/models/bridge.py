"""Bridge Diffusion model.

Implements the Gaussian Random Bridge diffusion process for generative modelling,
following Algorithm 2.2.1 (Training) and Algorithm 2.2.2 (Simulation) from the paper.
"""

import torch
import torch.nn as nn

from bridge_diffusion.config import BridgeConfig


class BridgeDiffusion(nn.Module):
    """Gaussian Random Bridge diffusion model.

    This implements the bridge process that connects samples from a prior distribution
    (Gaussian noise) to samples from the data distribution.

    The bridge process has (following paper notation):
    - Expectation: E_t = x + (y - x) * t / T
    - Variance: V_t = t * (T - t) / T

    Where:
    - x is a sample from the prior distribution (noise)
    - y is a sample from the data distribution
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

        E_t = x + (y - x) * t / T

        This interpolates from prior (x, noise) at t=0 to data (y) at t=T.

        Args:
            x: Prior samples (noise) of shape (batch, ...).
            y: Data samples of shape (batch, ...).
            t: Time values of shape (batch,).

        Returns:
            Expectation of shape (batch, ...).
        """
        # Reshape t for broadcasting
        t_shape = [t.shape[0]] + [1] * (x.ndim - 1)
        t = t.view(*t_shape)

        # Paper formula: E = x + (y - x) * t / T
        return x + (y - x) * t / self.T

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
            x: Prior samples (noise) of shape (batch, channels, height, width).
            y: Data samples of shape (batch, channels, height, width).
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

        The target is the drift towards data:
        b(xi_t, t) = (y - xi_t) / (T - t)

        Args:
            x: Prior samples (unused but kept for interface consistency).
            y: Data samples of shape (batch, channels, height, width).
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
        """Compute the training loss following Corollary 2.7 from the paper.

        The network learns to predict the data (y) directly from xi_t.
        Bridge interpolates: noise (x) at t=0 → data (y) at t=T.

        Steps:
        1. Sample t uniformly from [0, T]
        2. Sample xi_t from the bridge distribution
        3. Network predicts y (data) from xi_t
        4. Return MSE loss between network prediction and true y

        Args:
            x: Prior samples (noise) of shape (batch, channels, height, width).
            y: Data samples of shape (batch, channels, height, width).

        Returns:
            Scalar loss value.
        """
        batch_size = y.shape[0]
        device = y.device

        # Sample time uniformly in [0, T]
        t = torch.rand(batch_size, device=device) * self.T

        # Sample from bridge: xi_t ~ N(E_t, V_t)
        # E_t = x + (y - x) * t / T  (interpolation from noise to data)
        xi_t = self.sample_bridge(x, y, t)

        # Network predicts the data y
        prediction = self.network(xi_t, t)
        
        # Loss: predict the data y
        loss = torch.mean((prediction - y) ** 2)

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

    @torch.no_grad()
    def generate(
        self,
        x: torch.Tensor,
        num_steps: int = 100,
    ) -> torch.Tensor:
        """Generate samples using Corollary 2.9 from the paper.

        The network predicts the data y directly. The sampling follows:
        xi_{t+dt} = xi_t + (1/(T-t)) * (f(xi_t, t) - xi_t) * dt + dW

        where f(xi_t, t) is the network's prediction of the data.

        Args:
            x: Prior samples (noise) of shape (batch, channels, height, width).
            num_steps: Number of Euler-Maruyama steps.

        Returns:
            Generated samples of shape (batch, channels, height, width).
        """
        device = x.device
        batch_size = x.shape[0]

        # Start from noise at t = 0
        xi_t = x.clone()
        
        # Time step
        delta = (self.T - self.eps) / num_steps
        
        # Forward Euler from t=0 to t=T (approaching the data)
        for i in range(num_steps):
            t = i * delta
            t_tensor = torch.full((batch_size,), t, device=device)
            
            # Network predicts the data (y)
            y_pred = self.network(xi_t, t_tensor)
            
            # Drift: (y_pred - xi_t) / (T - t)
            inv_lambda = 1.0 / (self.T - t + self.eps)
            drift = inv_lambda * (y_pred - xi_t)
            
            # Euler-Maruyama step
            xi_t = xi_t + drift * delta
            
            # Add noise (except on last step)
            if i < num_steps - 1:
                noise = torch.randn_like(xi_t) * torch.sqrt(torch.tensor(delta, device=device))
                xi_t = xi_t + noise

        return xi_t
