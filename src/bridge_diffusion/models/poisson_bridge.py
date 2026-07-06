"""Poisson Bridge Diffusion model.

Implements the Poisson Random Bridge diffusion process for generative modelling
of discrete (count) data, following the Poisson Bridge Training and Simulation
algorithms from the paper.

The bridge connects a prior distribution (zeros or Poisson noise) to the data
distribution using Binomial interpolation and Poisson jump dynamics.
"""

import logging

import torch
import torch.nn as nn

from bridge_diffusion.config import PoissonBridgeConfig

logger = logging.getLogger(__name__)


class PoissonBridgeDiffusion(nn.Module):
    """Poisson Random Bridge diffusion model.

    This implements the bridge process for non-negative integer data (e.g. raw
    pixel counts) that connects samples from a prior distribution to samples
    from the data distribution using discrete Poisson/Binomial mechanics.

    Training (Algorithm — Poisson Bridge Training):
        1. Sample data y ~ Psi and prior x ~ Phi
        2. Sample time t ~ U[0, T)
        3. For each coordinate i: xi_t^(i) ~ Binomial(y^(i) - x^(i), t/T) + x^(i)
        4. Network predicts y_hat = f_theta(xi_t, t)
        5. Loss = ||y - y_hat||^2

    Simulation (Algorithm — Poisson Bridge Simulation):
        1. Start from prior xi_0 ~ Phi
        2. For each step k:
            a. Predict y_hat = f_theta(xi_t, t)
            b. lambda_i = max(0, (y_hat^(i) - xi_t^(i)) / (T - t))
            c. Delta_xi^(i) ~ Poisson(lambda_i * delta)
            d. xi_{t+delta}^(i) = xi_t^(i) + Delta_xi^(i)

    Note:
        The bridge is only well-defined for targets y in x + N_0^n, i.e. y must
        dominate the prior x coordinatewise (paper_v2 §5) — the driver only
        jumps upward. This means `prior="poisson"` is only valid when the data
        dominates the sampled prior coordinatewise; otherwise
        `compute_training_loss` emits a warning and clamps the negative
        Binomial count to 0, which generation cannot undo.

    Args:
        network: Neural network that predicts the target data.
        config: Poisson bridge configuration parameters.
    """

    def __init__(self, network: nn.Module, config: PoissonBridgeConfig):
        """Initialise PoissonBridgeDiffusion.

        Args:
            network: Neural network that predicts the target data.
            config: Poisson bridge configuration parameters.
        """
        super().__init__()
        self.network = network
        self.T = config.T
        self.eps = config.eps
        self.num_levels = config.num_levels
        self.prior_type = config.prior
        self.prior_lambda = config.prior_lambda

    def sample_prior(
        self,
        shape: tuple[int, ...],
        device: torch.device,
    ) -> torch.Tensor:
        """Sample from the prior distribution Phi.

        Args:
            shape: Shape of the sample (batch, channels, height, width).
            device: Device for the tensor.

        Returns:
            Prior samples as non-negative integer tensor of given shape.
        """
        if self.prior_type == "zeros":
            return torch.zeros(shape, device=device, dtype=torch.float32)
        elif self.prior_type == "poisson":
            # torch.poisson not supported on MPS — sample on CPU, move to device
            rate = torch.full(shape, self.prior_lambda, device=torch.device("cpu"))
            samples = torch.poisson(rate).to(device)
            return torch.clamp(samples, 0, self.num_levels - 1)
        else:
            raise ValueError(f"Unknown prior type: {self.prior_type}")

    def sample_bridge(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """Sample from the Poisson bridge distribution at time t.

        For each coordinate i:
            xi_t^(i) ~ Binomial(y^(i) - x^(i), t/T) + x^(i)

        Args:
            x: Prior samples of shape (batch, ...).
            y: Data samples of shape (batch, ...).
            t: Time values of shape (batch,).

        Returns:
            Bridge samples of shape (batch, ...).
        """
        # Reshape t for broadcasting
        t_shape = [t.shape[0]] + [1] * (x.ndim - 1)
        t_broadcast = t.view(*t_shape)

        # Binomial count parameter: n = y - x (must be non-negative integer)
        n = (y - x).clamp(min=0)

        # Binomial probability: p = t / T
        p = (t_broadcast / self.T).clamp(0, 1)

        # torch.binomial is not supported on MPS — sample on CPU, move back
        dev = x.device
        dist = torch.distributions.Binomial(total_count=n.cpu(), probs=p.cpu())
        increments = dist.sample().to(dev)

        # xi_t = x + Binomial(y - x, t/T)
        xi_t = x + increments

        return xi_t

    def compute_training_loss(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the training loss following the Poisson Bridge Training algorithm.

        Steps:
        1. Sample t uniformly from [0, T)
        2. Sample xi_t from the Binomial bridge distribution
        3. Network predicts y (data) from xi_t
        4. Return MSE loss between network prediction and true y

        Args:
            x: Prior samples of shape (batch, channels, height, width).
            y: Data samples of shape (batch, channels, height, width).

        Returns:
            Scalar loss value.
        """
        batch_size = y.shape[0]
        device = y.device

        if (y < x).any():
            logger.warning(
                "Poisson bridge assumes y >= x coordinatewise (paper_v2 §5, "
                "y in x + N_0^n); %d coordinates violate this and will be "
                "clamped, so generation cannot reach them from above.",
                int((y < x).sum()),
            )

        # Sample time uniformly in [0, T)
        t = torch.rand(batch_size, device=device) * (self.T - self.eps)

        # Sample from Binomial bridge (integer space)
        xi_t = self.sample_bridge(x, y, t)

        # Normalise to [0, 1] at network boundary
        scale = self.num_levels - 1
        prediction = self.network(xi_t / scale, t)

        # Loss in [0, 1] space — sensible scale for the network
        loss = torch.mean((prediction - y / scale) ** 2)

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
        """Generate samples using the Poisson Bridge Simulation algorithm.

        Traverses the dynamics via the conditioned jump intensity derived from
        Doob's h-transform. At each step:
            1. Predict y_hat = f_theta(xi_t, t)
            2. lambda_i = max(0, (y_hat^(i) - xi_t^(i)) / (T - t))
            3. Delta_xi^(i) ~ Poisson(lambda_i * delta)
            4. xi_{t+delta}^(i) = xi_t^(i) + Delta_xi^(i)

        Args:
            x: Prior samples of shape (batch, channels, height, width).
                Should be non-negative integers.
            num_steps: Number of simulation steps.

        Returns:
            Generated samples of shape (batch, channels, height, width).
        """
        device = x.device
        batch_size = x.shape[0]

        # Start from prior at t = 0
        xi_t = x.clone()

        # Step size
        delta = self.T / num_steps

        for k in range(num_steps):
            t = k * delta
            t_tensor = torch.full((batch_size,), t, device=device)

            # Network predicts E[Y | xi_t] — input/output in [0, 1]
            scale = self.num_levels - 1
            y_pred = self.network(xi_t / scale, t_tensor) * scale  # back to integer space

            # Remaining time (avoid division by zero)
            remaining = max(self.T - t, self.eps)

            # Jump intensity: lambda_i = max(0, (y_hat - xi_t) / (T - t))
            intensity = torch.clamp((y_pred - xi_t) / remaining, min=0)

            # Poisson increment: Delta_xi ~ Poisson(lambda * delta)
            # torch.poisson not supported on MPS — sample on CPU, move back
            rate = intensity * delta
            increment = torch.poisson(rate.cpu()).to(xi_t.device)

            # Update state
            xi_t = xi_t + increment

        return xi_t
