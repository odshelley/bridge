"""Sampling module for Bridge Diffusion.

Implements Algorithm 2.2.2 (Simulation) using Euler-Maruyama discretisation,
and the probability flow ODE variant with higher-order solvers.
"""

import logging
from enum import Enum
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torchdiffeq import odeint
from tqdm import tqdm

from bridge_diffusion.config import BridgeConfig, SamplingConfig

logger = logging.getLogger(__name__)


class ODESolver(Enum):
    """Available ODE solvers for probability flow sampling."""
    EULER = "euler"
    HEUN = "heun"  # 2nd order, a.k.a. improved Euler
    RK4 = "rk4"    # 4th order Runge-Kutta
    # torchdiffeq solvers
    DOPRI5 = "dopri5"  # Adaptive Dormand-Prince (RK45)
    DOPRI8 = "dopri8"  # Adaptive 8th order
    ADAPTIVE_HEUN = "adaptive_heun"  # Adaptive Heun


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
        x0: Optional[torch.Tensor] = None,
        num_steps: Optional[int] = None,
        return_trajectory: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
        """Generate samples using Euler-Maruyama discretisation of the SDE.

        Algorithm 2.2.2 (Simulation):
        xi_{t+dt} = xi_t + (E[Y|xi_t] - xi_t) / (T - t) * dt + dZ_t

        where dZ_t ~ N(0, dt * I) is the Brownian increment.

        Args:
            num_samples: Number of samples to generate.
            shape: Shape of each sample (channels, height, width).
            x0: Optional prior samples. If None, sample from N(0, I).
            num_steps: Number of discretisation steps. If None, use config.
            return_trajectory: Whether to return full trajectory.

        Returns:
            Generated samples of shape (num_samples, *shape).
            If return_trajectory, also returns list of intermediate samples.
        """
        if num_steps is None:
            num_steps = self.sampling_config.num_steps

        if x0 is None:
            xi = torch.randn(num_samples, *shape, device=self.device)
        else:
            xi = x0.to(self.device)

        dt = self.T / num_steps
        sqrt_dt = dt ** 0.5
        trajectory = [xi.clone()] if return_trajectory else []

        # Euler-Maruyama: start at prior and evolve to data distribution
        for step in tqdm(
            range(num_steps),
            desc="Sampling (SDE)",
            disable=not self.sampling_config.show_progress,
        ):
            t = step * dt
            t_tensor = torch.full((num_samples,), t, device=self.device)

            # Network predicts E[Y | xi_t]
            y_pred = self.model(xi, t_tensor)

            # Drift: (E[Y|xi_t] - xi_t) / (T - t)
            denom = max(self.T - t, 1e-6)
            drift = (y_pred - xi) / denom

            # Euler-Maruyama step with Brownian increment
            xi = xi + drift * dt
            if step < num_steps - 1:
                xi = xi + sqrt_dt * torch.randn_like(xi)

            if return_trajectory:
                trajectory.append(xi.clone())

        if self.sampling_config.clip_samples:
            xi = torch.clamp(xi, -1.0, 1.0)

        if return_trajectory:
            return xi, trajectory
        return xi

    def _ode_drift(
        self,
        xi: torch.Tensor,
        x0: torch.Tensor,
        t: float,
        t_tensor: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the probability flow ODE drift.

        From the paper, the ODE is:
        d xi_t = [1/2 * (xi_t - x)/t + 1/2 * (E[Y|xi_t] - xi_t)/(T-t)] dt

        Where the network predicts E[Y|xi_t] (the expected data given xi_t).

        Args:
            xi: Current state of shape (batch, ...).
            x0: Initial noise of shape (batch, ...).
            t: Current time (scalar).
            t_tensor: Time as tensor of shape (batch,).

        Returns:
            Drift of shape (batch, ...).
        """
        # Network predicts E[Y | xi_t]
        y_pred = self.model(xi, t_tensor)

        # First term: (xi - x0) / t  (drift from initial noise)
        # Avoid division by zero at t=0
        t_safe = max(t, 1e-6)
        term1 = (xi - x0) / t_safe

        # Second term: (E[Y|xi] - xi) / (T - t)  (drift towards data)
        denom = max(self.T - t, 1e-6)
        term2 = (y_pred - xi) / denom

        # Combined ODE drift (factor of 1/2 on each term)
        return 0.5 * term1 + 0.5 * term2

    @torch.no_grad()
    def sample_ode(
        self,
        num_samples: int,
        shape: tuple[int, ...],
        x0: Optional[torch.Tensor] = None,
        num_steps: Optional[int] = None,
        solver: ODESolver = ODESolver.HEUN,
        return_trajectory: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
        """Generate samples using the probability flow ODE.

        This uses the deterministic ODE formulation:
        d xi_t = [1/2 * (xi_t - x)/t + 1/2 * (E[Y|xi_t] - xi_t)/(T-t)] dt

        Being deterministic, this allows higher-order ODE solvers for
        better accuracy with fewer steps.

        Args:
            num_samples: Number of samples to generate.
            shape: Shape of each sample (channels, height, width).
            x0: Optional initial noise. If None, sample from N(0, I).
            num_steps: Number of discretisation steps. If None, use config.
            solver: ODE solver to use (euler, heun, rk4).
            return_trajectory: Whether to return full trajectory.

        Returns:
            Generated samples of shape (num_samples, *shape).
            If return_trajectory, also returns list of intermediate samples.
        """
        if num_steps is None:
            num_steps = self.sampling_config.num_steps

        # Initial noise x0 ~ N(0, I)
        if x0 is None:
            x0 = torch.randn(num_samples, *shape, device=self.device)
        else:
            x0 = x0.to(self.device)

        # Start at t=eps (avoid singularity at t=0)
        eps = 1e-4
        dt = (self.T - eps) / num_steps
        xi = x0.clone()

        trajectory = [xi.clone()] if return_trajectory else []

        for step in tqdm(
            range(num_steps),
            desc=f"Sampling (ODE {solver.value})",
            disable=not self.sampling_config.show_progress,
        ):
            t = eps + step * dt
            t_tensor = torch.full((num_samples,), t, device=self.device)

            if solver == ODESolver.EULER:
                # Simple Euler: xi_{n+1} = xi_n + f(xi_n, t_n) * dt
                drift = self._ode_drift(xi, x0, t, t_tensor)
                xi = xi + drift * dt

            elif solver == ODESolver.HEUN:
                # Heun's method (improved Euler / RK2):
                # k1 = f(xi_n, t_n)
                # k2 = f(xi_n + k1*dt, t_{n+1})
                # xi_{n+1} = xi_n + 0.5*(k1 + k2)*dt
                k1 = self._ode_drift(xi, x0, t, t_tensor)

                t_next = t + dt
                t_next_tensor = torch.full((num_samples,), t_next, device=self.device)
                xi_euler = xi + k1 * dt
                k2 = self._ode_drift(xi_euler, x0, t_next, t_next_tensor)

                xi = xi + 0.5 * (k1 + k2) * dt

            elif solver == ODESolver.RK4:
                # Classic 4th-order Runge-Kutta
                # k1 = f(xi_n, t_n)
                # k2 = f(xi_n + k1*dt/2, t_n + dt/2)
                # k3 = f(xi_n + k2*dt/2, t_n + dt/2)
                # k4 = f(xi_n + k3*dt, t_n + dt)
                # xi_{n+1} = xi_n + (k1 + 2*k2 + 2*k3 + k4)*dt/6
                k1 = self._ode_drift(xi, x0, t, t_tensor)

                t_mid = t + 0.5 * dt
                t_mid_tensor = torch.full((num_samples,), t_mid, device=self.device)
                k2 = self._ode_drift(xi + 0.5 * k1 * dt, x0, t_mid, t_mid_tensor)
                k3 = self._ode_drift(xi + 0.5 * k2 * dt, x0, t_mid, t_mid_tensor)

                t_next = t + dt
                t_next_tensor = torch.full((num_samples,), t_next, device=self.device)
                k4 = self._ode_drift(xi + k3 * dt, x0, t_next, t_next_tensor)

                xi = xi + (k1 + 2 * k2 + 2 * k3 + k4) * dt / 6

            if return_trajectory:
                trajectory.append(xi.clone())

        if self.sampling_config.clip_samples:
            xi = torch.clamp(xi, -1.0, 1.0)

        if return_trajectory:
            return xi, trajectory
        return xi

    @torch.no_grad()
    def sample_ode_torchdiffeq(
        self,
        num_samples: int,
        shape: tuple[int, ...],
        x0: Optional[torch.Tensor] = None,
        num_steps: Optional[int] = None,
        solver: str = "dopri5",
        rtol: float = 1e-5,
        atol: float = 1e-5,
    ) -> torch.Tensor:
        """Generate samples using torchdiffeq ODE solvers.

        Uses the well-tested torchdiffeq library for ODE integration.
        Supports adaptive solvers like dopri5 (Dormand-Prince RK45).

        Args:
            num_samples: Number of samples to generate.
            shape: Shape of each sample (channels, height, width).
            x0: Optional initial noise. If None, sample from N(0, I).
            num_steps: Number of evaluation points (for fixed-step solvers).
            solver: Solver name ('euler', 'heun', 'rk4', 'dopri5', 'dopri8', 'adaptive_heun').
            rtol: Relative tolerance for adaptive solvers.
            atol: Absolute tolerance for adaptive solvers.

        Returns:
            Generated samples of shape (num_samples, *shape).
        """
        if num_steps is None:
            num_steps = self.sampling_config.num_steps

        # Initial noise x0 ~ N(0, I)
        if x0 is None:
            x0 = torch.randn(num_samples, *shape, device=self.device)
        else:
            x0 = x0.to(self.device)

        # Time points: from eps to T
        eps = 1e-4
        t_span = torch.linspace(eps, self.T, num_steps + 1, device=self.device)

        # Store x0 for the drift function (needs to be accessible in closure)
        x0_stored = x0.clone()

        # Define the ODE function for torchdiffeq
        def ode_func(t: torch.Tensor, xi: torch.Tensor) -> torch.Tensor:
            """ODE drift function: d xi/dt = f(xi, t)."""
            t_scalar = t.item()
            batch_size = xi.shape[0]
            t_tensor = torch.full((batch_size,), t_scalar, device=self.device)
            return self._ode_drift(xi, x0_stored, t_scalar, t_tensor)

        # Integrate the ODE
        logger.info(f"Using torchdiffeq solver: {solver}")
        solution = odeint(
            ode_func,
            x0,
            t_span,
            method=solver,
            rtol=rtol,
            atol=atol,
        )

        # solution shape: (num_steps+1, num_samples, *shape)
        # Take the final state
        xi = solution[-1]

        if self.sampling_config.clip_samples:
            xi = torch.clamp(xi, -1.0, 1.0)

        return xi

    @torch.no_grad()
    def sample_batch_ode(
        self,
        total_samples: int,
        shape: tuple[int, ...],
        batch_size: int = 64,
        num_steps: Optional[int] = None,
        solver: ODESolver = ODESolver.HEUN,
        rtol: float = 1e-5,
        atol: float = 1e-5,
    ) -> torch.Tensor:
        """Generate samples in batches using the probability flow ODE.

        Args:
            total_samples: Total number of samples to generate.
            shape: Shape of each sample.
            batch_size: Batch size for generation.
            num_steps: Number of discretisation steps.
            solver: ODE solver to use.
            rtol: Relative tolerance (for torchdiffeq adaptive solvers).
            atol: Absolute tolerance (for torchdiffeq adaptive solvers).

        Returns:
            Generated samples of shape (total_samples, *shape).
        """
        # Check if using torchdiffeq solver
        torchdiffeq_solvers = {ODESolver.DOPRI5, ODESolver.DOPRI8, ODESolver.ADAPTIVE_HEUN}
        use_torchdiffeq = solver in torchdiffeq_solvers

        all_samples = []
        remaining = total_samples

        while remaining > 0:
            current_batch = min(batch_size, remaining)
            if use_torchdiffeq:
                samples = self.sample_ode_torchdiffeq(
                    current_batch, shape, num_steps=num_steps, 
                    solver=solver.value, rtol=rtol, atol=atol
                )
            else:
                samples = self.sample_ode(
                    current_batch, shape, num_steps=num_steps, solver=solver
                )
            all_samples.append(samples.cpu())
            remaining -= current_batch

        return torch.cat(all_samples, dim=0)

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
        return self.sample(num_samples, shape, x0=y, num_steps=num_steps)

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
