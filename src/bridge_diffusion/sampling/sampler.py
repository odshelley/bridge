"""Sampling module for Bridge Diffusion.

Implements Algorithm 2, Gaussian Bridge Simulation (paper_v2 §8) using
Euler-Maruyama discretisation, and the probability flow ODE variant with
higher-order solvers.
"""

import logging
from enum import Enum
from typing import Callable, Optional

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

    Implements the reverse-time simulation (Algorithm 2, Gaussian Bridge
    Simulation, paper_v2 §8) using Euler-Maruyama discretisation to generate
    samples from the data distribution.
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

        Algorithm 2, Gaussian Bridge Simulation (paper_v2 §8):
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

        From Prop. cor:prob_flow_ode (paper_v2 §4), the ODE is:
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

    def _solver_step(
        self,
        drift_fn,
        xi: torch.Tensor,
        t: float,
        dt: float,
        solver: ODESolver,
    ) -> torch.Tensor:
        """Advance xi by one fixed-step ODE update.

        Args:
            drift_fn: Callable (state, time) -> drift tensor.
            xi: Current state of shape (batch, ...).
            t: Current time (scalar).
            dt: Step size.
            solver: Fixed-step solver (EULER, HEUN, or RK4).

        Returns:
            Updated state of shape (batch, ...).
        """
        if solver == ODESolver.EULER:
            return xi + drift_fn(xi, t) * dt
        if solver == ODESolver.HEUN:
            k1 = drift_fn(xi, t)
            k2 = drift_fn(xi + k1 * dt, t + dt)
            return xi + 0.5 * (k1 + k2) * dt
        if solver == ODESolver.RK4:
            k1 = drift_fn(xi, t)
            k2 = drift_fn(xi + 0.5 * k1 * dt, t + 0.5 * dt)
            k3 = drift_fn(xi + 0.5 * k2 * dt, t + 0.5 * dt)
            k4 = drift_fn(xi + k3 * dt, t + dt)
            return xi + (k1 + 2 * k2 + 2 * k3 + k4) * dt / 6
        raise ValueError(f"Not a fixed-step solver: {solver}")

    def _batched(
        self,
        sample_fn: Callable[[int, int], torch.Tensor],
        total_samples: int,
        batch_size: int,
    ) -> torch.Tensor:
        """Generate total_samples in chunks of batch_size via sample_fn(start, n)."""
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        all_samples = []
        start = 0
        while start < total_samples:
            n = min(batch_size, total_samples - start)
            all_samples.append(sample_fn(start, n).cpu())
            start += n
        return torch.cat(all_samples, dim=0)

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

        def drift_fn(state: torch.Tensor, time: float) -> torch.Tensor:
            # Fixed-step solvers (HEUN, RK4) probe intermediate stages at
            # t + dt, which for the final step lands exactly on T. There,
            # denom = max(T - t, 1e-6) hits its floor and the drift explodes.
            # Clamp the evaluation time so it never reaches T.
            time = min(time, self.T - eps)
            time_tensor = torch.full((num_samples,), time, device=self.device)
            return self._ode_drift(state, x0, time, time_tensor)

        for step in tqdm(
            range(num_steps),
            desc=f"Sampling (ODE {solver.value})",
            disable=not self.sampling_config.show_progress,
        ):
            t = eps + step * dt
            xi = self._solver_step(drift_fn, xi, t, dt, solver)

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

        # Time points: from eps to T - eps. The last point feeds directly
        # into the ODE function (and, for stage-based methods, into
        # intermediate stages), so ending exactly at T would hit the
        # denom = max(T - t, 1e-6) floor and the drift would explode.
        eps = 1e-4
        t_span = torch.linspace(eps, self.T - eps, num_steps + 1, device=self.device)

        # Store x0 for the drift function (needs to be accessible in closure)
        x0_stored = x0.clone()

        # Define the ODE function for torchdiffeq
        def ode_func(t: torch.Tensor, xi: torch.Tensor) -> torch.Tensor:
            """ODE drift function: d xi/dt = f(xi, t)."""
            # Adaptive solvers (dopri5/dopri8) probe trial evaluations beyond
            # the requested t_span while controlling step size, so t here can
            # exceed T (verified: as far as ~1.6*T with a stiff drift) even
            # though t_span itself is clamped to [eps, T - eps]. Clamp here
            # too so denom = max(T - t, 1e-6) never sees a negative T - t.
            t_scalar = min(max(t.item(), eps), self.T - eps)
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
        torchdiffeq_solvers = {ODESolver.DOPRI5, ODESolver.DOPRI8, ODESolver.ADAPTIVE_HEUN}

        def sample_fn(start: int, n: int) -> torch.Tensor:
            if solver in torchdiffeq_solvers:
                return self.sample_ode_torchdiffeq(
                    n, shape, num_steps=num_steps, solver=solver.value, rtol=rtol, atol=atol
                )
            return self.sample_ode(n, shape, num_steps=num_steps, solver=solver)

        return self._batched(sample_fn, total_samples, batch_size)

    @torch.no_grad()
    def sample_batch(
        self,
        total_samples: int,
        shape: tuple[int, ...],
        batch_size: int = 64,
        num_steps: Optional[int] = None,
        x0: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Generate samples in batches to manage memory.

        Args:
            total_samples: Total number of samples to generate.
            shape: Shape of each sample.
            batch_size: Batch size for generation.
            num_steps: Number of discretisation steps.
            x0: Optional prior samples of shape (total_samples, *shape). Chunked
                alongside the batches; if None, each batch starts from N(0, I).

        Returns:
            Generated samples of shape (total_samples, *shape).
        """
        if x0 is not None and x0.shape[0] < total_samples:
            raise ValueError(f"x0 has {x0.shape[0]} samples but total_samples={total_samples}")

        def sample_fn(start: int, n: int) -> torch.Tensor:
            x0_batch = x0[start : start + n] if x0 is not None else None
            return self.sample(n, shape, x0=x0_batch, num_steps=num_steps)

        return self._batched(sample_fn, total_samples, batch_size)

