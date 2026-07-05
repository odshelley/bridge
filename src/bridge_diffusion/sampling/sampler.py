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
            raise ValueError(
                f"x0 has {x0.shape[0]} samples but total_samples={total_samples}"
            )

        all_samples = []
        start = 0

        while start < total_samples:
            current_batch = min(batch_size, total_samples - start)
            x0_batch = x0[start : start + current_batch] if x0 is not None else None
            samples = self.sample(current_batch, shape, x0=x0_batch, num_steps=num_steps)
            all_samples.append(samples.cpu())
            start += current_batch

        return torch.cat(all_samples, dim=0)

    def _compute_score(
        self,
        xi: torch.Tensor,
        x0: torch.Tensor,
        t: float,
        t_tensor: torch.Tensor,
        sigma: float = 1.0,
    ) -> torch.Tensor:
        """Compute the score function ∇log p_t(ξ).

        From Proposition 4.1 in the paper:
        (E[Y|ξ] - ξ)/(T-t) = (ξ - x)/t + σ² ∇log p_t(ξ)

        Rearranging:
        ∇log p_t(ξ) = (1/σ²) * [(E[Y|ξ] - ξ)/(T-t) - (ξ - x)/t]

        Args:
            xi: Current state of shape (batch, ...).
            x0: Initial noise of shape (batch, ...).
            t: Current time (scalar).
            t_tensor: Time as tensor of shape (batch,).
            sigma: Diffusion coefficient (default 1.0).

        Returns:
            Score ∇log p_t(ξ) of shape (batch, ...).
        """
        # Network predicts E[Y | xi_t]
        y_pred = self.model(xi, t_tensor)

        # Target pull: (E[Y|ξ] - ξ) / (T - t)
        denom_T = max(self.T - t, 1e-6)
        target_pull = (y_pred - xi) / denom_T

        # Source push: (ξ - x) / t
        t_safe = max(t, 1e-6)
        source_push = (xi - x0) / t_safe

        # Score: (1/σ²) * (target_pull - source_push)
        score = (target_pull - source_push) / (sigma ** 2)
        return score

    @torch.no_grad()
    def sample_predictor_corrector(
        self,
        num_samples: int,
        shape: tuple[int, ...],
        x0: Optional[torch.Tensor] = None,
        num_steps: Optional[int] = None,
        predictor: ODESolver = ODESolver.HEUN,
        corrector_steps: int = 1,
        corrector_snr: float = 0.1,
        sigma: float = 1.0,
        return_trajectory: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
        """Generate samples using Predictor-Corrector sampling.

        Combines ODE integration (predictor) with Langevin dynamics (corrector)
        for improved sample quality. Based on Song et al. "Score-Based Generative
        Modeling through Stochastic Differential Equations".

        Predictor: ODE step using probability flow
            dξ = [½(ξ-x)/t + ½(E[Y|ξ]-ξ)/(T-t)] dt

        Corrector: Langevin dynamics using the score
            ξ_new = ξ_old + ε∇log p_t(ξ) + √(2ε) z
        where:
            ∇log p_t(ξ) = (1/σ²) * [(E[Y|ξ]-ξ)/(T-t) - (ξ-x)/t]

        Args:
            num_samples: Number of samples to generate.
            shape: Shape of each sample (channels, height, width).
            x0: Optional initial noise. If None, sample from N(0, I).
            num_steps: Number of predictor steps. If None, use config.
            predictor: ODE solver for predictor step (euler, heun, rk4).
            corrector_steps: Number of Langevin corrector steps per predictor step.
            corrector_snr: Signal-to-noise ratio for corrector step size.
                Step size ε = (snr * ||noise||/||score||)².
            sigma: Diffusion coefficient (default 1.0).
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
            desc=f"Sampling (PC {predictor.value}+langevin)",
            disable=not self.sampling_config.show_progress,
        ):
            t = eps + step * dt
            t_tensor = torch.full((num_samples,), t, device=self.device)

            # === PREDICTOR STEP (ODE) ===
            if predictor == ODESolver.EULER:
                drift = self._ode_drift(xi, x0, t, t_tensor)
                xi = xi + drift * dt

            elif predictor == ODESolver.HEUN:
                k1 = self._ode_drift(xi, x0, t, t_tensor)
                t_next = t + dt
                t_next_tensor = torch.full((num_samples,), t_next, device=self.device)
                xi_euler = xi + k1 * dt
                k2 = self._ode_drift(xi_euler, x0, t_next, t_next_tensor)
                xi = xi + 0.5 * (k1 + k2) * dt

            elif predictor == ODESolver.RK4:
                k1 = self._ode_drift(xi, x0, t, t_tensor)
                t_mid = t + 0.5 * dt
                t_mid_tensor = torch.full((num_samples,), t_mid, device=self.device)
                k2 = self._ode_drift(xi + 0.5 * k1 * dt, x0, t_mid, t_mid_tensor)
                k3 = self._ode_drift(xi + 0.5 * k2 * dt, x0, t_mid, t_mid_tensor)
                t_next = t + dt
                t_next_tensor = torch.full((num_samples,), t_next, device=self.device)
                k4 = self._ode_drift(xi + k3 * dt, x0, t_next, t_next_tensor)
                xi = xi + (k1 + 2 * k2 + 2 * k3 + k4) * dt / 6

            # === CORRECTOR STEPS (Langevin dynamics) ===
            # Run corrector at the new time point after predictor
            # Skip corrector on the last step to avoid adding noise to final samples
            if step < num_steps - 1 and corrector_steps > 0:
                t_corrector = t + dt
                t_corrector_tensor = torch.full((num_samples,), t_corrector, device=self.device)

                # Bridge variance V_t = t(T-t)/T - use this to anneal the step size
                # This naturally goes to 0 at boundaries, preventing instability
                V_t = (t_corrector * (self.T - t_corrector)) / self.T

                for _ in range(corrector_steps):
                    # Compute score at current position
                    score = self._compute_score(xi, x0, t_corrector, t_corrector_tensor, sigma)

                    # Adaptive step size based on SNR, annealed by bridge variance
                    # ε = snr² * V_t (simpler and more stable than norm-based)
                    noise = torch.randn_like(xi)
                    step_size = corrector_snr ** 2 * V_t
                    sqrt_2_step = (2 * step_size) ** 0.5

                    # Langevin step: ξ_new = ξ + ε∇log p_t(ξ) + √(2ε) z
                    xi = xi + step_size * score + sqrt_2_step * noise

            if return_trajectory:
                trajectory.append(xi.clone())

        if self.sampling_config.clip_samples:
            xi = torch.clamp(xi, -1.0, 1.0)

        if return_trajectory:
            return xi, trajectory
        return xi

    @torch.no_grad()
    def sample_batch_predictor_corrector(
        self,
        total_samples: int,
        shape: tuple[int, ...],
        batch_size: int = 64,
        num_steps: Optional[int] = None,
        predictor: ODESolver = ODESolver.HEUN,
        corrector_steps: int = 1,
        corrector_snr: float = 0.1,
        sigma: float = 1.0,
    ) -> torch.Tensor:
        """Generate samples in batches using Predictor-Corrector sampling.

        Args:
            total_samples: Total number of samples to generate.
            shape: Shape of each sample.
            batch_size: Batch size for generation.
            num_steps: Number of predictor steps.
            predictor: ODE solver for predictor step.
            corrector_steps: Number of Langevin corrector steps.
            corrector_snr: Signal-to-noise ratio for corrector.
            sigma: Diffusion coefficient.

        Returns:
            Generated samples of shape (total_samples, *shape).
        """
        all_samples = []
        remaining = total_samples

        while remaining > 0:
            current_batch = min(batch_size, remaining)
            samples = self.sample_predictor_corrector(
                current_batch,
                shape,
                num_steps=num_steps,
                predictor=predictor,
                corrector_steps=corrector_steps,
                corrector_snr=corrector_snr,
                sigma=sigma,
            )
            all_samples.append(samples.cpu())
            remaining -= current_batch

        return torch.cat(all_samples, dim=0)

    @torch.no_grad()
    def sample_hybrid(
        self,
        num_samples: int,
        shape: tuple[int, ...],
        x0: Optional[torch.Tensor] = None,
        num_steps: Optional[int] = None,
        switch_fraction: float = 0.5,
        ode_solver: ODESolver = ODESolver.HEUN,
        return_trajectory: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
        """Generate samples using ODE then SDE (hybrid approach).

        Uses ODE for early steps (accurate when away from singularities),
        then switches to SDE for later steps (handles t→T singularity better).

        Args:
            num_samples: Number of samples to generate.
            shape: Shape of each sample (channels, height, width).
            x0: Optional initial noise. If None, sample from N(0, I).
            num_steps: Total number of steps. If None, use config.
            switch_fraction: Fraction of steps to use ODE before switching to SDE.
                0.5 means half ODE, half SDE. Default 0.5.
            ode_solver: ODE solver to use for the ODE phase (heun, rk4).
            return_trajectory: Whether to return full trajectory.

        Returns:
            Generated samples of shape (num_samples, *shape).
            If return_trajectory, also returns list of intermediate samples.
        """
        if num_steps is None:
            num_steps = self.sampling_config.num_steps

        # Initial noise
        if x0 is None:
            xi = torch.randn(num_samples, *shape, device=self.device)
        else:
            xi = x0.to(self.device)

        x0_saved = xi.clone()  # Save for ODE drift computation

        dt = self.T / num_steps
        sqrt_dt = dt ** 0.5
        switch_step = int(num_steps * switch_fraction)

        trajectory = [xi.clone()] if return_trajectory else []

        for step in tqdm(
            range(num_steps),
            desc=f"Sampling (hybrid ODE→SDE)",
            disable=not self.sampling_config.show_progress,
        ):
            t = step * dt
            t_tensor = torch.full((num_samples,), t, device=self.device)

            if step < switch_step:
                # === ODE PHASE ===
                if ode_solver == ODESolver.EULER:
                    drift = self._ode_drift(xi, x0_saved, t, t_tensor)
                    xi = xi + drift * dt

                elif ode_solver == ODESolver.HEUN:
                    k1 = self._ode_drift(xi, x0_saved, t, t_tensor)
                    t_next = t + dt
                    t_next_tensor = torch.full((num_samples,), t_next, device=self.device)
                    xi_euler = xi + k1 * dt
                    k2 = self._ode_drift(xi_euler, x0_saved, t_next, t_next_tensor)
                    xi = xi + 0.5 * (k1 + k2) * dt

                elif ode_solver == ODESolver.RK4:
                    k1 = self._ode_drift(xi, x0_saved, t, t_tensor)
                    t_mid = t + 0.5 * dt
                    t_mid_tensor = torch.full((num_samples,), t_mid, device=self.device)
                    k2 = self._ode_drift(xi + 0.5 * k1 * dt, x0_saved, t_mid, t_mid_tensor)
                    k3 = self._ode_drift(xi + 0.5 * k2 * dt, x0_saved, t_mid, t_mid_tensor)
                    t_next = t + dt
                    t_next_tensor = torch.full((num_samples,), t_next, device=self.device)
                    k4 = self._ode_drift(xi + k3 * dt, x0_saved, t_next, t_next_tensor)
                    xi = xi + (k1 + 2 * k2 + 2 * k3 + k4) * dt / 6
            else:
                # === SDE PHASE ===
                # Drift: (E[Y|xi] - xi) / (T - t)
                y_pred = self.model(xi, t_tensor)
                denom = max(self.T - t, 1e-6)
                drift = (y_pred - xi) / denom

                # Euler-Maruyama with noise
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

    @torch.no_grad()
    def sample_batch_hybrid(
        self,
        total_samples: int,
        shape: tuple[int, ...],
        batch_size: int = 64,
        num_steps: Optional[int] = None,
        switch_fraction: float = 0.5,
        ode_solver: ODESolver = ODESolver.HEUN,
    ) -> torch.Tensor:
        """Generate samples in batches using ODE→SDE hybrid.

        Args:
            total_samples: Total number of samples to generate.
            shape: Shape of each sample.
            batch_size: Batch size for generation.
            num_steps: Total number of steps.
            switch_fraction: Fraction of steps for ODE before SDE.
            ode_solver: ODE solver for the ODE phase.

        Returns:
            Generated samples of shape (total_samples, *shape).
        """
        all_samples = []
        remaining = total_samples

        while remaining > 0:
            current_batch = min(batch_size, remaining)
            samples = self.sample_hybrid(
                current_batch,
                shape,
                num_steps=num_steps,
                switch_fraction=switch_fraction,
                ode_solver=ode_solver,
            )
            all_samples.append(samples.cpu())
            remaining -= current_batch

        return torch.cat(all_samples, dim=0)

    @torch.no_grad()
    def sample_pc_sde(
        self,
        num_samples: int,
        shape: tuple[int, ...],
        x0: Optional[torch.Tensor] = None,
        num_steps: Optional[int] = None,
        predictor: ODESolver = ODESolver.HEUN,
        corrector_steps: int = 1,
        return_trajectory: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
        """Generate samples using ODE predictor + SDE corrector.

        This is more principled than Langevin corrector because it uses the
        exact SDE dynamics the model was trained on:
            dξ = (E[Y|ξ] - ξ)/(T-t) dt + dW

        Predictor: ODE step using probability flow (deterministic, accurate)
        Corrector: SDE sub-steps using trained dynamics (adds stochasticity)

        Args:
            num_samples: Number of samples to generate.
            shape: Shape of each sample (channels, height, width).
            x0: Optional initial noise. If None, sample from N(0, I).
            num_steps: Number of predictor steps. If None, use config.
            predictor: ODE solver for predictor step (euler, heun, rk4).
            corrector_steps: Number of SDE corrector sub-steps per predictor step.
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

        # Time discretization
        dt = self.T / num_steps
        xi = x0.clone()

        trajectory = [xi.clone()] if return_trajectory else []

        for step in tqdm(
            range(num_steps),
            desc=f"Sampling (PC {predictor.value}+sde)",
            disable=not self.sampling_config.show_progress,
        ):
            t = step * dt
            t_tensor = torch.full((num_samples,), t, device=self.device)

            # === PREDICTOR STEP (ODE) ===
            if predictor == ODESolver.EULER:
                drift = self._ode_drift(xi, x0, t, t_tensor)
                xi = xi + drift * dt

            elif predictor == ODESolver.HEUN:
                k1 = self._ode_drift(xi, x0, t, t_tensor)
                t_next = t + dt
                t_next_tensor = torch.full((num_samples,), t_next, device=self.device)
                xi_euler = xi + k1 * dt
                k2 = self._ode_drift(xi_euler, x0, t_next, t_next_tensor)
                xi = xi + 0.5 * (k1 + k2) * dt

            elif predictor == ODESolver.RK4:
                k1 = self._ode_drift(xi, x0, t, t_tensor)
                t_mid = t + 0.5 * dt
                t_mid_tensor = torch.full((num_samples,), t_mid, device=self.device)
                k2 = self._ode_drift(xi + 0.5 * k1 * dt, x0, t_mid, t_mid_tensor)
                k3 = self._ode_drift(xi + 0.5 * k2 * dt, x0, t_mid, t_mid_tensor)
                t_next = t + dt
                t_next_tensor = torch.full((num_samples,), t_next, device=self.device)
                k4 = self._ode_drift(xi + k3 * dt, x0, t_next, t_next_tensor)
                xi = xi + (k1 + 2 * k2 + 2 * k3 + k4) * dt / 6

            # === CORRECTOR STEPS (SDE sub-steps) ===
            # Use the trained SDE dynamics to add stochasticity
            # Skip on last step to avoid adding noise to final samples
            if step < num_steps - 1 and corrector_steps > 0:
                t_after_pred = t + dt
                # Sub-step size: we're at t_after_pred, taking small steps
                # that don't advance time (just refine the sample)
                sub_dt = dt / (corrector_steps + 1)
                sqrt_sub_dt = sub_dt ** 0.5

                for _ in range(corrector_steps):
                    t_sub_tensor = torch.full((num_samples,), t_after_pred, device=self.device)

                    # SDE drift: (E[Y|ξ] - ξ) / (T - t)
                    y_pred = self.model(xi, t_sub_tensor)
                    denom = max(self.T - t_after_pred, 1e-6)
                    sde_drift = (y_pred - xi) / denom

                    # Euler-Maruyama SDE step (adds noise for refinement)
                    xi = xi + sde_drift * sub_dt + sqrt_sub_dt * torch.randn_like(xi)

            if return_trajectory:
                trajectory.append(xi.clone())

        if self.sampling_config.clip_samples:
            xi = torch.clamp(xi, -1.0, 1.0)

        if return_trajectory:
            return xi, trajectory
        return xi

    @torch.no_grad()
    def sample_batch_pc_sde(
        self,
        total_samples: int,
        shape: tuple[int, ...],
        batch_size: int = 64,
        num_steps: Optional[int] = None,
        predictor: ODESolver = ODESolver.HEUN,
        corrector_steps: int = 1,
    ) -> torch.Tensor:
        """Generate samples in batches using ODE predictor + SDE corrector.

        Args:
            total_samples: Total number of samples to generate.
            shape: Shape of each sample.
            batch_size: Batch size for generation.
            num_steps: Number of predictor steps.
            predictor: ODE solver for predictor step.
            corrector_steps: Number of SDE corrector sub-steps.

        Returns:
            Generated samples of shape (total_samples, *shape).
        """
        all_samples = []
        remaining = total_samples

        while remaining > 0:
            current_batch = min(batch_size, remaining)
            samples = self.sample_pc_sde(
                current_batch,
                shape,
                num_steps=num_steps,
                predictor=predictor,
                corrector_steps=corrector_steps,
            )
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
