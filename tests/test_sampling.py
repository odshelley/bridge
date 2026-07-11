"""Tests for sampling module."""

import pytest
import torch

from bridge_diffusion.config import BridgeConfig, ModelConfig, SamplingConfig
from bridge_diffusion.models import BridgeDiffusion, DiffusersUNetWrapper
from bridge_diffusion.sampling import ODESolver, Sampler


class _IdentityNet(torch.nn.Module):
    """Predicts E[Y|xi] = xi, so the SDE drift is zero."""

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return x


class _OffsetNet(torch.nn.Module):
    """Predicts E[Y|xi] = xi + k, a huge fixed offset from the current state.

    Used to exercise the t -> T edge case: dividing this large, constant
    (y_pred - xi) difference by the 1e-6 denominator floor overflows to inf,
    while dividing by a denominator bounded below by eps (1e-4) stays finite.
    """

    def __init__(self, k: float) -> None:
        super().__init__()
        self.k = k

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return x + self.k


class TestSampler:
    """Tests for the Sampler class."""

    @pytest.fixture
    def sampler(self) -> Sampler:
        """Create a sampler for testing."""
        model_config = ModelConfig(
            in_channels=1,
            out_channels=1,
            sample_size=16,
            block_out_channels=(32, 64),
            layers_per_block=1,
            down_block_types=("DownBlock2D", "DownBlock2D"),
            up_block_types=("UpBlock2D", "UpBlock2D"),
        )
        network = DiffusersUNetWrapper(model_config)
        bridge_config = BridgeConfig(T=0.1, eps=1e-7)
        model = BridgeDiffusion(network, bridge_config)
        sampling_config = SamplingConfig(
            num_steps=10,
            num_samples=4,
        )
        return Sampler(
            model=model,
            bridge_config=bridge_config,
            sampling_config=sampling_config,
            device=torch.device("cpu"),
        )

    def test_sample_shape(self, sampler: Sampler) -> None:
        """Test that samples have correct shape."""
        num_samples = 4
        shape = (1, 16, 16)

        samples = sampler.sample(num_samples, shape)

        assert samples.shape == (num_samples, *shape)

    def test_sample_with_prior(self, sampler: Sampler) -> None:
        """Test sampling with provided prior samples."""
        num_samples = 4
        shape = (1, 16, 16)
        y = torch.randn(num_samples, *shape)

        samples = sampler.sample(num_samples, shape, x0=y)

        assert samples.shape == (num_samples, *shape)

    def test_sample_with_trajectory(self, sampler: Sampler) -> None:
        """Test that trajectory is returned correctly."""
        num_samples = 2
        shape = (1, 8, 8)
        num_steps = 5

        samples, trajectory = sampler.sample(
            num_samples,
            shape,
            num_steps=num_steps,
            return_trajectory=True,
        )

        # Trajectory should have num_steps + 1 entries (initial + each step)
        assert len(trajectory) == num_steps + 1

        # Each entry should have correct shape
        for t in trajectory:
            assert t.shape == (num_samples, *shape)

    def test_sample_batch(self, sampler: Sampler) -> None:
        """Test batch sampling."""
        total_samples = 10
        shape = (1, 8, 8)
        batch_size = 4

        samples = sampler.sample_batch(
            total_samples=total_samples,
            shape=shape,
            batch_size=batch_size,
        )

        assert samples.shape == (total_samples, *shape)

    def test_sample_clipping(self, sampler: Sampler) -> None:
        """Test that samples are clipped when configured."""
        # The sampler fixture has clip_samples=True by default
        num_samples = 4
        shape = (1, 16, 16)

        samples = sampler.sample(num_samples, shape)

        assert samples.min() >= -1.0
        assert samples.max() <= 1.0

    def test_sample_determinism_with_seed(self, sampler: Sampler) -> None:
        """Test that sampling is deterministic with same seed."""
        num_samples = 2
        shape = (1, 8, 8)

        # Sample twice with same seed
        torch.manual_seed(42)
        samples1 = sampler.sample(num_samples, shape)

        torch.manual_seed(42)
        samples2 = sampler.sample(num_samples, shape)

        assert torch.allclose(samples1, samples2)


class TestSamplerDifferentSteps:
    """Tests for sampling with different numbers of steps."""

    @pytest.fixture
    def model(self) -> BridgeDiffusion:
        """Create a bridge model for testing."""
        model_config = ModelConfig(
            in_channels=1,
            out_channels=1,
            sample_size=8,
            block_out_channels=(32, 64),
            layers_per_block=1,
            down_block_types=("DownBlock2D", "DownBlock2D"),
            up_block_types=("UpBlock2D", "UpBlock2D"),
        )
        network = DiffusersUNetWrapper(model_config)
        config = BridgeConfig(T=0.1, eps=1e-7)
        return BridgeDiffusion(network, config)

    @pytest.mark.parametrize("num_steps", [2, 10, 50])
    def test_various_step_counts(self, model: BridgeDiffusion, num_steps: int) -> None:
        """Test sampling with various step counts."""
        bridge_config = BridgeConfig(T=0.1, eps=1e-7)
        sampling_config = SamplingConfig(
            num_steps=num_steps,
            num_samples=2,
            show_progress=False,
        )

        sampler = Sampler(
            model=model,
            bridge_config=bridge_config,
            sampling_config=sampling_config,
            device=torch.device("cpu"),
        )

        samples = sampler.sample(2, (1, 8, 8))
        assert samples.shape == (2, 1, 8, 8)


class TestSampleBatchX0:
    def test_x0_is_chunked_and_used(self) -> None:
        sampler = Sampler(
            model=_IdentityNet(),
            bridge_config=BridgeConfig(),
            sampling_config=SamplingConfig(num_steps=1, show_progress=False),
            device=torch.device("cpu"),
        )
        # With num_steps=1 and zero drift, output == clamp(x0)
        x0 = torch.full((5, 1, 8, 8), 0.5)
        out = sampler.sample_batch(total_samples=5, shape=(1, 8, 8), batch_size=2, x0=x0)
        assert out.shape == (5, 1, 8, 8)
        assert torch.allclose(out, x0)

    def test_x0_none_still_works(self) -> None:
        sampler = Sampler(
            model=_IdentityNet(),
            bridge_config=BridgeConfig(),
            sampling_config=SamplingConfig(num_steps=1, show_progress=False),
            device=torch.device("cpu"),
        )
        out = sampler.sample_batch(total_samples=3, shape=(1, 8, 8), batch_size=2)
        assert out.shape == (3, 1, 8, 8)

    @pytest.mark.parametrize("bad_batch_size", [0, -2])
    def test_nonpositive_batch_size_raises(self, bad_batch_size: int) -> None:
        """batch_size <= 0 would loop forever; it must fail fast instead."""
        sampler = Sampler(
            model=_IdentityNet(),
            bridge_config=BridgeConfig(),
            sampling_config=SamplingConfig(num_steps=1, show_progress=False),
            device=torch.device("cpu"),
        )
        with pytest.raises(ValueError, match="batch_size must be positive"):
            sampler.sample_batch(total_samples=3, shape=(1, 8, 8), batch_size=bad_batch_size)
        with pytest.raises(ValueError, match="batch_size must be positive"):
            sampler.sample_batch_ode(total_samples=3, shape=(1, 8, 8), batch_size=bad_batch_size)


class TestSampleODE:
    """Characterization tests for the probability-flow ODE sampler."""

    @pytest.fixture
    def sampler(self) -> Sampler:
        return Sampler(
            model=_IdentityNet(),
            bridge_config=BridgeConfig(),
            sampling_config=SamplingConfig(num_steps=8, show_progress=False),
            device=torch.device("cpu"),
        )

    @pytest.mark.parametrize("solver", [ODESolver.EULER, ODESolver.HEUN, ODESolver.RK4])
    def test_identity_net_is_fixed_point(self, sampler: Sampler, solver: ODESolver) -> None:
        """With y_pred == xi and xi == x0, the ODE drift is identically zero."""
        x0 = torch.full((3, 1, 8, 8), 0.5)
        out = sampler.sample_ode(3, (1, 8, 8), x0=x0, solver=solver)
        assert out.shape == (3, 1, 8, 8)
        assert torch.allclose(out, x0)

    @pytest.mark.parametrize("solver", [ODESolver.EULER, ODESolver.HEUN, ODESolver.RK4])
    def test_determinism_with_seed(self, sampler: Sampler, solver: ODESolver) -> None:
        torch.manual_seed(7)
        out1 = sampler.sample_ode(2, (1, 8, 8), solver=solver)
        torch.manual_seed(7)
        out2 = sampler.sample_ode(2, (1, 8, 8), solver=solver)
        assert torch.allclose(out1, out2)

    def test_trajectory_length(self, sampler: Sampler) -> None:
        _, trajectory = sampler.sample_ode(
            2, (1, 8, 8), num_steps=5, return_trajectory=True
        )
        assert len(trajectory) == 6
        for entry in trajectory:
            assert entry.shape == (2, 1, 8, 8)

    def test_batch_ode_fixed_step(self, sampler: Sampler) -> None:
        out = sampler.sample_batch_ode(
            total_samples=5, shape=(1, 8, 8), batch_size=2, solver=ODESolver.HEUN
        )
        assert out.shape == (5, 1, 8, 8)

    def test_batch_ode_torchdiffeq_routing(self, sampler: Sampler) -> None:
        out = sampler.sample_batch_ode(
            total_samples=2, shape=(1, 8, 8), batch_size=2, solver=ODESolver.DOPRI5
        )
        assert out.shape == (2, 1, 8, 8)


class TestSampleODETimeClamp:
    """The last stage of HEUN/RK4 (and the torchdiffeq endpoint) evaluates the
    drift at t == T, where denom = max(T - t, 1e-6) hits its floor and the
    (y_pred - xi) / denom term explodes. The fix clamps the evaluation time
    away from T so the denominator is bounded below by eps (1e-4) instead.
    """

    @pytest.fixture
    def sampler(self) -> Sampler:
        # A large, constant (y_pred - xi) offset makes the t -> T blow-up
        # concrete: it overflows to inf when divided by the 1e-6 floor, but
        # stays finite when divided by the eps-bounded denominator.
        return Sampler(
            model=_OffsetNet(k=1e33),
            bridge_config=BridgeConfig(),  # T=0.1
            sampling_config=SamplingConfig(num_steps=1, show_progress=False, clip_samples=False),
            device=torch.device("cpu"),
        )

    @pytest.mark.parametrize("solver", [ODESolver.HEUN, ODESolver.RK4])
    def test_sample_ode_stays_finite_near_t(self, sampler: Sampler, solver: ODESolver) -> None:
        x0 = torch.zeros(2, 1, 8, 8)
        out = sampler.sample_ode(2, (1, 8, 8), x0=x0, solver=solver, num_steps=1)
        assert torch.isfinite(out).all()

    def test_sample_ode_torchdiffeq_stays_finite_near_t(self, sampler: Sampler) -> None:
        x0 = torch.zeros(2, 1, 8, 8)
        out = sampler.sample_ode_torchdiffeq(2, (1, 8, 8), x0=x0, solver="rk4", num_steps=1)
        assert torch.isfinite(out).all()
