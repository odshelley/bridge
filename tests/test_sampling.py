"""Tests for sampling module."""

import pytest
import torch

from bridge_diffusion.config import BridgeConfig, ModelConfig, SamplingConfig
from bridge_diffusion.models import BridgeDiffusion, DiffusersUNetWrapper
from bridge_diffusion.sampling import Sampler


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

        samples = sampler.sample(num_samples, shape, y=y)

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
