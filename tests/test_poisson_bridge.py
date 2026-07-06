"""Tests for the Poisson Bridge Diffusion model."""

import pytest
import torch

from bridge_diffusion.config import ModelConfig, PoissonBridgeConfig
from bridge_diffusion.models import DiffusersUNetWrapper, PoissonBridgeDiffusion


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def poisson_config() -> PoissonBridgeConfig:
    """Default Poisson bridge config for testing."""
    return PoissonBridgeConfig(T=1.0, eps=1e-7, num_levels=256, prior="zeros")


@pytest.fixture
def poisson_config_poisson_prior() -> PoissonBridgeConfig:
    """Poisson bridge config with Poisson prior."""
    return PoissonBridgeConfig(
        T=1.0, eps=1e-7, num_levels=256, prior="poisson", prior_lambda=2.0
    )


@pytest.fixture
def small_network() -> DiffusersUNetWrapper:
    """Small UNet for fast testing."""
    model_config = ModelConfig(
        in_channels=1,
        out_channels=1,
        sample_size=16,
        block_out_channels=(32, 64),
        layers_per_block=1,
        down_block_types=("DownBlock2D", "DownBlock2D"),
        up_block_types=("UpBlock2D", "UpBlock2D"),
    )
    return DiffusersUNetWrapper(model_config)


@pytest.fixture
def poisson_model(
    small_network: DiffusersUNetWrapper,
    poisson_config: PoissonBridgeConfig,
) -> PoissonBridgeDiffusion:
    """Poisson bridge model with zeros prior."""
    return PoissonBridgeDiffusion(small_network, poisson_config)


@pytest.fixture
def poisson_model_poisson_prior(
    poisson_config_poisson_prior: PoissonBridgeConfig,
) -> PoissonBridgeDiffusion:
    """Poisson bridge model with Poisson prior (needs its own network)."""
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
    return PoissonBridgeDiffusion(network, poisson_config_poisson_prior)


# ---------------------------------------------------------------------------
# Config tests
# ---------------------------------------------------------------------------


class TestPoissonBridgeConfig:
    """Tests for PoissonBridgeConfig."""

    def test_default_values(self) -> None:
        """Test default configuration values."""
        config = PoissonBridgeConfig()
        assert config.T == 1.0
        assert config.eps == 1e-7
        assert config.num_levels == 256
        assert config.prior == "zeros"
        assert config.prior_lambda == 1.0

    def test_custom_values(self) -> None:
        """Test custom configuration values."""
        config = PoissonBridgeConfig(
            T=2.0, eps=1e-5, num_levels=128, prior="poisson", prior_lambda=3.0
        )
        assert config.T == 2.0
        assert config.eps == 1e-5
        assert config.num_levels == 128
        assert config.prior == "poisson"
        assert config.prior_lambda == 3.0


# ---------------------------------------------------------------------------
# Prior sampling tests
# ---------------------------------------------------------------------------


class TestPriorSampling:
    """Tests for prior distribution sampling."""

    def test_zeros_prior(self, poisson_model: PoissonBridgeDiffusion) -> None:
        """Test that zeros prior returns all zeros."""
        shape = (4, 1, 16, 16)
        samples = poisson_model.sample_prior(shape, device=torch.device("cpu"))
        assert samples.shape == shape
        assert torch.all(samples == 0)

    def test_poisson_prior_shape(
        self, poisson_model_poisson_prior: PoissonBridgeDiffusion
    ) -> None:
        """Test that Poisson prior returns correctly shaped samples."""
        shape = (4, 1, 16, 16)
        samples = poisson_model_poisson_prior.sample_prior(
            shape, device=torch.device("cpu")
        )
        assert samples.shape == shape

    def test_poisson_prior_non_negative(
        self, poisson_model_poisson_prior: PoissonBridgeDiffusion
    ) -> None:
        """Test that Poisson prior samples are non-negative."""
        shape = (8, 1, 16, 16)
        samples = poisson_model_poisson_prior.sample_prior(
            shape, device=torch.device("cpu")
        )
        assert torch.all(samples >= 0)

    def test_poisson_prior_integer_valued(
        self, poisson_model_poisson_prior: PoissonBridgeDiffusion
    ) -> None:
        """Test that Poisson prior samples are integer-valued."""
        shape = (8, 1, 16, 16)
        samples = poisson_model_poisson_prior.sample_prior(
            shape, device=torch.device("cpu")
        )
        assert torch.allclose(samples, samples.round())

    def test_poisson_prior_within_range(
        self, poisson_model_poisson_prior: PoissonBridgeDiffusion
    ) -> None:
        """Test that Poisson prior samples are clamped to [0, num_levels-1]."""
        shape = (8, 1, 16, 16)
        samples = poisson_model_poisson_prior.sample_prior(
            shape, device=torch.device("cpu")
        )
        assert torch.all(samples <= 255)


# ---------------------------------------------------------------------------
# Bridge sampling (Binomial interpolation) tests
# ---------------------------------------------------------------------------


class TestBridgeSampling:
    """Tests for Binomial bridge sampling."""

    def test_bridge_sample_shape(
        self, poisson_model: PoissonBridgeDiffusion
    ) -> None:
        """Test that bridge samples have correct shape."""
        batch = 4
        x = torch.zeros(batch, 1, 16, 16)
        y = torch.randint(0, 256, (batch, 1, 16, 16)).float()
        t = torch.rand(batch) * poisson_model.T

        xi_t = poisson_model.sample_bridge(x, y, t)
        assert xi_t.shape == (batch, 1, 16, 16)

    def test_bridge_sample_non_negative(
        self, poisson_model: PoissonBridgeDiffusion
    ) -> None:
        """Test that bridge samples are non-negative."""
        batch = 8
        x = torch.zeros(batch, 1, 16, 16)
        y = torch.randint(0, 256, (batch, 1, 16, 16)).float()
        t = torch.rand(batch) * poisson_model.T

        xi_t = poisson_model.sample_bridge(x, y, t)
        assert torch.all(xi_t >= 0)

    def test_bridge_sample_integer_valued(
        self, poisson_model: PoissonBridgeDiffusion
    ) -> None:
        """Test that bridge samples are integer-valued."""
        batch = 8
        x = torch.zeros(batch, 1, 16, 16)
        y = torch.randint(0, 256, (batch, 1, 16, 16)).float()
        t = torch.rand(batch) * poisson_model.T

        xi_t = poisson_model.sample_bridge(x, y, t)
        assert torch.allclose(xi_t, xi_t.round())

    def test_bridge_at_t0_equals_prior(
        self, poisson_model: PoissonBridgeDiffusion
    ) -> None:
        """Test that bridge at t=0 returns x (the prior).

        With p=0/T=0 the Binomial(n, 0) is always 0, so xi_0 = x.
        """
        batch = 4
        x = torch.zeros(batch, 1, 16, 16)
        y = torch.randint(0, 256, (batch, 1, 16, 16)).float()
        t = torch.zeros(batch)

        xi_t = poisson_model.sample_bridge(x, y, t)
        assert torch.allclose(xi_t, x)

    def test_bridge_at_tT_equals_data(
        self, poisson_model: PoissonBridgeDiffusion
    ) -> None:
        """Test that bridge at t=T returns y (the data).

        With p=T/T=1 the Binomial(n, 1) always returns n, so
        xi_T = x + (y - x) = y.
        """
        batch = 4
        x = torch.zeros(batch, 1, 16, 16)
        y = torch.randint(0, 256, (batch, 1, 16, 16)).float()
        t = torch.full((batch,), poisson_model.T)

        xi_t = poisson_model.sample_bridge(x, y, t)
        assert torch.allclose(xi_t, y)

    def test_bridge_bounded_by_endpoints(
        self, poisson_model: PoissonBridgeDiffusion
    ) -> None:
        """Test that bridge samples lie between x and y (coordinate-wise)."""
        batch = 16
        x = torch.zeros(batch, 1, 16, 16)
        y = torch.randint(10, 200, (batch, 1, 16, 16)).float()
        t = torch.rand(batch) * poisson_model.T

        xi_t = poisson_model.sample_bridge(x, y, t)
        assert torch.all(xi_t >= x)
        assert torch.all(xi_t <= y)

    def test_bridge_with_nonzero_prior(
        self, poisson_model: PoissonBridgeDiffusion
    ) -> None:
        """Test bridge with non-zero prior x."""
        batch = 4
        x = torch.randint(0, 50, (batch, 1, 16, 16)).float()
        y = x + torch.randint(0, 100, (batch, 1, 16, 16)).float()
        t = torch.rand(batch) * poisson_model.T

        xi_t = poisson_model.sample_bridge(x, y, t)
        assert torch.all(xi_t >= x)
        assert torch.all(xi_t <= y)


# ---------------------------------------------------------------------------
# Training loss tests
# ---------------------------------------------------------------------------


class TestTrainingLoss:
    """Tests for compute_training_loss."""

    def test_loss_is_scalar(self, poisson_model: PoissonBridgeDiffusion) -> None:
        """Test that training loss is a scalar."""
        x = torch.zeros(4, 1, 16, 16)
        y = torch.randint(0, 256, (4, 1, 16, 16)).float()

        loss = poisson_model.compute_training_loss(x, y)
        assert loss.ndim == 0

    def test_loss_is_finite(self, poisson_model: PoissonBridgeDiffusion) -> None:
        """Test that training loss is finite."""
        x = torch.zeros(4, 1, 16, 16)
        y = torch.randint(0, 256, (4, 1, 16, 16)).float()

        loss = poisson_model.compute_training_loss(x, y)
        assert torch.isfinite(loss)

    def test_loss_is_non_negative(
        self, poisson_model: PoissonBridgeDiffusion
    ) -> None:
        """Test that MSE loss is non-negative."""
        x = torch.zeros(4, 1, 16, 16)
        y = torch.randint(0, 256, (4, 1, 16, 16)).float()

        loss = poisson_model.compute_training_loss(x, y)
        assert loss >= 0

    def test_loss_requires_grad(
        self, poisson_model: PoissonBridgeDiffusion
    ) -> None:
        """Test that loss supports backpropagation."""
        x = torch.zeros(4, 1, 16, 16)
        y = torch.randint(0, 256, (4, 1, 16, 16)).float()

        loss = poisson_model.compute_training_loss(x, y)
        assert loss.requires_grad

        # Should not raise
        loss.backward()


# ---------------------------------------------------------------------------
# Forward pass tests
# ---------------------------------------------------------------------------


class TestForward:
    """Tests for forward pass."""

    def test_forward_shape(self, poisson_model: PoissonBridgeDiffusion) -> None:
        """Test that forward pass outputs correct shape."""
        batch = 4
        x = torch.randint(0, 256, (batch, 1, 16, 16)).float()
        t = torch.rand(batch) * poisson_model.T

        out = poisson_model(x, t)
        assert out.shape == (batch, 1, 16, 16)

    def test_forward_different_timesteps(
        self, poisson_model: PoissonBridgeDiffusion
    ) -> None:
        """Test that different timesteps produce different outputs."""
        x = torch.randint(0, 256, (1, 1, 16, 16)).float()
        t1 = torch.tensor([0.1])
        t2 = torch.tensor([0.9])

        out1 = poisson_model(x, t1)
        out2 = poisson_model(x, t2)

        assert not torch.allclose(out1, out2)


# ---------------------------------------------------------------------------
# Generation (Poisson simulation) tests
# ---------------------------------------------------------------------------


class TestGeneration:
    """Tests for Poisson bridge simulation (sample generation)."""

    def test_generate_shape(self, poisson_model: PoissonBridgeDiffusion) -> None:
        """Test that generated samples have correct shape."""
        x = torch.zeros(4, 1, 16, 16)
        samples = poisson_model.generate(x, num_steps=5)
        assert samples.shape == (4, 1, 16, 16)

    def test_generate_non_negative(
        self, poisson_model: PoissonBridgeDiffusion
    ) -> None:
        """Test that generated samples are non-negative (Poisson increments >= 0)."""
        x = torch.zeros(4, 1, 16, 16)
        samples = poisson_model.generate(x, num_steps=5)
        assert torch.all(samples >= 0)

    def test_generate_integer_valued(
        self, poisson_model: PoissonBridgeDiffusion
    ) -> None:
        """Test that generated samples are integer-valued."""
        x = torch.zeros(4, 1, 16, 16)
        samples = poisson_model.generate(x, num_steps=5)
        assert torch.allclose(samples, samples.round())

    def test_generate_determinism_with_seed(
        self, poisson_model: PoissonBridgeDiffusion
    ) -> None:
        """Test that generation is deterministic with same seed."""
        x = torch.zeros(2, 1, 16, 16)

        torch.manual_seed(42)
        samples1 = poisson_model.generate(x, num_steps=5)

        torch.manual_seed(42)
        samples2 = poisson_model.generate(x, num_steps=5)

        assert torch.allclose(samples1, samples2)

    def test_generate_with_poisson_prior(
        self, poisson_model_poisson_prior: PoissonBridgeDiffusion
    ) -> None:
        """Test generation starting from Poisson prior."""
        torch.manual_seed(123)
        x = poisson_model_poisson_prior.sample_prior(
            (4, 1, 16, 16), device=torch.device("cpu")
        )
        samples = poisson_model_poisson_prior.generate(x, num_steps=5)
        assert samples.shape == (4, 1, 16, 16)
        assert torch.all(samples >= 0)

    @pytest.mark.parametrize("num_steps", [2, 10, 50])
    def test_generate_various_step_counts(
        self, poisson_model: PoissonBridgeDiffusion, num_steps: int
    ) -> None:
        """Test generation with various step counts."""
        x = torch.zeros(2, 1, 16, 16)
        samples = poisson_model.generate(x, num_steps=num_steps)
        assert samples.shape == (2, 1, 16, 16)
        assert torch.all(samples >= 0)


# ---------------------------------------------------------------------------
# Configurable num_levels tests
# ---------------------------------------------------------------------------


class TestNumLevels:
    """Tests for configurable num_levels."""

    def test_binary_levels(self) -> None:
        """Test with num_levels=2 (binary data)."""
        config = PoissonBridgeConfig(T=1.0, num_levels=2, prior="zeros")
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
        model = PoissonBridgeDiffusion(network, config)

        assert model.num_levels == 2

        # Binary data: y in {0, 1}
        x = torch.zeros(4, 1, 16, 16)
        y = torch.randint(0, 2, (4, 1, 16, 16)).float()
        t = torch.rand(4) * config.T

        xi_t = model.sample_bridge(x, y, t)
        # Bridge samples should be in {0, 1}
        assert torch.all((xi_t == 0) | (xi_t == 1))

    def test_small_levels(self) -> None:
        """Test with num_levels=16 (4-bit data)."""
        config = PoissonBridgeConfig(T=1.0, num_levels=16, prior="zeros")
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
        model = PoissonBridgeDiffusion(network, config)

        x = torch.zeros(4, 1, 16, 16)
        y = torch.randint(0, 16, (4, 1, 16, 16)).float()

        loss = model.compute_training_loss(x, y)
        assert torch.isfinite(loss)


# ---------------------------------------------------------------------------
# y >= x assumption tests
# ---------------------------------------------------------------------------


class TestYGreaterEqualXAssumption:
    """Tests for the y >= x coordinatewise assumption warning."""

    def test_warns_when_prior_exceeds_data(
        self, poisson_model: PoissonBridgeDiffusion, caplog: pytest.LogCaptureFixture
    ) -> None:
        """The paper requires y >= x coordinatewise; violating it should warn."""
        import logging

        x = torch.full((2, 1, 16, 16), 5.0)
        y = torch.zeros(2, 1, 16, 16)
        with caplog.at_level(
            logging.WARNING, logger="bridge_diffusion.models.poisson_bridge"
        ):
            poisson_model.compute_training_loss(x, y)
        assert any("y >= x" in r.message for r in caplog.records)
