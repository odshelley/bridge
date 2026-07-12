"""Direct unit tests for BridgeDiffusion (the paper's primary method)."""

import torch

from bridge_diffusion.config import BridgeConfig, ModelConfig
from bridge_diffusion.models import BridgeDiffusion, DiffusersUNetWrapper


def _tiny_network(image_size: int = 16) -> DiffusersUNetWrapper:
    model_config = ModelConfig(
        in_channels=3,
        out_channels=3,
        sample_size=image_size,
        block_out_channels=(32, 64),
        layers_per_block=1,
        down_block_types=("DownBlock2D", "DownBlock2D"),
        up_block_types=("UpBlock2D", "UpBlock2D"),
    )
    return DiffusersUNetWrapper(model_config)


def _bridge_model(t_terminal: float = 1.0) -> BridgeDiffusion:
    return BridgeDiffusion(_tiny_network(), BridgeConfig(T=t_terminal))


class TestComputeExpectation:
    def test_matches_linear_interpolation_formula(self) -> None:
        model = _bridge_model(t_terminal=1.0)
        x = torch.tensor([[0.0, 2.0], [1.0, 1.0]])
        y = torch.tensor([[4.0, 6.0], [3.0, -1.0]])
        t = torch.tensor([0.5, 0.25])

        expected = x + (y - x) * t.view(-1, 1) / model.T
        actual = model.compute_expectation(x, y, t)

        assert torch.allclose(actual, expected)

    def test_at_t_zero_equals_x(self) -> None:
        model = _bridge_model(t_terminal=1.0)
        x = torch.tensor([[1.0, -3.0]])
        y = torch.tensor([[5.0, 2.0]])
        t = torch.tensor([0.0])

        actual = model.compute_expectation(x, y, t)

        assert torch.allclose(actual, x)

    def test_at_terminal_time_equals_y(self) -> None:
        model = _bridge_model(t_terminal=1.0)
        x = torch.tensor([[1.0, -3.0]])
        y = torch.tensor([[5.0, 2.0]])
        t = torch.tensor([model.T])

        actual = model.compute_expectation(x, y, t)

        assert torch.allclose(actual, y)


class TestComputeVariance:
    def test_matches_formula(self) -> None:
        model = _bridge_model(t_terminal=1.0)
        t = torch.tensor([0.25, 0.5, 0.75])

        expected = t * (model.T - t) / model.T
        actual = model.compute_variance(t)

        assert torch.allclose(actual, expected)

    def test_zero_at_both_boundaries(self) -> None:
        model = _bridge_model(t_terminal=1.0)
        t = torch.tensor([0.0, model.T])

        actual = model.compute_variance(t)

        assert torch.allclose(actual, torch.zeros(2))


class TestSampleBridge:
    def test_at_t_zero_returns_x(self) -> None:
        model = _bridge_model(t_terminal=1.0)
        x = torch.randn(2, 3, 4, 4)
        y = torch.randn(2, 3, 4, 4)
        t = torch.zeros(2)

        sample = model.sample_bridge(x, y, t)

        # Variance is exactly zero at t=0, so no noise is added.
        assert torch.allclose(sample, x)

    def test_at_terminal_time_returns_y(self) -> None:
        model = _bridge_model(t_terminal=1.0)
        x = torch.randn(2, 3, 4, 4)
        y = torch.randn(2, 3, 4, 4)
        t = torch.full((2,), model.T)

        sample = model.sample_bridge(x, y, t)

        # Variance is exactly zero at t=T, so no noise is added.
        assert torch.allclose(sample, y)


class TestComputeTrainingLoss:
    def test_returns_finite_scalar_and_backpropagates(self) -> None:
        model = _bridge_model()
        x = torch.randn(2, 3, 16, 16)
        y = torch.randn(2, 3, 16, 16)

        loss = model.compute_training_loss(x, y)

        assert loss.ndim == 0
        assert torch.isfinite(loss)

        loss.backward()
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        assert len(grads) > 0
        assert all(torch.isfinite(g).all() for g in grads)


class TestGenerate:
    def test_shape_and_finiteness(self) -> None:
        model = _bridge_model()
        x = torch.randn(2, 3, 16, 16)

        samples = model.generate(x, num_steps=4)

        assert samples.shape == x.shape
        assert torch.isfinite(samples).all()
