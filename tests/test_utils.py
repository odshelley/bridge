"""Tests for utility functions."""

import torch

from bridge_diffusion.utils import count_parameters, get_device, set_seed


class TestSetSeed:
    """Tests for seed setting."""

    def test_reproducible_random(self) -> None:
        """Test that random numbers are reproducible."""
        set_seed(42)
        r1 = torch.rand(10)

        set_seed(42)
        r2 = torch.rand(10)

        assert torch.allclose(r1, r2)

    def test_different_seeds_different_results(self) -> None:
        """Test that different seeds produce different results."""
        set_seed(42)
        r1 = torch.rand(10)

        set_seed(123)
        r2 = torch.rand(10)

        assert not torch.allclose(r1, r2)


class TestGetDevice:
    """Tests for device detection."""

    def test_returns_valid_device(self) -> None:
        """Test that a valid device is returned."""
        device = get_device()
        assert isinstance(device, torch.device)

    def test_device_is_usable(self) -> None:
        """Test that the device can be used."""
        device = get_device()
        x = torch.randn(10, device=device)
        assert x.device.type == device.type


class TestCountParameters:
    """Tests for parameter counting."""

    def test_count_simple_model(self) -> None:
        """Test parameter counting on a simple model."""
        model = torch.nn.Linear(10, 5)
        count = count_parameters(model)

        # Linear(10, 5) has 10*5 weights + 5 biases = 55 parameters
        assert count == 55

    def test_count_excludes_non_trainable(self) -> None:
        """Test that non-trainable parameters are excluded."""
        model = torch.nn.Linear(10, 5)
        for param in model.parameters():
            param.requires_grad = False

        count = count_parameters(model)
        assert count == 0

    def test_count_nested_model(self) -> None:
        """Test parameter counting on nested models."""
        model = torch.nn.Sequential(
            torch.nn.Linear(10, 5),
            torch.nn.ReLU(),
            torch.nn.Linear(5, 2),
        )
        count = count_parameters(model)

        # Linear(10, 5): 55 params
        # ReLU: 0 params
        # Linear(5, 2): 12 params
        assert count == 55 + 12
