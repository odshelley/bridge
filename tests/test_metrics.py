"""Tests for evaluation metrics module.

These tests exercise `_to_uint8` directly and do not require torch-fidelity
to be installed: `_require_torch_fidelity` is only invoked lazily inside the
`compute_fid*` functions, so importing this module is safe even without the
optional dependency.
"""

import torch

from bridge_diffusion.evaluation.metrics import _to_uint8


class TestToUint8:
    """Tests for the explicit input_range based uint8 conversion."""

    def test_constant_value_maps_consistently_regardless_of_position(self) -> None:
        """A constant-valued tensor should map to the same uint8 value everywhere.

        This guards against any residual per-element or per-position branching:
        every entry of a constant tensor must round-trip to the same output.
        """
        images = torch.full((2, 3, 4, 4), 0.6)
        result = _to_uint8(images, input_range=(-1.0, 1.0))

        expected_value = result.flatten()[0]
        assert torch.all(result == expected_value)

    def test_zero_to_one_range_endpoints(self) -> None:
        """input_range=(0, 255) should map 0 -> 0 and 255 -> 255."""
        images = torch.tensor([0.0, 255.0])
        result = _to_uint8(images, input_range=(0.0, 255.0))

        assert result[0].item() == 0
        assert result[1].item() == 255

    def test_minus_one_to_one_range_endpoints(self) -> None:
        """input_range=(-1, 1) should map -1 -> 0 and 1 -> 255."""
        images = torch.tensor([-1.0, 1.0])
        result = _to_uint8(images, input_range=(-1.0, 1.0))

        assert result[0].item() == 0
        assert result[1].item() == 255

    def test_does_not_key_off_tensor_min(self) -> None:
        """Two batches with different mins but the same declared range agree.

        This is the regression this fix targets: previously `_to_uint8` guessed
        the input range from `images.min() < 0`, so a real batch (e.g. all >= 0)
        and a generated batch (e.g. containing negatives) could be normalised
        differently even though both are nominally in the same range, biasing
        FID. With an explicit `input_range`, both must be handled identically.
        """
        all_nonneg = torch.tensor([0.6, 0.6])
        has_negative = torch.tensor([0.6, -0.9])

        result_a = _to_uint8(all_nonneg, input_range=(-1.0, 1.0))
        result_b = _to_uint8(has_negative, input_range=(-1.0, 1.0))

        assert result_a[0].item() == result_b[0].item()
