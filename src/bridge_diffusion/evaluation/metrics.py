"""Evaluation metrics for Bridge Diffusion.

Implements FID (Fréchet Inception Distance) and other metrics
for evaluating generated samples.
"""

import logging
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import DataLoader, TensorDataset

logger = logging.getLogger(__name__)


def _require_torch_fidelity():
    """Import torch_fidelity or fail with an actionable message."""
    try:
        import torch_fidelity
    except ImportError:
        logger.error("torch-fidelity not installed. Install with: pip install torch-fidelity")
        raise
    return torch_fidelity


def _to_uint8(images: torch.Tensor, input_range: tuple[float, float]) -> torch.Tensor:
    """Convert float images in an explicit range to [0, 255] uint8 (NCHW).

    Args:
        images: Float image tensor.
        input_range: (lo, hi) bounds of `images`' value range, e.g. (-1.0, 1.0)
            or (0.0, 1.0). The caller must supply this explicitly rather than
            having it inferred from the tensor's values, so that real and
            generated batches are always normalised identically.

    Returns:
        uint8 tensor with values in [0, 255].
    """
    lo, hi = input_range
    images = (images - lo) / (hi - lo)
    return (images * 255).clamp(0, 255).to(torch.uint8)


def compute_fid(
    real_images: torch.Tensor,
    generated_images: torch.Tensor,
    batch_size: int = 64,
    device: Optional[torch.device] = None,
    input_range: tuple[float, float] = (-1.0, 1.0),
) -> float:
    """Compute FID between real and generated images using torch-fidelity.

    Args:
        real_images: Real images of shape (N, C, H, W) in range `input_range`.
        generated_images: Generated images of same shape and range.
        batch_size: Batch size for feature extraction.
        device: Device for computation.
        input_range: (lo, hi) value range shared by both `real_images` and
            `generated_images`. Both tensors are normalised using this same
            range so FID is not biased by per-tensor range guessing.

    Returns:
        FID score (lower is better).
    """
    torch_fidelity = _require_torch_fidelity()

    # Ensure images are in [0, 255] uint8 format as expected by torch-fidelity
    real_images = _to_uint8(real_images, input_range)
    generated_images = _to_uint8(generated_images, input_range)

    # torch-fidelity expects images in NCHW format with uint8 values
    metrics = torch_fidelity.calculate_metrics(
        input1=TensorDataset(generated_images),
        input2=TensorDataset(real_images),
        cuda=device is not None and device.type == "cuda",
        fid=True,
        verbose=False,
        batch_size=batch_size,
    )

    return metrics["frechet_inception_distance"]


def compute_fid_from_paths(
    real_path: Path,
    generated_path: Path,
    batch_size: int = 64,
    device: Optional[torch.device] = None,
) -> float:
    """Compute FID from directories of images.

    Args:
        real_path: Path to directory of real images.
        generated_path: Path to directory of generated images.
        batch_size: Batch size for feature extraction.
        device: Device for computation.

    Returns:
        FID score.
    """
    torch_fidelity = _require_torch_fidelity()

    metrics = torch_fidelity.calculate_metrics(
        input1=str(generated_path),
        input2=str(real_path),
        cuda=device is not None and device.type == "cuda",
        fid=True,
        verbose=False,
        batch_size=batch_size,
    )

    return metrics["frechet_inception_distance"]


def compute_fid_against_dataset(
    generated_images: torch.Tensor,
    dataset_name: str,
    batch_size: int = 64,
    device: Optional[torch.device] = None,
    input_range: tuple[float, float] = (-1.0, 1.0),
) -> float:
    """Compute FID against a standard dataset (CIFAR-10, etc.).

    Args:
        generated_images: Generated images of shape (N, C, H, W) in range `input_range`.
        dataset_name: Name of reference dataset ("cifar10", etc.). torch-fidelity
            fetches and normalises this dataset itself, so `input_range` only
            applies to `generated_images`.
        batch_size: Batch size for feature extraction.
        device: Device for computation.
        input_range: (lo, hi) value range of `generated_images`.

    Returns:
        FID score.
    """
    torch_fidelity = _require_torch_fidelity()

    # Prepare generated images
    generated_images = _to_uint8(generated_images, input_range)

    metrics = torch_fidelity.calculate_metrics(
        input1=TensorDataset(generated_images),
        input2=dataset_name,  # torch-fidelity handles standard datasets
        cuda=device is not None and device.type == "cuda",
        fid=True,
        verbose=False,
        batch_size=batch_size,
    )

    return metrics["frechet_inception_distance"]


class MetricsLogger:
    """Logger for evaluation metrics with MLflow integration."""

    def __init__(self, use_mlflow: bool = True):
        """Initialise MetricsLogger.

        Args:
            use_mlflow: Whether to log to MLflow.
        """
        self.use_mlflow = use_mlflow
        self.metrics_history: dict[str, list[float]] = {}

    def log(self, metrics: dict[str, float], step: Optional[int] = None) -> None:
        """Log metrics.

        Args:
            metrics: Dictionary of metric names to values.
            step: Optional step number.
        """
        for name, value in metrics.items():
            if name not in self.metrics_history:
                self.metrics_history[name] = []
            self.metrics_history[name].append(value)

            logger.info(f"{name}: {value:.4f}")

        if self.use_mlflow:
            try:
                import mlflow

                if mlflow.active_run():
                    mlflow.log_metrics(metrics, step=step)
            except ImportError:
                pass

    def get_history(self, name: str) -> list[float]:
        """Get history for a metric.

        Args:
            name: Metric name.

        Returns:
            List of values.
        """
        return self.metrics_history.get(name, [])
