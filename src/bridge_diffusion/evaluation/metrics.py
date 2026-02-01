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


def compute_fid(
    real_images: torch.Tensor,
    generated_images: torch.Tensor,
    batch_size: int = 64,
    device: Optional[torch.device] = None,
) -> float:
    """Compute FID between real and generated images using torch-fidelity.

    Args:
        real_images: Real images of shape (N, C, H, W) in range [-1, 1] or [0, 1].
        generated_images: Generated images of same shape.
        batch_size: Batch size for feature extraction.
        device: Device for computation.

    Returns:
        FID score (lower is better).
    """
    try:
        import torch_fidelity
    except ImportError:
        logger.error("torch-fidelity not installed. Install with: pip install torch-fidelity")
        raise

    # Ensure images are in [0, 255] uint8 format as expected by torch-fidelity
    def prepare_images(images: torch.Tensor) -> torch.Tensor:
        # If in [-1, 1], convert to [0, 1]
        if images.min() < 0:
            images = (images + 1) / 2
        # Convert to [0, 255] uint8
        images = (images * 255).clamp(0, 255).to(torch.uint8)
        return images

    real_images = prepare_images(real_images)
    generated_images = prepare_images(generated_images)

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
    try:
        import torch_fidelity
    except ImportError:
        logger.error("torch-fidelity not installed. Install with: pip install torch-fidelity")
        raise

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
) -> float:
    """Compute FID against a standard dataset (CIFAR-10, etc.).

    Args:
        generated_images: Generated images of shape (N, C, H, W).
        dataset_name: Name of reference dataset ("cifar10", etc.).
        batch_size: Batch size for feature extraction.
        device: Device for computation.

    Returns:
        FID score.
    """
    try:
        import torch_fidelity
    except ImportError:
        logger.error("torch-fidelity not installed. Install with: pip install torch-fidelity")
        raise

    # Prepare generated images
    if generated_images.min() < 0:
        generated_images = (generated_images + 1) / 2
    generated_images = (generated_images * 255).clamp(0, 255).to(torch.uint8)

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
