"""Evaluation module for Bridge Diffusion."""

from bridge_diffusion.evaluation.metrics import (
    MetricsLogger,
    compute_fid,
    compute_fid_against_dataset,
    compute_fid_from_paths,
)

__all__ = [
    "compute_fid",
    "compute_fid_from_paths",
    "compute_fid_against_dataset",
    "MetricsLogger",
]
