"""Utility functions for reproducibility and device management."""

import random
from typing import Literal

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility across all libraries.

    Args:
        seed: The random seed to use.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # For deterministic algorithms (may impact performance)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device(preferred: Literal["auto", "cuda", "mps", "cpu"] = "auto") -> torch.device:
    """Get the best available device for computation.

    Args:
        preferred: The preferred device type. "auto" will select the best available.

    Returns:
        The torch device to use.
    """
    if preferred == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")
    elif preferred == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available")
        return torch.device("cuda")
    elif preferred == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS is not available")
        return torch.device("mps")
    else:
        return torch.device("cpu")


def get_device_info(device: torch.device) -> dict:
    """Get information about the specified device.

    Args:
        device: The torch device.

    Returns:
        Dictionary containing device information.
    """
    info = {"device": str(device), "type": device.type}

    if device.type == "cuda":
        info.update(
            {
                "name": torch.cuda.get_device_name(device),
                "memory_total": torch.cuda.get_device_properties(device).total_memory,
                "memory_allocated": torch.cuda.memory_allocated(device),
                "memory_cached": torch.cuda.memory_reserved(device),
            }
        )
    elif device.type == "mps":
        info["name"] = "Apple Silicon GPU"

    return info


def count_parameters(model: torch.nn.Module) -> int:
    """Count the number of trainable parameters in a model.

    Args:
        model: The PyTorch model.

    Returns:
        Number of trainable parameters.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
