"""Bridge Diffusion - Gaussian Random Bridge Diffusion Models."""

__version__ = "0.1.0"

# Lazy imports — submodules with heavy deps (torchvision, diffusers) are only
# loaded when their symbols are actually accessed.

_LAZY_IMPORTS: dict[str, tuple[str, str]] = {
    # Config
    "BridgeConfig": ("bridge_diffusion.config", "BridgeConfig"),
    "DataConfig": ("bridge_diffusion.config", "DataConfig"),
    "DDPMConfig": ("bridge_diffusion.config", "DDPMConfig"),
    "ExperimentConfig": ("bridge_diffusion.config", "ExperimentConfig"),
    "ModelConfig": ("bridge_diffusion.config", "ModelConfig"),
    "PoissonBridgeConfig": ("bridge_diffusion.config", "PoissonBridgeConfig"),
    "SamplingConfig": ("bridge_diffusion.config", "SamplingConfig"),
    "TrainingConfig": ("bridge_diffusion.config", "TrainingConfig"),
    # Data
    "get_cifar10_eval_transforms": ("bridge_diffusion.data", "get_cifar10_eval_transforms"),
    "get_cifar10_transforms": ("bridge_diffusion.data", "get_cifar10_transforms"),
    "get_data_info": ("bridge_diffusion.data", "get_data_info"),
    "get_dataloader": ("bridge_diffusion.data", "get_dataloader"),
    "get_dataset": ("bridge_diffusion.data", "get_dataset"),
    "get_mnist_int_transforms": ("bridge_diffusion.data", "get_mnist_int_transforms"),
    "get_mnist_transforms": ("bridge_diffusion.data", "get_mnist_transforms"),
    # Evaluation
    "MetricsLogger": ("bridge_diffusion.evaluation", "MetricsLogger"),
    "compute_fid": ("bridge_diffusion.evaluation", "compute_fid"),
    "compute_fid_against_dataset": ("bridge_diffusion.evaluation", "compute_fid_against_dataset"),
    "compute_fid_from_paths": ("bridge_diffusion.evaluation", "compute_fid_from_paths"),
    # Models
    "CIFAR10_CONFIG": ("bridge_diffusion.models", "CIFAR10_CONFIG"),
    "MNIST_CONFIG": ("bridge_diffusion.models", "MNIST_CONFIG"),
    "BridgeDiffusion": ("bridge_diffusion.models", "BridgeDiffusion"),
    "DDPMDiffusion": ("bridge_diffusion.models", "DDPMDiffusion"),
    "DiffusersUNetWrapper": ("bridge_diffusion.models", "DiffusersUNetWrapper"),
    "PoissonBridgeDiffusion": ("bridge_diffusion.models", "PoissonBridgeDiffusion"),
    # Sampling
    "ODESolver": ("bridge_diffusion.sampling", "ODESolver"),
    "Sampler": ("bridge_diffusion.sampling", "Sampler"),
    # Training
    "Trainer": ("bridge_diffusion.training", "Trainer"),
    # Utils
    "count_parameters": ("bridge_diffusion.utils", "count_parameters"),
    "get_device": ("bridge_diffusion.utils", "get_device"),
    "get_device_info": ("bridge_diffusion.utils", "get_device_info"),
    "set_seed": ("bridge_diffusion.utils", "set_seed"),
}


def __getattr__(name: str):
    if name in _LAZY_IMPORTS:
        import importlib

        module_path, attr = _LAZY_IMPORTS[name]
        mod = importlib.import_module(module_path)
        val = getattr(mod, attr)
        globals()[name] = val  # cache so subsequent access is fast
        return val
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["__version__", *_LAZY_IMPORTS.keys()]
