"""Configuration dataclasses for Bridge Diffusion."""

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

import yaml


@dataclass
class ModelConfig:
    """Configuration for the UNet model architecture (diffusers-compatible).
    
    These parameters map to diffusers.UNet2DModel for standard benchmarks.
    """

    # Input/output
    in_channels: int = 1
    out_channels: int = 1
    sample_size: int = 32  # Image resolution
    
    # Architecture (diffusers UNet2DModel parameters)
    block_out_channels: tuple[int, ...] = (128, 256, 256, 256)
    layers_per_block: int = 2
    down_block_types: tuple[str, ...] = (
        "DownBlock2D",
        "AttnDownBlock2D",
        "AttnDownBlock2D",
        "AttnDownBlock2D",
    )
    up_block_types: tuple[str, ...] = (
        "AttnUpBlock2D",
        "AttnUpBlock2D",
        "AttnUpBlock2D",
        "UpBlock2D",
    )
    
    # Attention
    attention_head_dim: int = 8
    
    # Regularisation
    dropout: float = 0.0
    
    # Class conditioning (optional)
    num_class_embeds: int | None = None


@dataclass
class TrainingConfig:
    """Configuration for training."""

    batch_size: int = 128
    num_steps: int = 40_000
    learning_rate: float = 1e-4
    weight_decay: float = 0.0
    grad_clip_norm: float | None = 1.0
    checkpoint_every: int = 5000
    log_every: int = 100
    seed: int = 42
    use_ema: bool = True
    ema_decay: float = 0.9999


@dataclass
class BridgeConfig:
    """Configuration for the Bridge diffusion process."""

    T: float = 0.1  # Terminal time
    eps: float = 1e-7  # Small epsilon to avoid division by zero


@dataclass
class PoissonBridgeConfig:
    """Configuration for the Poisson Bridge diffusion process.

    The Poisson bridge operates on non-negative integer data (e.g. raw pixel
    counts) and uses Binomial interpolation for training and Poisson jumps
    for simulation.
    """

    T: float = 1.0  # Terminal time
    eps: float = 1e-7  # Small epsilon to avoid division by zero
    num_levels: int = 256  # Number of discrete levels (e.g. 256 for 0-255 pixels)
    prior: Literal["zeros", "poisson"] = "zeros"  # Prior distribution type
    prior_lambda: float = 1.0  # Rate parameter when prior="poisson"


@dataclass
class DDPMConfig:
    """Configuration for DDPM baseline."""

    num_train_timesteps: int = 1000
    beta_schedule: Literal["linear", "cosine", "squaredcos_cap_v2"] = "linear"


@dataclass
class SamplingConfig:
    """Configuration for sampling."""

    num_steps: int = 100
    num_samples: int = 64
    deterministic: bool = False
    seed: int | None = None
    show_progress: bool = True
    clip_samples: bool = True


@dataclass
class DataConfig:
    """Configuration for data loading.

    When ``source_dataset`` is set, training runs in transport mode: the bridge
    prior x is drawn from the source dataset instead of Gaussian noise, and the
    dataloader yields (x_source, y_target) pairs (independent coupling).
    """

    dataset: Literal["mnist", "cifar10", "afhq"] = "mnist"
    data_dir: Path = field(default_factory=lambda: Path("./data"))
    image_size: int = 32  # Resize images to this size
    num_workers: int = 4
    pin_memory: bool = True
    raw_pixels: bool = False  # If True, return integer pixel values [0, 255] (for Poisson bridge)
    classes: list[str] | None = None  # Filter target dataset to these class names
    source_dataset: Literal["mnist", "cifar10", "afhq"] | None = None
    source_classes: list[str] | None = None  # Filter source dataset to these class names

    def __post_init__(self) -> None:
        """Validate transport-mode field combinations."""
        if self.source_classes is not None and self.source_dataset is None:
            raise ValueError("source_classes requires source_dataset to be set")


@dataclass
class ExperimentConfig:
    """Full experiment configuration."""

    name: str = "bridge_experiment"
    output_dir: Path = field(default_factory=lambda: Path("./outputs"))
    method: Literal["bridge", "ddpm", "poisson_bridge"] = "bridge"
    mlflow_tracking_uri: str | None = None  # e.g., "sqlite:///mlflow.db" or None for default

    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    bridge: BridgeConfig = field(default_factory=BridgeConfig)
    poisson_bridge: PoissonBridgeConfig = field(default_factory=PoissonBridgeConfig)
    ddpm: DDPMConfig = field(default_factory=DDPMConfig)
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    data: DataConfig = field(default_factory=DataConfig)

    def __post_init__(self) -> None:
        """Ensure paths are Path objects."""
        if isinstance(self.output_dir, str):
            self.output_dir = Path(self.output_dir)
        if isinstance(self.data.data_dir, str):
            self.data.data_dir = Path(self.data.data_dir)
        # Default mlflow tracking to output_dir if not specified
        if self.mlflow_tracking_uri is None:
            self.mlflow_tracking_uri = f"sqlite:///{self.output_dir}/mlflow.db"
        # Transport mode (source_dataset) is only mathematically valid for the
        # Gaussian bridge: ddpm would silently train on the wrong distribution,
        # and poisson_bridge would feed [-1, 1] floats where it expects counts.
        if self.data.source_dataset is not None and self.method != "bridge":
            raise ValueError(
                f"Transport mode (source_dataset) requires method='bridge', got method='{self.method}'"
            )
        # raw_pixels (integer pixel counts) is only meaningful for the Poisson
        # bridge; every other method expects floats normalised to [-1, 1].
        if self.method == "poisson_bridge" and not self.data.raw_pixels:
            raise ValueError(
                "method='poisson_bridge' requires data.raw_pixels=true (integer pixel counts)"
            )
        if self.method != "poisson_bridge" and self.data.raw_pixels:
            raise ValueError(
                f"data.raw_pixels=true requires method='poisson_bridge', got method='{self.method}'"
            )

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ExperimentConfig":
        """Load configuration from a YAML file."""
        with open(path) as f:
            data = yaml.safe_load(f)

        known = {
            "name",
            "output_dir",
            "method",
            "mlflow_tracking_uri",
            "model",
            "training",
            "bridge",
            "poisson_bridge",
            "ddpm",
            "sampling",
            "data",
        }
        unknown = set(data) - known
        if unknown:
            raise ValueError(f"Unknown top-level config keys in {path}: {sorted(unknown)}")

        # Parse nested configs, converting lists to tuples for model config
        model_data = data.get("model", {})
        for key in ["block_out_channels", "down_block_types", "up_block_types"]:
            if key in model_data and isinstance(model_data[key], list):
                model_data[key] = tuple(model_data[key])
        
        model_cfg = ModelConfig(**model_data)
        training_cfg = TrainingConfig(**data.get("training", {}))
        bridge_cfg = BridgeConfig(**data.get("bridge", {}))
        poisson_bridge_cfg = PoissonBridgeConfig(**data.get("poisson_bridge", {}))
        ddpm_cfg = DDPMConfig(**data.get("ddpm", {}))
        sampling_cfg = SamplingConfig(**data.get("sampling", {}))
        data_cfg = DataConfig(**data.get("data", {}))

        return cls(
            name=data.get("name", "bridge_experiment"),
            output_dir=Path(data.get("output_dir", "./outputs")),
            method=data.get("method", "bridge"),
            mlflow_tracking_uri=data.get("mlflow_tracking_uri"),
            model=model_cfg,
            training=training_cfg,
            bridge=bridge_cfg,
            poisson_bridge=poisson_bridge_cfg,
            ddpm=ddpm_cfg,
            sampling=sampling_cfg,
            data=data_cfg,
        )

    def to_yaml(self, path: str | Path) -> None:
        """Save configuration to a YAML file."""

        def serialise(obj: Any) -> Any:
            """Convert dataclass fields to YAML-serialisable types."""
            if isinstance(obj, Path):
                return str(obj)
            if isinstance(obj, tuple):
                return list(obj)
            if isinstance(obj, dict):
                return {k: serialise(v) for k, v in obj.items()}
            return obj

        data = {
            "name": self.name,
            "output_dir": str(self.output_dir),
            "method": self.method,
            "mlflow_tracking_uri": self.mlflow_tracking_uri,
            "model": serialise(asdict(self.model)),
            "training": serialise(asdict(self.training)),
            "bridge": serialise(asdict(self.bridge)),
            "poisson_bridge": serialise(asdict(self.poisson_bridge)),
            "ddpm": serialise(asdict(self.ddpm)),
            "sampling": serialise(asdict(self.sampling)),
            "data": serialise(asdict(self.data)),
        }

        with open(path, "w") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)
