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
    """Configuration for data loading."""

    dataset: Literal["mnist", "cifar10"] = "mnist"
    data_dir: Path = field(default_factory=lambda: Path("./data"))
    image_size: int = 32  # Resize images to this size
    num_workers: int = 4
    pin_memory: bool = True


@dataclass
class ExperimentConfig:
    """Full experiment configuration."""

    name: str = "bridge_experiment"
    output_dir: Path = field(default_factory=lambda: Path("./outputs"))
    method: Literal["bridge", "ddpm"] = "bridge"
    mlflow_tracking_uri: str | None = None  # e.g., "sqlite:///mlflow.db" or None for default

    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    bridge: BridgeConfig = field(default_factory=BridgeConfig)
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

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ExperimentConfig":
        """Load configuration from a YAML file."""
        with open(path) as f:
            data = yaml.safe_load(f)

        # Parse nested configs, converting lists to tuples for model config
        model_data = data.get("model", {})
        for key in ["block_out_channels", "down_block_types", "up_block_types"]:
            if key in model_data and isinstance(model_data[key], list):
                model_data[key] = tuple(model_data[key])
        
        model_cfg = ModelConfig(**model_data)
        training_cfg = TrainingConfig(**data.get("training", {}))
        bridge_cfg = BridgeConfig(**data.get("bridge", {}))
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
            "ddpm": serialise(asdict(self.ddpm)),
            "sampling": serialise(asdict(self.sampling)),
            "data": serialise(asdict(self.data)),
        }

        with open(path, "w") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)
