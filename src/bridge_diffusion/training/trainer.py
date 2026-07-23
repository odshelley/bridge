"""Training module for Bridge Diffusion.

Implements the training loop following Algorithm 1, Gaussian Bridge Training
(paper_v2 §8), with MLflow experiment tracking and checkpointing.
"""

import logging
import random
from pathlib import Path
from typing import Optional

import mlflow
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

from bridge_diffusion.config import ExperimentConfig

logger = logging.getLogger(__name__)


def _normalise_samples(samples: torch.Tensor, model: nn.Module) -> torch.Tensor:
    """Map generated samples to [0, 1] for image logging.

    Poisson models (exposing num_levels) output integer counts in
    [0, num_levels - 1]; others output [-1, 1] floats.

    Args:
        samples: Generated samples from the model.
        model: The model that produced the samples (checked for num_levels).

    Returns:
        Samples mapped to [0, 1], suitable for make_grid/save_image.
    """
    if hasattr(model, "num_levels"):
        return (samples / (model.num_levels - 1)).clamp(0, 1)
    return (samples.clamp(-1, 1) + 1) / 2


def _sample_prior(model: nn.Module, y: torch.Tensor) -> torch.Tensor:
    """Sample prior x for the training step.

    For PoissonBridgeDiffusion, calls model.sample_prior() to get integer-valued
    prior samples. For all other models, falls back to standard Gaussian noise.

    Args:
        model: The diffusion model.
        y: Data batch (used for shape and device).

    Returns:
        Prior samples with same shape as y.
    """
    if hasattr(model, "sample_prior"):
        return model.sample_prior(y.shape, device=y.device)
    return torch.randn_like(y)


class Trainer:
    """Trainer for Bridge Diffusion models."""

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        config: ExperimentConfig,
        device: torch.device,
        checkpoint_dir: Optional[Path] = None,
    ):
        """Initialise Trainer.

        Args:
            model: Diffusion model exposing compute_training_loss, called as
                compute_training_loss(x=prior, y=data). This convention holds
                for all three methods (bridge, Poisson bridge, and DDPM).
            train_loader: DataLoader for training data.
            config: Experiment configuration.
            device: Device to train on.
            checkpoint_dir: Directory for saving checkpoints. Defaults to
                <config.output_dir>/checkpoints so concurrent or sequential
                experiments never overwrite each other's checkpoints (the
                files are named only by step number).
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.config = config
        self.device = device
        self.checkpoint_dir = checkpoint_dir or Path(config.output_dir) / "checkpoints"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.optimiser = AdamW(
            self.model.parameters(),
            lr=config.training.learning_rate,
            weight_decay=config.training.weight_decay,
        )

        # Use constant learning rate (matching paper's original implementation)
        self.scheduler = None

        self.ema_model: Optional[nn.Module] = None
        if config.training.use_ema:
            self.ema_model = self._create_ema_model()
            self.ema_decay = config.training.ema_decay

        self.global_step = 0
        self.best_loss = float("inf")

        # Transport mode: bridge prior x comes from the source dataset, not noise
        self.transport = config.data.source_dataset is not None
        self._sample_sources: Optional[torch.Tensor] = None
        if self.transport:
            self._sample_sources = self._load_sample_sources()

    def _load_sample_sources(self, num_samples: int = 16) -> torch.Tensor:
        """Load a fixed batch of held-out source (val) images for sample logging."""
        from bridge_diffusion.data import load_source_val_images

        return load_source_val_images(self.config.data, num_samples, spread=True)

    def _prepare_batch(self, batch: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """Extract (x, y) for the bridge loss from a dataloader batch.

        Transport mode: x is the source image from the paired batch.
        Otherwise: x is a prior sample (Gaussian noise, or the model's own prior).
        """
        if self.transport:
            x = batch[0].to(self.device)
            y = batch[1].to(self.device)
        else:
            y = batch[0].to(self.device)
            x = _sample_prior(self.model, y)
        return x, y

    def _create_ema_model(self) -> nn.Module:
        """Create EMA copy of the model."""
        import copy

        ema_model = copy.deepcopy(self.model)
        for param in ema_model.parameters():
            param.requires_grad = False
        return ema_model

    def _update_ema(self) -> None:
        """Update EMA model parameters."""
        if self.ema_model is None:
            return

        with torch.no_grad():
            for ema_param, param in zip(
                self.ema_model.parameters(), self.model.parameters()
            ):
                ema_param.data.mul_(self.ema_decay).add_(
                    param.data, alpha=1 - self.ema_decay
                )

    def _log_samples(self, num_samples: int = 16) -> None:
        """Generate and log sample images to MLflow."""
        import tempfile
        from torchvision.utils import make_grid, save_image
        
        self.model.eval()
        model_to_sample = self.ema_model if self.ema_model is not None else self.model
        
        with torch.no_grad():
            # Generate samples
            shape = (num_samples, self.config.model.in_channels,
                     self.config.model.sample_size, self.config.model.sample_size)
            if self.transport and self._sample_sources is not None:
                x = self._sample_sources.to(self.device)
            else:
                x = _sample_prior(model_to_sample, torch.zeros(shape, device=self.device))
            
            # Sample using the model's generate method (Euler-Maruyama simulation)
            samples = model_to_sample.generate(x, num_steps=100)
            
            # Map to [0, 1] for image logging (Poisson-aware)
            samples = _normalise_samples(samples, model_to_sample)

            # Create grid
            grid = make_grid(samples, nrow=4, padding=2, normalize=False)
            
            # Save and log to MLflow
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
                save_image(grid, f.name)
                mlflow.log_artifact(f.name, artifact_path=f"samples/step_{self.global_step}")
                logger.info(f"Logged sample images at step {self.global_step}")
        
        self.model.train()

    def train(self) -> None:
        """Run the training loop.

        Implements Algorithm 1, Gaussian Bridge Training (paper_v2 §8):
        1. Sample (x, y) pairs where x is the prior (noise), y is data
        2. Sample t uniformly
        3. Compute bridge samples and targets
        4. Minimise MSE loss
        """
        self.model.train()
        
        # Set up MLflow tracking from config
        mlflow.set_tracking_uri(self.config.mlflow_tracking_uri)
        logger.info(f"MLflow tracking URI: {self.config.mlflow_tracking_uri}")
        
        mlflow.set_experiment(self.config.name)

        with mlflow.start_run(run_name=f"{self.config.name}_{self.config.method}_run"):
            # Log all relevant parameters for reproducibility
            mlflow.log_params({
                "method": self.config.method,
                "dataset": self.config.data.dataset,
                "image_size": self.config.data.image_size,
                "batch_size": self.config.training.batch_size,
                "learning_rate": self.config.training.learning_rate,
                "num_steps": self.config.training.num_steps,
                "weight_decay": self.config.training.weight_decay,
                "grad_clip_norm": self.config.training.grad_clip_norm,
                "use_ema": self.config.training.use_ema,
                "ema_decay": self.config.training.ema_decay,
                "bridge_T": self.config.bridge.T,
                "bridge_eps": self.config.bridge.eps,
                "block_out_channels": str(self.config.model.block_out_channels),
                "layers_per_block": self.config.model.layers_per_block,
                "attention_head_dim": self.config.model.attention_head_dim,
                "dropout": self.config.model.dropout,
                "classes": str(self.config.data.classes),
                "source_dataset": str(self.config.data.source_dataset),
                "source_classes": str(self.config.data.source_classes),
            })
            
            # Log DDPM-specific params if using DDPM
            if self.config.method == "ddpm":
                mlflow.log_params({
                    "ddpm_num_train_timesteps": self.config.ddpm.num_train_timesteps,
                    "ddpm_beta_schedule": self.config.ddpm.beta_schedule,
                })

            data_iter = iter(self.train_loader)
            
            # Calculate remaining steps if resuming
            start_step = self.global_step
            remaining_steps = self.config.training.num_steps - start_step
            
            pbar = tqdm(
                range(remaining_steps),
                desc="Training",
                unit="step",
                initial=start_step,
                total=self.config.training.num_steps,
            )

            running_loss = 0.0
            log_interval = self.config.training.log_every

            for step in pbar:
                try:
                    batch = next(data_iter)
                except StopIteration:
                    data_iter = iter(self.train_loader)
                    batch = next(data_iter)

                # Paper notation: x = prior (noise or source image), y = data (target)
                x, y = self._prepare_batch(batch)

                self.optimiser.zero_grad()
                loss = self.model.compute_training_loss(x, y)
                loss.backward()

                if self.config.training.grad_clip_norm and self.config.training.grad_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.training.grad_clip_norm,
                    )

                self.optimiser.step()

                if self.ema_model is not None:
                    self._update_ema()

                running_loss += loss.item()
                self.global_step += 1

                if self.global_step % log_interval == 0:
                    avg_loss = running_loss / log_interval
                    current_lr = self.optimiser.param_groups[0]["lr"]

                    mlflow.log_metrics(
                        {
                            "train_loss": avg_loss,
                            "learning_rate": current_lr,
                        },
                        step=self.global_step,
                    )

                    pbar.set_postfix({"loss": f"{avg_loss:.4f}", "lr": f"{current_lr:.2e}"})
                    running_loss = 0.0

                if self.global_step % self.config.training.checkpoint_every == 0:
                    self.save_checkpoint()
                    self._log_samples()

            self.save_checkpoint(final=True)
            self._log_samples()  # Final samples
            mlflow.pytorch.log_model(self.model, "model")

    def save_checkpoint(self, final: bool = False) -> None:
        """Save model checkpoint.

        Note: only torch (CPU), numpy, and python RNG state is captured, plus
        CUDA generator state when available. MPS has no equivalent
        get_rng_state API, so device-local RNG state is not captured and
        resume determinism is partial when training on MPS.

        Args:
            final: Whether this is the final checkpoint.
        """
        suffix = "final" if final else f"step_{self.global_step}"
        checkpoint_path = self.checkpoint_dir / f"checkpoint_{suffix}.pt"

        checkpoint = {
            "global_step": self.global_step,
            "model_state_dict": self.model.state_dict(),
            "optimiser_state_dict": self.optimiser.state_dict(),
            "config": self.config,
            "rng_state": {
                "torch": torch.get_rng_state(),
                "numpy": np.random.get_state(),
                "python": random.getstate(),
            },
        }

        if torch.cuda.is_available():
            checkpoint["rng_state"]["cuda"] = torch.cuda.get_rng_state_all()

        if self.ema_model is not None:
            checkpoint["ema_model_state_dict"] = self.ema_model.state_dict()

        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Saved checkpoint to {checkpoint_path}")

    def load_checkpoint(self, checkpoint_path: Path) -> None:
        """Load model checkpoint.

        Args:
            checkpoint_path: Path to checkpoint file.
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimiser.load_state_dict(checkpoint["optimiser_state_dict"])
        self.global_step = checkpoint["global_step"]

        if self.ema_model is not None and "ema_model_state_dict" in checkpoint:
            self.ema_model.load_state_dict(checkpoint["ema_model_state_dict"])

        rng_state = checkpoint.get("rng_state")
        if rng_state is not None:
            # torch.load(map_location=self.device) moves every tensor in the
            # checkpoint onto that device, including these RNG ByteTensors.
            # torch.set_rng_state/torch.cuda.set_rng_state_all both require
            # CPU ByteTensors, so they must be moved back before restoring.
            torch.set_rng_state(rng_state["torch"].cpu())
            np.random.set_state(rng_state["numpy"])
            random.setstate(rng_state["python"])
            if torch.cuda.is_available() and "cuda" in rng_state:
                saved_states = rng_state["cuda"]
                current_count = torch.cuda.device_count()
                if len(saved_states) != current_count:
                    logger.warning(
                        f"Checkpoint saved CUDA RNG state for {len(saved_states)} device(s) "
                        f"but {current_count} are available now; restoring only the first "
                        f"{min(len(saved_states), current_count)}. CUDA RNG replay on resume "
                        "will not be exact."
                    )
                usable = saved_states[:current_count]
                torch.cuda.set_rng_state_all([s.cpu() for s in usable])

        logger.info(f"Loaded checkpoint from {checkpoint_path} at step {self.global_step}")
