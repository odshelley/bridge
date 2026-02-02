"""Training module for Bridge Diffusion.

Implements the training loop following Algorithm 2.2.1 from the paper,
with MLflow experiment tracking and checkpointing.
"""

import logging
from pathlib import Path
from typing import Optional

import mlflow
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

from bridge_diffusion.config import ExperimentConfig
from bridge_diffusion.models import BridgeDiffusion

logger = logging.getLogger(__name__)


class Trainer:
    """Trainer for Bridge Diffusion models."""

    def __init__(
        self,
        model: BridgeDiffusion,
        train_loader: DataLoader,
        config: ExperimentConfig,
        device: torch.device,
        checkpoint_dir: Optional[Path] = None,
    ):
        """Initialise Trainer.

        Args:
            model: Bridge diffusion model to train.
            train_loader: DataLoader for training data.
            config: Experiment configuration.
            device: Device to train on.
            checkpoint_dir: Directory for saving checkpoints.
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.config = config
        self.device = device
        self.checkpoint_dir = checkpoint_dir or Path("checkpoints")
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
            y = torch.randn(shape, device=self.device)
            
            # Sample using the model's generate method (Euler-Maruyama simulation)
            samples = model_to_sample.generate(y, num_steps=100)
            
            # Clamp to valid range
            samples = samples.clamp(-1, 1)
            samples = (samples + 1) / 2  # Convert from [-1, 1] to [0, 1]
            
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

        Implements Algorithm 2.2.1:
        1. Sample (x, y) pairs where x is data, y is noise
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

                x = batch[0].to(self.device)
                y = torch.randn_like(x)

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
                    current_lr = self.config.training.learning_rate  # constant LR

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
        }

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

        logger.info(f"Loaded checkpoint from {checkpoint_path} at step {self.global_step}")
