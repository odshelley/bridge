"""End-to-end training test exercising the real Trainer.train() loop.

Before this test existed, Trainer.train() was executed by zero tests, which is
how two critical DDPM bugs shipped. This test drives the full loop (forward,
backward, optimiser step, EMA update, checkpointing, and sample logging) for
all three methods on a tiny model/dataset, then verifies a checkpoint can be
reloaded and used to generate samples.
"""

import functools
from pathlib import Path

import pytest
import torch

from bridge_diffusion.config import (
    DataConfig,
    ExperimentConfig,
    ModelConfig,
    PoissonBridgeConfig,
    TrainingConfig,
)
from bridge_diffusion.data import get_dataloader
from bridge_diffusion.models import (
    BridgeDiffusion,
    DDPMDiffusion,
    DiffusersUNetWrapper,
    PoissonBridgeDiffusion,
)
from bridge_diffusion.training import Trainer
from bridge_diffusion.training.trainer import _sample_prior


def _build_config(method: str, afhq_dir: Path, tmp_path: Path) -> ExperimentConfig:
    model_config = ModelConfig(
        in_channels=3,
        out_channels=3,
        sample_size=16,
        block_out_channels=(32, 64),
        layers_per_block=1,
        down_block_types=("DownBlock2D", "DownBlock2D"),
        up_block_types=("UpBlock2D", "UpBlock2D"),
    )
    data_config = DataConfig(
        dataset="afhq",
        data_dir=afhq_dir,
        image_size=16,
        num_workers=0,
        # No classes filter -> all 12 fake train images across cat/dog/wild.
        raw_pixels=(method == "poisson_bridge"),
    )
    training_config = TrainingConfig(
        batch_size=2,
        num_steps=3,
        checkpoint_every=2,
        log_every=1,
        use_ema=True,
    )

    kwargs: dict = dict(
        name=f"e2e_{method}",
        output_dir=tmp_path,
        method=method,
        mlflow_tracking_uri=f"sqlite:///{tmp_path}/mlflow.db",
        model=model_config,
        training=training_config,
        data=data_config,
    )
    if method == "poisson_bridge":
        kwargs["poisson_bridge"] = PoissonBridgeConfig(num_levels=256, prior="zeros")

    return ExperimentConfig(**kwargs)


def _build_model(config: ExperimentConfig) -> torch.nn.Module:
    """Mirror cli.create_model's method -> model-class dispatch."""
    network = DiffusersUNetWrapper(config.model)
    if config.method == "bridge":
        return BridgeDiffusion(network, config.bridge)
    if config.method == "poisson_bridge":
        return PoissonBridgeDiffusion(network, config.poisson_bridge)
    if config.method == "ddpm":
        return DDPMDiffusion(
            network,
            num_train_timesteps=config.ddpm.num_train_timesteps,
            beta_schedule=config.ddpm.beta_schedule,
        )
    raise ValueError(f"Unknown method: {config.method}")


@pytest.mark.parametrize("method", ["bridge", "poisson_bridge", "ddpm"])
def test_trainer_train_end_to_end(method, afhq_dir, tmp_path, monkeypatch) -> None:
    # MLflow's sqlite backend defaults new experiments' artifact root to
    # ./mlruns relative to cwd; chdir into tmp_path so no state leaks into the repo.
    monkeypatch.chdir(tmp_path)

    config = _build_config(method, afhq_dir, tmp_path)
    loader = get_dataloader(config.data, batch_size=config.training.batch_size, train=True)
    model = _build_model(config)

    trainer = Trainer(
        model=model,
        train_loader=loader,
        config=config,
        device=torch.device("cpu"),
        checkpoint_dir=tmp_path / "checkpoints",
    )

    # _log_samples defaults to 16 samples at 100 Euler/diffusion steps each,
    # generated twice (mid-training checkpoint + final). Shrink the batch via
    # the documented monkeypatch escape hatch; the call itself still runs on
    # every checkpoint, which is the DDPM crash's regression target.
    trainer._log_samples = functools.partial(trainer._log_samples, num_samples=2)

    trainer.train()

    checkpoint_path = trainer.checkpoint_dir / "checkpoint_final.pt"
    assert checkpoint_path.exists()

    reloaded_model = _build_model(config)
    reloaded_trainer = Trainer(
        model=reloaded_model,
        train_loader=loader,
        config=config,
        device=torch.device("cpu"),
        checkpoint_dir=tmp_path / "checkpoints_reload",
    )
    reloaded_trainer.load_checkpoint(checkpoint_path)

    prior = _sample_prior(reloaded_trainer.model, torch.zeros(2, 3, 16, 16))
    samples = reloaded_trainer.model.generate(prior, num_steps=2)

    assert samples.shape == (2, 3, 16, 16)
    assert torch.isfinite(samples).all()
