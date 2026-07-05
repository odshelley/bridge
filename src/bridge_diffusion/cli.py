"""Command-line interface for Bridge Diffusion."""

import argparse
import logging
from pathlib import Path

import torch

from bridge_diffusion.config import BridgeConfig, ExperimentConfig, SamplingConfig
from bridge_diffusion.data import get_data_info, get_dataloader
from bridge_diffusion.models import BridgeDiffusion, DDPMDiffusion, DiffusersUNetWrapper, PoissonBridgeDiffusion
from bridge_diffusion.sampling import Sampler
from bridge_diffusion.training import Trainer
from bridge_diffusion.utils import get_device, set_seed

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def create_model(config: ExperimentConfig, network: DiffusersUNetWrapper):
    """Create the appropriate model based on config.method."""
    if config.method == "bridge":
        return BridgeDiffusion(network, config.bridge)
    elif config.method == "poisson_bridge":
        return PoissonBridgeDiffusion(network, config.poisson_bridge)
    elif config.method == "ddpm":
        return DDPMDiffusion(
            network,
            config.bridge,  # passed for compatibility
            num_train_timesteps=config.ddpm.num_train_timesteps,
            beta_schedule=config.ddpm.beta_schedule,
        )
    else:
        raise ValueError(f"Unknown method: {config.method}")


def _load_source_images(
    source_dir: Path,
    image_size: int,
    num_channels: int,
    num_samples: int,
) -> tuple[torch.Tensor, list[str]]:
    """Load up to num_samples images from a folder as prior samples x0.

    Args:
        source_dir: Folder containing source images.
        image_size: Target image size (resize + center crop).
        num_channels: Number of channels (3 for RGB, 1 for grayscale).
        num_samples: Maximum number of images to load.

    Returns:
        (images, stems): tensor of shape (n, C, H, W) in [-1, 1], and the
        source file stems for naming outputs.
    """
    from PIL import Image
    from torchvision import transforms

    extensions = {".png", ".jpg", ".jpeg"}
    paths = sorted(
        p for p in Path(source_dir).iterdir() if p.suffix.lower() in extensions
    )[:num_samples]
    if not paths:
        raise ValueError(f"No images found in {source_dir}")

    mode = "RGB" if num_channels == 3 else "L"
    mean = (0.5,) * num_channels
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean, mean),
    ])
    images = torch.stack([transform(Image.open(p).convert(mode)) for p in paths])
    return images, [p.stem for p in paths]


def train_main(args: argparse.Namespace) -> None:
    """Main training function."""
    config = ExperimentConfig.from_yaml(args.config)
    set_seed(config.training.seed)

    device = get_device()
    logger.info(f"Using device: {device}")
    logger.info(f"Method: {config.method}")

    data_info = get_data_info(config.data)
    logger.info(f"Dataset: {config.data.dataset}")
    logger.info(f"Image size: {data_info['image_size']}x{data_info['image_size']}")
    logger.info(f"Channels: {data_info['num_channels']}")

    train_loader = get_dataloader(
        config.data,
        batch_size=config.training.batch_size,
        train=True,
    )

    # Override config with actual data info
    config.model.in_channels = data_info["num_channels"]
    config.model.out_channels = data_info["num_channels"]
    config.model.sample_size = data_info["image_size"]

    network = DiffusersUNetWrapper(config.model)
    model = create_model(config, network)

    num_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model parameters: {num_params:,}")

    checkpoint_dir = Path(args.checkpoint_dir) if args.checkpoint_dir else None
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        config=config,
        device=device,
        checkpoint_dir=checkpoint_dir,
    )

    if args.resume:
        trainer.load_checkpoint(Path(args.resume))

    logger.info("Starting training...")
    trainer.train()
    logger.info("Training complete!")


def sample_main(args: argparse.Namespace) -> None:
    """Main sampling function."""
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = checkpoint.get("config")

    if config is None:
        if args.config:
            config = ExperimentConfig.from_yaml(args.config)
        else:
            raise ValueError("No config found in checkpoint and no config file provided")

    set_seed(args.seed or config.training.seed)

    device = get_device()
    logger.info(f"Using device: {device}")
    logger.info(f"Method: {config.method}")

    data_info = get_data_info(config.data)

    config.model.in_channels = data_info["num_channels"]
    config.model.out_channels = data_info["num_channels"]
    config.model.sample_size = data_info["image_size"]

    network = DiffusersUNetWrapper(config.model)
    model = create_model(config, network)

    if args.use_ema and "ema_model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["ema_model_state_dict"])
        logger.info("Loaded EMA weights")
    else:
        model.load_state_dict(checkpoint["model_state_dict"])
        logger.info("Loaded model weights")

    sampling_config = SamplingConfig(
        num_steps=args.num_steps,
        num_samples=args.num_samples,
        show_progress=True,
        clip_samples=True,
    )

    sampler = Sampler(
        model=model,
        bridge_config=config.bridge,
        sampling_config=sampling_config,
        device=device,
    )

    shape = (data_info["num_channels"], data_info["image_size"], data_info["image_size"])

    # Transport mode: start the bridge from source images instead of noise
    x0 = None
    source_stems: list[str] | None = None
    if args.source_dir:
        x0, source_stems = _load_source_images(
            Path(args.source_dir),
            image_size=data_info["image_size"],
            num_channels=data_info["num_channels"],
            num_samples=args.num_samples,
        )
        logger.info(f"Loaded {x0.shape[0]} source images from {args.source_dir}")
    elif config.data.source_dataset is not None:
        import dataclasses

        from bridge_diffusion.data import get_dataset

        source_config = dataclasses.replace(
            config.data,
            dataset=config.data.source_dataset,
            classes=config.data.source_classes,
            source_dataset=None,
            source_classes=None,
        )
        val_dataset = get_dataset(source_config, train=False)
        count = min(args.num_samples, len(val_dataset))
        x0 = torch.stack([val_dataset[i][0] for i in range(count)])
        source_stems = [f"{i:04d}" for i in range(count)]
        logger.info(f"Using {count} val images from source dataset as priors")

    if x0 is not None and config.data.source_dataset is None:
        logger.warning(
            "Checkpoint was not trained in transport mode; sampling from "
            "--source-dir images anyway."
        )
    num_to_sample = x0.shape[0] if x0 is not None else args.num_samples

    if config.method == "poisson_bridge":
        if x0 is not None:
            logger.warning(
                "poisson_bridge does not support source images; ignoring "
                "--source-dir / source priors."
            )
        logger.info(f"Generating {args.num_samples} samples with {args.num_steps} steps...")
        # Poisson bridge uses the model's own generate() (Poisson jump simulation)
        from torchvision.utils import make_grid, save_image

        all_samples = []
        model = model.to(device)
        model.eval()
        for start in range(0, args.num_samples, args.batch_size):
            n = min(args.batch_size, args.num_samples - start)
            x = model.sample_prior(
                (n, *shape),
                device=device,
            )
            all_samples.append(model.generate(x, num_steps=args.num_steps))
        samples = torch.cat(all_samples, dim=0)
        # Normalise from [0, num_levels-1] to [0, 1] for saving
        samples = (samples / (model.num_levels - 1)).clamp(0, 1)

        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        for i, sample in enumerate(samples):
            save_image(sample, output_dir / f"sample_{i:04d}.png")

        grid = make_grid(samples[:64], nrow=8, padding=2, normalize=False)
        save_image(grid, output_dir / "grid.png")
    else:
        logger.info(f"Generating {num_to_sample} samples with {args.num_steps} steps...")
        samples = sampler.sample_batch(
            total_samples=num_to_sample,
            shape=shape,
            batch_size=args.batch_size,
            num_steps=args.num_steps,
            x0=x0,
        )

        output_dir = Path(args.output_dir)
        if source_stems is not None:
            from torchvision.utils import save_image

            output_dir.mkdir(parents=True, exist_ok=True)
            for stem, sample in zip(source_stems, samples):
                save_image((sample.clamp(-1, 1) + 1) / 2, output_dir / f"{stem}_translated.png")
            sampler.save_grid(samples[:64], output_dir / "grid.png")
        else:
            sampler.save_samples(samples, output_dir)
            sampler.save_grid(samples[:64], output_dir / "grid.png")

    logger.info(f"Saved samples to {output_dir}")


def evaluate_main(args: argparse.Namespace) -> None:
    """Main evaluation function."""
    from bridge_diffusion.evaluation import compute_fid_from_paths

    device = get_device()
    logger.info(f"Computing FID between {args.real_dir} and {args.generated_dir}")

    fid = compute_fid_from_paths(
        real_path=Path(args.real_dir),
        generated_path=Path(args.generated_dir),
        batch_size=args.batch_size,
        device=device,
    )

    logger.info(f"FID: {fid:.2f}")


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Bridge Diffusion")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Train command
    train_parser = subparsers.add_parser("train", help="Train a model")
    train_parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to config YAML file",
    )
    train_parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="checkpoints",
        help="Directory for saving checkpoints",
    )
    train_parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume from",
    )
    train_parser.set_defaults(func=train_main)

    # Sample command
    sample_parser = subparsers.add_parser("sample", help="Generate samples")
    sample_parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint",
    )
    sample_parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to config YAML file (optional if in checkpoint)",
    )
    sample_parser.add_argument(
        "--output-dir",
        type=str,
        default="samples",
        help="Directory for saving samples",
    )
    sample_parser.add_argument(
        "--num-samples",
        type=int,
        default=64,
        help="Number of samples to generate",
    )
    sample_parser.add_argument(
        "--num-steps",
        type=int,
        default=100,
        help="Number of sampling steps",
    )
    sample_parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size for sampling",
    )
    sample_parser.add_argument(
        "--use-ema",
        action="store_true",
        help="Use EMA weights if available",
    )
    sample_parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for sampling",
    )
    sample_parser.add_argument(
        "--source-dir",
        type=str,
        default=None,
        help="Folder of images to use as bridge priors x0 (transport mode)",
    )
    sample_parser.set_defaults(func=sample_main)

    # Evaluate command
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate samples")
    eval_parser.add_argument(
        "--real-dir",
        type=str,
        required=True,
        help="Directory of real images",
    )
    eval_parser.add_argument(
        "--generated-dir",
        type=str,
        required=True,
        help="Directory of generated images",
    )
    eval_parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size for FID computation",
    )
    eval_parser.set_defaults(func=evaluate_main)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
