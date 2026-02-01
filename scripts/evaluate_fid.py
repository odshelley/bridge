#!/usr/bin/env python3
"""Evaluate trained models and compute FID scores.

Reproduces Table 1 from the paper:
- Generate 50,000 samples
- Compute FID against train and test sets
- Evaluate at different sampling step counts (2, 10, 100, 1000)
"""

import argparse
import logging
from pathlib import Path

import mlflow
import torch
from torchmetrics.image.fid import FrechetInceptionDistance
from torchvision import transforms
from tqdm import tqdm

from bridge_diffusion.config import ExperimentConfig, SamplingConfig
from bridge_diffusion.data import get_dataloader
from bridge_diffusion.models import BridgeDiffusion, DDPMDiffusion, DiffusersUNetWrapper
from bridge_diffusion.sampling import Sampler
from bridge_diffusion.utils import get_device, set_seed

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Paper evaluation settings
EVAL_STEPS = [2, 10, 100, 1000]
NUM_SAMPLES = 50000
BATCH_SIZE = 128


def create_model(config: ExperimentConfig, network: DiffusersUNetWrapper):
    """Create the appropriate model based on config.method."""
    if config.method == "bridge":
        return BridgeDiffusion(network, config.bridge)
    elif config.method == "ddpm":
        return DDPMDiffusion(
            network,
            config.bridge,
            num_train_timesteps=config.ddpm.num_train_timesteps,
            beta_schedule=config.ddpm.beta_schedule,
        )
    else:
        raise ValueError(f"Unknown method: {config.method}")


def load_model_from_checkpoint(checkpoint_path: Path, device: torch.device, use_ema: bool = True):
    """Load model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    config = checkpoint["config"]
    
    network = DiffusersUNetWrapper(config.model)
    model = create_model(config, network)
    
    if use_ema and "ema_model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["ema_model_state_dict"])
        logger.info("Loaded EMA weights")
    else:
        model.load_state_dict(checkpoint["model_state_dict"])
        logger.info("Loaded model weights")
    
    model = model.to(device)
    model.eval()
    return model, config


def generate_samples(
    model,
    config: ExperimentConfig,
    num_samples: int,
    num_steps: int,
    device: torch.device,
    batch_size: int = 128,
) -> torch.Tensor:
    """Generate samples from the model."""
    sampling_config = SamplingConfig(
        num_steps=num_steps,
        num_samples=num_samples,
        show_progress=True,
        clip_samples=True,
    )
    
    sampler = Sampler(
        model=model,
        bridge_config=config.bridge,
        sampling_config=sampling_config,
        device=device,
    )
    
    shape = (config.model.in_channels, config.model.sample_size, config.model.sample_size)
    samples = sampler.sample_batch(
        total_samples=num_samples,
        shape=shape,
        batch_size=batch_size,
        num_steps=num_steps,
    )
    
    return samples


def prepare_images_for_fid(images: torch.Tensor) -> torch.Tensor:
    """Prepare images for FID computation.
    
    FID expects:
    - Images in [0, 255] range as uint8
    - RGB format (3 channels)
    - Size at least 64x64 (will be resized to 299x299 internally)
    """
    # Ensure in [0, 1] range
    if images.min() < 0:
        images = (images + 1) / 2  # From [-1, 1] to [0, 1]
    images = images.clamp(0, 1)
    
    # Convert grayscale to RGB if needed
    if images.shape[1] == 1:
        images = images.repeat(1, 3, 1, 1)
    
    # Resize to 299x299 for InceptionV3
    if images.shape[2] != 299 or images.shape[3] != 299:
        images = torch.nn.functional.interpolate(
            images, size=(299, 299), mode="bilinear", align_corners=False
        )
    
    # Convert to uint8 [0, 255]
    images = (images * 255).to(torch.uint8)
    
    return images


def compute_fid(
    real_loader,
    generated_samples: torch.Tensor,
    device: torch.device,
    max_real_samples: int = 50000,
) -> float:
    """Compute FID between real and generated samples."""
    fid = FrechetInceptionDistance(feature=2048, normalize=False).to(device)
    
    # Add real images
    num_real = 0
    for batch in tqdm(real_loader, desc="Processing real images"):
        images = batch[0]
        images = prepare_images_for_fid(images)
        fid.update(images.to(device), real=True)
        num_real += images.shape[0]
        if num_real >= max_real_samples:
            break
    
    # Add generated images in batches
    batch_size = 128
    for i in tqdm(range(0, len(generated_samples), batch_size), desc="Processing generated"):
        batch = generated_samples[i:i + batch_size]
        batch = prepare_images_for_fid(batch)
        fid.update(batch.to(device), real=False)
    
    return fid.compute().item()


def main():
    parser = argparse.ArgumentParser(description="Evaluate model and compute FID scores")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--num-samples", type=int, default=NUM_SAMPLES, help="Number of samples to generate")
    parser.add_argument("--steps", type=int, nargs="+", default=EVAL_STEPS, help="Sampling steps to evaluate")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="Batch size for generation")
    parser.add_argument("--output-dir", type=str, default="./eval_results", help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--no-ema", action="store_true", help="Don't use EMA weights")
    parser.add_argument("--save-samples", action="store_true", help="Save generated samples")
    args = parser.parse_args()
    
    set_seed(args.seed)
    device = get_device()
    logger.info(f"Using device: {device}")
    
    # Load model
    model, config = load_model_from_checkpoint(
        Path(args.checkpoint), device, use_ema=not args.no_ema
    )
    logger.info(f"Method: {config.method}")
    logger.info(f"Dataset: {config.data.dataset}")
    
    # Load real data
    train_loader = get_dataloader(config.data, batch_size=args.batch_size, train=True)
    test_loader = get_dataloader(config.data, batch_size=args.batch_size, train=False)
    
    # Output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # MLflow logging - use tracking URI from config
    mlflow.set_tracking_uri(config.mlflow_tracking_uri)
    logger.info(f"MLflow tracking URI: {config.mlflow_tracking_uri}")
    mlflow.set_experiment(f"{config.name}_evaluation")
    
    results = {}
    
    with mlflow.start_run(run_name=f"{config.method}_fid_evaluation"):
        mlflow.log_params({
            "method": config.method,
            "checkpoint": str(args.checkpoint),
            "num_samples": args.num_samples,
            "eval_steps": str(args.steps),
            "use_ema": not args.no_ema,
        })
        
        for num_steps in args.steps:
            logger.info(f"\n{'='*60}")
            logger.info(f"Evaluating with {num_steps} sampling steps")
            logger.info(f"{'='*60}")
            
            # Generate samples
            logger.info(f"Generating {args.num_samples} samples...")
            samples = generate_samples(
                model, config, args.num_samples, num_steps, device, args.batch_size
            )
            
            # Optionally save samples
            if args.save_samples:
                samples_path = output_dir / f"samples_steps_{num_steps}.pt"
                torch.save(samples, samples_path)
                logger.info(f"Saved samples to {samples_path}")
            
            # Compute FID against train set
            logger.info("Computing FID against training set...")
            fid_train = compute_fid(train_loader, samples, device)
            logger.info(f"FID (train): {fid_train:.2f}")
            
            # Compute FID against test set
            logger.info("Computing FID against test set...")
            fid_test = compute_fid(test_loader, samples, device)
            logger.info(f"FID (test): {fid_test:.2f}")
            
            results[num_steps] = {"train": fid_train, "test": fid_test}
            
            # Log to MLflow
            mlflow.log_metrics({
                f"fid_train_steps_{num_steps}": fid_train,
                f"fid_test_steps_{num_steps}": fid_test,
            })
        
        # Print summary table (matching paper format)
        logger.info("\n" + "="*70)
        logger.info("RESULTS SUMMARY (FID train / test)")
        logger.info("="*70)
        logger.info(f"{'Model':<25} " + " ".join(f"{'Steps='+str(s):>12}" for s in args.steps))
        logger.info("-"*70)
        
        row = f"{config.method.upper():<25} "
        for s in args.steps:
            if s in results:
                row += f"{results[s]['train']:.2f} / {results[s]['test']:.2f}  "
        logger.info(row)
        logger.info("="*70)
        
        # Save results to file
        results_file = output_dir / f"fid_results_{config.method}.txt"
        with open(results_file, "w") as f:
            f.write(f"Model: {config.method}\n")
            f.write(f"Checkpoint: {args.checkpoint}\n")
            f.write(f"Num samples: {args.num_samples}\n")
            f.write(f"Use EMA: {not args.no_ema}\n\n")
            f.write("Steps\tFID (train)\tFID (test)\n")
            for s in args.steps:
                if s in results:
                    f.write(f"{s}\t{results[s]['train']:.2f}\t{results[s]['test']:.2f}\n")
        
        mlflow.log_artifact(str(results_file))
        logger.info(f"\nResults saved to {results_file}")


if __name__ == "__main__":
    main()
