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

import numpy as np
import mlflow
import torch
from scipy import linalg
from torchmetrics.image.fid import NoTrainInceptionV3
from torchvision import transforms
from tqdm import tqdm

from bridge_diffusion.config import ExperimentConfig, SamplingConfig
from bridge_diffusion.data import get_dataloader
from bridge_diffusion.models import BridgeDiffusion, DDPMDiffusion, DiffusersUNetWrapper
from bridge_diffusion.sampling import ODESolver, Sampler
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
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
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
    use_ode: bool = False,
    ode_solver: str = "heun",
    use_pc: bool = False,
    pc_corrector_steps: int = 1,
    pc_snr: float = 0.1,
    use_hybrid: bool = False,
    hybrid_switch_fraction: float = 0.5,
) -> torch.Tensor:
    """Generate samples from the model.

    Args:
        model: Trained model.
        config: Experiment configuration.
        num_samples: Number of samples to generate.
        num_steps: Number of sampling steps.
        device: Device for computation.
        batch_size: Batch size for generation.
        use_ode: If True, use probability flow ODE instead of SDE.
        ode_solver: ODE solver to use ('euler', 'heun', 'rk4').
        use_pc: If True, use Predictor-Corrector sampling.
        pc_corrector_steps: Number of Langevin corrector steps (for PC).
        pc_snr: Signal-to-noise ratio for corrector step size (for PC).
        use_hybrid: If True, use ODE→SDE hybrid sampling.
        hybrid_switch_fraction: Fraction of steps for ODE before switching to SDE.
    """
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

    if use_hybrid:
        # Hybrid ODE→SDE sampling
        solver = ODESolver(ode_solver)
        samples = sampler.sample_batch_hybrid(
            total_samples=num_samples,
            shape=shape,
            batch_size=batch_size,
            num_steps=num_steps,
            switch_fraction=hybrid_switch_fraction,
            ode_solver=solver,
        )
    elif use_pc:
        # Predictor-Corrector sampling
        solver = ODESolver(ode_solver)
        samples = sampler.sample_batch_predictor_corrector(
            total_samples=num_samples,
            shape=shape,
            batch_size=batch_size,
            num_steps=num_steps,
            predictor=solver,
            corrector_steps=pc_corrector_steps,
            corrector_snr=pc_snr,
        )
    elif use_ode:
        solver = ODESolver(ode_solver)
        samples = sampler.sample_batch_ode(
            total_samples=num_samples,
            shape=shape,
            batch_size=batch_size,
            num_steps=num_steps,
            solver=solver,
        )
    else:
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
    """Compute FID between real and generated samples.

    InceptionV3 feature extraction runs on the compute device (e.g. MPS) for speed.
    FID statistics (mean, covariance, matrix sqrt) are computed on CPU with float64.
    """
    # Load InceptionV3 on compute device for fast feature extraction
    inception = NoTrainInceptionV3(name="inception-v3-compat", features_list=["2048"])
    inception = inception.to(device)
    inception.eval()

    def extract_features(images_uint8: torch.Tensor) -> np.ndarray:
        """Run InceptionV3 on device, return numpy features on CPU [batch, 2048]."""
        with torch.no_grad():
            feats = inception(images_uint8.to(device))
        if isinstance(feats, (tuple, list)):
            feats = torch.stack(feats)
        return feats.cpu().numpy()

    # Collect real features
    real_feats = []
    num_real = 0
    for batch in tqdm(real_loader, desc="Processing real images"):
        images = batch[0]
        images = prepare_images_for_fid(images)
        real_feats.append(extract_features(images))
        num_real += images.shape[0]
        if num_real >= max_real_samples:
            break
    real_feats = np.concatenate(real_feats, axis=0)[:max_real_samples]

    # Collect generated features
    fake_feats = []
    batch_size = 128
    for i in tqdm(range(0, len(generated_samples), batch_size), desc="Processing generated"):
        batch = generated_samples[i:i + batch_size]
        batch = prepare_images_for_fid(batch)
        fake_feats.append(extract_features(batch))
    fake_feats = np.concatenate(fake_feats, axis=0)

    # Compute FID on CPU with float64
    mu_real, sigma_real = real_feats.mean(axis=0), np.cov(real_feats, rowvar=False)
    mu_fake, sigma_fake = fake_feats.mean(axis=0), np.cov(fake_feats, rowvar=False)

    diff = mu_real - mu_fake
    covmean, _ = linalg.sqrtm(sigma_real @ sigma_fake, disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    fid = float(diff @ diff + np.trace(sigma_real + sigma_fake - 2 * covmean))
    return fid


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
    parser.add_argument("--ode", action="store_true", help="Use probability flow ODE instead of SDE")
    parser.add_argument("--ode-solver", type=str, default="heun",
                        choices=["euler", "heun", "rk4", "dopri5", "dopri8", "adaptive_heun"],
                        help="ODE solver to use (default: heun). dopri5/dopri8/adaptive_heun use torchdiffeq.")
    parser.add_argument("--pc", action="store_true",
                        help="Use Predictor-Corrector sampling (ODE predictor + Langevin corrector)")
    parser.add_argument("--pc-corrector-steps", type=int, default=1,
                        help="Number of Langevin corrector steps per predictor step (default: 1)")
    parser.add_argument("--pc-snr", type=float, default=0.1,
                        help="Signal-to-noise ratio for corrector step size (default: 0.1)")
    parser.add_argument("--hybrid", action="store_true",
                        help="Use hybrid ODE→SDE sampling (ODE first, then SDE)")
    parser.add_argument("--hybrid-switch", type=float, default=0.5,
                        help="Fraction of steps to use ODE before switching to SDE (default: 0.5)")
    parser.add_argument("--max-real-samples", type=int, default=50000,
                        help="Max real samples for FID computation (fewer = faster but noisier)")
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
            "use_ode": args.ode,
            "use_pc": args.pc,
            "use_hybrid": args.hybrid,
            "ode_solver": args.ode_solver if (args.ode or args.pc or args.hybrid) else "n/a",
            "pc_corrector_steps": args.pc_corrector_steps if args.pc else "n/a",
            "pc_snr": args.pc_snr if args.pc else "n/a",
            "hybrid_switch": args.hybrid_switch if args.hybrid else "n/a",
        })

        for num_steps in args.steps:
            logger.info(f"\n{'='*60}")
            if args.hybrid:
                sampler_type = f"Hybrid ODE→SDE ({args.ode_solver}, switch at {int(args.hybrid_switch*100)}%)"
            elif args.pc:
                sampler_type = f"PC ({args.ode_solver}+langevin, {args.pc_corrector_steps} corrector steps, snr={args.pc_snr})"
            elif args.ode:
                sampler_type = f"ODE ({args.ode_solver})"
            else:
                sampler_type = "SDE (Euler-Maruyama)"
            logger.info(f"Evaluating with {num_steps} sampling steps [{sampler_type}]")
            logger.info(f"{'='*60}")

            # Generate samples
            logger.info(f"Generating {args.num_samples} samples...")
            samples = generate_samples(
                model, config, args.num_samples, num_steps, device, args.batch_size,
                use_ode=args.ode, ode_solver=args.ode_solver,
                use_pc=args.pc, pc_corrector_steps=args.pc_corrector_steps, pc_snr=args.pc_snr,
                use_hybrid=args.hybrid, hybrid_switch_fraction=args.hybrid_switch
            )
            
            # Optionally save samples
            if args.save_samples:
                samples_path = output_dir / f"samples_steps_{num_steps}.pt"
                torch.save(samples, samples_path)
                logger.info(f"Saved samples to {samples_path}")
            
            # Compute FID against train set
            logger.info("Computing FID against training set...")
            fid_train = compute_fid(train_loader, samples, device, max_real_samples=args.max_real_samples)
            logger.info(f"FID (train): {fid_train:.2f}")

            # Compute FID against test set
            logger.info("Computing FID against test set...")
            fid_test = compute_fid(test_loader, samples, device, max_real_samples=args.max_real_samples)
            logger.info(f"FID (test): {fid_test:.2f}")
            
            results[num_steps] = {"train": fid_train, "test": fid_test}
            
            # Log to MLflow
            mlflow.log_metrics({
                f"fid_train_steps_{num_steps}": fid_train,
                f"fid_test_steps_{num_steps}": fid_test,
            })
            
            # Save results incrementally after each step count
            results_file = output_dir / f"fid_results_{config.method}.txt"
            with open(results_file, "w") as f:
                f.write(f"Model: {config.method}\n")
                f.write(f"Checkpoint: {args.checkpoint}\n")
                f.write(f"Num samples: {args.num_samples}\n")
                f.write(f"Use EMA: {not args.no_ema}\n")
                f.write(f"Use ODE: {args.ode}\n")
                f.write(f"Use PC: {args.pc}\n")
                if args.ode or args.pc:
                    f.write(f"ODE Solver: {args.ode_solver}\n")
                if args.pc:
                    f.write(f"PC Corrector Steps: {args.pc_corrector_steps}\n")
                    f.write(f"PC SNR: {args.pc_snr}\n")
                f.write("\nSteps\tFID (train)\tFID (test)\n")
                for s in sorted(results.keys()):
                    f.write(f"{s}\t{results[s]['train']:.2f}\t{results[s]['test']:.2f}\n")
            logger.info(f"Saved intermediate results to {results_file}")
        
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
