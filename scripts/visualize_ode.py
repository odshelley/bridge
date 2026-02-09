#!/usr/bin/env python3
"""Generate and visualize samples comparing SDE vs ODE sampling."""

import torch
from pathlib import Path
from torchvision.utils import make_grid, save_image

from bridge_diffusion.config import SamplingConfig
from bridge_diffusion.models import DiffusersUNetWrapper
from bridge_diffusion.sampling import Sampler, ODESolver


def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load checkpoint
    checkpoint_path = Path("./checkpoints/checkpoint_final.pt")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    
    # Create model
    from bridge_diffusion.models import BridgeDiffusion
    network = DiffusersUNetWrapper(config.model)
    model = BridgeDiffusion(network, config.bridge)
    
    # Load EMA weights
    if "ema_model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["ema_model_state_dict"])
        print("Loaded EMA weights")
    else:
        model.load_state_dict(checkpoint["model_state_dict"])
    
    model = model.to(device)
    model.eval()
    
    # Sampling config
    sampling_config = SamplingConfig(
        num_steps=10,
        num_samples=10,
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
    
    # Use same initial noise for fair comparison
    torch.manual_seed(42)
    x0 = torch.randn(10, *shape, device=device)
    
    output_dir = Path("./outputs/ode_comparison")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save grids
    def save_grid(samples, name):
        # Denormalize from [-1, 1] to [0, 1]
        samples = (samples + 1) / 2
        samples = torch.clamp(samples, 0, 1)
        grid = make_grid(samples, nrow=5, padding=2)
        path = output_dir / f"{name}.png"
        save_image(grid, path)
        print(f"Saved: {path}")

    step_counts = [10, 50, 100]

    for num_steps in step_counts:
        # Generate with SDE (original method)
        print(f"\n=== Generating with SDE ({num_steps} steps) ===")
        torch.manual_seed(42)
        samples_sde = sampler.sample(10, shape, x0=x0.clone(), num_steps=num_steps)
        save_grid(samples_sde, f"sde_{num_steps}steps")

        # Generate with ODE Heun
        print(f"\n=== Generating with ODE Heun ({num_steps} steps) ===")
        samples_ode_heun = sampler.sample_ode(10, shape, x0=x0.clone(), num_steps=num_steps, solver=ODESolver.HEUN)
        save_grid(samples_ode_heun, f"ode_heun_{num_steps}steps")

        # Generate with ODE RK4
        print(f"\n=== Generating with ODE RK4 ({num_steps} steps) ===")
        samples_ode_rk4 = sampler.sample_ode(10, shape, x0=x0.clone(), num_steps=num_steps, solver=ODESolver.RK4)
        save_grid(samples_ode_rk4, f"ode_rk4_{num_steps}steps")

    print(f"\nAll images saved to {output_dir}")
    print("Compare the grids to see SDE vs ODE sampling quality at different step counts!")


if __name__ == "__main__":
    main()
