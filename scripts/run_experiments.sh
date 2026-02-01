#!/usr/bin/env bash
# Run all experiments from the paper (Table 1)
# 
# This script trains:
# 1. Bridge model (T=0.1)
# 2. DDPM baseline (improved denoising model with cosine schedule)
# 3. Simple DDPM baseline (linear schedule)
#
# Then evaluates FID at steps 2, 10, 100, 1000

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_DIR="$(dirname "$SCRIPT_DIR")"
cd "$CODE_DIR"

echo "=============================================="
echo "Bridge Diffusion - Paper Experiments"
echo "=============================================="
echo "Working directory: $CODE_DIR"
echo ""

# Create output directories
mkdir -p outputs/mnist
mkdir -p outputs/ddpm_mnist  
mkdir -p outputs/ddpm_mnist_simple
mkdir -p eval_results

# Training settings from paper
# - 40,000 steps
# - Batch size 128
# - Same architecture for all models

echo "=============================================="
echo "Step 1: Training Bridge Model (T=0.1)"
echo "=============================================="
if [ ! -f "checkpoints/bridge_diffusion_mnist/checkpoint_final.pt" ]; then
    uv run bridge-diffusion train --config configs/mnist.yaml
else
    echo "Bridge checkpoint exists, skipping training"
fi

echo ""
echo "=============================================="
echo "Step 2: Training Improved DDPM (cosine schedule)"
echo "=============================================="
if [ ! -f "checkpoints/ddpm_mnist/checkpoint_final.pt" ]; then
    uv run bridge-diffusion train --config configs/mnist_ddpm.yaml
else
    echo "DDPM checkpoint exists, skipping training"
fi

echo ""
echo "=============================================="  
echo "Step 3: Training Simple DDPM (linear schedule)"
echo "=============================================="
if [ ! -f "checkpoints/ddpm_mnist_simple/checkpoint_final.pt" ]; then
    uv run bridge-diffusion train --config configs/mnist_ddpm_simple.yaml
else
    echo "Simple DDPM checkpoint exists, skipping training"
fi

echo ""
echo "=============================================="
echo "Step 4: Evaluating FID Scores"
echo "=============================================="

echo "Evaluating Bridge model..."
uv run python scripts/evaluate_fid.py \
    --checkpoint checkpoints/bridge_diffusion_mnist/checkpoint_final.pt \
    --num-samples 50000 \
    --steps 2 10 100 1000 \
    --output-dir eval_results/bridge

echo ""
echo "Evaluating Improved DDPM..."
uv run python scripts/evaluate_fid.py \
    --checkpoint checkpoints/ddpm_mnist/checkpoint_final.pt \
    --num-samples 50000 \
    --steps 2 10 100 1000 \
    --output-dir eval_results/ddpm_improved

echo ""
echo "Evaluating Simple DDPM..."
uv run python scripts/evaluate_fid.py \
    --checkpoint checkpoints/ddpm_mnist_simple/checkpoint_final.pt \
    --num-samples 50000 \
    --steps 2 10 100 1000 \
    --output-dir eval_results/ddpm_simple

echo ""
echo "=============================================="
echo "All experiments complete!"
echo "=============================================="
echo ""
echo "Results saved to eval_results/"
echo ""
echo "To view experiment tracking in MLflow UI:"
echo "  cd $CODE_DIR && mlflow ui --backend-store-uri sqlite:///outputs/mlflow.db"
echo ""
echo "Then open http://127.0.0.1:5000 in your browser"
