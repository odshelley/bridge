#!/usr/bin/env bash
# Train and evaluate the Poisson Bridge model on MNIST
#
# 1. Train for 40,000 steps
# 2. Generate sample grid for visual inspection
# 3. Evaluate FID at steps 10, 100, 1000

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_DIR="$(dirname "$SCRIPT_DIR")"
cd "$CODE_DIR"

# Use venv Python to avoid conda conflicts
PYTHON=".venv/bin/python"
CLI=".venv/bin/bridge-diffusion"

echo "=============================================="
echo "Poisson Bridge - MNIST Experiments"
echo "=============================================="
echo "Working directory: $CODE_DIR"
echo ""

# Create output directories
mkdir -p outputs/poisson_samples
mkdir -p eval_results/poisson_bridge

# ──────────────────────────────────────────────────
# Step 1: Train
# ──────────────────────────────────────────────────
echo "=============================================="
echo "Step 1: Training Poisson Bridge (40k steps)"
echo "=============================================="
CKPT="checkpoints/poisson/checkpoint_final.pt"
if [ ! -f "$CKPT" ]; then
    $CLI train --config configs/mnist_poisson.yaml
else
    echo "Checkpoint exists at $CKPT, skipping training"
    echo "(Delete it to retrain)"
fi

# ──────────────────────────────────────────────────
# Step 2: Generate sample grid
# ──────────────────────────────────────────────────
echo ""
echo "=============================================="
echo "Step 2: Generating sample grid (64 images)"
echo "=============================================="
$CLI sample \
    --checkpoint "$CKPT" \
    --output-dir outputs/poisson_samples \
    --num-samples 64 \
    --num-steps 100

echo "Sample grid saved to outputs/poisson_samples/grid.png"

# ──────────────────────────────────────────────────
# Step 3: Evaluate FID
# ──────────────────────────────────────────────────
echo ""
echo "=============================================="
echo "Step 3: Evaluating FID scores"
echo "=============================================="
$PYTHON scripts/evaluate_fid.py \
    --checkpoint "$CKPT" \
    --num-samples 10000 \
    --steps 10 100 1000 \
    --batch-size 64 \
    --output-dir eval_results/poisson_bridge

echo ""
echo "=============================================="
echo "All done!"
echo "=============================================="
echo ""
echo "Results:"
echo "  Samples:  outputs/poisson_samples/grid.png"
echo "  FID:      eval_results/poisson_bridge/fid_results_poisson_bridge.txt"
echo ""
echo "MLflow UI:"
echo "  cd $CODE_DIR && mlflow ui --backend-store-uri sqlite:///outputs/mlflow.db"
