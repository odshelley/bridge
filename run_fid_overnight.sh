#!/bin/bash
# FID Evaluation Script - Run overnight
# Evaluates both SDE and ODE sampling schemes with 2, 10, 100, 1000 steps
#
# Usage: ./run_fid_overnight.sh
# Must be run from the code/ directory

set -e  # Exit on error

# Verify we're in the right directory
if [ ! -f "pyproject.toml" ] || [ ! -d "checkpoints" ]; then
    echo "ERROR: This script must be run from the code/ directory"
    echo "Current directory: $(pwd)"
    exit 1
fi

# Verify checkpoint exists
CHECKPOINT="./checkpoints/checkpoint_final.pt"
if [ ! -f "${CHECKPOINT}" ]; then
    echo "ERROR: Checkpoint not found at ${CHECKPOINT}"
    exit 1
fi

NUM_SAMPLES=50000
BATCH_SIZE=128
STEPS=(2 10 100 1000)
BASE_OUTPUT_DIR="./eval_results"
SCRIPT_PATH="./scripts/evaluate_fid.py"

# Verify script exists
if [ ! -f "${SCRIPT_PATH}" ]; then
    echo "ERROR: Evaluation script not found at ${SCRIPT_PATH}"
    exit 1
fi

# Create timestamp for this run
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
RUN_DIR="${BASE_OUTPUT_DIR}/run_${TIMESTAMP}"

echo "=============================================="
echo "FID Evaluation - Started at $(date)"
echo "=============================================="
echo "Checkpoint: ${CHECKPOINT}"
echo "Samples: ${NUM_SAMPLES}"
echo "Batch size: ${BATCH_SIZE}"
echo "Steps: ${STEPS[*]}"
echo "Output directory: ${RUN_DIR}"
echo "=============================================="
echo ""

# Create output directories
mkdir -p "${RUN_DIR}/sde"
mkdir -p "${RUN_DIR}/ode"

# Track total runs
TOTAL_RUNS=$((${#STEPS[@]} * 2))
CURRENT_RUN=0

# Run SDE evaluations
echo "=============================================="
echo "STARTING SDE EVALUATIONS"
echo "=============================================="

for step in "${STEPS[@]}"; do
    CURRENT_RUN=$((CURRENT_RUN + 1))
    echo ""
    echo "[${CURRENT_RUN}/${TOTAL_RUNS}] SDE with ${step} steps - Started at $(date)"
    echo "----------------------------------------------"

    uv run python "${SCRIPT_PATH}" \
        --checkpoint "${CHECKPOINT}" \
        --num-samples "${NUM_SAMPLES}" \
        --batch-size "${BATCH_SIZE}" \
        --steps "${step}" \
        --output-dir "${RUN_DIR}/sde/steps_${step}" \
        --save-samples

    echo "[${CURRENT_RUN}/${TOTAL_RUNS}] SDE with ${step} steps - Completed at $(date)"
done

# Run ODE evaluations
echo ""
echo "=============================================="
echo "STARTING ODE EVALUATIONS"
echo "=============================================="

for step in "${STEPS[@]}"; do
    CURRENT_RUN=$((CURRENT_RUN + 1))
    echo ""
    echo "[${CURRENT_RUN}/${TOTAL_RUNS}] ODE with ${step} steps - Started at $(date)"
    echo "----------------------------------------------"

    uv run python "${SCRIPT_PATH}" \
        --checkpoint "${CHECKPOINT}" \
        --num-samples "${NUM_SAMPLES}" \
        --batch-size "${BATCH_SIZE}" \
        --steps "${step}" \
        --output-dir "${RUN_DIR}/ode/steps_${step}" \
        --ode \
        --ode-solver heun \
        --save-samples

    echo "[${CURRENT_RUN}/${TOTAL_RUNS}] ODE with ${step} steps - Completed at $(date)"
done

# Summary
echo ""
echo "=============================================="
echo "ALL EVALUATIONS COMPLETED"
echo "Finished at $(date)"
echo "=============================================="
echo ""
echo "Results saved in: ${RUN_DIR}"
echo ""
echo "Directory structure:"
find "${RUN_DIR}" -type f \( -name "*.txt" -o -name "*.pt" \) 2>/dev/null | head -20
echo ""
echo "=== SDE Results ==="
for step in "${STEPS[@]}"; do
    results_file=$(find "${RUN_DIR}/sde/steps_${step}" -name "fid_results_*.txt" 2>/dev/null | head -1)
    if [ -n "${results_file}" ] && [ -f "${results_file}" ]; then
        echo "--- Steps: ${step} ---"
        cat "${results_file}"
        echo ""
    fi
done

echo "=== ODE Results ==="
for step in "${STEPS[@]}"; do
    results_file=$(find "${RUN_DIR}/ode/steps_${step}" -name "fid_results_*.txt" 2>/dev/null | head -1)
    if [ -n "${results_file}" ] && [ -f "${results_file}" ]; then
        echo "--- Steps: ${step} ---"
        cat "${results_file}"
        echo ""
    fi
done
