#!/usr/bin/env bash
# Run X-FUSE GPU integration tests on an HPC cluster node with environment modules.
# Usage:
#   CUDA_MODULE="cuda/12.4" TORCH_INDEX="https://download.pytorch.org/whl/cu124" ./scripts/run_hpc_gpu_tests.sh

set -euo pipefail

CUDA_MODULE="${CUDA_MODULE:-cuda/12.4}"
TORCH_INDEX="${TORCH_INDEX:-https://download.pytorch.org/whl/cu124}"

if command -v module >/dev/null 2>&1; then
    echo "Loading cluster module: ${CUDA_MODULE}"
    module load "${CUDA_MODULE}"
fi

echo "Syncing X-FUSE developer environment via uv..."
uv sync --extra dev

echo "Ensuring CUDA-enabled PyTorch wheels from ${TORCH_INDEX}..."
uv pip install --upgrade torch torchvision --index-url "${TORCH_INDEX}"

echo "Checking X-FUSE runtime info..."
uv run xfuse info

echo "Running GPU test suite..."
uv run pytest -m gpu -v
