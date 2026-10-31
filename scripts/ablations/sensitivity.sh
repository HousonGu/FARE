#!/usr/bin/env bash
# One-at-a-time sensitivity of lambda, clip, smoothing constant and L2 weight.
# Usage: bash scripts/ablations/sensitivity.sh [DATASET ...]
set -euo pipefail
repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"
datasets=("$@"); if [ ${#datasets[@]} -eq 0 ]; then datasets=(favorita rossmann rohlik m5); fi
for dataset in "${datasets[@]}"; do
  python ablations/sensitivity/sensitivity.py --dataset "$dataset" --out results/ablations/sensitivity --threads 8
done
