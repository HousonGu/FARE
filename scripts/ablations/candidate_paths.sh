#!/usr/bin/env bash
# Learned B+S, learned B+F, and constant alpha and beta. Usage: bash scripts/ablations/candidate_paths.sh [DATASET ...]
set -euo pipefail
repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"
datasets=("$@"); if [ ${#datasets[@]} -eq 0 ]; then datasets=(favorita m5 rossmann rohlik); fi
for dataset in "${datasets[@]}"; do
  out=results/ablations/candidate_paths/$dataset
  python ablations/candidate_paths/prepare.py --dataset "$dataset" --out "$out/prepared"
  python ablations/candidate_paths/run.py --prepared "$out/prepared" --out "$out" --threads 8
done
