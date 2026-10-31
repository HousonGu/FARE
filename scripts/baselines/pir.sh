#!/usr/bin/env bash
# Usage: bash scripts/baselines/pir.sh [DATASET ...] ; requires bash scripts/fetch_third_party.sh first
set -euo pipefail
repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"
datasets=("$@"); if [ ${#datasets[@]} -eq 0 ]; then datasets=(favorita m5 rossmann rohlik); fi
for dataset in "${datasets[@]}"; do
  python baselines/pir/run.py --data "data/baselines/$dataset" --out "results/baselines/pir/$dataset" --device cuda
done
