#!/usr/bin/env bash
# Usage: bash scripts/baselines/pfrp.sh [DATASET ...] ; requires bash scripts/fetch_third_party.sh first
set -euo pipefail
repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"
datasets=("$@"); if [ ${#datasets[@]} -eq 0 ]; then datasets=(favorita m5 rossmann rohlik); fi
for dataset in "${datasets[@]}"; do
  python baselines/pfrp/run.py --data "data/baselines/$dataset" --out "results/baselines/pfrp/$dataset" --device cuda
done
