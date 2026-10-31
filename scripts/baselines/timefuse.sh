#!/usr/bin/env bash
# TimeFuse on the stored B/S/F candidates. Favorita/M5 read data/baselines/DATASET
# (bash scripts/baselines/prepare_inputs.sh DATASET); Rossmann/Rohlik read the completed
# Chronos-2 run results/DATASET/chronos and the raw CSV files.
# Usage: bash scripts/baselines/timefuse.sh [DATASET ...] [--device DEVICE]
set -euo pipefail
repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"
datasets=(); device=cuda
while [ $# -gt 0 ]; do
  case "$1" in
    --device) device="$2"; shift 2 ;;
    --dataset) datasets+=("$2"); shift 2 ;;
    *) datasets+=("$1"); shift ;;
  esac
done
if [ ${#datasets[@]} -eq 0 ]; then datasets=(favorita m5 rossmann rohlik); fi
for dataset in "${datasets[@]}"; do
  case "$dataset" in
    favorita|m5) python baselines/timefuse/run.py --dataset "$dataset" --out results/baselines/timefuse --device "$device" ;;
    rossmann|rohlik) python baselines/timefuse/rossmann_rohlik.py --dataset "$dataset" --out results/baselines/timefuse --device "$device" ;;
    *) echo "unknown dataset $dataset" >&2; exit 2 ;;
  esac
done
