#!/usr/bin/env bash
# Build data/prepared/DATASET and data/baselines/DATASET from a finished main Chronos-2 run
# (bash scripts/fare/DATASET/chronos.sh). Usage: bash scripts/baselines/prepare_inputs.sh DATASET
set -euo pipefail
repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"
dataset="${1:?dataset: favorita, m5, rossmann or rohlik}"
case "$dataset" in
  favorita) run=results/favorita/chronos/favorita_original; gates=$run/gates_chronos ;;
  m5)       run=results/m5/chronos/m5_full; gates=$run/gates_chronos ;;
  rossmann|rohlik) run=results/$dataset/chronos; gates=$run ;;
  *) echo "unknown dataset $dataset" >&2; exit 2 ;;
esac
if [ ! -d "data/prepared/$dataset" ]; then
  python -m exp.prepare_panel --dataset "$dataset" --raw-root data/raw \
    --reference-root "$run" --out "data/prepared/$dataset"
fi
python baselines/prepare_inputs.py --dataset "$dataset" --panel "data/prepared/$dataset" \
  --raw-root data/raw --chronos "$run/chronos" \
  --fare-predictions "$gates/predictions.parquet" --fare-fit "$gates/fit.json" \
  --out "data/baselines/$dataset"
