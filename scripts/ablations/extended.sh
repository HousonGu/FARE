#!/usr/bin/env bash
# Retrieval and error-estimation ablations. Requires the completed Chronos-2 run of each
# dataset (bash scripts/fare/DATASET/chronos.sh). Usage: bash scripts/ablations/extended.sh [DATASET ...]
set -euo pipefail
repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"
datasets=("$@"); if [ ${#datasets[@]} -eq 0 ]; then datasets=(favorita m5 rossmann rohlik); fi
for dataset in "${datasets[@]}"; do
  case "$dataset" in
    favorita|m5)
      if [ "$dataset" = favorita ]; then source=results/favorita/chronos/favorita_original; else source=results/m5/chronos/m5_full; fi
      out=results/ablations/extended/$dataset
      for variant in A5_constant_alpha A6_constant_beta A3_global_errors A4_no_shrink A1_joint; do
        python ablations/extended/run.py --source "$source" --out "$out/${variant}_20261005" --variant "$variant" --seed 20261005
      done
      for seed in 20261005 20261006 20261007; do
        python ablations/extended/run.py --source "$source" --out "$out/A2_random_$seed" --variant A2_random --seed "$seed"
      done ;;
    rossmann|rohlik)
      for variant in no_factor no_disagreement A5_constant_alpha A6_constant_beta A3_global_errors A4_no_shrink A1_joint; do
        python ablations/extended/rossmann_rohlik.py --dataset "$dataset" --variant "$variant" --seed 20261005
      done
      for seed in 20261005 20261006 20261007; do
        python ablations/extended/rossmann_rohlik.py --dataset "$dataset" --variant A2_random --seed "$seed"
      done ;;
    *) echo "unknown dataset $dataset" >&2; exit 2 ;;
  esac
done
