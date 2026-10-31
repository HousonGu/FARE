#!/usr/bin/env bash
# Runtime and memory of FARE, TimeFuse and RAF on one machine, one measurement per process.
# Requires the shared input folders (bash scripts/baselines/prepare_inputs.sh DATASET), the raw CSV
# files for the reference stage, and for RAF the Chronos-2 weights (bash scripts/download_chronos.sh).
#
# Usage: bash scripts/efficiency/run.sh {check|smoke|run|scale}
#   check  resolve every measured function; no data is read
#   smoke  at most 8 query series per stage (pipeline test, not a measurement)
#   run    full query sets at the first, middle and last test origin, EFFICIENCY_REPS processes each
#   scale  FARE reference stage at the last origin with 1/8, 1/4, 1/2 and the full library
#
# Environment: EFFICIENCY_DEVICE (cpu|cuda|cuda:0), EFFICIENCY_THREADS, EFFICIENCY_DATASETS,
# EFFICIENCY_METHODS, EFFICIENCY_POSITIONS, EFFICIENCY_REPS, EFFICIENCY_SCALE_DATASETS,
# EFFICIENCY_OUTPUT (a new directory), EFFICIENCY_INPUTS, EFFICIENCY_MODEL_PATH.
set -euo pipefail
repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"
MODE="${1:-check}"
PYTHON="${EFFICIENCY_PYTHON:-python}"
export PYTHONDONTWRITEBYTECODE=1
export EFFICIENCY_DEVICE="${EFFICIENCY_DEVICE:-cuda}"
case "$EFFICIENCY_DEVICE" in cpu|cuda|cuda:0) ;; *) echo 'Use cpu/cuda/cuda:0 and select the GPU with CUDA_VISIBLE_DEVICES.' >&2; exit 2 ;; esac
THREADS="${EFFICIENCY_THREADS:-8}"
export OMP_NUM_THREADS="$THREADS" MKL_NUM_THREADS="$THREADS" OPENBLAS_NUM_THREADS="$THREADS" NUMEXPR_NUM_THREADS="$THREADS"
INPUTS="${EFFICIENCY_INPUTS:-data/baselines}"
MODEL="${EFFICIENCY_MODEL_PATH:-models/chronos-2}"
DATASETS="${EFFICIENCY_DATASETS:-favorita m5 rossmann rohlik}"
METHODS="${EFFICIENCY_METHODS:-fare timefuse raf}"
POSITIONS="${EFFICIENCY_POSITIONS:-first middle last}"
REPS="${EFFICIENCY_REPS:-3}"
BATCH="${EFFICIENCY_BATCH_SIZE:-64}"
OUTPUT="${EFFICIENCY_OUTPUT:-results/efficiency/$(date -u +%Y%m%dT%H%M%SZ)}"

check() {
  for dataset in $DATASETS; do
    for method in $METHODS; do
      env EFFICIENCY_DATASET="$dataset" EFFICIENCY_METHOD="$method" "$PYTHON" -B efficiency/entry.py \
        --dataset "$dataset" --method "$method" --check
      env EFFICIENCY_DATASET="$dataset" EFFICIENCY_METHOD="$method" "$PYTHON" -B efficiency/profiler.py \
        --spec "efficiency/specs/$method.json" --check
    done
  done
}

measure() {
  local dataset="$1" method="$2" position="$3" repeat="$4" phase="$5" fraction="$6" mode="$7"
  local label="${mode}_${dataset}_${method}_${position}_${phase}_f${fraction}_r${repeat}"
  local smoke=()
  [[ "$mode" == smoke ]] && smoke=(--smoke-queries 8)
  echo "efficiency $label"
  env EFFICIENCY_DATASET="$dataset" EFFICIENCY_METHOD="$method" "$PYTHON" -B efficiency/profiler.py \
    --spec "efficiency/specs/$method.json" --label "$label" --out-dir "$OUTPUT/profiles" \
    --bank-fraction "$fraction" --seed 20261027 \
    --meta "dataset=$dataset" --meta "position=$position" --meta "phase=$phase" --meta "mode=$mode" \
    --meta "device=$EFFICIENCY_DEVICE" --meta "threads=$THREADS" \
    -- efficiency/entry.py --method "$method" --dataset "$dataset" --phase "$phase" --position "$position" \
    --inputs "$INPUTS" --model-path "$MODEL" --device "$EFFICIENCY_DEVICE" --batch-size "$BATCH" \
    --output "$OUTPUT/stages/$label" "${smoke[@]}"
}

new_output() {
  [[ ! -e "$OUTPUT" ]] || { echo "Use a new EFFICIENCY_OUTPUT: $OUTPUT" >&2; exit 1; }
}

case "$MODE" in
  check) check ;;
  smoke|run)
    check; new_output
    repeats="$REPS"; [[ "$MODE" == smoke ]] && repeats=1
    for dataset in $DATASETS; do
      for position in $POSITIONS; do
        for method in $METHODS; do
          for ((repeat=1; repeat<=repeats; repeat++)); do
            if [[ "$method" == raf ]]; then
              measure "$dataset" raf "$position" "$repeat" raf 1 "$MODE"
            else
              measure "$dataset" "$method" "$position" "$repeat" reference 1 "$MODE"
              measure "$dataset" "$method" "$position" "$repeat" fusion 1 "$MODE"
            fi
          done
        done
      done
    done ;;
  scale)
    METHODS=fare; DATASETS="${EFFICIENCY_SCALE_DATASETS:-m5 rossmann}"
    check; new_output
    for dataset in $DATASETS; do
      for fraction in 0.125 0.25 0.5 1; do
        for ((repeat=1; repeat<=REPS; repeat++)); do
          measure "$dataset" fare last "$repeat" reference "$fraction" scale
        done
      done
    done ;;
  *) sed -n '2,15p' "$0"; exit 2 ;;
esac
