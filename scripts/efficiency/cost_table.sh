#!/usr/bin/env bash
# Once/Origin cost table at the last test origin: one-time (development) stages and one new origin.
# Every repetition uses a fresh process and output. Usage: bash scripts/efficiency/cost_table.sh check|smoke|run
set -euo pipefail
repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"
mode="${1:-check}"
case "$mode" in check|smoke|run) ;; *) echo 'Usage: cost_table.sh check|smoke|run' >&2; exit 2;; esac
python_bin="${EFFICIENCY_PYTHON:-python}"
datasets="${EFFICIENCY_DATASETS:-favorita m5 rossmann rohlik}"
methods="${EFFICIENCY_METHODS:-fare timefuse backbone residual pir pfrp saraf raf}"
output="${EFFICIENCY_OUTPUT:-results/efficiency/cost_$(date -u +%Y%m%dT%H%M%SZ)}"
inputs="${EFFICIENCY_INPUTS:-data/baselines}"
model="${EFFICIENCY_MODEL_PATH:-models/chronos-2}"
threads="${EFFICIENCY_THREADS:-8}"
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS="$threads" MKL_NUM_THREADS="$threads" OPENBLAS_NUM_THREADS="$threads" NUMEXPR_NUM_THREADS="$threads"
export EFFICIENCY_DEVICE="${EFFICIENCY_DEVICE:-cuda}"
if [[ "$mode" != check && -e "$output" ]]; then echo "Use a new EFFICIENCY_OUTPUT: $output" >&2; exit 2; fi
profile() {
  local method="$1" phase="$2" spec="$3" label="$4" fit="${5:-}"
  local extra=(); [[ "$mode" == smoke ]] && extra+=(--smoke-queries 8)
  [[ -n "$fit" ]] && extra+=(--fit-dir "$fit")
  env EFFICIENCY_METHOD="$method" EFFICIENCY_DATASET="$dataset"     "$python_bin" -B efficiency/profiler.py --spec "efficiency/specs/$spec.json" --label "$label"     --out-dir "$output/profiles" --meta "dataset=$dataset" --meta "phase=$phase" --meta "mode=$mode"     -- efficiency/entry.py --method "$method" --dataset "$dataset" --phase "$phase" --position last     --inputs "$inputs" --model-path "$model" --output "$output/stages/$label" --device "$EFFICIENCY_DEVICE" --threads "$threads" "${extra[@]}"
}
for dataset in $datasets; do
 for method in $methods; do
  case "$method" in
    fare) once=dev_reference; test=reference; spec=cost_fare ;;
    timefuse) once=fusion; test=reference; spec=timefuse ;;
    backbone) once=backbone_dev; test=backbone; spec=cost_backbone ;;
    raf) once=raf_plan; test=raf_test; spec=cost_raf ;;
    residual|pir|pfrp|saraf) once=fit; test=test; spec="cost_$method" ;;
    *) echo "Unsupported method $method" >&2; exit 2;;
  esac
  if [[ "$mode" == check ]]; then
    env EFFICIENCY_METHOD="$method" EFFICIENCY_DATASET="$dataset" "$python_bin" -B efficiency/entry.py --method "$method" --dataset "$dataset" --phase "$once" --check
    env EFFICIENCY_METHOD="$method" EFFICIENCY_DATASET="$dataset" "$python_bin" -B efficiency/profiler.py --spec "efficiency/specs/$spec.json" --check
    continue
  fi
  reps="${EFFICIENCY_REPS:-3}"; [[ "$mode" == smoke ]] && reps=1
  for ((rep=1; rep<=reps; rep++)); do
    label="${dataset}_${method}_${once}_r$rep"
    profile "$method" "$once" "$spec" "$label"
    case "$method" in
      fare)
        profile fare fusion fare "${dataset}_fare_fusion_r$rep"
        profile fare reference fare "${dataset}_fare_reference_r$rep" ;;
      timefuse) profile timefuse reference timefuse "${dataset}_timefuse_reference_r$rep" ;;
      *) profile "$method" "$test" "$spec" "${dataset}_${method}_${test}_r$rep" "$output/stages/$label" ;;
    esac
  done
 done
done
