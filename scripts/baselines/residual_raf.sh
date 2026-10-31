#!/usr/bin/env bash
# Usage: bash scripts/baselines/residual_raf.sh [--dataset DATASET] [extra arguments]
set -euo pipefail
repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"
exec python baselines/residual_raf/run.py --out results/baselines/residual_raf "$@"
