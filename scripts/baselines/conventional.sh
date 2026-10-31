#!/usr/bin/env bash
# Usage: bash scripts/baselines/conventional.sh [--dataset DATASET] [extra arguments]
set -euo pipefail
repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"
exec python baselines/conventional/run.py --out results/baselines/conventional "$@"
