#!/usr/bin/env bash
# Download the Chronos-2 weights used by RAF and the LightGBM residual corrector.
set -euo pipefail
repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"
huggingface-cli download amazon/chronos-2 --local-dir models/chronos-2
