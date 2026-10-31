#!/usr/bin/env bash
# Fetch the comparison-method sources into third_party/.
# Usage: bash scripts/fetch_third_party.sh [TimeFuse PFRP PIR SARAF]
set -euo pipefail
repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"
names=("$@")
if [ ${#names[@]} -eq 0 ]; then names=(TimeFuse PFRP PIR SARAF); fi
for name in "${names[@]}"; do
  url="$(python -c "import json,sys; print(json.load(open('configs/third_party.json'))[sys.argv[1]]['repository'])" "$name")"
  target="third_party/$name"
  if [ -d "$target/.git" ]; then
    echo "$name already in $target"; continue
  fi
  git clone -q --depth 1 "$url" "$target"
  echo "$name fetched into $target"
done
