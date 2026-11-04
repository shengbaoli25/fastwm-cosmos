#!/usr/bin/env bash
# This file launches the exact variant selected by this experiment branch.
set -euo pipefail
JEPA_SOURCE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
JEPA_PY=/inspire/hdd/project/long-range-embodied-collaborative-intelligence/ky26340/realman-fastwam/.venv/bin/python
mapfile -t JEPA_SELECTION < <("$JEPA_PY" -c 'import json,sys; c=json.load(open(sys.argv[1])); print(c["variant"]); print(c["checkpoint"]); print(c["sha256"])' "$JEPA_SOURCE/configs/experiments/jepa_selected.json")
test "${#JEPA_SELECTION[@]}" -eq 3
exec bash "$JEPA_SOURCE/scripts/cosmos/launch_rm71d78_jepa.sh" \
  --variant "${JEPA_SELECTION[0]}" --jepa-checkpoint "${JEPA_SELECTION[1]}" \
  --jepa-sha256 "${JEPA_SELECTION[2]}" "$@"
