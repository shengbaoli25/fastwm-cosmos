#!/usr/bin/env bash
# One process controls all four GPUs; no torchrun and no gradient accumulation.
set -euo pipefail
if [[ $# -lt 2 ]]; then
  printf '%s\n' 'Usage: bash scripts/ablation/launch_pi05.sh {check-hardware|stats|train|resume} CONFIG [--stop-after-steps N]' >&2
  exit 2
fi
action="$1"
case "$action" in check-hardware|stats|train|resume) ;; *) exit 2 ;; esac
config_path="$(realpath "$2")"
shift 2
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
upstream_root="$(python3 - "$config_path" <<'PY'
import json, pathlib, sys
print(pathlib.Path(json.load(open(sys.argv[1], encoding="utf-8"))["upstream_root"]).resolve())
PY
)"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.90}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export PYTHONUNBUFFERED=1
cd "$repo_root"
exec "$upstream_root/.venv/bin/python" scripts/ablation/pi05.py "$action" --config "$config_path" "$@"
