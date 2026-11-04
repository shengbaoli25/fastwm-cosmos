#!/usr/bin/env bash
# Calls the existing Accelerate/DeepSpeed launcher: four ranks, configured real microbatch 32 or 64.
set -euo pipefail
if [[ $# -lt 2 ]]; then
  printf '%s\n' 'Usage: bash scripts/ablation/launch_fastwam_h200.sh /absolute/env/bin/python /absolute/config.yaml [check-hardware|train|resume] [CLI arguments]' >&2
  exit 2
fi
[[ $(uname -s) == Linux ]] || { echo 'Linux CUDA worker required' >&2; exit 2; }
python_env="$1"
config_path="$(realpath "$2")"
action="${3:-train}"
if [[ $# -ge 3 ]]; then shift 3; else shift 2; fi
case "$action" in check-hardware|train|resume) ;; *) exit 2 ;; esac
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export PYTHONUNBUFFERED=1
export DIFFSYNTH_SKIP_DOWNLOAD=true
cd "$repo_root"
exec "$python_env" scripts/ablation/fastwam.py "$action" --config "$config_path" "$@"
