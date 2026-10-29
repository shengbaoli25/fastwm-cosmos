#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
EXPERIMENT="${1:?Usage: bash scripts/dense/launch.sh <main|control> <weights> [teacher_source] [dataset_root] [completed_cache]}"
WEIGHTS="${2:?Pre-staged DINOv3 ViT-H+/16 model.safetensors is required}"
TEACHER_SOURCE="${3:-${ROOT}/third_party/dinov3}"
DATASET="${4:-/inspire/ssd/project/long-range-embodied-collaborative-intelligence/ky26340/realman-joint8-pluginsertion71-20261023/data/2026_10_23_lerobot_plug_insertion_100s}"
case "${EXPERIMENT}" in main|control) ;; *) echo "Expected main or control" >&2; exit 2 ;; esac
if [[ -z "${TRAIN_JOB_ID:-}" ]]; then
  echo "This launcher must run inside a platform distributed-training task." >&2
  exit 2
fi
export CUDA_VISIBLE_DEVICES="0,1,2,3"
export OMP_NUM_THREADS="4"
export TOKENIZERS_PARALLELISM="false"
export EXPECTED_CODE_REVISION="${EXPECTED_CODE_REVISION:-$(git -C "${ROOT}" rev-parse HEAD)}"
export GIT_TERMINAL_PROMPT="0"
export PYTHONPATH="${ROOT}/src:${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
ARGS=(--teacher dinov3_vith16plus --experiment "${EXPERIMENT}" --weights "${WEIGHTS}"
      --dataset "${DATASET}" --teacher-source "${TEACHER_SOURCE}")
if [[ -n "${5:-}" ]]; then ARGS+=(--cache-root "$5"); fi
exec python -u "${ROOT}/scripts/dense/run_experiment.py" "${ARGS[@]}"
