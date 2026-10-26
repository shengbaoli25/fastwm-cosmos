#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
TEACHER="${1:?Usage: bash scripts/dense/launch.sh <vjepa21|dinov3> [teacher_source] <weights> [dataset_root]}"
case "${TEACHER}" in
  dinov3|vjepa21) ;;
  *) echo "Unsupported teacher: ${TEACHER}" >&2; exit 2 ;;
esac
TEACHER_SOURCE="${2:-${ROOT}/third_party/${TEACHER}}"
WEIGHTS="${3:?Usage: bash scripts/dense/launch.sh <vjepa21|dinov3> [teacher_source] <weights> [dataset_root]}"
DATASET="${4:-/inspire/ssd/project/long-range-embodied-collaborative-intelligence/ky26340/realman-joint8-pluginsertion71-20261023/data/2026_10_23_lerobot_plug_insertion_100s}"

if [[ -z "${TRAIN_JOB_ID:-}" ]]; then
  echo "This launcher must run inside a platform distributed-training task." >&2
  exit 2
fi

export CUDA_VISIBLE_DEVICES="0,1,2,3"
export OMP_NUM_THREADS="4"
export TOKENIZERS_PARALLELISM="false"
export EXPECTED_CODE_REVISION="${EXPECTED_CODE_REVISION:-$(git -C "${ROOT}" rev-parse HEAD)}"
export GIT_TERMINAL_PROMPT="0"
if [[ "${TEACHER}" == "dinov3" ]]; then
  export DINOV3_LICENSE_ACCEPTED="1"
fi
export PYTHONPATH="${ROOT}/src:${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
ARGS=(--teacher "${TEACHER}" --dataset "${DATASET}" --teacher-source "${TEACHER_SOURCE}")
ARGS+=(--weights "${WEIGHTS}")
exec python -u "${ROOT}/scripts/dense/run_experiment.py" "${ARGS[@]}"
