#!/usr/bin/env bash
# Qizhi distributed training job: 1 node x 4 H200; foreground supervisor owns ranks.
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
NATIVE_PYTHON="${NATIVE_PYTHON:-/inspire/hdd/project/long-range-embodied-collaborative-intelligence/ky26340/realman-fastwam/.venv/bin/python}"
export PYTHONPATH="${ROOT}/src:${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
cd -- "$ROOT"
exec "$NATIVE_PYTHON" -u scripts/cosmos/train_reference_audit.py "$@"
