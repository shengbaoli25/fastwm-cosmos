#!/usr/bin/env bash
# Foreground command of a new four-H200 job. Pass --variant explicitly.
set -euo pipefail
: "${TRAIN_JOB_ID:?Use a new distributed training job}"
JEPA_SOURCE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
JEPA_PY=/inspire/hdd/project/long-range-embodied-collaborative-intelligence/ky26340/realman-fastwam/.venv/bin/python
cd "$JEPA_SOURCE"
export PYTHONPATH="$JEPA_SOURCE/src:$JEPA_SOURCE${PYTHONPATH:+:$PYTHONPATH}"
export NCCL_NVLS_ENABLE=0
exec "$JEPA_PY" -u "$JEPA_SOURCE/scripts/cosmos/train_rm71d78_jepa.py" "$@" --execute
