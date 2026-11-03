#!/usr/bin/env bash
# Execute as the foreground command of one new four-H200 platform job.
set -euo pipefail
: "${TRAIN_JOB_ID:?Use a new distributed training job}"
COSMOS_SOURCE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
COSMOS_PY=/inspire/hdd/project/long-range-embodied-collaborative-intelligence/ky26340/realman-fastwam/.venv/bin/python
COSMOS_RUN_ROOT=/inspire/ssd/project/long-range-embodied-collaborative-intelligence/ky26340/rm71d-cosmos-videoisolated-delta30k-20261103
cd "$COSMOS_SOURCE"
export PYTHONPATH="$COSMOS_SOURCE/src:$COSMOS_SOURCE${PYTHONPATH:+:$PYTHONPATH}"
export NCCL_NVLS_ENABLE=0
exec "$COSMOS_PY" -u "$COSMOS_SOURCE/scripts/cosmos/train_rm71d78_delta.py" \
  --output-root "$COSMOS_RUN_ROOT/online/$TRAIN_JOB_ID" --execute
