#!/usr/bin/env bash
# Fresh wipe30 data, reference auditor and JEPA; one node with four H200 GPUs.
set -euo pipefail
: "${TRAIN_JOB_ID:?Use a new four-H200 distributed training job}"
JEPA_SOURCE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
JEPA_PY=/inspire/hdd/project/long-range-embodied-collaborative-intelligence/ky26340/realman-fastwam/.venv/bin/python
cd "$JEPA_SOURCE"
if [[ -n "${JEPA_DEPENDENCY_DIR:-}" ]]; then
  test -d "$JEPA_DEPENDENCY_DIR"
  export PYTHONPATH="$JEPA_DEPENDENCY_DIR${PYTHONPATH:+:$PYTHONPATH}"
fi
export PYTHONPATH="$JEPA_SOURCE/src:$JEPA_SOURCE${PYTHONPATH:+:$PYTHONPATH}"
export NCCL_NVLS_ENABLE=0
"$JEPA_PY" -u "$JEPA_SOURCE/scripts/cosmos/check_jepa_dependencies.py"
exec "$JEPA_PY" -u "$JEPA_SOURCE/scripts/cosmos/train_jepa_dataset.py" --reference-audit "$@" --execute
