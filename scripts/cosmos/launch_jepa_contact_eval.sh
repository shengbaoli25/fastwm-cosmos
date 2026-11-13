#!/usr/bin/env bash
# Foreground single-GPU evaluation. Never launches a trainer or modifies its files.
set -euo pipefail
JEPA_EVAL_SOURCE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
JEPA_PY=${JEPA_PY:-/inspire/hdd/project/long-range-embodied-collaborative-intelligence/ky26340/realman-fastwam/.venv/bin/python}
JEPA_DEPENDENCY_DIR=${JEPA_DEPENDENCY_DIR:-/inspire/ssd/project/long-range-embodied-collaborative-intelligence/ky26340/rm71d-jepa-force-latent-20261104/runtime/timm-1.0.15}
export PYTHONPATH="$JEPA_EVAL_SOURCE/src:$JEPA_EVAL_SOURCE:$JEPA_DEPENDENCY_DIR${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=4
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
cd "$JEPA_EVAL_SOURCE"
exec "$JEPA_PY" -u "$JEPA_EVAL_SOURCE/scripts/cosmos/evaluate_jepa_contact.py" "$@"
