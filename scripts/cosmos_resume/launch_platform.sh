#!/usr/bin/env bash
# Default prints a plan. Pass --execute only after reviewing remote-audit.json.
set -euo pipefail
COSMOS_ROOT=/inspire/ssd/project/long-range-embodied-collaborative-intelligence/ky26340/rm71d-jepa-cosmos-50k-20261029/cosmos
COSMOS_PY=/inspire/hdd/project/long-range-embodied-collaborative-intelligence/ky26340/realman-fastwam/.venv/bin/python
COSMOS_SOURCE="$COSMOS_ROOT/code"
COSMOS_OLD="$COSMOS_ROOT/online/job-187fc8ee-5443-4786-963b-d48882d3f8ff/runs/long/training"
COSMOS_KIT="$COSMOS_ROOT/resume30k/tools"
COSMOS_AUDIT="$COSMOS_KIT/remote-audit.json"
COSMOS_MODE="${1:---plan}"
if [[ "$COSMOS_MODE" != --plan && "$COSMOS_MODE" != --execute ]]; then
  echo 'Usage: bash launch_platform.sh [--plan|--execute]' >&2
  exit 2
fi
if [[ "$COSMOS_MODE" == --execute ]]; then
  : "${TRAIN_JOB_ID:?Use the new four-H200 training job environment}"
fi
COSMOS_OUTPUT="$COSMOS_ROOT/resume30k/online/${TRAIN_JOB_ID:-PLAN_JOB}/training"
COSMOS_ARGS=(
  --config "$COSMOS_OLD/config.yaml"
  --state "$COSMOS_OLD/checkpoints/state/step_025000"
  --output "$COSMOS_OUTPUT"
  --source-root "$COSMOS_SOURCE"
  --audit "$COSMOS_AUDIT"
)
if [[ "$COSMOS_MODE" == --plan ]]; then
  exec "$COSMOS_PY" "$COSMOS_KIT/resume_to_step.py" "${COSMOS_ARGS[@]}"
fi
cd "$COSMOS_SOURCE"
# The foreground platform command starts exactly one local four-rank launcher.
unset RANK LOCAL_RANK WORLD_SIZE LOCAL_WORLD_SIZE GROUP_RANK ROLE_RANK ROLE_WORLD_SIZE MASTER_ADDR MASTER_PORT
unset TORCHELASTIC_RUN_ID TORCHELASTIC_RESTART_COUNT
export CUDA_VISIBLE_DEVICES=0,1,2,3
export PYTHONPATH="$COSMOS_SOURCE/src:$COSMOS_SOURCE${PYTHONPATH:+:$PYTHONPATH}"
export NCCL_NVLS_ENABLE=0
exec "$COSMOS_PY" -m accelerate.commands.launch \
  --config_file "$COSMOS_SOURCE/scripts/accelerate_configs/accelerate_zero1_ds_4gpu.yaml" \
  --num_processes 4 \
  "$COSMOS_KIT/resume_to_step.py" "${COSMOS_ARGS[@]}" --execute
