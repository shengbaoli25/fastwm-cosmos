#!/usr/bin/env bash
set -euo pipefail
joint8_hdd=/inspire/hdd/project/long-range-embodied-collaborative-intelligence/ky26340
joint8_eval="$joint8_hdd/joint8-eval-5k20k-20261021"
joint8_source="$joint8_hdd/realman-noaction-joint8/code-82cb9d2"
joint8_python="$joint8_hdd/realman-fastwam/.venv/bin/python"
joint8_train=/inspire/ssd/project/long-range-embodied-collaborative-intelligence/ky26340/realman-joint8-merged50-b64-20k-ssd-r1
cd "$joint8_source"
test "$(git rev-parse HEAD)" = 82cb9d2fe4c855d146ab13b0045b175cd833f48a
export PYTHONPATH="$joint8_source/src:$joint8_source"
export OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false
"$joint8_python" -c 'import PIL, av; print("PREDICTION_DEPENDENCIES_OK", flush=True)'
exec "$joint8_python" -u "$joint8_eval/package/run.py" \
  --root "$joint8_eval/results" --training-run "$joint8_train" --steps 5000 15000 --defer-render --no-preview \
  --dataset "$joint8_hdd/realman-noaction-joint8/data/1020_cup_merged50" \
  --source-sha e4feb413686fdcc44e1bc046e4510250d187ef0bd6ae1dc6c1c32e1aad02d60d \
  --tokenizer "$joint8_hdd/realman-dualwm/checkpoints/tokenizer.pth" \
  2>&1 | tee "$joint8_eval/evaluation.log"
