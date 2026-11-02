#!/usr/bin/env bash
# Install the pinned Linux environment only. Does not download weights or train.
set -euo pipefail
if [[ $# -ne 2 ]]; then
  echo "Usage: $0 /absolute/upstream/ForceVLA /absolute/envs/forcevla" >&2
  exit 2
fi
repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
upstream_root=$1
environment_root=$2
revision=79de88fdd0651d957a2b5212cae8995159c5ad65
[[ $upstream_root = /* && $environment_root = /* ]] || { echo 'Use absolute source/environment paths' >&2; exit 2; }
[[ $(uname -s) == Linux ]] || { echo 'This bootstrap requires Linux CUDA' >&2; exit 2; }
export GIT_LFS_SKIP_SMUDGE=1
if [[ ! -e $upstream_root ]]; then
  git clone --filter=blob:none --no-checkout https://github.com/ft-robotic/ForceVLA.git "$upstream_root"
  git -C "$upstream_root" checkout --detach "$revision"
fi
[[ $(git -C "$upstream_root" rev-parse HEAD) == "$revision" ]] || { echo 'Upstream revision mismatch' >&2; exit 2; }
git -C "$upstream_root" diff --quiet HEAD -- src scripts flaxformer
python_bootstrap=${PYTHON_BOOTSTRAP:-python3.11}
"$python_bootstrap" -c 'import sys; assert sys.version_info[:2] == (3,11), "Use Python 3.11"'
if [[ ! -e $environment_root ]]; then
  "$python_bootstrap" -m venv "$environment_root"
fi
python_env="$environment_root/bin/python"
"$python_env" -c 'import sys; assert sys.version_info[:2] == (3,11), "Use a dedicated Python 3.11 environment"'
"$python_env" -m pip install --upgrade pip setuptools wheel
# The official lerobot gitlink lacks a .gitmodules entry. Install its exact
# recorded revision explicitly, in the same resolver transaction as the model.
"$python_env" -m pip install -c "$repo_root/configs/ablation/forcevla-linux.constraints" \
  -e "$upstream_root" -e "$upstream_root/flaxformer" -e "$upstream_root/packages/openpi-client" \
  'aqtp==0.9.0' 'tensorflow-cpu==2.17.1' \
  'lerobot @ git+https://github.com/huggingface/lerobot@0cf864870cf29f4738d3ade893e6fd13fbd7cdb5'
"$python_env" -m pip check
"$python_env" -c 'from tensorflow.compiler.xla.service import hlo_pb2; from aqt.jax_legacy.jax import flax_layers; import openpi.models.pi0_force, openpi.training.config, openpi.training.data_loader; print("Pinned ForceVLA AQT/model/trainer/data imports passed")'
"$python_env" -m pip freeze > "$environment_root/installed-requirements.txt"
echo "Environment prepared: $python_env. Training has not started."
