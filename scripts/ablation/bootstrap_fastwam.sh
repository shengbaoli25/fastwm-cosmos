#!/usr/bin/env bash
# Prepare an isolated environment from the worker image's Python; no model download/training.
set -euo pipefail
if [[ $# -lt 1 || $# -gt 3 ]]; then
  printf '%s\n' 'Usage: bash scripts/ablation/bootstrap_fastwam.sh /absolute/envs/fastwam [python3] [/absolute/upstream/FastWAM]' >&2
  exit 2
fi
[[ $(uname -s) == Linux ]] || { echo 'Linux CUDA worker required' >&2; exit 2; }
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
env_root="$1"
python_bootstrap="${2:-python3}"
upstream_root="${3:-/work/upstream/FastWAM}"
revision=7faa71108368fbb3b6885649f112af607427a2d4
bounded() {
  local seconds="$1" label="$2"
  shift 2
  "$python_bootstrap" "$repo_root/scripts/ablation/bounded_command.py" --seconds "$seconds" --label "$label" -- "$@"
}
[[ "$env_root" = /* && "$upstream_root" = /* ]] || { echo 'Use absolute environment/source paths' >&2; exit 2; }
"$python_bootstrap" -c 'import sys; assert (3,10) <= sys.version_info[:2] <= (3,12), "Use Python 3.10, 3.11, or 3.12"'
export GIT_LFS_SKIP_SMUDGE=1
if [[ ! -e "$upstream_root" ]]; then
  mkdir -p "$(dirname "$upstream_root")"
  bounded 180 official-source-clone git clone --filter=blob:none --no-checkout https://github.com/yuantianyuan01/FastWAM "$upstream_root"
  bounded 180 official-source-checkout git -C "$upstream_root" checkout --detach "$revision"
fi
[[ "$(git -C "$upstream_root" rev-parse HEAD)" == "$revision" ]] || { echo 'Official FastWAM source pin differs' >&2; exit 2; }
git -C "$upstream_root" diff --quiet HEAD --
if [[ ! -e "$env_root" ]]; then
  "$python_bootstrap" -m venv "$env_root"
fi
python_env="$env_root/bin/python"
"$python_env" - "$env_root" <<'PY'
from pathlib import Path
import sys
root = Path(sys.argv[1]).resolve()
assert Path(sys.prefix).resolve() == root and sys.prefix != sys.base_prefix, "Use a dedicated virtual environment"
assert "include-system-site-packages = false" in (root / "pyvenv.cfg").read_text().lower(), "Do not inherit the worker image's site-packages"
PY
export PIP_DEFAULT_TIMEOUT="${PIP_DEFAULT_TIMEOUT:-30}"
export PIP_RETRIES="${PIP_RETRIES:-2}"
bounded 180 packaging-tools "$python_env" -m pip install --upgrade pip setuptools wheel
wheelhouse="${FASTWAM_WHEELHOUSE:-$env_root/wheelhouse}"
wheel_args=(--wheelhouse "$wheelhouse" --timeout-seconds "${FASTWAM_WHEEL_TIMEOUT:-600}" --reuse-root "${PIP_CACHE_DIR:-$HOME/.cache/pip}")
if [[ -n "${FASTWAM_WHEEL_REUSE_ROOT:-}" ]]; then wheel_args+=(--reuse-root "$FASTWAM_WHEEL_REUSE_ROOT"); fi
if [[ -n "${FASTWAM_TORCH_WHEEL_BASE_URL:-}" ]]; then wheel_args+=(--base-url "$FASTWAM_TORCH_WHEEL_BASE_URL"); fi
bounded "$(( ${FASTWAM_WHEEL_TIMEOUT:-600} + 60 ))" official-torch-wheel-cache \
  "$python_env" "$repo_root/scripts/ablation/prefetch_fastwam_wheels.py" "${wheel_args[@]}"
# Install verified official local wheels. CUDA/NVIDIA and all other dependencies
# inherit PIP_INDEX_URL; never route the entire resolution to PyTorch.
bounded "${FASTWAM_PIP_TIMEOUT:-900}" torch-and-cuda-dependencies \
  "$python_env" -m pip install --only-binary=:all: -r "$wheelhouse/requirements.txt"
bounded 180 build-prerequisites "$python_env" -m pip install 'packaging==25.0' 'numpy==2.2.6' ninja psutil py-cpuinfo
# DeepSpeed's build metadata needs the already installed Torch. CUDA ops are
# not eagerly built; this recipe uses the existing AdamW/ZeRO1 implementation.
DS_BUILD_OPS=0 bounded "${FASTWAM_PIP_TIMEOUT:-900}" fastwam-dependencies "$python_env" -m pip install --no-build-isolation -e "$repo_root" \
  'sentencepiece==0.2.0' 'ftfy==6.3.1'
"$python_env" -m pip check
"$python_env" - "$repo_root" "$upstream_root" "$env_root" <<'PY'
import hashlib, importlib.metadata as metadata, json, pathlib, sys
root, upstream, environment = map(pathlib.Path, sys.argv[1:])
core = {}
for relative in ("src/fastwam/models/wan22/action_dit.py", "src/fastwam/models/wan22/mot.py"):
    current = (root / relative).read_bytes().replace(b"\r\n", b"\n")
    official = (upstream / relative).read_bytes().replace(b"\r\n", b"\n")
    assert current == official, f"Official core source differs: {relative}"
    core[relative] = hashlib.sha256(current).hexdigest()
import fastwam.models.wan22.fastwam, fastwam.runtime
import torch, torchvision, deepspeed, accelerate, datasets
versions = {key: metadata.version(key) for key in ("torch", "torchvision", "accelerate", "deepspeed", "datasets", "transformers", "sentencepiece", "ftfy")}
receipt = {"official_revision": "7faa71108368fbb3b6885649f112af607427a2d4", "core_source_sha256": core,
           "versions": versions, "torch_cuda_version": torch.version.cuda, "training_started": False}
(environment / "fastwam-environment.json").write_text(json.dumps(receipt, indent=2) + "\n")
print(json.dumps(receipt, indent=2))
PY
"$python_env" -m pip freeze > "$env_root/installed-requirements.txt"
