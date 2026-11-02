#!/usr/bin/env bash
# Explicit environment preparation; does not download model weights or train.
set -euo pipefail
upstream_root="${1:-/work/upstream/openpi}"
revision=215abfb217dbac7d5f1273282331b9b1866c0479
command -v git >/dev/null
command -v uv >/dev/null
if [[ ! -e "$upstream_root" ]]; then
  mkdir -p "$(dirname "$upstream_root")"
  GIT_LFS_SKIP_SMUDGE=1 git clone --no-checkout https://github.com/Physical-Intelligence/openpi "$upstream_root"
  GIT_LFS_SKIP_SMUDGE=1 git -C "$upstream_root" checkout "$revision"
fi
[[ "$(git -C "$upstream_root" rev-parse HEAD)" == "$revision" ]] || {
  printf '%s\n' "OpenPI must already be at reviewed revision $revision" >&2
  exit 1
}
git -C "$upstream_root" diff --exit-code HEAD --
GIT_LFS_SKIP_SMUDGE=1 git -C "$upstream_root" submodule update --init --recursive
cd "$upstream_root"
GIT_LFS_SKIP_SMUDGE=1 uv sync --frozen --python 3.11
.venv/bin/python -c 'import importlib.metadata as m; print({k: m.version(k) for k in ("jax", "jaxlib", "flax", "orbax-checkpoint", "torch", "transformers", "lerobot")})'
