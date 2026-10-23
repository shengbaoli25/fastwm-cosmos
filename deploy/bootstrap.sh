#!/usr/bin/env bash
# Install only the inference runtime. Never installs drivers or opens hardware.
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON=python3
ENV_DIR="${HOME}/.venvs/fastwm-realman"
CUDA_WHEEL=cu128
TORCH_VERSION=""
VISION_VERSION=""
HARDWARE=0
IN_PLACE=0
DRY_RUN=0
usage() {
  cat <<'EOF'
Usage: bash deploy/bootstrap.sh [options]
  --cuda cu126|cu128|cu130|keep  Torch wheel family (default cu128)
  --python /path/to/python      Python >=3.10 (default python3)
  --venv PATH                  Install in a venv (default ~/.venvs/fastwm-realman)
  --in-place                   Explicitly use --python's existing environment
  --torch-version VERSION      Override Torch (also set --vision-version)
  --vision-version VERSION     Matching torchvision version
  --hardware                   Also install RM75 and RealSense Python bindings
  --dry-run                    Print selected plan, do not install anything
  -h, --help                   Show this help

Defaults: cu126/cu128 -> Torch 2.7.1 + torchvision 0.22.1;
          cu130 -> Torch 2.9.1 + torchvision 0.24.1.
ARM64/GB10: prefer --cuda keep --in-place --python <vendor-env>/bin/python;
           existing packages are preserved and only missing runtime packages added.
No driver, system CUDA toolkit, camera firmware or robot settings are changed.
EOF
}
while (($#)); do
  case "$1" in
    --cuda|--python|--venv|--torch-version|--vision-version)
      if (($# < 2)); then echo "Missing value for $1" >&2; exit 2; fi
      case "$1" in
        --cuda) CUDA_WHEEL="$2";; --python) PYTHON="$2";; --venv) ENV_DIR="$2";;
        --torch-version) TORCH_VERSION="$2";; --vision-version) VISION_VERSION="$2";;
      esac
      shift 2;;
    --hardware) HARDWARE=1; shift;; --in-place) IN_PLACE=1; shift;;
    --dry-run) DRY_RUN=1; shift;; -h|--help) usage; exit 0;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2;;
  esac
done
case "$CUDA_WHEEL" in cu126|cu128|cu130|keep) ;; *) echo 'Unsupported --cuda family' >&2; exit 2;; esac
if [[ -n "$TORCH_VERSION" && -z "$VISION_VERSION" || -z "$TORCH_VERSION" && -n "$VISION_VERSION" ]]; then
  echo 'Override both --torch-version and --vision-version together.' >&2; exit 2
fi
if [[ "$CUDA_WHEEL" == keep && -n "$TORCH_VERSION" ]]; then
  echo '--cuda keep cannot be combined with Torch version overrides.' >&2; exit 2
fi
if [[ -z "$TORCH_VERSION" ]]; then
  if [[ "$CUDA_WHEEL" == cu130 ]]; then TORCH_VERSION=2.9.1; VISION_VERSION=0.24.1;
  else TORCH_VERSION=2.7.1; VISION_VERSION=0.22.1; fi
fi
"$PYTHON" -c 'import sys; assert sys.version_info >= (3, 10), "Python >=3.10 required"'
MACHINE="$("$PYTHON" -c 'import platform; print(platform.machine())')"
printf 'Platform: %s; CUDA wheels: %s; hardware bindings: %s\n' "$MACHINE" "$CUDA_WHEEL" "$HARDWARE"
if [[ "$CUDA_WHEEL" == keep ]]; then
  echo 'Keep the selected environment Torch/torchvision; no Torch package replacement.'
else
  printf 'Torch %s / torchvision %s; wheel availability will be checked for this Python/platform.\n' "$TORCH_VERSION" "$VISION_VERSION"
fi
if ((IN_PLACE)); then printf 'Environment: existing %s\n' "$PYTHON";
else printf 'Environment: %s\n' "$ENV_DIR"; fi
if ((DRY_RUN)); then exit 0; fi
if (( ! IN_PLACE )); then
  if [[ ! -x "$ENV_DIR/bin/python" ]]; then
    if [[ "$CUDA_WHEEL" == keep ]]; then
      echo '--cuda keep needs an existing venv or --in-place; a fresh venv has no Torch.' >&2; exit 2
    fi
    "$PYTHON" -m venv "$ENV_DIR"
  fi
  PYTHON="$ENV_DIR/bin/python"
fi
if [[ "$CUDA_WHEEL" != keep ]]; then
  "$PYTHON" -m pip install --upgrade pip
  WHEELS=(--index-url "https://download.pytorch.org/whl/$CUDA_WHEEL" --only-binary=:all:
          "torch==$TORCH_VERSION" "torchvision==$VISION_VERSION")
  # Fail on unavailable Python/ARM64 wheels before replacing an existing Torch.
  "$PYTHON" -m pip install --dry-run "${WHEELS[@]}"
  "$PYTHON" -m pip install "${WHEELS[@]}"
else
  "$PYTHON" -c 'import torch, torchvision; print("Keeping Torch", torch.__version__, "torchvision", torchvision.__version__)'
fi
MISSING_REQUIREMENTS="$(mktemp)"
trap 'rm -f "$MISSING_REQUIREMENTS"' EXIT
if [[ "$CUDA_WHEEL" == keep ]]; then
  # In a vendor-managed CUDA environment, do not upgrade pip or replace any
  # installed package. Refuse conflicts and install only absent requirements.
  if ! "$PYTHON" - "$ROOT/deploy/requirements.txt" "$MISSING_REQUIREMENTS" <<'PY'
import importlib.metadata as metadata
from pathlib import Path
import sys
from packaging.requirements import Requirement

requirements = [Requirement(line.strip()) for line in Path(sys.argv[1]).read_text().splitlines()
                if line.strip() and not line.lstrip().startswith('#')]
missing, conflicts = [], []
for req in requirements:
    try:
        version = metadata.version(req.name)
    except metadata.PackageNotFoundError:
        missing.append(str(req))
    else:
        if req.specifier and version not in req.specifier:
            conflicts.append(f'{req.name}=={version} conflicts with {req}')
if conflicts:
    print('\n'.join(conflicts), file=sys.stderr)
    raise SystemExit('Resolve vendor environment conflicts explicitly; no packages were changed.')
Path(sys.argv[2]).write_text('\n'.join(missing) + ('\n' if missing else ''))
print('Existing runtime packages preserved; missing:', ', '.join(missing) or 'none')
PY
  then exit 2; fi
  if [[ -s "$MISSING_REQUIREMENTS" ]]; then
    "$PYTHON" -m pip install --no-deps -r "$MISSING_REQUIREMENTS"
  fi
else
  "$PYTHON" -m pip install -r "$ROOT/deploy/requirements.txt"
fi
# The service scripts add src/ and deploy/ to PYTHONPATH directly. Do not
# install the training pyproject, which pins a specific CUDA wheel family and
# adds DeepSpeed/datasets that are not needed for inference.
if ((HARDWARE)); then
  if [[ "$CUDA_WHEEL" == keep ]]; then
    if ! "$PYTHON" - "$ROOT/deploy/requirements-hardware.txt" "$MISSING_REQUIREMENTS" <<'PY'
import importlib.metadata as metadata
from pathlib import Path
import sys
from packaging.requirements import Requirement

missing = []
for line in Path(sys.argv[1]).read_text().splitlines():
    if not line.strip() or line.lstrip().startswith('#'): continue
    req = Requirement(line.strip())
    try: version = metadata.version(req.name)
    except metadata.PackageNotFoundError: missing.append(str(req))
    else:
        if req.specifier and version not in req.specifier:
            raise SystemExit(f'{req.name}=={version} conflicts with {req}; install matching vendor bindings manually')
Path(sys.argv[2]).write_text('\n'.join(missing) + ('\n' if missing else ''))
print('Missing optional hardware wheels:', ', '.join(missing) or 'none')
PY
    then exit 2; fi
    if [[ -s "$MISSING_REQUIREMENTS" ]]; then
      "$PYTHON" -m pip install --dry-run --only-binary=:all: -r "$MISSING_REQUIREMENTS"
      "$PYTHON" -m pip install --only-binary=:all: -r "$MISSING_REQUIREMENTS"
    fi
  else
    "$PYTHON" -m pip install --dry-run --only-binary=:all: -r "$ROOT/deploy/requirements-hardware.txt"
    "$PYTHON" -m pip install --only-binary=:all: -r "$ROOT/deploy/requirements-hardware.txt"
  fi
fi
printf '\nRuntime installed. Next: %q %q --config /path/to/runtime.json --check-cuda\n' "$PYTHON" "$ROOT/deploy/doctor.py"
echo 'No model was loaded. No robot or camera connection was opened.'
