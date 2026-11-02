#!/usr/bin/env bash
# Explicit single-process, four-GPU JAX launch; no torchrun or batch accumulation.
set -euo pipefail
if [[ $# -lt 2 || $# -gt 4 ]]; then
  echo "Usage: $0 /absolute/envs/forcevla/bin/python /absolute/run.json [train|resume] [stop_after_steps]" >&2
  exit 2
fi
python_env=$1
config=$2
operation=${3:-train}
extra_args=()
if [[ $# -eq 4 ]]; then extra_args=(--stop-after-steps "$4"); fi
[[ $operation == train || $operation == resume ]] || { echo 'Operation must be train or resume' >&2; exit 2; }
[[ $(uname -s) == Linux ]] || { echo 'This launch requires Linux CUDA' >&2; exit 2; }
repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
cd -- "$repo_root"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
export XLA_PYTHON_CLIENT_MEM_FRACTION=${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.9}
export PYTHONUNBUFFERED=1
# Refuse stale plans before initialization. The Python runner also verifies
# exactly four H200s, one JAX process, pinned source, data and weight identities.
"$python_env" - "$config" <<'PY'
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd() / "src"))
from fastwam.ablation.forcevla import execution_profile
profile = execution_profile(json.loads(Path(sys.argv[1]).read_text()))
expected = {"gpu_count": 4, "gpu_model": "H200", "per_gpu_batch_size": 64,
            "global_batch_size": 256, "fsdp_devices": 4}
for name, value in expected.items():
    if profile[name] != value:
        raise ValueError(f"Requested 4-H200 recipe mismatch: {name}={profile[name]!r}, expected {value!r}")
print(json.dumps(profile, indent=2))
PY
exec "$python_env" scripts/ablation/forcevla.py --config "$config" --operation "$operation" "${extra_args[@]}"
