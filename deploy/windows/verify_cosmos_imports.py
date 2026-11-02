"""CPU-only Cosmos imports from Desktop/infer; no model allocation or hardware I/O."""
from __future__ import annotations

import importlib
import importlib.metadata
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "cosmos/code"
UPSTREAM = ROOT / "runtime/upstream/cosmos-predict2.5"


def main():
    sys.path[:0] = [str(SOURCE / "src"), str(SOURCE / "deploy"), str(SOURCE)]
    versions = {}
    for name in ("torch", "torchvision", "numpy", "Pillow", "einops", "safetensors",
                 "hydra-core", "omegaconf", "rich", "loguru", "nvidia-ml-py",
                 "attrs", "imageio", "imageio-ffmpeg"):
        versions[name] = importlib.metadata.version(name)
    import torch
    from fastwam.cosmos.native import nonaction_native_imports, verify_upstream
    hashes = verify_upstream(UPSTREAM)
    nonaction_native_imports(UPSTREAM)
    modules = {}
    for name in ("fastwam.cosmos.noaction", "fastwam.cosmos.assets",
                 "realman_deploy.model_adapter", "realman_sync.cosmos_policy",
                 "realman_sync.cosmos_checkpoint_policy", "realman_sync.server"):
        module = importlib.import_module(name)
        path = Path(module.__file__).resolve()
        if not path.is_relative_to(SOURCE):
            raise RuntimeError(f"Import escaped the selected infer source: {name}: {path}")
        modules[name] = str(path)
    from fastwam.cosmos.noaction import SCHEMA
    from realman_sync.wire import PROTOCOL
    if torch.cuda.is_initialized() or torch.distributed.is_initialized():
        raise RuntimeError("Import check unexpectedly initialized CUDA or distributed execution")
    print(json.dumps(dict(passed=True, python=sys.executable, versions=versions,
        source=str(SOURCE), upstream=str(UPSTREAM), source_modules=modules,
        pinned_upstream_files=len(hashes), checkpoint_schema=SCHEMA,
        wire_protocol=PROTOCOL, cuda_initialized=False, models_loaded=False,
        hardware_opened=False, server_started=False), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
