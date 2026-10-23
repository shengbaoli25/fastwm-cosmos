"""Validate the selected CUDA device before allocating model weights."""
from __future__ import annotations

import os
import hashlib
import platform
from pathlib import Path
import subprocess


def configure_compile_cache(device: str, repo_root: str | Path) -> dict[str, str]:
    """Give this checkout, host and visible CUDA device private compiler caches.

    Call before importing Torch. CUDA_VISIBLE_DEVICES logical ordinals are
    translated to physical GPU UUIDs when nvidia-smi is available.
    """
    root = Path(repo_root).expanduser().resolve()
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    logical = device.split(":", 1)[1] if ":" in device else "0"
    selector = logical
    if visible and visible not in {"-1", "NoDevFiles"}:
        entries = [item.strip() for item in visible.split(",") if item.strip()]
        if logical.isdigit() and int(logical) < len(entries):
            selector = entries[int(logical)]
    gpu_identity = ""
    try:
        result = subprocess.run(["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader,nounits"],
                                check=True, capture_output=True, text=True, timeout=3)
        for line in result.stdout.splitlines():
            physical, uuid = [part.strip() for part in line.split(",", 1)]
            if selector in {physical, uuid}:
                gpu_identity = uuid
                break
    except (OSError, subprocess.SubprocessError, ValueError):
        pass
    scope = hashlib.sha256("\0".join((str(root), platform.node(), platform.machine(),
        device, gpu_identity or selector, visible)).encode()).hexdigest()[:20]
    default_root = Path.home() / ".cache" / "fastwam"
    cache_root = Path(os.environ.get("FASTWAM_CACHE_ROOT", default_root))
    scoped = cache_root.expanduser().resolve() / scope
    paths = {"TORCHINDUCTOR_CACHE_DIR": str(scoped / "inductor"),
             "TRITON_CACHE_DIR": str(scoped / "triton"),
             "CUDA_CACHE_PATH": str(scoped / "cuda")}
    for name, value in paths.items():
        Path(value).mkdir(parents=True, exist_ok=True)
        os.environ[name] = value
    return paths


def validate_device(device: str) -> dict:
    import torch
    selected = torch.device(device)
    if selected.type != "cuda":
        raise ValueError("模型部署需要 CUDA GPU；CPU 仅用于无硬件单元测试")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA 不可用；检查 NVIDIA 驱动及当前环境的 PyTorch CUDA wheel")
    index = torch.cuda.current_device() if selected.index is None else selected.index
    if not 0 <= index < torch.cuda.device_count():
        raise ValueError(f"CUDA 设备索引 {index} 不存在（受 CUDA_VISIBLE_DEVICES 影响）")
    selected = torch.device("cuda", index)
    with torch.cuda.device(selected):
        props = torch.cuda.get_device_properties(selected)
        if props.major < 8 or not torch.cuda.is_bf16_supported():
            raise RuntimeError("此模型要求原生 BF16（Ampere 或更新架构）；不会静默改为 FP16")
        # Exercise an actual kernel. Availability alone does not establish that
        # this wheel supports the architecture (especially newer Blackwell GPUs).
        try:
            x = torch.ones((16, 16), device=selected, dtype=torch.bfloat16)
            y = x @ x
            torch.cuda.synchronize(selected)
            if y[0, 0].item() != 16:
                raise RuntimeError("BF16 矩阵乘法返回异常结果")
            del x, y
        except RuntimeError as exc:
            raise RuntimeError(f"{props.name} 的 BF16 CUDA 自检失败；安装支持该架构及驱动的 PyTorch wheel：{exc}") from exc
        free_bytes, total_bytes = torch.cuda.mem_get_info(selected)
    return {"device": str(selected), "selected_device": str(selected), "name": props.name,
            "capability": [props.major, props.minor], "total_bytes": props.total_memory,
            "free_bytes": free_bytes, "driver_total_bytes": total_bytes,
            "torch": torch.__version__, "torch_cuda": torch.version.cuda, "dtype": "bfloat16",
            "wheel_architectures": torch.cuda.get_arch_list(),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"), "kernel_check": "passed"}
