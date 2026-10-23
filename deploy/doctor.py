#!/usr/bin/env python3
"""Read-only deployment checks; never connects cameras/robot or sends commands."""
from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import platform
import re
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "deploy"))
sys.path.insert(0, str(ROOT / "src"))


def resolved_path(value, base):
    path = Path(value).expanduser()
    return (path if path.is_absolute() else Path(base) / path).resolve()


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def inspect_config(config_path, *, checkpoint=None, hardware_libraries=False):
    """Only reads files and Python package metadata. Does not import torch."""
    config_path = Path(config_path).expanduser().resolve()
    checks = []

    def record(name, action):
        try:
            details = action()
            checks.append(dict(name=name, status="ok", detail=details))
            return details
        except Exception as exc:
            checks.append(dict(name=name, status="error", detail=str(exc)))
            return None

    def warn(name, detail):
        checks.append(dict(name=name, status="warning", detail=detail))

    def read_json(path):
        data = json.loads(Path(path).read_text())
        if not isinstance(data, dict):
            raise ValueError(f"JSON 顶层必须是对象: {path}")
        return data

    config = record("runtime_config", lambda: read_json(config_path))
    report = dict(config=str(config_path), platform=platform.platform(), machine=platform.machine(),
                  python=sys.version.split()[0], executable=sys.executable, checks=checks,
                  cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"))
    if config is None:
        return report
    # Keep config out of reports: it may contain local paths or unrelated secrets.
    checks[-1]["detail"] = str(config_path)
    base = config_path.parent
    device = str(config.get("device", "cuda"))
    report["requested_device"] = device
    if not re.fullmatch(r"cuda(?::\d+)?", device):
        checks.append(dict(name="device", status="error", detail="device 必须为 cuda 或 cuda:N"))
    else:
        checks.append(dict(name="device", status="ok", detail=device + "（逻辑编号，受 CUDA_VISIBLE_DEVICES 影响）"))

    def require_file(value):
        path = resolved_path(value, base)
        if not path.is_file():
            raise FileNotFoundError(str(path))
        return str(path)

    def require_dir(value):
        path = resolved_path(value, base)
        if not path.is_dir():
            raise FileNotFoundError(str(path))
        return str(path)

    paths = {}
    for key in ("normalization_file", "tokenizer", "hardware_config"):
        paths[key] = record(key, lambda key=key: require_file(config[key]))
    kwargs = config.get("model_kwargs", {})
    if not isinstance(kwargs, dict):
        checks.append(dict(name="model_kwargs", status="error", detail="model_kwargs 必须是对象"))
        kwargs = {}
    for key, validator in (("upstream", require_dir), ("checkpoint", require_file), ("text_cache", require_dir)):
        paths[key] = record("model_kwargs." + key, lambda key=key, validator=validator: validator(kwargs[key]))
    if paths.get("upstream"):
        def upstream_check():
            from fastwam.cosmos.native import verify_upstream
            verify_upstream(Path(paths["upstream"]))
            return "Cosmos 核心源码 SHA256 与仓库固定版本一致"
        record("upstream_identity", upstream_check)
    if paths.get("checkpoint") and paths.get("tokenizer"):
        def official_assets_check():
            source = ROOT / "src/fastwam/cosmos/assets.py"
            tree = ast.parse(source.read_text())
            digests = {}
            for node in tree.body:
                if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
                    if node.targets[0].id in {"WEIGHT_SHA", "NONACTION_WEIGHT_SHA", "TOKENIZER_SHA"}:
                        digests[node.targets[0].id] = ast.literal_eval(node.value)
            if len(digests) != 3:
                raise ValueError("仓库中缺少官方 asset SHA pin")
            base_digest = sha256_file(paths["checkpoint"])
            tokenizer_digest = sha256_file(paths["tokenizer"])
            variants = {digests["WEIGHT_SHA"]: "action-conditioned",
                        digests["NONACTION_WEIGHT_SHA"]: "base/post-trained"}
            if base_digest not in variants:
                raise ValueError("Cosmos 基座权重 SHA256 与固定的官方版本不匹配")
            if tokenizer_digest != digests["TOKENIZER_SHA"]:
                raise ValueError("Cosmos tokenizer SHA256 不匹配")
            # The loader selects the factory from the trained checkpoint schema
            # and enforces that factory's exact base identity again.
            return dict(base_checkpoint_sha256=base_digest, base_variant=variants[base_digest],
                        tokenizer_sha256=tokenizer_digest)
        record("official_cosmos_assets", official_assets_check)

    normalizer = None
    if paths.get("normalization_file"):
        def normalization_check():
            from realman_deploy.normalization import DatasetNormalizer
            return DatasetNormalizer.from_json(paths["normalization_file"])
        normalizer = record("normalization", normalization_check)
        if normalizer is not None:
            checks[-1]["detail"] = dict(action_space=normalizer.action_space, action_dim=normalizer.action_dim,
                force_preprocessing_id=(normalizer.force_preprocessing or {}).get("id"))
    if paths.get("text_cache"):
        def text_check():
            manifest = read_json(Path(paths["text_cache"]) / "manifest.json")
            if manifest.get("schema") != "cosmos-reason1-full-concat-v1" or manifest.get("status") != "complete":
                raise ValueError("需要完整的 Cosmos Reason1 text cache")
            prompts = [entry["prompt"] for entry in manifest.get("texts", [])]
            if not prompts or len(prompts) != len(set(prompts)):
                raise ValueError("text cache 必须包含唯一任务文本")
            prompt = config.get("task_prompt")
            if prompt not in prompts and not (prompt is None and len(prompts) == 1):
                raise ValueError("task_prompt 不存在于 text cache")
            expected = kwargs.get("expected_text_identity")
            digest = hashlib.sha256(json.dumps(manifest, sort_keys=True, separators=(",", ":"),
                ensure_ascii=False, allow_nan=False).encode()).hexdigest()
            if expected and digest != expected:
                raise ValueError("text cache identity 与配置不一致")
            pinned = read_json(ROOT / "src/fastwam/cosmos/reason1_weights.json")
            if manifest.get("encoder_revision") != pinned.get("revision"):
                raise ValueError("Reason1 encoder revision 与仓库固定版本不一致")
            if any(manifest.get("encoder_files", {}).get(key) != value
                   for key, value in pinned.get("weights", {}).items()):
                raise ValueError("Reason1 encoder 权重文件 identity 与仓库固定版本不一致")
            payload_hashes = {}
            for entry in manifest["texts"]:
                path = (Path(paths["text_cache"]) / entry["path"]).resolve()
                if not path.is_relative_to(Path(paths["text_cache"])):
                    raise ValueError("text payload 路径超出 cache 目录")
                if not path.is_file():
                    raise FileNotFoundError(path)
                payload_digest = sha256_file(path)
                if payload_digest != entry.get("sha256"):
                    raise ValueError("text cache SafeTensors payload SHA256 不匹配: " + entry["path"])
                payload_hashes[entry["prompt"]] = payload_digest
            return dict(prompts=prompts, identity_sha256=digest, payload_sha256=payload_hashes)
        record("text_manifest", text_check)

    if paths.get("hardware_config") and config.get("hardware_factory", "realman_deploy.hardware:make_hardware") is None:
        warn("hardware_config_only", "离线部署：hardware_factory 为 null，硬件连接已禁用；未验证实机 IP、相机或 SDK 配置。")
    elif paths.get("hardware_config"):
        def hardware_check():
            from realman_deploy.hardware import RealManHardwareAdapter
            # __init__ opens real hardware: call only the pure config validator.
            hardware = RealManHardwareAdapter._validate_config(read_json(paths["hardware_config"]))
            cameras = hardware["cameras"]
            if any(not isinstance(c.get("serial"), str) or not c["serial"].strip() for c in cameras):
                raise ValueError("两相机 serial 必须填写非空实机序列号")
            if config.get("camera_order", ["global", "wrist"]) != [c["name"] for c in cameras]:
                raise ValueError("runtime 与硬件的相机顺序不一致")
            for key, default in (("camera_width", 640), ("camera_height", 480)):
                if config.get(key, default) != hardware[key]:
                    raise ValueError("runtime 与硬件图像尺寸不一致: " + key)
            if normalizer is not None and hardware["action_space"] != normalizer.action_space:
                raise ValueError("hardware 与 normalization 的 action_space 不一致")
            if hardware.get("force_preprocessing_required"):
                from realman_force.force_calibration import validate_profile
                from realman_force.force_filter import validate_config, make_manifest
                from realman_force.contract import input_contract
                hbase = Path(paths["hardware_config"]).parent
                calibration = validate_profile(read_json(resolved_path(hardware["force_calibration_file"], hbase)))
                filter_config = validate_config(read_json(resolved_path(hardware["force_filter_file"], hbase)))
                contract = input_contract(calibration, make_manifest(filter_config, calibration))
                if normalizer is not None and contract != normalizer.force_preprocessing:
                    raise ValueError("力标定/滤波与训练 normalization 的预处理身份不一致")
            return dict(robot_ip=hardware["robot_ip"], robot_port=hardware["robot_port"],
                        feedback_port=hardware["feedback_port"], cameras=cameras,
                        action_space=hardware["action_space"], connected=False)
        record("hardware_config_only", hardware_check)

    if "sampling_profile" in config:
        def sampling_check():
            from realman_deploy.sampling import PROFILES, profile_steps
            profile = str(config["sampling_profile"])
            steps = profile_steps(profile)
            mode = config.get("execution_mode", "synchronous")
            if mode not in PROFILES[profile]["execution_modes"]:
                raise ValueError("此采样配置不支持所选执行模式")
            return dict(steps, execution_mode=mode)
        record("sampling", sampling_check)
    else:
        warn("sampling", "未设置 sampling_profile；旧配置按旧路径运行。新 joint8 使用界面选择 4416 后重新加载模型。")

    checkpoint_root = record("checkpoint_directory", lambda: require_dir(config["checkpoint_directory"]))
    if checkpoint_root:
        candidates = sorted(str(path) for path in Path(checkpoint_root).glob("*.pt"))
        if not candidates:
            checks.append(dict(name="trained_checkpoints", status="error", detail="checkpoint_directory 中没有 .pt 文件"))
        else:
            checks.append(dict(name="trained_checkpoints", status="ok", detail=candidates))
    if checkpoint is not None:
        def checkpoint_check():
            path = Path(require_file(checkpoint))
            if normalizer is not None and normalizer.force_preprocessing is not None:
                from realman_deploy.preprocessing import verify_binding
                verify_binding(path, paths["normalization_file"])
                return "checkpoint + normalization + force sidecar 的 SHA256 绑定通过（未加载模型）"
            return str(path)
        record("selected_checkpoint_binding", checkpoint_check)
    else:
        warn("checkpoint_binding", "未指定 --checkpoint：只检查权重文件存在；加载模型时才校验实际 checkpoint 结构和绑定。")

    versions = {}
    for name in ("torch", "torchvision", "numpy", "Pillow", "safetensors", "hydra-core", "omegaconf", "einops", "loguru", "attrs"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            checks.append(dict(name="package:" + name, status="error", detail="当前 Python 环境未安装"))
    report["packages"] = versions
    if hardware_libraries:
        for name in ("Robotic_Arm", "pyrealsense2"):
            def library_check(name=name):
                if importlib.util.find_spec(name) is None:
                    raise ImportError("未安装硬件库 " + name)
                return "模块可定位；未加载 SDK、未连接设备"
            record("hardware_library:" + name, library_check)
    warn("memory", "显存余量由 --check-cuda 报告；模型加载与 warmup 的实测峰值才决定是否容纳，不按卡名保证。")
    return report


def check_cuda(device, timeout=15.0):
    from realman_deploy.cuda_status import CudaStatusProbe
    probe = CudaStatusProbe(device=device, timeout=timeout)
    try:
        result = probe.status()
        while result.get("checking"):
            time.sleep(0.05)
            result = probe.status()
        return result
    finally:
        probe.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, help="Optional exact trained checkpoint; verifies force binding by streaming hashes")
    parser.add_argument("--check-cuda", action="store_true", help="Probe the configured GPU and BF16 kernel in an isolated subprocess")
    parser.add_argument("--cuda-timeout", type=float, default=15.0)
    parser.add_argument("--hardware-libraries", action="store_true", help="Check optional SDK installation; never connects devices")
    args = parser.parse_args(argv)
    if args.cuda_timeout <= 0:
        parser.error("--cuda-timeout must be positive")
    report = inspect_config(args.config, checkpoint=args.checkpoint, hardware_libraries=args.hardware_libraries)
    if args.check_cuda:
        result = check_cuda(report.get("requested_device", "cuda"), args.cuda_timeout)
        report["checks"].append(dict(name="cuda_bf16_kernel", status="ok" if result.get("available") else "error", detail=result))
    report["ok"] = all(item["status"] != "error" for item in report["checks"])
    report["hardware_connected"] = False
    report["model_loaded"] = False
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
