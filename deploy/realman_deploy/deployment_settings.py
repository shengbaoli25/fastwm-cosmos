"""Validated deployment choices shared by the dashboard and launch script."""
from __future__ import annotations

import math
import re
from urllib.parse import urlsplit


REMOTE_DEFAULTS = {
    "url": "", "token_env": "REALMAN_REMOTE_TOKEN", "checkpoint_sha256": "",
    "timeout_s": 5.0, "max_latency_s": 1.0, "image_encoding": "zlib_rgb",
}


def _number(value, name, low, high):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not low <= value <= high:
        raise ValueError(f"{name} 必须为 {low}..{high} 的有限数值")
    return float(value)


def deployment_settings(config: dict) -> dict:
    device = config.get("inference_device", "local")
    if device not in {"local", "qizhi_h200"}:
        raise ValueError("推理设备必须为 local 或 qizhi_h200")
    if device == "qizhi_h200":
        from .sampling import updated_config
        if "sampling_profile" not in config or "execution_mode" not in config:
            raise ValueError("远端配置必须明确 sampling_profile（4416 或 161616）和 execution_mode")
        updated_config(config, config["sampling_profile"], config["execution_mode"])
    supplied = config.get("remote_inference", {})
    if not isinstance(supplied, dict):
        raise ValueError("remote_inference 必须为对象")
    if set(supplied) - REMOTE_DEFAULTS.keys():
        raise ValueError("remote_inference 含未知设置；认证请使用 token_env")
    remote = {**REMOTE_DEFAULTS, **supplied}
    if not isinstance(remote["url"], str):
        raise ValueError("远端地址必须为字符串")
    remote["url"] = remote["url"].strip().rstrip("/")
    if remote["url"]:
        parsed = urlsplit(remote["url"])
        if (parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.username
                or parsed.password or parsed.query or parsed.fragment):
            raise ValueError("远端地址需为 http(s) URL，可含转发前缀，不可含账号、查询参数或片段")
        _ = parsed.port
    if device == "qizhi_h200" and not remote["url"]:
        raise ValueError("请填写启智端口转发地址")
    if not isinstance(remote["token_env"], str) or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", remote["token_env"]):
        raise ValueError("token_env 必须为有效环境变量名")
    sha = remote["checkpoint_sha256"]
    if not isinstance(sha, str) or (sha and not re.fullmatch(r"[0-9a-f]{64}", sha)):
        raise ValueError("权重 SHA256 必须为 64 位小写十六进制")
    remote["timeout_s"] = _number(remote["timeout_s"], "timeout_s", .1, 120)
    remote["max_latency_s"] = _number(remote["max_latency_s"], "max_latency_s", .01, 10)
    if remote["image_encoding"] not in {"raw_rgb", "zlib_rgb"}:
        raise ValueError("image_encoding 必须为 raw_rgb 或 zlib_rgb")
    alpha = _number(config.get("action_smoothing_alpha", 1.0), "动作平滑系数", .01, 1)
    age = config.get("max_result_age_s")
    if age is None and device == "qizhi_h200":
        age = .6
    if age is not None:
        age = _number(age, "max_result_age_s", .05, 10)
    return dict(inference_device=device, remote_inference=remote,
                action_smoothing_alpha=alpha, max_result_age_s=age)


def update_deployment_settings(config: dict, body: dict) -> dict:
    keys = {"inference_device", "remote_inference", "action_smoothing_alpha", "max_result_age_s"}
    merged = {**config, **{key: body[key] for key in keys if key in body}}
    if "remote_inference" in body:
        if not isinstance(body["remote_inference"], dict):
            raise ValueError("remote_inference 必须为对象")
        merged["remote_inference"] = {**config.get("remote_inference", {}), **body["remote_inference"]}
    return {**merged, **deployment_settings(merged)}
