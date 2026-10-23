"""Deployment choices; sampling schedules are independent of checkpoint identity."""
from __future__ import annotations

PROFILES = {
    "161616": {"id": "161616", "label": "标准 · 16 / 16 / 16",
               "video_steps": 16, "force_steps": 16, "action_steps": 16,
               "execution_modes": ["synchronous"]},
    "4416": {"id": "4416", "label": "平衡 · 4 / 4 / 16",
             "video_steps": 4, "force_steps": 4, "action_steps": 16,
             "execution_modes": ["synchronous", "async_relaxed"]},
}


def profile_steps(profile: str) -> dict:
    if profile not in PROFILES:
        raise ValueError("采样配置必须为 161616 或 4416")
    return {k: PROFILES[profile][k] for k in ("video_steps", "force_steps", "action_steps")}


def settings(config: dict) -> dict:
    return {"sampling_profile": config.get("sampling_profile", "161616"),
            "execution_mode": config.get("execution_mode", "synchronous"),
            "profiles": list(PROFILES.values())}


def updated_config(config: dict, profile: str, mode: str) -> dict:
    profile_steps(profile)
    if mode not in PROFILES[profile]["execution_modes"]:
        raise ValueError("异步模式仅支持 4 / 4 / 16；标准配置请选择同步推理")
    # Do not change model_kwargs/manifest.num_inference_steps. Those identify
    # the trained model; inference schedules are a separate runtime choice.
    options = dict(config.get("inference_optimization", {}))
    options.setdefault("mode", "auto")
    return {**config, "sampling_profile": profile, "execution_mode": mode,
            "inference_backend": "joint_multirate", "inference_optimization": options}
