"""Explicit online timing experiments; immutable defaults, no device writes.

These profiles do not change model sampling, force-filter coefficients, device
alarms, physical limits, or the 30 Hz action grid. Server acceptance envelopes
must be checked separately before applying a profile.
"""
from types import MappingProxyType


_HARDWARE = MappingProxyType({
    'strict': MappingProxyType(dict(gripper_control_max_age_s=.5,
        gripper_feedback_max_age_s=1.5, feedback_max_age_s=.03,
        force_source_max_age_s=.03, camera_max_age_s=.12, camera_pair_skew_s=.05)),
    'relaxed': MappingProxyType(dict(gripper_control_max_age_s=1.5,
        gripper_feedback_max_age_s=2., feedback_max_age_s=.1,
        force_source_max_age_s=.1, camera_max_age_s=.25, camera_pair_skew_s=.15)),
})
_MOTION = MappingProxyType({
    'strict': MappingProxyType(dict(max_observation_age_s=.12, settle_window_s=.2,
        settle_tolerance_deg=.1, settle_velocity_deg_s=.5, max_inference_drift_deg=.5)),
    'relaxed': MappingProxyType(dict(max_observation_age_s=.3, settle_window_s=.1,
        settle_tolerance_deg=.3, settle_velocity_deg_s=1.5, max_inference_drift_deg=2.)),
})
_LABELS = MappingProxyType({'strict': '原始时序档', 'relaxed': '宽松实验档'})
_DESCRIPTIONS = MappingProxyType({
    'strict': '原始动作反馈、相机对齐与停稳阈值；作为成功率对照。',
    'relaxed': '允许较旧反馈和较大相机时间差，同时放宽停稳及推理期间漂移；实际报警和关节限位继续生效。',
})


def validate_profile(profile):
    if type(profile) is not str or profile not in _HARDWARE:
        raise ValueError('时序档位必须为 strict 或 relaxed')
    return profile


def hardware_settings(profile):
    return dict(_HARDWARE[validate_profile(profile)])


def motion_settings(profile):
    return dict(_MOTION[validate_profile(profile)])


def profile_status(profile='strict', *, busy=False, error=None):
    profile = validate_profile(profile)
    return dict(supported=True, profile=profile,
        profiles=[dict(id=key, label=_LABELS[key], description=_DESCRIPTIONS[key])
                  for key in _HARDWARE],
        limits=dict(hardware_settings(profile), **motion_settings(profile)),
        busy=bool(busy), error=error)
