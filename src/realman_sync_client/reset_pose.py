"""Persisted measured reset target, independent of model action output."""
import json
import math
from pathlib import Path


def load_reset_pose(path):
    pose = json.loads(Path(path).read_text(encoding='utf-8-sig'))
    if not isinstance(pose, dict) or pose.get('schema') != 'realman-reset-pose-v1':
        raise ValueError('复位位置文件格式不正确')
    for key in ('joint_radians', 'joint_degrees'):
        values = pose.get(key)
        if (not isinstance(values, list) or len(values) != 7 or any(
                isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v)
                for v in values)):
            raise ValueError('复位位置必须包含七个有限关节角度')
    if any(abs(math.degrees(r)-d) > 1e-5 for r,d in zip(pose['joint_radians'],pose['joint_degrees'])):
        raise ValueError('复位位置角度单位不一致')
    if (type(pose.get('speed_percent')) is not int or not 1 <= pose['speed_percent'] <= 10
            or pose.get('gripper_action') != 'unchanged' or pose.get('model') != 'RM_75'
            or not isinstance(pose.get('recorded_at'), str) or not pose['recorded_at']):
        raise ValueError('复位位置速度、机械臂或记录信息不正确')
    return pose
