"""Passive, contact-free residual wrench calibration, outside the motion loop.

The controller already returns zero_force. Fit its *remaining* bias and gravity,
not a second copy of the physical payload. All torques retain the controller's
wrench origin; this module only rotates axes and never translates a wrench.
"""
from collections import deque
from datetime import datetime, timezone
import copy
import hashlib
import json
from pathlib import Path
import time

import numpy as np

SCHEMA = 'realman_residual_wrench_v1'
SOURCE = 'RealMan zero_force coordinate=1, World=base, Arm_Tip identity'
MIN_FRAMES = 32
WINDOW_S = 1.5
MAX_FORCE_RMS = 1.5  # N, vector RMS at contact-free validation poses
MAX_TORQUE_RMS = .12  # Nm


def rotation(pose):
    p = np.asarray(pose, dtype=float)
    if p.shape != (6,) or not np.isfinite(p).all():
        raise ValueError('六轴力补偿需要同包的有效末端姿态')
    x, y, z = p[3:]
    cx, cy, cz = np.cos([x, y, z]); sx, sy, sz = np.sin([x, y, z])
    # RealMan XYZ Euler: Rz(yaw) Ry(pitch) Rx(roll), radians.
    return np.array([[cz*cy, cz*sy*sx-sz*cx, cz*sy*cx+sz*sx],
                     [sz*cy, sz*sy*sx+cz*cx, sz*sy*cx-cz*sx],
                     [-sy, cy*sx, cy*cx]])


def vector(value, n):
    a = np.asarray(value, dtype=float)
    if a.shape != (n,) or not np.isfinite(a).all():
        raise ValueError('无效六轴力标定数值')
    return a


def skew(v):
    x, y, z = v
    return np.array([[0., -z, y], [z, 0., -x], [-y, x, 0.]])


def gravity_axis(pose):
    return rotation(pose).T @ np.array([0., 0., -1.])


def design(pose):
    u = gravity_axis(pose)
    # p = [residual weight N, residual first moment Nm (3), bias F (3), bias T (3)]
    a = np.zeros((6, 10))
    a[:3, 0] = u
    a[:3, 4:7] = np.eye(3)
    a[3:, 1:4] = -skew(u)
    a[3:, 7:10] = np.eye(3)
    return a


def tool_wrench(wrench, pose):
    r = rotation(pose)
    return (vector(wrench, 6).reshape(2, 3) @ r).reshape(6)


def base_wrench(wrench, pose):
    return (vector(wrench, 6).reshape(2, 3) @ rotation(pose).T).reshape(6)


def profile_id(profile):
    data = {k: v for k, v in profile.items() if k != 'id'}
    return hashlib.sha256(json.dumps(data, sort_keys=True, allow_nan=False,
                                     separators=(',', ':')).encode()).hexdigest()


def validate_profile(profile):
    if (profile.get('schema') != SCHEMA or profile.get('source') != SOURCE
            or profile.get('id') != profile_id(profile)):
        raise ValueError('力标定文件格式、来源或校验值无效')
    vector(profile['parameters'], 10)
    axes = np.asarray(profile['gravity_axes'], dtype=float)
    if (axes.ndim != 2 or axes.shape[1] != 3 or len(axes) < 6 or not np.isfinite(axes).all()
            or not np.allclose(np.linalg.norm(axes, axis=1), 1, atol=1e-6)):
        raise ValueError('力标定姿态范围无效')
    q = profile['quality']
    if (q['rank'] != 10 or not np.isfinite(q['condition']) or q['condition'] > 100
            or len(q['validation_errors']) < 2):
        raise ValueError('力标定缺少可辨识性或独立验证')
    for e in q['validation_errors']:
        errors = vector(e, 2)
        if np.any(errors < 0) or errors[0] > MAX_FORCE_RMS or errors[1] > MAX_TORQUE_RMS:
            raise ValueError('空载标定验证误差超限')
    return profile


def compensation(profile, pose):
    validate_profile(profile)
    u = gravity_axis(pose)
    if np.max(np.asarray(profile['gravity_axes']) @ u) < np.cos(np.deg2rad(60)):
        raise ValueError('末端姿态超出力标定覆盖范围，请补充空载多姿态标定')
    return base_wrench(design(pose) @ np.asarray(profile['parameters']), pose)


def corrected(profile, wrench, pose):
    return vector(wrench, 6)-compensation(profile, pose)


def stationary_point(samples):
    """A window is usable only after the human explicitly declares no contact."""
    if len(samples) < MIN_FRAMES:
        raise ValueError('请保持空载静止约 2 秒，等待至少 32 个力数据帧')
    times = np.array([s['force_time'] for s in samples], dtype=float)
    if (not np.isfinite(times).all() or times[-1]-times[0] < 1.
            or np.any(np.diff(times) <= 0) or np.max(np.diff(times)) > .12):
        raise ValueError('标定窗口的力时间戳不连续')
    if any(s['force_coordinate'] != 1 or s['robot_time'] != s['force_time']
           or not 0 <= s['sample_time']-s['force_time'] <= .03 for s in samples):
        raise ValueError('标定窗口力/姿态未同步或坐标系错误')
    q = np.array([vector(s['joints_deg'], 7) for s in samples])
    poses = np.array([vector(s['pose'], 6) for s in samples])
    grips = np.array([s['gripper_open'] for s in samples], dtype=float)
    targets = np.array([s['gripper_target_open'] for s in samples], dtype=float)
    if (np.max(np.ptp(q, axis=0)) > .25 or np.max(np.ptp(poses[:, :3], axis=0)) > .002
            or not np.isfinite(grips).all() or not np.isfinite(targets).all()
            or np.ptp(grips) > .02 or np.ptp(targets) > .02):
        raise ValueError('机械臂或夹爪仍在运动，请松开 B 并保持静止')
    r = rotation(poses[-1])
    angles = [np.arccos(np.clip((np.trace(r.T @ rotation(p))-1)/2, -1, 1)) for p in poses]
    if max(angles) > np.deg2rad(.5):
        raise ValueError('末端姿态未静止')
    values = np.array([tool_wrench(s['force'], s['pose']) for s in samples])
    center = np.median(values, axis=0)
    # Use all samples for a noise gate; a median must not hide contact transients.
    rms = np.sqrt(np.mean((values-center)**2, axis=0))
    if np.linalg.norm(rms[:3]) > 1. or np.linalg.norm(rms[3:]) > .08:
        raise ValueError('空载力波动过大，请检查接触、线缆拉扯并等待稳定')
    return {'pose': poses[-1].tolist(), 'tool_wrench': center.tolist(),
            'noise_rms': rms.tolist(), 'frames': len(samples),
            'first_time': float(times[0]), 'last_time': float(times[-1]),
            'contact_free_confirmed': True,
            'gripper_open': float(np.median(grips))}


def fit_profile(points, context=None):
    if len(points) < 8 or any(p.get('contact_free_confirmed') is not True for p in points):
        raise ValueError('需要至少 6 个空载训练姿态和 2 个独立验证姿态')
    train, validation = points[:-2], points[-2:]
    a = np.vstack([design(p['pose']) for p in train])
    b = np.concatenate([vector(p['tool_wrench'], 6) for p in train])
    params, _, rank, singular = np.linalg.lstsq(a, b, rcond=None)
    condition = float(singular[0]/singular[-1]) if singular[-1] > 0 else float('inf')
    if rank != 10 or condition > 100:
        raise ValueError('姿态变化不足，无法区分偏置与重力；请改变手腕倾斜方向后重新采点')
    errors = []
    for p in points:
        residual = vector(p['tool_wrench'], 6)-design(p['pose']) @ params
        errors.append([float(np.linalg.norm(residual[:3])), float(np.linalg.norm(residual[3:]))])
    if any(e[0] > MAX_FORCE_RMS or e[1] > MAX_TORQUE_RMS for e in errors):
        raise ValueError('空载拟合或独立验证误差超限；检查接触、线缆和装配后重新标定')
    profile = {'schema': SCHEMA, 'source': SOURCE, 'created_utc': datetime.now(timezone.utc).isoformat(),
               'parameters': params.tolist(), 'gravity_axes': [gravity_axis(p['pose']).tolist() for p in train],
               'context': copy.deepcopy(context or {}), 'points': copy.deepcopy(points),
               'quality': {'rank': int(rank), 'condition': condition,
                           'training_errors': errors[:-2], 'validation_errors': errors[-2:]},
               'notes': 'Residual parameters after controller compensation; not physical payload mass/COM. Static gravity and bias only; no inertia/contact removal or adaptive baseline.'}
    profile['id'] = profile_id(profile)
    return validate_profile(profile)


def retare_profile(profile, point):
    validate_profile(profile)
    if point.get('contact_free_confirmed') is not True:
        raise ValueError('漂移校零必须明确确认空载无接触')
    compensation(profile, point['pose'])  # Enforce the calibrated orientation envelope.
    delta = vector(point['tool_wrench'], 6)-design(point['pose']) @ profile['parameters']
    if np.linalg.norm(delta[:3]) > 3. or np.linalg.norm(delta[3:]) > .25:
        raise ValueError('偏移超过小幅零漂范围；检查负载/接触/控制器清零并重新多姿态标定')
    result = copy.deepcopy(profile)
    p = np.asarray(result['parameters'])
    p[4:7] += delta[:3]; p[7:10] += delta[3:]
    result.update(parameters=p.tolist(), parent_id=profile['id'],
                  created_utc=datetime.now(timezone.utc).isoformat())
    result['tare'] = {'point': point, 'delta_tool': delta.tolist(),
                      'scope': 'Constant tool-axis bias only; gravity parameters frozen; contact-free operator declaration'}
    result['id'] = profile_id(result)
    return validate_profile(result)


class ForceCalibration:
    """Owned by RecordingClient's lock. Bounded 2 s history; no robot API calls."""
    def __init__(self, path, context=None, required=False):
        self.path = Path(path)
        self.context = context or {}
        self.required = required
        self.samples = deque(maxlen=64)
        self.points = []
        self.profile = None
        self.verified = False
        self.loaded_from_disk = False
        self.error = None
        self.load_error = None
        self.latest = None
        if self.path.is_file():
            try:
                self.profile = validate_profile(json.loads(self.path.read_text()))
                if self.profile['context'] != self.context:
                    raise ValueError('标定机器人/工具配置与当前配置不一致，请重新标定')
                # A successful fit includes independent validation and remains
                # active across restarts. Loading never estimates a new zero.
                self.verified = self.loaded_from_disk = True
            except Exception as exc:
                self.profile = None
                self.error = self.load_error = str(exc)
        elif required:
            self.error = self.load_error = '缺少持久力标定文件，请恢复 active.json 或重新完成空载多姿态标定'

    def ingest(self, sample):
        self.latest = None
        if sample is None or 'force' not in sample:
            self.samples.clear()
            return
        try:
            from force_recording import pack_force
            pack_force(sample)
            rotation(sample['pose'])
        except (KeyError, TypeError, ValueError) as exc:
            self.samples.clear()
            self.error = '无效力标定反馈：'+str(exc)
            return
        if self.samples and sample['force_time'] <= self.samples[-1]['force_time']:
            self.samples.clear()
        self.samples.append(copy.deepcopy(sample))
        self.error = self.load_error
        if self.profile:
            try:
                f = corrected(self.profile, sample['force'], sample['pose'])
                self.latest = f.tolist()
                self.error = None
            except ValueError as exc:
                self.error = str(exc)

    def point(self, now):
        if not self.samples or not 0 <= now-self.samples[-1]['force_time'] < .1:
            raise ValueError('没有新鲜六轴力反馈，请先准备遥操作并保持 B 松开')
        end = self.samples[-1]['force_time']
        return stationary_point([s for s in self.samples if s['force_time'] >= end-WINDOW_S])

    def operation(self, op, now, contact_free=False):
        if op == 'reset':
            self.points.clear()
            return
        if contact_free is not True:
            raise ValueError('请先确认夹爪空载、末端悬空且已静止约 2 秒')
        if op == 'capture':
            if len(self.points) >= 12:
                raise ValueError('最多 12 个标定点，请拟合或清空本轮采点')
            point = self.point(now)
            u = gravity_axis(point['pose'])
            if any(np.dot(u, gravity_axis(p['pose'])) > np.cos(np.deg2rad(8)) for p in self.points):
                raise ValueError('与已有采点倾斜方向太接近，请改变手腕倾斜姿态')
            self.points.append(point)
        elif op == 'fit':
            if self.points and 'last_time' in self.points[-1] and not 0 <= now-self.points[-1]['last_time'] < 60:
                raise ValueError('最后的验证点已过期，请重新采集空载姿态')
            self._save(fit_profile(self.points, self.context))
        elif op in ('verify', 'tare'):
            if self.profile is None:
                raise ValueError('请先完成空载多姿态标定')
            point = self.point(now)
            if op == 'tare':
                self._save(retare_profile(self.profile, point))
            else:
                predicted = compensation(self.profile, point['pose'])
                residual = base_wrench(point['tool_wrench'], point['pose'])-predicted
                if np.linalg.norm(residual[:3]) > MAX_FORCE_RMS or np.linalg.norm(residual[3:]) > MAX_TORQUE_RMS:
                    self.verified = False
                    raise ValueError('空载复核不通过；小幅零漂可校零，改变装配/负载需重新多姿态标定')
                self.verified, self.error = True, None
        else:
            raise ValueError('未知力标定操作')

    def _save(self, profile):
        validate_profile(profile)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        text = json.dumps(profile, ensure_ascii=False, indent=2, allow_nan=False)+'\n'
        # Keep immutable versions for recovery and provenance.
        archive = self.path.parent/(profile['id']+'.json')
        if not archive.exists():
            with archive.open('x') as f: f.write(text)
        temporary = self.path.with_suffix('.tmp')
        temporary.write_text(text); temporary.replace(self.path)
        self.profile, self.verified, self.error = profile, True, None
        self.loaded_from_disk = False
        self.load_error = None
        self.samples.clear()  # Never reuse pre-tare/pre-fit samples for a second calibration.
        self.points.clear()

    def status(self):
        fresh = self.samples and 0 <= time.monotonic()-self.samples[-1]['force_time'] < .1
        return {'calibrated': self.profile is not None, 'verified': self.verified,
                'persistent': True, 'required': self.required,
                'loaded_from_disk': self.loaded_from_disk,
                'id': self.profile['id'] if self.profile else None, 'points': len(self.points),
                'needed_points': 8, 'corrected': self.latest if fresh else None, 'error': self.error,
                'quality': self.profile['quality'] if self.profile else None}
