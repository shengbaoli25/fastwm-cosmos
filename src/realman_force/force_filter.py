"""Causal wrench-only filtering, outside the robot's motion process.

UDP history is bounded by the producer. Compensate every input with its own
pose, then filter before selecting the 30 Hz recording row. No time shifting.
"""
import copy
import hashlib
import json

import numpy as np

from .force_calibration import compensation, validate_profile

# Engineering priors, not measured ADC noise. Q/R at a nominal 5 ms step is
# 1/16 on each axis. Keep them fixed during contact; never auto-fit to a task.
DEFAULT_CONFIG = {
    'type': 'kalman', 'model': 'random_walk',
    'measurement_variance': [.25, .25, .25, .0004, .0004, .0004],
    'process_variance_per_s': [3.125, 3.125, 3.125, .005, .005, .005],
    'sample_rate_hz': 200., 'warmup_s': .5, 'max_gap_s': .02,
}
LEGACY_BUTTERWORTH_CONFIG = {'type': 'butterworth', 'order': 2, 'cutoff_hz': 8.,
                           'sample_rate_hz': 200., 'warmup_s': .5, 'max_gap_s': .02}


class FilterWarmup(ValueError):
    pass


def validate_config(config):
    if not isinstance(config, dict):
        raise ValueError('力滤波配置必须为对象')
    kind = config.get('type')
    numeric = ['sample_rate_hz', 'warmup_s', 'max_gap_s']
    if kind == 'kalman':
        if set(config) != set(DEFAULT_CONFIG) or config['model'] != 'random_walk':
            raise ValueError('卡尔曼力滤波必须为六轴独立 random_walk 模型')
        for key in ('measurement_variance', 'process_variance_per_s'):
            values = config[key]
            if (not isinstance(values, list) or len(values) != 6
                    or any(type(v) not in (int, float) or not np.isfinite(v) or v <= 0 for v in values)):
                raise ValueError('卡尔曼噪声方差必须为六个正有限数值')
        # Reject numeric extremes that can overflow/underflow covariance updates.
        if any(not 1e-12 <= v <= 1e8 for key in
               ('measurement_variance', 'process_variance_per_s') for v in config[key]):
            raise ValueError('卡尔曼噪声方差超出数值范围')
    elif kind == 'butterworth':
        if (set(config) != set(LEGACY_BUTTERWORTH_CONFIG)
                or type(config['order']) is not int or config['order'] != 2):
            raise ValueError('历史力滤波配置必须为二阶 Butterworth')
        numeric.append('cutoff_hz')
    else:
        raise ValueError('未知力滤波类型')
    for k in numeric:
        if type(config[k]) not in (int, float) or not np.isfinite(config[k]):
            raise ValueError('力滤波配置必须为有限数值')
    if (config['sample_rate_hz'] != 200
            or (kind == 'butterworth' and not 1 <= config['cutoff_hz'] <= 10)
            or not .5 <= config['warmup_s'] <= 2 or not .01 <= config['max_gap_s'] <= .03):
        raise ValueError('力滤波频率/预热/断流配置超限')
    return copy.deepcopy(config)


def digest(manifest):
    return hashlib.sha256(json.dumps({k: v for k, v in manifest.items() if k != 'id'},
        sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def make_manifest(config, calibration, *, schema=None):
    config = validate_config(config)
    validate_profile(calibration)
    if config['type'] == 'kalman':
        schema = schema or 'realman_wrench_filter_v3'
        if schema not in ('realman_wrench_filter_v2', 'realman_wrench_filter_v3'):
            raise ValueError('未知卡尔曼滤波版本')
        result = {
            'schema': schema, 'config': config,
            'calibration_id': calibration['id'],
            'input': 'Controller World/base zero_force minus per-input-pose static calibration',
            'output': 'Causal six independent scalar Kalman wrench estimates; original force/action row timestamps retained',
            'state_model': 'x[k]=x[k-1]+w; z[k]=x[k]+v; A=H=1; no axis coupling',
            'prediction': 'P_pred=P+process_variance_per_s*actual_dt_seconds',
            'update': 'K=P_pred/(P_pred+R); x=x+K*(z-x); P=(1-K)^2*P_pred+K^2*R (Joseph form)',
            'variance_units': 'R and P: [N^2,N^2,N^2,(Nm)^2,(Nm)^2,(Nm)^2]; process_variance_per_s: same units per second',
            'noise_parameters': 'Fixed engineering priors, not measured sensor noise or ForceVLA settings; never adapted from task/contact data',
            'initialization': 'First corrected input x=z, P=R; then >=warmup_s and >=200*warmup_s inputs before recording',
            'reset': 'Input fault, nonmonotonic timestamp, gap > max_gap_s, changed calibration; never reset at episode start',
            'sampling': 'Nominal 200 Hz host UDP arrival; Q scales with actual positive timestamp interval; rolling median interval must stay within 4..6 ms',
            'limitations': 'Causal estimation has phase delay and is not a complete anti-alias filter for 30 Hz rows; no inertia removal or automatic baseline. Posterior covariance is model uncertainty, not calibrated sensor accuracy.',
        }
        if schema == 'realman_wrench_filter_v3':
            result['sampling'] = 'Nominal 200 Hz host UDP arrival; Q scales with actual positive timestamp interval; measured arrival rate is diagnostic, not a fixed-coefficient gate'
            result['readiness'] = 'Require warmup_s elapsed and >=sample_rate_hz*warmup_s inputs after reset; retain max_gap_s, monotonicity, freshness and source validity checks'
        result['id'] = digest(result)
        return result
    # Keep v1 byte-for-byte compatible with manifests in existing recordings.
    if schema not in (None, 'realman_wrench_filter_v1'):
        raise ValueError('未知 Butterworth 滤波版本')
    from scipy.signal import butter
    result = {'schema': 'realman_wrench_filter_v1', 'config': config,
        'calibration_id': calibration['id'],
        'sos': butter(2, config['cutoff_hz'], fs=200., output='sos').tolist(),
        'input': 'Controller World/base zero_force minus per-input-pose static calibration',
        'output': 'Causal six-axis SOS low-pass, sampled at force/action row timestamp; no timestamp shift',
        'initialization': 'Steady state at first input, then >=warmup_s and >=200*warmup_s inputs before recording',
        'reset': 'Input fault, nonmonotonic timestamp, gap > max_gap_s, changed calibration; never reset at episode start',
        'sampling': 'Nominal 200 Hz host UDP arrival; fixed coefficients, actual timestamps retained; rolling median interval must stay within 4..6 ms',
        'limitations': 'Frequency-dependent phase delay; no removal of real contact/inertia or automatic baseline. Source raw and corrected unfiltered row remain unchanged.'}
    result['id'] = digest(result)
    return result


def validate_manifest(manifest, calibration):
    if manifest != make_manifest(manifest['config'], calibration, schema=manifest['schema']):
        raise ValueError('力滤波版本或标定版本不一致')
    return manifest


class ForceFilter:
    def __init__(self, config):
        from collections import deque
        self.config = validate_config(config)
        self.is_kalman = self.config['type'] == 'kalman'
        if self.is_kalman:
            self.measurement_variance = np.asarray(self.config['measurement_variance'])
            self.process_variance_per_s = np.asarray(self.config['process_variance_per_s'])
        else:
            from scipy.signal import butter, sosfilt_zi
            self.sos = butter(2, self.config['cutoff_hz'], fs=200., output='sos')
            self.initial = sosfilt_zi(self.sos)
        self.intervals = deque(maxlen=100)
        self.input_history = deque(maxlen=128)
        self.manifest = None
        self.epoch = 0
        self.reset()

    def reset(self, reason='等待高频力反馈及滤波预热'):
        self.epoch += 1
        self.zi = None
        self.covariance = None
        self.stamp = self.first = None
        self.count = 0
        self.value = None
        self.ready = False
        self.error = reason
        self.reset_reason = reason
        self.intervals.clear()
        self.input_history.clear()

    def apply(self, sample, calibration):
        if calibration is None:
            self.reset()
            raise ValueError('力滤波需要有效的空载标定参数')
        if self.manifest is None or self.manifest['calibration_id'] != calibration['id']:
            self.manifest = make_manifest(self.config, calibration)
            self.reset()
        history = sample.get('force_history')
        if not history or len(history) > 256:
            raise ValueError('缺少有界的 200 Hz 六轴力历史')
        stamps = np.asarray([p['time'] for p in history], dtype=float)
        if (not np.isfinite(stamps).all() or np.any(np.diff(stamps) <= 0)
                or stamps[-1] != sample['force_time']):
            raise ValueError('高频力历史乱序或未对齐当前 force 时间戳')
        if (not np.array_equal(history[-1]['force'], sample['force'])
                or not np.array_equal(history[-1]['pose'], sample['pose'])):
            raise ValueError('滤波输入与当前同包力/姿态不一致')
        # Older saved streams have only controller force. New producers keep
        # sensor raw as a diagnostic channel, never an input to this filter.
        raw_present = ['force_raw' in p for p in history]
        if any(raw_present):
            if not all(raw_present):
                raise ValueError('高频原始力历史不完整')
            raw = np.asarray([p['force_raw'] for p in history], dtype=float)
            if (raw.shape != (len(history), 6) or not np.isfinite(raw).all()
                    or any(type(p.get('force_coordinate')) is not int or p['force_coordinate'] != 1
                           for p in history)
                    or not np.array_equal(history[-1]['force_raw'], sample.get('force_raw'))):
                raise ValueError('高频原始力无效或未对齐当前数据包')
        # The source resends a small overlap to tolerate one delayed telemetry
        # tick. Only strictly newer inputs enter filter state.
        if self.stamp is not None and stamps[-1] <= self.stamp:
            raise ValueError('高频力反馈重复或时钟倒退')
        for p in history:
            t = p['time']
            if self.stamp is not None and t <= self.stamp:
                continue
            v = np.asarray(p['force'], dtype=float)
            if v.shape != (6,) or not np.isfinite(v).all():
                raise ValueError('高频力数据无效')
            x = v-compensation(calibration, p['pose'])
            if self.stamp is not None and t-self.stamp > self.config['max_gap_s']:
                self.reset(f'高频力反馈间隔 {(t-self.stamp)*1000:.1f} ms，重新预热')
            if self.value is None:
                self.first = t
                if self.is_kalman:
                    self.value = x.copy()
                    self.covariance = self.measurement_variance.copy()
                else:
                    self.zi = self.initial[:, :, None]*x[None, None, :]
            else:
                dt = t-self.stamp
                self.intervals.append(dt)
                if self.is_kalman:
                    predicted = self.covariance+self.process_variance_per_s*dt
                    gain = predicted/(predicted+self.measurement_variance)
                    self.value += gain*(x-self.value)
                    self.covariance = (1-gain)**2*predicted+gain**2*self.measurement_variance
            if not self.is_kalman:
                from scipy.signal import sosfilt
                y, self.zi = sosfilt(self.sos, x[None, :], axis=0, zi=self.zi)
                self.value = y[0]
            self.stamp = t
            self.count += 1
            # The arm process sends only new packets plus one overlap. Retain
            # bounded prehistory here for replayable recording-start snapshots.
            self.input_history.append(copy.deepcopy(p))
        nominal = self.config['sample_rate_hz']
        rate_ok = self.is_kalman or (bool(self.intervals) and .004 <= np.median(self.intervals) <= .006)
        self.ready = bool(self.count >= int(nominal*self.config['warmup_s'])
                      and self.stamp-self.first >= self.config['warmup_s'] and rate_ok)
        self.error = None if self.ready else ('高频力速率不是约 200 Hz' if not rate_ok else '力滤波预热中（约 0.5 秒）')
        if not self.ready:
            raise FilterWarmup(self.error)
        return dict(sample, force_history=list(self.input_history),
                    force_filtered=self.value.tolist(), force_filter_id=self.manifest['id'],
                    force_filter_time=self.stamp, force_filter_epoch=self.epoch,
                    force_filter_input_count=self.count, force_filter_state=(
                        {'estimate': self.value.tolist(), 'covariance': self.covariance.tolist()}
                        if self.is_kalman else self.zi.tolist()))

    def status(self):
        mean_interval = float(np.mean(self.intervals)) if self.intervals else None
        return {'enabled': True, 'ready': self.ready, 'error': self.error, 'config': self.config,
                'id': self.manifest['id'] if self.manifest else None,
                'epoch': self.epoch, 'input_count': self.count,
                'filtered': self.value.tolist() if self.ready else None,
                'input_rate_hz': round(1/mean_interval, 1) if mean_interval else None,
                'median_interval_ms': round(float(np.median(self.intervals))*1000, 3) if self.intervals else None,
                'max_interval_ms': round(max(self.intervals)*1000, 3) if self.intervals else None,
                'last_reset_reason': self.reset_reason}
