"""Web telemetry around the existing hardware/control contract.

Construction starts cameras independently and attempts robot monitoring. The
only automatic controller change is the reversible UDP subscription. Gripper
setup verifies existing power/bus/parameters and never repairs or enables them.
Motion methods remain explicit and retain the original adapter's checks.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math
import threading
import time
from typing import Any

from . import clock
from .camera import RealSenseCamera
from .client import validate_metadata
from .gripper import Ctag2F120Worker, FeedbackUnavailable, _signed32, verify_arm_state
from .feedback_metrics import FeedbackMetrics, HostSchedulerProbe
from .hardware import RealManHardwareAdapter, _RobotFeedback, _finite
from .wire import PROTOCOL, temporal_tolerances


@dataclass(frozen=True)
class _ResetToken:
    hardware: object
    sdk: object
    feedback: object
    control_generation: int
    arm_fault_generation: int


@dataclass(frozen=True)
class _ManualControlToken:
    worker: object
    cancel_generation: int


@dataclass(frozen=True)
class _FeedbackRecoveryToken:
    hardware: object
    gripper: object
    feedback: object
    control_generation: int
    gripper_hard_generation: int
    arm_hard_generation: int
    gripper_connection_generation: int = 0


class _Rate:
    """Accepted source timestamps, never HTTP polls or configured sample rates."""

    def __init__(self):
        self.lock = threading.Lock()
        self.times = deque(maxlen=2048)
        self.samples = 0

    def add(self, stamp):
        with self.lock:
            if not math.isfinite(stamp) or stamp <= 0 or (self.times and stamp <= self.times[-1]):
                return
            self.times.append(stamp)
            self.samples += 1

    def snapshot(self, now, max_age):
        with self.lock:
            recent = [value for value in self.times if now - 2.0 <= value <= now]
            last = self.times[-1] if self.times else None
            samples = self.samples
        age = now - last if last is not None else None
        hz = ((len(recent) - 1) / (recent[-1] - recent[0])
              if len(recent) > 1 and age is not None and 0 <= age <= max_age else 0.0)
        return dict(hz=round(hz, 2), samples=samples,
                    age_ms=round(age * 1000, 2) if age is not None else None)


class _MonitoredCamera(RealSenseCamera):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.rate = _Rate()

    def _accept(self, message):
        super()._accept(message)
        if not message[2]:
            self.rate.add(float(message[0]['received']))

    def diagnostics(self):
        result = self.status()
        rate = self.rate.snapshot(clock.now(), self.max_age_s)
        result.update(label=self.spec.get('label', self.spec['name']), fps=rate['hz'],
                      samples=rate['samples'], width=self.width, height=self.height)
        return result

    def frame(self):
        with self.lock:
            if self.rgb is None:
                return None
            return self.rgb, self.width, self.height, self.meta.get('frame_number')


class _MonitoredFeedback(_RobotFeedback):
    def __init__(self, sdk, config):
        super().__init__(sdk, config)
        self.telemetry_lock = threading.Lock()
        self.state_rate, self.force_rate = _Rate(), _Rate()
        self.state_values = self.force_values = None
        self.state_diagnostic_error = self.force_diagnostic_error = '等待真实 UDP 反馈'
        self.hard_fault_generation = 0
        self.arm_fault_generation = 0
        self.state_received = None
        self.feedback_metrics = FeedbackMetrics(max_gap_s=float((config.get('_force_filter') or {}).get('max_gap_s', .02)))
        self.scheduler_probe = HostSchedulerProbe(max_gap_s=float((config.get('_force_filter') or {}).get('max_gap_s', .02)))

    def start(self):
        self.scheduler_probe.start()
        try:
            return super().start()
        except Exception:
            self.scheduler_probe.close()
            raise

    def restore(self):
        try:
            return super().restore()
        finally:
            self.scheduler_probe.close()

    def ingest(self, state, now):
        started = clock.now()
        try:
            return self._ingest(state, now)
        finally:
            # Telemetry never supplies timestamps or data to the filter.
            try:
                self.feedback_metrics.record(now, max(0., clock.now()-started)*1000)
            except ValueError:
                pass

    def _ingest(self, state, now):
        # Keep state diagnostics useful even when the independent force field is
        # invalid. The inherited model loader still rejects the entire sample.
        try:
            arm_ip = state.arm_ip.decode() if isinstance(state.arm_ip, bytes) else str(state.arm_ip)
        except Exception as exc:
            with self.telemetry_lock:
                self.hard_fault_generation += 1
                self.arm_fault_generation += 1
                self.state_diagnostic_error = self.force_diagnostic_error = str(exc)
            super().ingest(state, now)
            return
        if arm_ip and arm_ip != str(self.config['robot_ip']):
            return
        try:
            if int(state.errCode) != 0:
                raise ValueError(f'UDP 数据解析错误: {state.errCode}')
            error_values = list(state.err.err)
            error_count = int(state.err.err_len)
            if not 0 <= error_count <= len(error_values):
                raise ValueError('UDP 控制器错误数组长度无效')
            system_errors = [int(value) for value in error_values[:error_count]]
            if any(system_errors):
                raise ValueError(f'UDP 控制器系统错误: {system_errors}')
        except Exception as exc:
            # The original adapter checks errCode (packet parsing), not state.err
            # (controller faults). Never let its loader accept this packet.
            error = str(exc)
            with self.telemetry_lock:
                self.hard_fault_generation += 1
                self.arm_fault_generation += 1
                self.state_diagnostic_error = self.force_diagnostic_error = error
            with self.lock:
                self.error = self.force_error = error
                self.q = self.pose = self.force = None
                self.processed.clear()
                if self.force_filter is not None:
                    self.force_filter.reset(error)
            return
        try:
            joint = state.joint_status
            errors, enabled = list(joint.joint_err_code), list(joint.joint_en_flag)
            if len(errors) != 7 or len(enabled) != 7:
                raise ValueError('RM75 七轴错误/使能反馈长度无效')
            if int(state.errCode) != 0 or any(int(x) != 0 for x in errors):
                raise ValueError('RM75 控制器或关节反馈报错')
            if not all(int(x) == 1 for x in enabled):
                raise ValueError('RM75 存在未使能关节')
            values = _finite(list(joint.joint_position), 7, 'joint feedback')
            with self.telemetry_lock:
                self.state_values, self.state_diagnostic_error = values, None
                self.state_received = now
            self.state_rate.add(now)
        except Exception as exc:
            with self.telemetry_lock:
                self.hard_fault_generation += 1
                self.arm_fault_generation += 1
                self.state_diagnostic_error = str(exc)
        try:
            values = _finite(list(state.force_sensor.zero_force), 6, 'force feedback')
            coordinate = int(state.force_sensor.coordinate)
            if coordinate != int(self.config.get('force_coordinate', 1)):
                raise ValueError(f'力反馈坐标系错误: {coordinate}')
            with self.telemetry_lock:
                self.force_values, self.force_diagnostic_error = values, None
            self.force_rate.add(now)
        except Exception as exc:
            with self.telemetry_lock:
                self.hard_fault_generation += 1
                self.force_diagnostic_error = str(exc)
        super().ingest(state, now)

    def diagnostics(self):
        now = clock.now()
        limit = float(self.config['feedback_max_age_s'])
        state = self.state_rate.snapshot(now, limit)
        force = self.force_rate.snapshot(now, limit)
        with self.telemetry_lock:
            q, raw = self.state_values, self.force_values
            state_error, force_error = self.state_diagnostic_error, self.force_diagnostic_error
        with self.lock:
            filtered = dict(self.processed[-1]) if self.processed else None
            filter_error = self.force_error or self.error
            filter_state = dict(epoch=self.force_filter.epoch,
                last_reset_reason=self.force_filter.reset_reason) if self.force_filter is not None else None
        timing = self.feedback_metrics.snapshot()
        state['callback_timing'] = timing
        force['callback_timing'] = timing
        force['filter_state'] = filter_state
        force['scheduler_timing'] = self.scheduler_probe.snapshot()
        state.update(values=list(q) if q is not None else None, units='degrees',
                     kind='measured_joint_positions', error=state_error)
        # Fall back to genuine raw measurements during filter warm-up, with an
        # explicit kind/error. Raw values never become model observations.
        use_filtered = filtered is not None and not filter_error and 0 <= now-filtered['time'] <= limit
        values = filtered['force'] if use_filtered else raw
        force.update(values=list(values) if values is not None else None,
                     kind='calibrated_filtered' if use_filtered else 'controller_zero_force',
                     units=['N', 'N', 'N', 'Nm', 'Nm', 'Nm'],
                     error=force_error or filter_error)
        for item in (state, force):
            if item['age_ms'] is None or not 0 <= item['age_ms'] <= limit * 1000:
                item['error'] = item['error'] or 'UDP 反馈过期'
            item['ready'] = item['error'] is None
        return state, force


class _CheckedGripper(Ctag2F120Worker):
    """Read-only setup; explicit later commands use the existing worker."""

    def __init__(self, config):
        super().__init__(config)
        self._feedback_rate = _Rate()
        self._observed_alarm = None

    def _setup(self, sdk, factory):
        verify_arm_state(sdk)
        voltage = self._code(sdk.rm_get_tool_voltage(), '读取末端供电')
        if voltage != 3:
            raise RuntimeError('监测模式未改变末端供电：需要现有 24V 供电')
        bus = self._code(sdk.rm_get_tool_rs485_mode(), '读取末端 RS485 模式')
        if not isinstance(bus, dict) or any(bus.get(key) != value for key, value in
                                          (('mode', 1), ('baudrate', 115200), ('timeout', 2))):
            raise RuntimeError('监测模式未改变总线：需要现有 Modbus RTU 115200 / timeout=2')
        alarm = self._read(sdk, factory, 0x0610, 3)[2]
        self._setup_alarm = alarm
        self._observed_alarm = alarm
        with self.lock:
            self._observe_device_locked(alarm=alarm)
        # An existing device alarm blocks control, not read-only monitoring.
        # Still verify all mappings/parameters; never repair or clear them here.
        mapping = self._read(sdk, factory, 0x0404, 4)
        actuator = self._read(sdk, factory, 0x0301, 4)
        lower, upper = _signed32(*mapping[:2]), _signed32(*mapping[2:])
        if lower >= upper or _signed32(*actuator[:2]) >= _signed32(*actuator[2:]):
            raise RuntimeError('夹爪实机行程映射无效')
        speed = int(self.config.get('gripper_speed', 30))
        torque = int(self.config.get('gripper_torque', 50))
        if not 1 <= speed <= 100 or not 1 <= torque <= 100:
            raise ValueError('gripper_speed / gripper_torque 必须在 1..100')
        actual_parameters = self._read(sdk, factory, 0x0104, 4)
        expected_parameters = [speed, torque, 100, 100]
        if actual_parameters != expected_parameters:
            raise RuntimeError('监测模式未改写夹爪参数：0x0104..0x0107 '
                               f'实测 {actual_parameters}，配置要求 {expected_parameters} '
                               '(速度、力矩、加速度、减速度)')
        position = self._read(sdk, factory, 0x060D, 3)
        opening = 1.0 - (_signed32(*position[:2]) - lower) / (upper - lower)
        if not math.isfinite(opening) or not -0.02 <= opening <= 1.02:
            raise RuntimeError('夹爪反馈位置超出已保存的开合映射')
        return lower, upper, max(0.0, min(1.0, opening))

    def _poll_feedback(self, *args):
        sample = super()._poll_feedback(*args)
        if sample is not None:
            with self.lock:
                self._observed_alarm = self.state.get('alarm')
        return sample

    def _verify_readonly_reconnect(self, sdk, factory):
        # Unlike the base worker setup, this subclass only reads and verifies.
        # No power/bus/parameter writes, enable, clear, or target replay.
        lower, upper, opening = self._setup(sdk, factory)
        if (lower, upper) != (self.mapping_min, self.mapping_max):
            raise RuntimeError('夹爪重连行程映射不匹配，保持停止')
        return lower, upper, opening

    def status(self, max_age_s=None):
        max_age_s = self.control_max_age_s if max_age_s is None else max_age_s
        result = super().status(max_age_s)
        # The base stale check must not erase the actual failed setup reason.
        with self.lock:
            source_error = self.state.get('error')
            communication_fault = bool(self.state.get('communication_fault'))
            observed_alarm = self.state.get('alarm', self._observed_alarm)
            if not self.state.get('initialized') and self.state.get('error'):
                result['error'] = self.state['error']
            # Enable acknowledgements update state.received in the original
            # worker. Only a position-history entry is a new measured opening.
            history = [(stamp, opening) for stamp, opening in self.history
                       if stamp <= result.get('received', 0)]
            request_times = dict(self.history_started)
        now = clock.now()
        for stamp, _ in history:
            self._feedback_rate.add(stamp)
        metrics = self._feedback_rate.snapshot(now, max_age_s)
        stamp = history[-1][0] if history else None
        age_s = now-request_times.get(stamp, stamp) if stamp is not None else math.inf
        result['age_ms'] = round(age_s*1000, 2) if math.isfinite(age_s) else None
        result['hz'], result['samples'] = metrics['hz'], metrics['samples']
        recent = [stamp for stamp, _ in history if now-2 <= stamp <= now]
        intervals = [1000*(right-left) for left, right in zip(recent, recent[1:])]
        result['last_interval_ms'] = round(intervals[-1], 2) if intervals else None
        result['max_interval_ms'] = round(max(intervals), 2) if intervals else None
        result['observation_max_age_ms'] = max_age_s*1000
        # Observation and motion budgets are independent profile settings.
        control_age_s = min(max_age_s, self.control_max_age_s)
        if result.get('initialized') and not 0 <= age_s <= control_age_s:
            result.update(ready=False, error='夹爪运动反馈过期')
        result['alarm'] = observed_alarm
        opening, age = result.get('opening'), result.get('age_ms')
        # A read timeout cannot invalidate the historical fact of an earlier
        # measurement that is still inside the configured observation budget.
        # Motion is separately blocked by ready, enable age and fault generation.
        position_fresh = bool(result.get('initialized')
            and isinstance(opening, (int, float)) and not isinstance(opening, bool)
            and math.isfinite(opening) and 0 <= opening <= 1
            and isinstance(age, (int, float)) and math.isfinite(age) and 0 <= age <= max_age_s * 1000)
        result['measurement_ready'] = position_fresh and result.get('alarm') == 0
        # Observation failures must name the observation contract, not the
        # tighter motion gate. A valid older measurement can still condition
        # inference while fresh feedback is required before any motor command.
        if result['measurement_ready']:
            result['measurement_error'] = None
        elif type(observed_alarm) is int and observed_alarm != 0:
            result['measurement_error'] = f'夹爪报警 0x{observed_alarm:04x}'
        elif not result.get('initialized'):
            result['measurement_error'] = source_error or '缺少已初始化的夹爪实测开度'
        elif not position_fresh:
            result['measurement_error'] = ('夹爪观测反馈过期或无效'
                f'（年龄 {age} ms，观测上限 {max_age_s*1000:g} ms）')
        else:
            result['measurement_error'] = '夹爪报警状态未知'
        enable_age = result.get('enabled_age_ms')
        enable_fresh = isinstance(enable_age, (int, float)) and math.isfinite(enable_age) and 0 <= enable_age <= self.control_max_age_s*1000
        result['control_feedback_ready'] = bool(result['measurement_ready'] and 0 <= age_s <= control_age_s
            and enable_fresh)
        result['monitoring_ready'] = bool(position_fresh and enable_fresh
            and type(result.get('enabled')) is bool and not communication_fault)
        freshness = []
        if result.get('initialized'):
            if not 0 <= age_s <= control_age_s:
                freshness.append('夹爪运动反馈过期')
            if not enable_fresh:
                freshness.append('夹爪使能反馈过期')
        result['freshness_error'] = '；'.join(freshness) or None
        transport = result.get('transport') or {}
        event = transport.get('last_event') or {}
        transport_error = event.get('error') if transport.get('needs_sync') or transport.get('failed') else None
        # communication_fault also marks an old otherwise-valid response. Its
        # state.error may prioritize the device alarm; neither is evidence of a
        # transport failure. Preserve actual read exceptions separately.
        state_only_errors = {'夹爪反馈过期', '夹爪运动反馈过期', '夹爪使能反馈过期'}
        if type(observed_alarm) is int and observed_alarm:
            state_only_errors.add(f'夹爪报警 0x{observed_alarm:04x}')
        result['communication_error'] = transport_error or (
            source_error if communication_fault and source_error not in state_only_errors else None)
        if type(observed_alarm) is int and observed_alarm != 0:
            result.update(ready=False, error=f'夹爪报警 0x{observed_alarm:04x}')
        return result

    def snapshot(self, anchor, max_age_s=0.5):
        """Measured openness is valid while disabled; commands still require ready."""
        now = clock.now()
        state = self.status(max_age_s)
        with self.lock:
            eligible = [(stamp, opening) for stamp, opening in self.history if stamp <= anchor]
            request_times = dict(self.history_started)
        if not state.get('measurement_ready'):
            raise RuntimeError(state.get('measurement_error') or '缺少已初始化、无报警的夹爪实测开度')
        if not eligible:
            raise RuntimeError('没有机械臂状态锚点之前的夹爪反馈')
        stamp, opening = eligible[-1]
        source_start = request_times.get(stamp, stamp)
        if not 0 <= anchor-source_start <= max_age_s or not 0 <= now-source_start <= max_age_s:
            raise RuntimeError('夹爪观测反馈过期或与机械臂状态不同步')
        return opening

    def _send_target(self, sdk, factory, requested, bounded, previous, sequence, source):
        # Recheck every ramp write, not just admission of its original target.
        # status() uses position history; an enable ACK cannot refresh its age.
        state = self.status()
        if not state.get('ready') or not state.get('control_feedback_ready'):
            self.hold()
            raise FeedbackUnavailable('夹爪目标未写入：真实位置反馈过期或运动状态未就绪；'
                               + str(state.get('error') or f'需要{self.control_max_age_s*1000:g}ms内有效位置、使能且无报警'))
        return super()._send_target(sdk, factory, requested, bounded, previous, sequence, source)


class WebHardware(RealManHardwareAdapter):
    """One capture owner shared by dashboard and the existing 30 Hz collector."""

    def __init__(self, config_path):
        self._robot_error = None
        self._gripper_error = None
        self._robot_identity_valid = False
        self._ever_armed = False
        self._closing = False
        self._diagnostic_lock = threading.Lock()
        self._diagnostic_active = False
        self._last_gripper_diagnosis = None
        self._clearing_gripper = False
        self._reconnecting_gripper = False
        self._last_gripper_result = None
        self._manual_gripper_operation = None
        self._last_manual_gripper_result = None
        self._feedback_recovery_generation = 0
        self._reset_active = False
        self.frame_geometry = dict(verified=False, error='尚未核对标定坐标系和 DH')
        super().__init__(config_path)

    def _connect(self):
        # Start each camera even if its peer or the robot cannot connect.
        for spec in self.config['cameras']:
            camera = _MonitoredCamera(spec, width=self.config['camera_width'],
                                      height=self.config['camera_height'], fps=self.config['camera_fps'],
                                      max_age_s=self.config['camera_max_age_s'])
            self.cameras.append(camera)
            try:
                camera.start()
            except Exception as exc:
                with camera.lock:
                    camera.error = str(exc)
        try:
            self._connect_robot()
        except Exception as exc:
            self._robot_error = str(exc)
        # A failed UDP warm-up does not prevent a separate gripper monitor.
        if self._robot_identity_valid:
            self.gripper = _CheckedGripper(self.config)
            try:
                self.gripper.start()
                self.gripper_feedback = self.gripper.wait_initialized(
                    float(self.config.get('gripper_start_timeout_s', 8.0)))
            except Exception as exc:
                self._gripper_error = str(exc)

    def _connect_robot(self):
        from Robotic_Arm.rm_robot_interface import rm_inverse_kinematics_params_t
        from .sdk import create_sdk
        self.ik_params_type = rm_inverse_kinematics_params_t
        self.sdk = create_sdk()
        self.sdk.rm_set_timeout(int(self.config.get('connect_timeout_ms', 1000)))
        self.handle = self.sdk.rm_create_robot_arm(str(self.config['robot_ip']),
                                                  int(self.config['robot_port']), 3)
        if self.handle is None or int(getattr(self.handle, 'id', 0)) <= 0:
            raise RuntimeError('RM75 连接失败')
        code, info = self.sdk.rm_get_robot_info()
        if code != 0:
            raise RuntimeError(f'读取 RM75 信息失败: {code}')
        for expected_key, actual_key in (('expected_model', 'arm_model'),
                                          ('expected_force_type', 'force_type'),
                                          ('expected_controller_version', 'robot_controller_version')):
            expected = self.config.get(expected_key)
            if expected is not None and info.get(actual_key) != expected:
                raise RuntimeError(f'RM75 {expected_key} 不匹配: {info.get(actual_key)} != {expected}')
        if int(info.get('arm_dof', 0)) != 7:
            raise RuntimeError('模型要求 RM75 七轴')
        self._verify_frames_and_dh()
        lo_code, lo = self.sdk.rm_get_joint_min_pos()
        hi_code, hi = self.sdk.rm_get_joint_max_pos()
        if lo_code != 0 or hi_code != 0:
            raise RuntimeError(f'读取 RM75 关节软限位失败: min={lo_code}, max={hi_code}')
        self.joint_min, self.joint_max = _finite(lo, 7, 'joint minimum'), _finite(hi, 7, 'joint maximum')
        if any(a >= b for a, b in zip(self.joint_min, self.joint_max)):
            raise RuntimeError('RM75 关节软限位无效')
        self._robot_identity_valid = self.connected = True
        self.feedback = _MonitoredFeedback(self.sdk, self.config)
        self.feedback.start()
        _, pose, _, _ = self.feedback.snapshot(float(self.config['feedback_max_age_s']))
        center = self.config.get('workspace_center_pose')
        self.workspace_center = _finite(center, 6, 'workspace_center_pose') if center is not None else pose

    def _verify_frames_and_dh(self):
        """Read-only parity with collection Robot.verify_frames/DH validation.

        Expected names and physical d lengths come from the validated force
        calibration, so a different config cannot silently redefine its frame.
        """
        report = dict(verified=False, checked_at=clock.now())
        try:
            calibration = self.config.get('_force_calibration')
            context = calibration.get('context') if isinstance(calibration, dict) else None
            if not isinstance(context, dict):
                raise RuntimeError('力标定缺少坐标系与物理几何 context')
            for method, key, label in (('rm_get_current_tool_frame', 'expected_tool_frame', 'tool'),
                                        ('rm_get_current_work_frame', 'expected_work_frame', 'work')):
                expected = context.get(key)
                if not isinstance(expected, str) or not expected:
                    raise RuntimeError(f'力标定缺少 {key}')
                if key in self.config and self.config[key] != expected:
                    raise RuntimeError(f'硬件配置 {key} 与已绑定力标定不一致')
                frame = Ctag2F120Worker._code(getattr(self.sdk, method)(), method)
                if not isinstance(frame, dict):
                    raise RuntimeError(f'{method} 没有返回坐标系字典')
                pose = _finite(frame.get('pose', ()), 6, f'{label} frame pose')
                report[label] = dict(name=frame.get('name'), pose=list(pose), expected_name=expected)
                if frame.get('name') != expected or math.sqrt(sum(value*value for value in pose)) > 1e-6:
                    raise RuntimeError(f'{label} 坐标系与力标定不一致：实测 {report[label]}，要求同名 identity pose')
            expected_d = _finite(context.get('expected_dh_d_m', ()), 7, 'calibration expected DH d')
            if 'expected_dh_d_m' in self.config:
                configured = _finite(self.config['expected_dh_d_m'], 7, 'configured DH d')
                if any(abs(a-b) > 1e-6 for a, b in zip(configured, expected_d)):
                    raise RuntimeError('硬件配置 DH d 与已绑定力标定不一致')
            controller = Ctag2F120Worker._code(self.sdk.rm_get_DH_data(), '读取控制器 DH')
            algorithm = self.sdk.rm_algo_get_dh()
            if not isinstance(controller, dict) or not isinstance(algorithm, dict):
                raise RuntimeError('控制器/SDK DH 没有返回参数字典')
            report.update(controller_dh={}, sdk_dh={}, expected_dh_d_m=list(expected_d))
            for key in ('d', 'a', 'alpha', 'offset'):
                values = _finite(controller.get(key, ()), 7, f'controller DH {key}')
                sdk_values = _finite(algorithm.get(key, ()), 7, f'SDK DH {key}')
                report['controller_dh'][key], report['sdk_dh'][key] = list(values), list(sdk_values)
                if any(abs(a-b) > 1e-6 for a, b in zip(values, sdk_values)):
                    raise RuntimeError(f'控制器与 SDK DH {key} 不一致')
            if any(abs(a-b) > 1e-6 for a, b in zip(report['controller_dh']['d'], expected_d)):
                raise RuntimeError(f"机械臂物理 DH d 与力标定不一致：{report['controller_dh']['d']} != {list(expected_d)}")
            report.update(verified=True, error=None)
        except Exception as exc:
            report['error'] = str(exc)
            raise
        finally:
            self.frame_geometry = report
        return report

    def status(self):
        result = super().status()
        if self.feedback is None and self._robot_error:
            result['robot']['error'] = self._robot_error
            result['error'] = self._robot_error
        if self.gripper is None and self._gripper_error:
            result['gripper']['error'] = self._gripper_error
        available, reason = self._local_ready(result)
        result.update(execution_available=available, execution_reason=reason, frame_geometry=self.frame_geometry)
        return result

    def _local_ready(self, status, *, for_execution=True):
        if self.closed or self._closing:
            return False, '硬件监测已关闭'
        if self._diagnostic_active:
            return False, '正在诊断或恢复夹爪'
        if self.estop_latched:
            return False, '硬件急停已锁存'
        if not self._robot_identity_valid or not status.get('connected'):
            return False, self._robot_error or '机械臂身份与连接未验证'
        if not status.get('joint_limits_ready'):
            return False, '机械臂关节限位未验证'
        for key, component in (('robot_ready', 'robot'), ('force_ready', 'robot')):
            if not status.get(key):
                return False, status.get(component, {}).get('error') or f'{component} 未就绪'
        # observe_at always requires measured gripper data, including when an
        # old config marks gripper_required=false. Never invent its eighth DOF.
        gripper_key = 'ready' if for_execution else 'measurement_ready'
        if self.gripper is None or not status.get('gripper', {}).get(gripper_key):
            error_key = 'error' if for_execution else 'measurement_error'
            return False, status.get('gripper', {}).get(error_key) or '缺少真实夹爪开度'
        if (not status.get('cameras_ready') or status.get('camera_order') != ['global', 'wrist']
                or len(self.cameras) != 2):
            return False, next((c.get('error') for c in status['cameras'] if not c.get('ready')),
                               '双相机未就绪')
        if not status.get('force_preprocessing') or self.feedback.force_filter is None:
            return False, '缺少已验证的力校准/滤波合同'
        if self.config['action_space'] != 'joint8':
            return False, '网页客户端仅接受 joint8'
        return True, None

    @property
    def execution_available(self):
        return self.status()['execution_available']

    @property
    def execution_reason(self):
        return self.status()['execution_reason']

    def can_infer(self, metadata):
        try:
            metadata = validate_metadata(dict(protocol=PROTOCOL, ready=True, metadata=metadata))
            limits = temporal_tolerances(metadata)
            for config_key, server_key in (('force_source_max_age_s', 'force_max_age_s'),
                    ('camera_max_age_s', 'camera_max_age_s'), ('camera_pair_skew_s', 'camera_pair_skew_s')):
                if float(self.config.get(config_key, limits[server_key])) > limits[server_key]+1e-9:
                    return False, '当前 Server 不支持此时序档位，请切回原始档或更新 Server'
            status = self.status()
            ready, reason = self._local_ready(status, for_execution=False)
            if not ready:
                return False, reason
            if status['force_preprocessing'] != metadata.get('force_preprocessing'):
                return False, '客户端力预处理合同与服务端不一致'
            if (self.config['camera_width'], self.config['camera_height']) != (metadata['width'], metadata['height']):
                return False, '客户端相机尺寸与服务端不一致'
            if not math.isclose(float(self.config['command_period_s']), 1/30, rel_tol=0, abs_tol=1e-9):
                return False, '客户端控制周期与训练 30 Hz 不一致'
            # Preflight only: never call observe/observe_at here or increment
            # the sequence owned by ObservationCollector.
            now = clock.now()
            _, _, _, stamp, _ = self.feedback.model_snapshot(now)
            self._aligned_camera_frames(now, stamp, wait_s=0.)
            self.gripper.snapshot(stamp, float(self.config.get('gripper_feedback_max_age_s', .5)))
            return True, None
        except Exception as exc:
            return False, str(exc)

    def diagnostics(self):
        if self.feedback is not None:
            state, force = self.feedback.diagnostics()
        else:
            error = self._robot_error or '机械臂 UDP 未连接'
            state = dict(hz=0.0, samples=0, age_ms=None, values=None, ready=False,
                         units='degrees', kind='measured_joint_positions', error=error)
            force = dict(hz=0.0, samples=0, age_ms=None, values=None, ready=False,
                         units=['N', 'N', 'N', 'Nm', 'Nm', 'Nm'], kind='unavailable', error=error)
        # Dashboard values use the same units as joint8 model observations.
        # A missing gripper is explicit null, never the example initial value.
        joints = state['values']
        gripper = (self.gripper.status(float(self.config.get('gripper_feedback_max_age_s', .5)))
                   if self.gripper is not None else {})
        opening = gripper.get('opening')
        age = gripper.get('age_ms')
        measured = (gripper.get('initialized') and isinstance(opening, (int, float))
                    and not isinstance(opening, bool) and math.isfinite(opening) and 0 <= opening <= 1
                    and isinstance(age, (int, float)) and math.isfinite(age)
                    and 0 <= age <= float(self.config.get('gripper_feedback_max_age_s', .5)) * 1000)
        opening = float(opening) if measured else None
        state['values'] = ([math.radians(x) for x in joints] + [opening] if joints is not None
                           else [None] * 7 + [opening] if opening is not None else None)
        state.update(units=['rad'] * 7 + ['0..1'], kind='measured_joint8',
                     gripper_age_ms=age, gripper_error=gripper.get('measurement_error'))
        if opening is None or not gripper.get('measurement_ready', gripper.get('ready')):
            state['ready'] = False
            state['error'] = state['error'] or gripper.get('measurement_error') or '缺少新鲜实测夹爪开度'
        return dict(state=state, force=force, cameras=[camera.diagnostics() for camera in self.cameras],
                    reference_checks=dict(self.frame_geometry,
                                          passed=bool(self.frame_geometry.get('verified'))))

    def frame(self, name):
        for camera in self.cameras:
            if camera.spec['name'] == name:
                return camera.frame()
        return None

    def arm(self):
        with self.lock:
            ready, reason = self._local_ready(self.status())
            if not ready:
                raise RuntimeError(reason)
            super().arm()
            self._ever_armed = True

    def acknowledge_gripper_faults(self):
        """Explicit execute_start acknowledges recovery; arm per chunk cannot."""
        state = self.gripper.status() if self.gripper is not None else {}
        if not state.get('ready') or not state.get('control_feedback_ready'):
            raise RuntimeError('夹爪反馈尚未恢复，不能恢复动作执行')
        self._acknowledged_gripper_faults = state.get('fault_generation', 0)

    def invalidate_feedback_recovery(self):
        """Called for user intent changes; normal chunk slow_stop is exempt."""
        with self.lock:
            self._feedback_recovery_generation = getattr(self, '_feedback_recovery_generation', 0)+1

    def _recovery_token_matches(self, token):
        return (isinstance(token, _FeedbackRecoveryToken) and token.hardware is self
            and token.gripper is self.gripper and token.feedback is self.feedback
            and token.control_generation == getattr(self, '_feedback_recovery_generation', 0)
            and token.gripper_hard_generation == self.gripper.hard_fault_generation
            and token.arm_hard_generation == self.feedback.hard_fault_generation
            and token.gripper_connection_generation == getattr(self.gripper, 'connection_generation', 0))

    def recovery_requires_execution_restart(self, token):
        """A new TCP connection can restore sensing, never the old motion permit."""
        return (isinstance(token, _FeedbackRecoveryToken) and token.hardware is self
            and token.gripper is self.gripper and token.feedback is self.feedback
            and token.control_generation == getattr(self, '_feedback_recovery_generation', 0)
            and token.gripper_connection_generation != getattr(self.gripper, 'connection_generation', 0))

    def _recovery_feedback_ready_locked(self, state):
        """Recheck a status snapshot while both feedback owners are locked."""
        worker = self.gripper
        now = clock.now()
        return (not self.closed and not self._closing and not self.estop_latched
            and not self._diagnostic_active and not worker.stop.is_set()
            and worker.thread is not None and worker.thread.is_alive()
            and state.get('ready') and state.get('control_feedback_ready')
            and state.get('fault_generation') == worker.fault_generation
            and state.get('hard_fault_generation') == worker.hard_fault_generation
            and worker.state.get('ready') and not worker.state.get('communication_fault')
            and worker.enabled_value is True and type(worker.state.get('alarm')) is int
            and worker.state.get('alarm') == 0
            and worker.enabled_received is not None and 0 <= now-worker.enabled_received <= worker.control_max_age_s
            and 0 <= now-worker.state.get('measurement_started', -math.inf) <= worker.control_max_age_s
            and self.feedback.state_diagnostic_error is None
            and self.feedback.force_diagnostic_error is None)

    def capture_feedback_recovery_token(self):
        """Capture once at explicit execute_start, never per healthy chunk."""
        with self.lock:
            ready, _ = self._local_ready(self.status())
            if not ready or self.gripper is None or self.feedback is None:
                raise RuntimeError('硬件尚未就绪，不能创建反馈恢复许可')
            state = self.gripper.status()
            with self.feedback.telemetry_lock, self.gripper.lock:
                if not self._recovery_feedback_ready_locked(state):
                    raise RuntimeError('硬件反馈已改变，不能创建反馈恢复许可')
                token = _FeedbackRecoveryToken(self, self.gripper, self.feedback,
                    getattr(self, '_feedback_recovery_generation', 0),
                    self.gripper.hard_fault_generation, self.feedback.hard_fault_generation,
                    getattr(self.gripper, 'connection_generation', 0))
                self._active_feedback_recovery_token = token
                return token

    def acknowledge_recovered_feedback(self, token):
        """Confirm only a transient gap; never enable, clear, or send motion."""
        with self.lock:
            if self.gripper is None or self.feedback is None or not self._recovery_token_matches(token):
                return False
            ready, _ = self._local_ready(self.status())
            if not ready:
                return False
            state = self.gripper.status()
            with self.feedback.telemetry_lock, self.gripper.lock:
                if not self._recovery_token_matches(token) or not self._recovery_feedback_ready_locked(state):
                    return False
                self._acknowledged_gripper_faults = state['fault_generation']
                return True

    def gripper_diagnostics(self):
        state = (self.gripper.status(float(self.config.get('gripper_feedback_max_age_s', .5)))
                 if self.gripper is not None else {})
        thread = self.gripper.thread if self.gripper is not None else None
        worker_alive = thread is not None and thread.is_alive()
        with self.lock:
            last = self._last_gripper_diagnosis
            fault = state.get('alarm')
            # A successful diagnostic register read is evidence, whereas the
            # zero-filled buffer accompanying an SDK failure is not evidence.
            if type(fault) is not int and last is not None:
                alarm = next((row for row in last['registers'] if row['address'] == 0x0612), None)
                fault = alarm['word'] if alarm and alarm['ok'] else None
            result = self._last_gripper_result
            if type(fault) is not int and result is not None and result.get('after_alarm') is not None:
                fault = result['after_alarm']
        if type(fault) is not int:
            fault = None
        available = (self.connected and self._robot_identity_valid and self.sdk is not None
                     and not self.closed and not self._closing and not self.armed
                     and not self.estop_latched and not self._diagnostic_active)
        enabled = state.get('enabled')
        if type(enabled) is not bool:
            enabled = result.get('enabled') if result else None
        live_enabled = state.get('enabled')
        live_measurement = bool(state.get('initialized') and state.get('measurement_ready')
                                and state.get('alarm') == 0 and type(live_enabled) is bool and worker_alive)
        manual_available = bool(available and live_measurement and state.get('control_feedback_ready'))
        return dict(ready=bool(state.get('ready')), error=state.get('error') or self._gripper_error,
                    fault_generation=state.get('fault_generation'),
                    hard_fault_generation=state.get('hard_fault_generation'),
                    communication_error=state.get('communication_error'), freshness_error=state.get('freshness_error'),
                    monitoring_ready=bool(state.get('monitoring_ready')),
                    fault_code=fault, enabled=enabled, measurement_ready=bool(state.get('measurement_ready')),
                    measurement_error=state.get('measurement_error'),
                    auto_reconnect=state.get('auto_reconnect'),
                    connection_generation=state.get('connection_generation', 0),
                    opening=state.get('opening'), requested_opening=state.get('requested_opening'),
                    age_ms=state.get('age_ms'),
                    enabled_age_ms=state.get('enabled_age_ms'),
                    read_duration_ms=state.get('read_duration_ms'),
                    enable_poll_s=state.get('enable_poll_s', self.config.get('gripper_enable_poll_s')),
                    feedback_schedule=state.get('feedback_schedule'),
                    transport=state.get('transport') if isinstance(state.get('transport'), dict) else {},
                    hz=state.get('hz', 0.0), samples=state.get('samples', 0),
                    last_interval_ms=state.get('last_interval_ms'), max_interval_ms=state.get('max_interval_ms'),
                    observation_max_age_ms=state.get('observation_max_age_ms'), control_max_age_ms=state.get('control_max_age_ms', 500),
                    enable_available=manual_available and live_enabled is False,
                    disable_available=manual_available and live_enabled is True,
                    motion_available=manual_available and live_enabled is True and bool(state.get('ready')),
                    manual_operation=self._manual_gripper_operation,
                    last_manual_result=self._last_manual_gripper_result,
                    clear_available=bool(available), clearing=self._clearing_gripper,
                    reconnect_available=bool(available),
                    reconnecting=getattr(self, '_reconnecting_gripper', False),
                    reconnect_reason=('只读重连通信；保留设备实际使能状态，不清错、不改参数、不发送开合目标'
                                      if available else '需已连接、停止动作执行、无急停且没有诊断/恢复任务'),
                    clear_reason=('重建末端通信，确认失能后必要时单次清错并对齐采集参数；不发送使能或开合目标'
                                  if available else '需已连接、停止动作执行、无急停且没有诊断/恢复任务'),
                    diagnostic_available=(self.connected and not self.closed and not self.armed
                                          and not self._diagnostic_active and not worker_alive),
                    last_diagnostic=last, last_result=result,
                    message=result.get('message') if result else None)

    def capture_manual_control_token(self):
        """Reserve the cancellation boundary when the UI request is queued."""
        with self.lock:
            worker = self.gripper
            if worker is None or self.closed or self._closing:
                raise RuntimeError('夹爪未连接，不能排队手动操作')
            with worker.lock:
                return _ManualControlToken(worker, worker.cancel_generation)

    def _reset_joint_state(self, *, active=False):
        if (self.closed or self._closing or not self.connected or not self._robot_identity_valid
                or self.sdk is None or self.feedback is None):
            raise RuntimeError('复位需要身份已验证且已连接的机械臂')
        if self.estop_latched:
            raise RuntimeError('急停已锁存，禁止复位')
        if self._diagnostic_active and not (active and getattr(self, '_reset_active', False)):
            raise RuntimeError('硬件操作正在进行，暂不能复位')
        if self.joint_min is None or self.joint_max is None:
            raise RuntimeError('机械臂原生关节软限位尚未读取')
        with self.feedback.telemetry_lock:
            q = self.feedback.state_values
            stamp = self.feedback.state_received
            error = self.feedback.state_diagnostic_error
            generation = self.feedback.arm_fault_generation
        if error:
            raise RuntimeError(error)
        if stamp is None or not 0 <= clock.now()-stamp <= float(self.config['feedback_max_age_s']):
            raise RuntimeError('机械臂关节 UDP 反馈过期，禁止复位')
        return _finite(q or (), 7, 'reset joint feedback'), stamp, generation

    def reset_available(self):
        try:
            with self.lock:
                self._reset_joint_state()
            return True, None
        except (RuntimeError, ValueError, TypeError) as exc:
            return False, str(exc)

    def capture_reset_token(self):
        with self.lock:
            _, _, generation = self._reset_joint_state()
            return _ResetToken(self, self.sdk, self.feedback,
                getattr(self, '_feedback_recovery_generation', 0), generation)

    def _require_reset_token(self, token):
        if (not isinstance(token, _ResetToken) or token.hardware is not self
                or token.sdk is not self.sdk or token.feedback is not self.feedback
                or token.control_generation != getattr(self, '_feedback_recovery_generation', 0)
                or token.arm_fault_generation != self.feedback.arm_fault_generation):
            raise RuntimeError('复位已被停止、更新操作或机械臂故障取消')

    def _reset_target(self, pose, lower, upper):
        if not isinstance(pose, dict) or pose.get('schema') != 'realman-reset-pose-v1':
            raise ValueError('复位记录格式无效')
        if pose.get('robot_ip') != str(self.config['robot_ip']):
            raise ValueError('复位记录的机械臂地址不匹配')
        expected_model = self.config.get('expected_model', 'RM_75')
        if pose.get('model') != expected_model:
            raise ValueError('复位记录的机械臂型号不匹配')
        if type(pose.get('speed_percent')) is not int or pose['speed_percent'] != 10:
            raise ValueError('复位速度必须为 10%')
        if pose.get('gripper_action') != 'unchanged':
            raise ValueError('复位不得包含夹爪动作')
        values = pose.get('joint_degrees')
        if (not isinstance(values, (list, tuple)) or len(values) != 7
                or any(type(x) not in (int, float) or not math.isfinite(x) for x in values)):
            raise ValueError('复位需要七个有限的关节角度')
        target = _finite(values, 7, 'reset target')
        lo, hi = _finite(lower or (), 7, 'joint minimum'), _finite(upper or (), 7, 'joint maximum')
        margin = float(self.config.get('joint_limit_margin_deg', 2.))
        if (not math.isfinite(margin) or margin < 0 or any(a >= b for a, b in zip(lo, hi))
                or any(x < a+margin or x > b-margin for x, a, b in zip(target, lo, hi))):
            raise ValueError('复位目标超出机械臂原生关节软限位')
        return target

    def reset_to_pose(self, pose, *, expected_token):
        """One explicit planned joint move; completion comes from real UDP q."""
        target = self._reset_target(pose, self.joint_min, self.joint_max)
        if not self._diagnostic_lock.acquire(blocking=False):
            raise RuntimeError('其他硬件操作正在进行，复位未发送')
        started = clock.now()
        try:
            with self.lock:
                self._require_reset_token(expected_token)
                self._reset_joint_state()
                self._diagnostic_active = self._reset_active = True
            # slow_stop cancels pending gripper targets without communicating
            # with the gripper, so a failed tool link cannot block joint reset.
            self.slow_stop()
            verify_arm_state(self.sdk)
            lo_code, lower = self.sdk.rm_get_joint_min_pos()
            hi_code, upper = self.sdk.rm_get_joint_max_pos()
            if lo_code != 0 or hi_code != 0:
                raise RuntimeError('复位前重新读取原生关节软限位失败')
            target = self._reset_target(pose, lower, upper)
            with self.lock:
                self._require_reset_token(expected_token)
                self._reset_joint_state(active=True)
                self._require_reset_token(expected_token)
                # Serialize only the final check + nonblocking dispatch with
                # user epoch invalidation. Never hold this lock during motion.
                self._ever_armed = True
                code = self.sdk.rm_movej(list(target), v=10, r=0, connect=0, block=0)
                if type(code) is not int or code != 0:
                    raise RuntimeError(f'复位规划请求未确认: {code!r}；不会重发')
            deadline = clock.now()+30.
            stable_since = last_stamp = None
            while clock.now() < deadline:
                with self.lock:
                    self._require_reset_token(expected_token)
                    measured, stamp, _ = self._reset_joint_state(active=True)
                    self._require_reset_token(expected_token)
                error = max(abs(a-b) for a, b in zip(target, measured))
                if stamp != last_stamp:
                    stable_since = (stable_since if stable_since is not None else stamp) if error <= .5 else None
                    last_stamp = stamp
                    if stable_since is not None and stamp-stable_since >= .2:
                        return dict(ok=True, completed=True, dispatched=True, joint_degrees=list(measured),
                            max_error_deg=round(error, 4), duration_ms=round((clock.now()-started)*1000, 2),
                            speed_percent=10, gripper_action='unchanged')
                time.sleep(.01)
            raise RuntimeError('复位 30 秒内未达到目标，已请求缓停')
        except Exception as exc:
            try:
                self.slow_stop()
            except Exception as stop_error:
                raise RuntimeError(f'{exc}；复位缓停未确认: {stop_error}') from exc
            raise
        finally:
            with self.lock:
                self._diagnostic_active = self._reset_active = False
            self._diagnostic_lock.release()

    def control_gripper(self, operation, *, expected_token=None):
        self.invalidate_feedback_recovery()
        if expected_token is None:
            return super().control_gripper(operation)
        worker = self.gripper
        if not isinstance(expected_token, _ManualControlToken) or expected_token.worker is not worker:
            from .gripper_transport import MotionCancelled
            raise MotionCancelled('夹爪手动请求所属连接已改变；未入队')
        if self.closed or self._closing or not self.connected or worker is None:
            raise RuntimeError('夹爪未连接')
        if self.armed or self.estop_latched:
            raise RuntimeError('先暂停推理；急停锁定时禁止夹爪操作')
        # Same adapter operations, with the captured cancellation boundary
        # carried to the worker's final lock-protected admission. Do not keep
        # the hardware lock while waiting for an enable/readback Future.
        expected = expected_token.cancel_generation
        if operation in ('enable', 'disable'):
            return worker.set_enabled(operation == 'enable', expected_cancel_generation=expected)
        if operation == 'open' and self.config.get('gripper_manual_release') is True:
            return worker.manual_open(timeout_s=6., expected_cancel_generation=expected)
        if operation in ('open', 'close'):
            worker.command(1. if operation == 'open' else 0., source='manual',
                           expected_cancel_generation=expected)
            return worker.status()
        raise ValueError('未知夹爪操作')

    def manual_gripper(self, operation, *, expected_token=None):
        """An explicit button request, serialized with diagnostics and recovery.

        Enable/disable wait for the existing worker's register readback future.
        Open/close enqueue one target and do not claim physical completion.
        """
        self.invalidate_feedback_recovery()
        if operation not in ('enable', 'disable', 'open', 'close'):
            raise ValueError('夹爪手动操作仅支持 enable/disable/open/close')
        if not self._diagnostic_lock.acquire(blocking=False):
            raise RuntimeError('夹爪诊断、恢复或手动操作正在进行')
        started = clock.now()
        result = dict(ok=False, operation=operation, queued=False, acknowledged=False,
                      moving=None, confirmed=False, completed=False)
        try:
            with self.lock:
                diagnostic = self.gripper_diagnostics()
                key = ('motion_available' if operation in ('open', 'close') else
                       'enable_available' if operation == 'enable' else 'disable_available')
                if not diagnostic.get(key):
                    raise RuntimeError('夹爪手动操作条件未满足：需停止动作执行、无急停、'
                                       '通信线程正常且反馈新鲜无报警，并处于对应使能状态')
                self._diagnostic_active = True
                self._manual_gripper_operation = operation
            # This uses the original worker queue/validation/target ramp, never
            # another SDK connection and never a retry after an uncertain reply.
            state = (super().control_gripper(operation) if expected_token is None
                     else self.control_gripper(operation, expected_token=expected_token))
            if operation in ('enable', 'disable'):
                expected = operation == 'enable'
                if state.get('enabled') is not expected:
                    raise RuntimeError('夹爪使能状态读回未确认；未重发操作')
                result.update(ok=True, acknowledged=True, confirmed=True, completed=True, enabled=expected,
                    message=('夹爪使能状态已读回确认；未提交开合目标' if expected else
                             '夹爪失能状态已读回确认；未提交开合目标'))
            else:
                result.update(ok=True, queued=True, requested_opening=1.0 if operation == 'open' else 0.0,
                    message=('张开' if operation == 'open' else '闭合') +
                            '目标已提交；动作尚未确认完成，请查看夹爪实测开度')
                if operation == 'open' and self.config.get('gripper_manual_release') is True:
                    for key in ('manual_release_performed', 'manual_open_prepared'):
                        result[key] = state.get(key)
                    result['manual_open_result'] = dict(queued=True, acknowledged=False, moving=None,
                        completed=False, source='manual', hold_released=state.get('manual_release_performed'))
            result['opening'] = state.get('opening')
            return result
        except Exception as exc:
            result['message'] = f'夹爪手动 {operation} 失败：{exc}'
            raise RuntimeError(result['message']) from exc
        finally:
            result['duration_ms'] = round((clock.now()-started)*1000, 2)
            with self.lock:
                self._last_manual_gripper_result = result
                self._manual_gripper_operation = None
                self._diagnostic_active = False
            self._diagnostic_lock.release()

    def reconnect_gripper(self):
        """Replace a stopped communication owner using read-only setup only.

        A slow old worker must actually exit before another bus owner starts.
        Reconnection invalidates all earlier execution acknowledgements, while
        preserving the physical enable state read back from the device.
        """
        self.invalidate_feedback_recovery()
        if not self._diagnostic_lock.acquire(blocking=False):
            raise RuntimeError('夹爪诊断、恢复或手动操作正在进行')
        started = clock.now()
        result = dict(ok=False, operation='reconnect', stage='preconditions',
                      writes_performed=False, writes=[], enabled=None, after_alarm=None)
        candidate = None

        def guard():
            if (self.closed or self._closing or not self.connected or self.armed
                    or self.estop_latched):
                raise RuntimeError('通信重连已取消：设备关闭、动作执行或急停状态改变')

        def stop_worker(worker):
            thread = worker.thread
            worker.hold()
            worker.close()
            if thread is not None and thread.is_alive():
                # Older close() implementations clear .thread even if join
                # times out. Preserve the owner so another click cannot bypass
                # this check and create two users of the same tool bus.
                worker.thread = thread
                raise RuntimeError('旧夹爪线程未退出；未创建第二个通信连接')

        try:
            with self.lock:
                diagnostic = self.gripper_diagnostics()
                if not diagnostic.get('reconnect_available'):
                    raise RuntimeError(diagnostic.get('reconnect_reason') or '只读重连条件未满足')
                self._diagnostic_active = self._reconnecting_gripper = True
                old = self.gripper
            guard()
            result['stage'] = 'stop_previous_worker'
            if old is not None:
                stop_worker(old)
            guard()
            old_generation = old.fault_generation if old is not None else 0
            generation = max(old_generation, getattr(self, '_acknowledged_gripper_faults', 0)) + 1
            result['stage'] = 'read_only_setup'
            candidate = _CheckedGripper(self.config)
            candidate.fault_generation = generation
            with self.lock:
                guard()
                self.gripper = candidate
                self._gripper_error = None
            candidate.start()
            candidate.wait_initialized(float(self.config.get('gripper_start_timeout_s', 8.)))
            guard()
            state = candidate.status()
            if not state.get('monitoring_ready'):
                raise RuntimeError(state.get('error') or '重连后尚无新鲜、完整的夹爪反馈')
            result.update(ok=True, stage='monitoring', enabled=state['enabled'],
                opening=state.get('opening'), after_alarm=state.get('alarm'),
                fault_generation=state.get('fault_generation', generation),
                message='只读通信重连成功；实际状态为' + ('已使能' if state['enabled'] else '已失能') +
                        (f"，设备仍报警 0x{state['alarm']:04x}，动作继续禁止" if state.get('alarm') else '') +
                        '，未改电压、总线或参数，未清错、使能或开合；恢复推理和执行需手动开启')
            return result
        except Exception as exc:
            cleanup_error = None
            if candidate is not None:
                try:
                    stop_worker(candidate)
                except Exception as cleanup_exc:
                    cleanup_error = str(cleanup_exc)
            result['message'] = f'只读通信重连失败：{exc}' + (
                f'；新线程清理未确认：{cleanup_error}' if cleanup_error else '')
            with self.lock:
                self._gripper_error = result['message']
            raise RuntimeError(result['message']) from exc
        finally:
            result['duration_ms'] = round((clock.now()-started)*1000, 2)
            with self.lock:
                self._last_gripper_result = result
                self._diagnostic_active = self._reconnecting_gripper = False
            self._diagnostic_lock.release()

    def clear_gripper_error(self):
        """Explicit user recovery, reproduced from verified prior-machine code.

        Evidence: references/gripper-clear-history.json. No write is retried.
        Reset 0x0403=1 is issued only after confirmed disable and a nonzero alarm;
        0x0402 (home), target/trigger and enable=1 are never written here.
        """
        self.invalidate_feedback_recovery()
        if not self._diagnostic_lock.acquire(blocking=False):
            raise RuntimeError('夹爪诊断或恢复正在进行')
        started = clock.now()
        result = dict(ok=False, before_alarm=None, after_alarm=None, enabled=None,
                      reset_written=False, reset_attempted=False, stage='preconditions', writes=[])
        try:
            with self.lock:
                if not self.gripper_diagnostics()['clear_available']:
                    raise RuntimeError(self.gripper_diagnostics()['clear_reason'])
                self._diagnostic_active = self._clearing_gripper = True

            def guard():
                if self.closed or self._closing or self.armed or self.estop_latched:
                    raise RuntimeError('恢复已取消：硬件关闭、动作执行或急停状态改变')

            result['stage'] = 'stop_previous_worker'
            old = self.gripper
            thread = old.thread if old is not None else None
            if old is not None:
                old.hold()
                old.close()
                if thread is not None and thread.is_alive():
                    raise RuntimeError('旧夹爪线程未退出，未访问总线或写入寄存器')
            from Robotic_Arm.rm_robot_interface import rm_peripheral_read_write_params_t
            device = self.config.get('gripper_device', 1)
            if type(device) is not int or not 1 <= device <= 247:
                raise ValueError('夹爪从站地址必须是 1..247 的整数')

            def read(address):
                guard()
                value = Ctag2F120Worker._code(self.sdk.rm_read_holding_registers(
                    rm_peripheral_read_write_params_t(1, address, device, 1)), f'读取 0x{address:04x}')
                if type(value) is not int:
                    raise RuntimeError(f'0x{address:04x} 读取没有返回有效整数')
                return value & 0xffff

            def write(address, value):
                guard()
                entry = dict(kind='single', address=address, address_hex=f'0x{address:04x}',
                             values=[value], acknowledged=False)
                result['writes'].append(entry)
                Ctag2F120Worker._code(self.sdk.rm_write_single_register(
                    rm_peripheral_read_write_params_t(1, address, device, 1), value),
                    f'单次写入 0x{address:04x}={value}（失败不重发）')
                entry['acknowledged'] = True

            result['stage'] = 'verify_arm_and_power'
            guard()
            verify_arm_state(self.sdk)
            voltage = Ctag2F120Worker._code(self.sdk.rm_get_tool_voltage(), '读取末端供电')
            if voltage != 3:
                raise RuntimeError('当前末端不是 24V；未自动更改供电')
            result['stage'] = 'reinitialize_modbus'
            guard()
            Ctag2F120Worker._code(self.sdk.rm_set_modbus_mode(1, 115200, 2), '重建末端 Modbus')
            time.sleep(.3)
            guard()
            bus = Ctag2F120Worker._code(self.sdk.rm_get_tool_rs485_mode(), '回读末端 Modbus')
            if any(bus.get(k) != v for k, v in (('mode', 1), ('baudrate', 115200), ('timeout', 2))):
                raise RuntimeError('末端 Modbus 设置回读不一致')
            result['stage'] = 'disable_and_confirm'
            write(0x0100, 0)
            actual_enabled = read(0x0100)
            result['enabled'] = bool(actual_enabled)
            if actual_enabled != 0:
                raise RuntimeError('失能写入虽返回，但 0x0100 尚非 0；未清错或重发')
            result['stage'] = 'read_alarm_before'
            result['before_alarm'] = read(0x0612)
            if result['before_alarm'] != 0:
                result['stage'] = 'single_alarm_reset'
                result['reset_attempted'] = True
                write(0x0403, 1)
                result['reset_written'] = True
            result['stage'] = 'verify_alarm_and_disabled'
            deadline = clock.now() + 2.0
            while True:
                if clock.now() >= deadline:
                    raise RuntimeError('报警/失能回读未在 2 秒内确认')
                result['after_alarm'] = read(0x0612)
                actual_enabled = read(0x0100)
                result['enabled'] = bool(actual_enabled)
                if actual_enabled != 0:
                    # The old verified flow also disables again on this fault.
                    # This is a fail-stop, never a retry of reset or enable.
                    result['stage'] = 'unexpected_enable_fail_stop'
                    result['failsafe_disable_attempted'] = True
                    write(0x0100, 0)
                    stopped = read(0x0100)
                    result['enabled'] = bool(stopped)
                    result['failsafe_disable_confirmed'] = stopped == 0
                    raise RuntimeError('清错后意外出现使能；' +
                        ('已单次再次失能并读回 0，仍保持故障状态' if stopped == 0 else
                         '再次失能后仍未读回 0，请检查设备；未重发清错或运动指令'))
                if clock.now() > deadline:
                    raise RuntimeError('报警/失能回读超过 2 秒，未报告恢复成功')
                if result['after_alarm'] == 0:
                    break
                time.sleep(min(.05, max(0., deadline-clock.now())))
            result['recovery_verified'] = True
            result['stage'] = 'align_volatile_motion_parameters'
            guard()
            register_io = Ctag2F120Worker(self.config)
            expected = [int(self.config.get('gripper_speed', 30)),
                        int(self.config.get('gripper_torque', 50)), 100, 100]
            if not all(1 <= value <= 100 for value in expected):
                raise ValueError('夹爪速度/力矩/加减速参数必须在 1..100')
            result['params_expected'] = expected
            result['params_before'] = register_io._read(self.sdk, rm_peripheral_read_write_params_t, 0x0104, 4)
            result['params_written'] = False
            if result['params_before'] != expected:
                result['enabled'] = bool(read(0x0100))
                if result['enabled']:
                    raise RuntimeError('参数对齐前夹爪已不是失能状态；未写参数')
                guard()
                entry = dict(kind='multiple', address=0x0104, address_hex='0x0104',
                             count=4, values=list(expected), acknowledged=False)
                result['writes'].append(entry)
                # Only the four volatile speed/torque/acceleration parameters.
                # The block ends at 0x0107, before the 0x0108 motion trigger.
                register_io._write(self.sdk, rm_peripheral_read_write_params_t, 0x0104, expected)
                entry['acknowledged'] = True
                result['params_written'] = True
            guard()
            result['params_after'] = register_io._read(self.sdk, rm_peripheral_read_write_params_t, 0x0104, 4)
            if result['params_after'] != expected:
                raise RuntimeError(f"易失参数回读不一致：{result['params_after']} != {expected}；未重发")
            actual_enabled = read(0x0100)
            result['enabled'] = bool(actual_enabled)
            if actual_enabled != 0:
                result['stage'] = 'parameters_unexpected_enable_fail_stop'
                result['failsafe_disable_attempted'] = True
                write(0x0100, 0)
                stopped = read(0x0100)
                result['enabled'] = bool(stopped)
                result['failsafe_disable_confirmed'] = stopped == 0
                raise RuntimeError('参数对齐后意外出现使能；已单次失能并回读，仍保持故障状态')
            result['stage'] = 'restart_readonly_monitor'
            guard()
            worker = _CheckedGripper(self.config)
            self.gripper = worker
            worker.start()
            self.gripper_feedback = worker.wait_initialized(float(self.config.get('gripper_start_timeout_s', 8.0)))
            state = worker.status(float(self.config.get('gripper_feedback_max_age_s', .5)))
            if state.get('enabled') is not False or not state.get('measurement_ready'):
                raise RuntimeError(state.get('error') or '只读监测未确认失能且无报警的真实开度')
            self._gripper_error = None
            result.update(ok=True, stage='complete', enabled=False,
                message=('通信已恢复，报警已清除并确认失能；未发送使能或开合目标' if result['reset_written'] else
                         '通信已恢复，原报警为 0，无需写清错寄存器；已确认失能，未发送使能或开合目标'))
            if result['params_written']:
                result['message'] += f"；易失运动参数已回读确认 {result['params_after']}"
            return result
        except Exception as exc:
            result['message'] = f"夹爪恢复停在 {result['stage']}: {exc}"
            self._gripper_error = result['message']
            raise RuntimeError(result['message']) from exc
        finally:
            result['duration_ms'] = round((clock.now()-started)*1000, 2)
            with self.lock:
                self._last_gripper_result = result
                self._diagnostic_active = self._clearing_gripper = False
            self._diagnostic_lock.release()

    def diagnose_gripper(self):
        """Bounded read-only checks on the existing handle, with no bus repair.

        The dashboard pauses inference/collector first. The gripper worker must
        have exited; otherwise its independent handle could contend on RS485.
        No power, baud, parameter, enable, target, or reset write is performed.
        """
        if not self._diagnostic_lock.acquire(blocking=False):
            raise RuntimeError('夹爪只读诊断正在进行')
        try:
            with self.lock:
                if self.closed or self._closing or not self.connected or self.sdk is None:
                    raise RuntimeError('机械臂现有连接不可用')
                if self.armed:
                    raise RuntimeError('先停止动作执行，再诊断夹爪')
                thread = self.gripper.thread if self.gripper is not None else None
                if thread is not None and thread.is_alive():
                    raise RuntimeError('夹爪轮询线程仍在运行；不能并发访问同一 RS485 总线')
                self._diagnostic_active = True
            from Robotic_Arm.rm_robot_interface import rm_peripheral_read_write_params_t
            device = self.config.get('gripper_device', 1)
            if type(device) is not int or not 1 <= device <= 247:
                raise ValueError('夹爪从站地址必须是 1..247 的整数')
            result = dict(ok=False, port=1, device=device, checked_at=clock.now(),
                          transport='existing_realman_tool_rs485_handle', settings={}, registers=[],
                          writes_performed=False)
            for name, method in (('tool_voltage', 'rm_get_tool_voltage'),
                                  ('tool_rs485_mode', 'rm_get_tool_rs485_mode')):
                started = clock.now()
                try:
                    reply = getattr(self.sdk, method)()
                    # SDK getters are inconsistent by design: tool voltage is
                    # (code, value), tool RS485 mode is a return_code dict.
                    if isinstance(reply, dict):
                        code = reply.get('return_code')
                        payload = {key: value for key, value in reply.items() if key != 'return_code'}
                    elif isinstance(reply, tuple) and len(reply) == 2:
                        code, payload = reply
                    else:
                        raise RuntimeError('SDK 查询返回结构不匹配')
                    result['settings'][name] = dict(ok=type(code) is int and code == 0,
                        return_code=code, value=payload if type(code) is int and code == 0 else None,
                        elapsed_ms=round((clock.now()-started)*1000, 2))
                except Exception as exc:
                    result['settings'][name] = dict(ok=False, return_code=None, value=None, error=str(exc),
                        elapsed_ms=round((clock.now()-started)*1000, 2))
            # Addresses are already used by the original CTAG worker: alarm is
            # [0x0610, 0x0611, 0x0612][2], enable=0x0100, speed/force=0x0104/5.
            # Do not reconstruct a 32-bit position from non-atomic word reads.
            for name, address in (('alarm', 0x0612), ('enabled', 0x0100),
                                  ('speed_parameter', 0x0104), ('force_parameter', 0x0105)):
                started = clock.now()
                row = dict(name=name, address=address, address_hex=f'0x{address:04x}',
                           ok=False, return_code=None, word=None)
                try:
                    reply = self.sdk.rm_read_holding_registers(
                        rm_peripheral_read_write_params_t(1, address, device, 1))
                    if not isinstance(reply, tuple) or len(reply) != 2:
                        raise RuntimeError('SDK 单寄存器读取返回结构不匹配')
                    code, payload = reply
                    row['return_code'] = code
                    if type(code) is not int or code != 0:
                        raise RuntimeError(f'控制器拒绝/通信失败，返回码 {code}；返回缓冲不可作为测量')
                    if type(payload) is not int:
                        raise RuntimeError('SDK 寄存器值不是整数')
                    row.update(ok=True, word=payload & 0xffff)
                except Exception as exc:
                    row['error'] = str(exc)
                row['elapsed_ms'] = round((clock.now()-started)*1000, 2)
                result['registers'].append(row)
            result['ok'] = all(row['ok'] for row in result['registers'])
            result['message'] = ('单寄存器只读检查成功；未清错、使能或移动夹爪' if result['ok'] else
                                 '部分寄存器读取失败；未更改夹爪设置，失败缓冲未作为测量')
            with self.lock:
                self._last_gripper_diagnosis = result
            return result
        finally:
            with self.lock:
                self._diagnostic_active = False
            self._diagnostic_lock.release()

    def _check_command_dispatch(self):
        if self._closing:
            raise RuntimeError('硬件正在关闭')
        token = getattr(self, '_active_feedback_recovery_token', None)
        if self.recovery_requires_execution_restart(token):
            raise FeedbackUnavailable('夹爪通信连接已重建，需要重新开始执行')
        if token is not None and not self._recovery_token_matches(token):
            raise RuntimeError('执行会话已失效：设备报警、失能或用户操作后需要重新开始执行')
        state = self.gripper.status() if self.gripper is not None else {}
        if state.get('fault_generation', 0) != getattr(self, '_acknowledged_gripper_faults', 0):
            raise RuntimeError('关节目标未写入：夹爪通信曾中断；反馈恢复后请重新点击开始执行')
        def detail(value):
            return {key: value.get(key) for key in ('age_ms', 'hz', 'last_interval_ms',
                'max_interval_ms', 'enabled', 'alarm', 'ready', 'error')}
        if not state.get('ready') or not state.get('control_feedback_ready'):
            raise RuntimeError(f'关节目标未写入：夹爪运动反馈未就绪 {detail(state)}')

    def command(self, action):
        self._check_command_dispatch()
        state = self.gripper.status() if self.gripper is not None else {}
        def detail(value):
            return {key: value.get(key) for key in ('age_ms', 'enabled_age_ms', 'hz', 'last_interval_ms',
                'max_interval_ms', 'enabled', 'alarm', 'ready', 'error', 'fault_generation')}
        started = clock.now()
        try:
            return super().command(action)
        except Exception as exc:
            after = self.gripper.status() if self.gripper is not None else {}
            raise RuntimeError(f'{exc}; command_ms={(clock.now()-started)*1000:.2f}; '
                               f'gripper_before={detail(state)}; gripper_after={detail(after)}') from exc

    def emergency_stop(self):
        self.invalidate_feedback_recovery()
        return super().emergency_stop()

    def close(self):
        self.invalidate_feedback_recovery()
        with self.lock:
            if self.closed or self._closing:
                return
            self._closing = True
            needs_stop = self._ever_armed or self.armed or self.last_command is not None
        # Existing SDK reads have the adapter's bounded timeout. Finish them
        # before deleting their handle; _closing rejects any new diagnostics.
        with self._diagnostic_lock:
            self._close_resources(needs_stop)

    def _close_resources(self, needs_stop):
        errors = []
        if needs_stop:
            try:
                self.slow_stop()
            except Exception as exc:
                errors.append(f'缓停失败: {exc}')
        with self.lock:
            self.closed, self.connected, self.armed = True, False, False
        for camera in self.cameras:
            try:
                camera.close()
            except Exception as exc:
                errors.append(f'相机关闭失败: {exc}')
        if self.gripper is not None:
            try:
                self.gripper.close()
            except Exception as exc:
                errors.append(f'夹爪关闭失败: {exc}')
        if self.feedback is not None:
            try:
                self.feedback.restore()
            except Exception as exc:
                errors.append(f'恢复 UDP 配置失败: {exc}')
        if self.sdk is not None and self.handle is not None and int(getattr(self.handle, 'id', 0)) > 0:
            try:
                code = self.sdk.rm_delete_robot_arm()
                if isinstance(code, int) and code != 0:
                    raise RuntimeError(str(code))
            except Exception as exc:
                errors.append(f'机械臂连接关闭失败: {exc}')
        self._stop_pool.shutdown(wait=False, cancel_futures=True)
        if errors:
            raise RuntimeError('; '.join(errors))


def make_hardware(config_path, **_):
    return WebHardware(config_path)
