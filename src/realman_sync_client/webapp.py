"""Loopback dashboard; telemetry and synchronous inference stay off HTTP threads."""
from __future__ import annotations

import argparse
from collections import deque
from copy import deepcopy
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO
import json
import logging
import math
from pathlib import Path
import secrets
import signal
import threading
import time
from urllib.parse import urlsplit

from . import clock
from .client import SynchronousClient, ClientOptions, validate_actions, validate_metadata
from .cli import add_motion_arguments, motion_options
from .http import InferenceHTTPClient
from .mock import MockHardware
from .recovery import transient_failure
from .reset_pose import load_reset_pose

LOG = logging.getLogger(__name__)
WEB_ROOT = Path(__file__).with_name('web')


def sampling_profile_ids(sampling):
    """Only the connected server can declare its supported sampling profiles."""
    profiles = sampling.get('profiles') if isinstance(sampling, dict) else None
    if not isinstance(profiles, list):
        return frozenset()
    return frozenset(profile['id'] for profile in profiles
                     if isinstance(profile, dict) and isinstance(profile.get('id'), str)
                     and profile['id'].strip())


class Dashboard:
    RECOVERY_STABLE_S = .5
    RECOVERY_TIMEOUT_S = 30.

    def __init__(self, transport, *, mode='realman', hardware_config=None, hardware_factory=None,
                 auto_infer=True, execute_steps=32, motion_settings=None):
        self.transport, self.mode = transport, mode
        self.hardware_config, self.hardware_factory = hardware_config, hardware_factory
        self.execute_steps = execute_steps
        self.client_options = ClientOptions(execute_steps=execute_steps, **(motion_settings or {}))
        self.lock = threading.RLock()
        self.stop_event = threading.Event()
        self.want_connect, self.auto_infer, self.execute = True, auto_infer, False
        self.generation = 0
        self.hardware = self.client = None
        self.metadata = None
        self.thread = None
        self.server = dict(url=transport.server_url, ready=False, backend=None, error=None, ping_ms=None)
        self.inference = dict(enabled=auto_infer, phase='waiting', requests_completed=0,
                              inference_ms=None, round_trip_ms=None, obs_age_ms=None,
                              last_error=None, actions=[], latencies=[])
        self.events = deque(maxlen=100)
        self.phase = 'connecting'
        self.blocked_reason = '正在连接设备和 Server'
        self.execution_available = False
        self._jpeg_lock = threading.Lock()
        self._jpeg = {}
        self._last_event = None
        self._session_active = False
        self._hardware_action = None
        self._hardware_action_token = None
        self._hardware_busy = False
        self._hardware_operation = None
        self._sampling_action = None
        self._sampling_busy = False
        self.sampling = dict(supported=False, busy=False, error=None)
        self.recovery = dict(active=False, reason=None, resume_execution=False,
                             attempts=0, last_recovered_at=None)
        self._execution_recovery_token = None
        self._recovery_generation = None
        self._recovery_deadline = self._recovery_ready_since = None
        self.saved_reset_pose = None
        self._reset_error = None
        self._reset_preparing = False
        self._reset_last_result = None
        if isinstance(hardware_config, (str, Path)):
            reset_path = Path(hardware_config).parent/'reset_pose.json'
            if reset_path.exists():
                try:
                    self.saved_reset_pose = load_reset_pose(reset_path)
                except Exception as exc:
                    self._reset_error = str(exc)

    def reset_pose_status(self):
        with self.lock:
            pose = deepcopy(self.saved_reset_pose)
            busy = self._reset_preparing or 'reset_to_pose' in (self._hardware_action, self._hardware_operation)
            hw = self.hardware
            reason = self._reset_error or ('尚未记录复位位置' if pose is None else None)
            available = False
            if pose is not None and hw is not None and hasattr(hw, 'reset_available'):
                try:
                    available, reason = hw.reset_available()
                except Exception as exc:
                    reason = str(exc)
            elif pose is not None:
                reason = '设备未连接或尚不支持复位'
            occupied = self._hardware_busy or self._hardware_action or self._sampling_busy or self._sampling_action
            return dict(saved=pose is not None, available=bool(available and not occupied and not busy),
                busy=bool(busy), reason=reason, recorded_at=pose.get('recorded_at') if pose else None,
                joint_degrees=pose.get('joint_degrees') if pose else None,
                speed_percent=pose.get('speed_percent') if pose else None,
                last_result=deepcopy(self._reset_last_result))

    def _queue_reset_pose(self):
        with self.lock:
            info = self.reset_pose_status()
            if not info['available'] or not self.want_connect:
                raise ValueError(info['reason'] or '请等待当前操作完成后复位')
            self._cancel_recovery()
            self.generation += 1
            generation = self.generation
            self.auto_infer = self.execute = False
            self.inference['enabled'] = False
            if self.client:
                self.client.stop_event.set()
            self._reset_preparing = self._hardware_busy = True
            self._hardware_operation = 'reset_to_pose'
            self._reset_last_result = None
            hw = self.hardware
        try:
            # Do not publish the queued move until this stop returns. Otherwise
            # the stop could arrive after the new reset motion was dispatched.
            if getattr(hw, 'armed', False):
                hw.slow_stop()
            else:
                gripper = getattr(hw, 'gripper', None)
                if gripper is not None:
                    gripper.hold()
            with self.lock:
                if generation != self.generation or not self.want_connect or self.stop_event.is_set():
                    raise InterruptedError('复位已取消')
                token = hw.capture_reset_token()
                self._hardware_action = 'reset_to_pose'
                self._hardware_action_token = token
        except Exception as exc:
            with self.lock:
                self._reset_last_result = dict(ok=False, message=str(exc))
            raise
        finally:
            with self.lock:
                self._reset_preparing = self._hardware_busy = False
                self._hardware_operation = None
        self.event('info', '复位已排队：暂停推理与执行，10% 速度回到记录位置，夹爪保持')
        return dict(message='复位已排队；可随时点击停止执行或紧急停止')

    def _cancel_recovery(self):
        # Called under the dashboard lock for every accepted user transition.
        self.recovery.update(active=False, reason=None, resume_execution=False)
        self._recovery_generation = None
        self._recovery_deadline = self._recovery_ready_since = None
        self._execution_recovery_token = None
        invalidate = getattr(self.hardware, 'invalidate_feedback_recovery', None)
        if invalidate:
            invalidate()

    def _stop_cycle(self):
        if self.client:
            self.client.stop_event.set()
        if getattr(self.hardware, 'armed', False):
            self.hardware.slow_stop()
        else:
            gripper = getattr(self.hardware, 'gripper', None)
            if gripper is not None:
                gripper.hold()
        self._close_client()

    def _pause_failure(self, error):
        with self.lock:
            self.auto_infer = self.execute = False
            self._cancel_recovery()
            self.inference.update(enabled=False, phase='error', last_error=str(error))
        self._stop_cycle()
        self.event('error', '推理已暂停：' + str(error))

    def _begin_recovery(self, error, generation):
        with self.lock:
            # A late exception must not revive a user-cancelled session.
            if generation != self.generation or not self.auto_infer or not self.want_connect:
                return True
            if not transient_failure(error):
                return False
            if self.recovery['active']:
                return True
            resume = self.execute
            if resume and self._execution_recovery_token is None:
                return False
            self.generation += 1
            self._recovery_generation = self.generation
            if self._recovery_deadline is None:
                self._recovery_deadline = clock.now()+self.RECOVERY_TIMEOUT_S
            self._recovery_ready_since = None
            self.recovery.update(active=True, reason=str(error), resume_execution=resume,
                                 attempts=self.recovery['attempts']+1)
            self.execute = False
            self.inference.update(enabled=True, phase='recovering', last_error=str(error), actions=[])
            self.phase = 'recovering'
        self._stop_cycle()
        self.event('warning', '暂时中断，等待反馈恢复后自动继续：'+str(error))
        return True

    def _poll_recovery(self, ready):
        with self.lock:
            if not self.recovery['active']:
                return False
            generation = self._recovery_generation
            if (generation != self.generation or not self.auto_infer or not self.want_connect
                    or self.stop_event.is_set()):
                self._cancel_recovery()
                return True
            resume = self.recovery['resume_execution']
            now = clock.now()
            if now >= self._recovery_deadline:
                self._pause_failure(RuntimeError('自动恢复等待超过 30 秒，保持停止：'+self.recovery['reason']))
                return True
            usable = ready and self.server['ready'] and (not resume or self.execution_available)
            if not usable:
                self._recovery_ready_since = None
                self.phase = self.inference['phase'] = 'recovering'
                return True
            if self._recovery_ready_since is None:
                self._recovery_ready_since = now
            if now-self._recovery_ready_since < self.RECOVERY_STABLE_S:
                return True
            if resume:
                acknowledge = getattr(self.hardware, 'acknowledge_recovered_feedback', None)
                if acknowledge is None or not acknowledge(self._execution_recovery_token):
                    self._recovery_ready_since = None
                    return True
            # Same user generation, verified feedback, no old collector/action.
            self.execute = bool(resume)
            self.recovery.update(active=False, reason=None, resume_execution=False,
                                 last_recovered_at=time.strftime('%H:%M:%S'))
            # The budget ends only after a new full cycle succeeds, not just
            # a brief ready signal; repeated collector failures stay bounded.
            self._recovery_ready_since = None
            self._recovery_generation = None
            self.inference.update(last_error=None, phase='monitoring')
            self.event('info', '反馈已恢复，自动重建观测并继续'+('推理与执行' if resume else '推理'))
            return False

    def event(self, level, message):
        with self.lock:
            if self._last_event == (level, message):
                return
            self._last_event = (level, message)
            self.events.append(dict(time=time.strftime('%H:%M:%S'), level=level, message=str(message)))
        LOG.info('%s: %s', level, message)

    def start(self):
        self.thread = threading.Thread(target=self._run, name='web-sync-inference', daemon=True)
        self.thread.start()

    def control(self, action, *, sampling_profile=None):
        if action == 'reset_pose':
            return self._queue_reset_pose()
        allowed = {'connect', 'disconnect', 'inference_start', 'inference_stop',
                   'execute_start', 'execute_stop', 'emergency_stop',
                   'gripper_clear_error', 'gripper_diagnose', 'gripper_reconnect',
                   'gripper_enable', 'gripper_disable', 'gripper_open', 'gripper_close', 'sampling_set'}
        if action not in allowed:
            raise ValueError('未知操作')
        with self.lock:
            if action == 'sampling_set':
                if not self.sampling.get('supported') or not hasattr(self.transport, 'set_sampling'):
                    raise ValueError('当前 Server 尚不支持采样切换')
                if not isinstance(sampling_profile, str) or sampling_profile not in sampling_profile_ids(self.sampling):
                    raise ValueError('采样档位必须由当前 Server 的 profiles 声明')
                if self.execute or getattr(self.hardware, 'armed', False):
                    raise ValueError('请先停止动作执行')
                if self._reset_preparing or self._hardware_action or self._hardware_busy or self._sampling_action or self._sampling_busy:
                    raise ValueError('请等待当前操作完成')
                self._cancel_recovery()
                self.generation += 1
                self.auto_infer = self.execute = False
                if self.client:
                    self.client.stop_event.set()
                self._sampling_action = sampling_profile
                self.sampling['error'] = None
                return {'message': '采样切换已排队；将暂停推理并释放会话，预热完成后请恢复自动推理'}
            if action.startswith('gripper_'):
                if not self.hardware or self.execute or getattr(self.hardware, 'armed', False):
                    raise ValueError('请先停止动作执行并连接设备')
                if self._reset_preparing or self._hardware_action or self._hardware_busy or self._sampling_action or self._sampling_busy:
                    raise ValueError('请等待当前操作完成')
                method = {'gripper_clear_error':'clear_gripper_error',
                          'gripper_reconnect':'reconnect_gripper',
                          'gripper_diagnose':'diagnose_gripper'}.get(action, action)
                if not hasattr(self.hardware, 'manual_gripper' if method.startswith('gripper_') else method):
                    raise ValueError('当前硬件未提供此夹爪操作')
                if action == 'gripper_clear_error':
                    diag = self.hardware.gripper_diagnostics()
                    if not diag.get('clear_available'):
                        raise ValueError(diag.get('clear_reason') or '夹爪清错条件未满足')
                elif action == 'gripper_reconnect':
                    diag = self.hardware.gripper_diagnostics()
                    if not diag.get('reconnect_available'):
                        raise ValueError(diag.get('reconnect_reason') or '夹爪只读重连条件未满足')
                elif method.startswith('gripper_'):
                    available = {'gripper_enable':'enable_available', 'gripper_disable':'disable_available',
                                 'gripper_open':'motion_available', 'gripper_close':'motion_available'}[action]
                    diag = self.hardware.gripper_diagnostics()
                    if not diag.get(available):
                        raise ValueError(diag.get('error') or '当前夹爪状态不允许此操作')
                capture_manual = getattr(self.hardware, 'capture_manual_control_token', None)
                hardware_token = (capture_manual() if method.startswith('gripper_')
                                  and capture_manual else None)
                self._cancel_recovery()
                self.generation += 1
                self.auto_infer = self.execute = False
                if self.client:
                    self.client.stop_event.set()
                self._hardware_action = method
                self._hardware_action_token = hardware_token
                if action == 'gripper_reconnect':
                    return {'message': '只读通信重连已排队；暂停推理，不使能、不开合、不清错；完成后需手动恢复自动推理'}
                return {'message': '夹爪操作已排队，结果将显示在运行事件；完成后需手动恢复自动推理'}
            if action in {'inference_start', 'execute_start', 'connect'} and (
                    self._reset_preparing or self._hardware_action or self._hardware_busy or self._sampling_action or self._sampling_busy):
                raise ValueError('请等待当前操作完成')
            if action == 'execute_start' and not getattr(self.hardware, 'execution_available', True):
                raise ValueError(getattr(self.hardware, 'execution_reason', None) or '设备执行未就绪')
            if action == 'execute_start' and not self.execution_available:
                raise ValueError(self.blocked_reason or '当前设备不允许执行')
            if action == 'execute_start' and not self.auto_infer:
                raise ValueError('请先开启自动推理')
            self._cancel_recovery()
            self.generation += 1
            self.execute = False
            if action == 'connect':
                self.want_connect = True
            elif action == 'disconnect':
                self.want_connect = self.auto_infer = self.execute = False
                self._hardware_action = None
                self._sampling_action = None
            elif action == 'inference_start':
                self.auto_infer = True
                self.inference['last_error'] = None
            elif action == 'execute_start':
                if hasattr(self.hardware, 'acknowledge_gripper_faults'):
                    self.hardware.acknowledge_gripper_faults()
                capture = getattr(self.hardware, 'capture_feedback_recovery_token', None)
                self._execution_recovery_token = capture() if capture else None
                self.execute = True
            else:
                self.execute = False
                if action in {'inference_stop', 'emergency_stop'}:
                    self.auto_infer = False
                if action in {'execute_stop', 'inference_stop'} and self._hardware_action in {
                        'gripper_enable', 'gripper_open', 'gripper_close', 'reset_to_pose'}:
                    self._hardware_action = None
                if action == 'emergency_stop':
                    self._hardware_action = None
                    self._sampling_action = None
                    self.execution_available = False
            self.inference['enabled'] = self.auto_infer
            hw, client = self.hardware, self.client
            reset_active = self._reset_preparing or self._hardware_operation == 'reset_to_pose'
            if client is not None:
                client.stop_event.set()
        # Native stop never waits for the blocked inference HTTP request.
        if hw is not None and action in {'execute_stop', 'inference_stop', 'disconnect', 'emergency_stop'}:
            if action == 'emergency_stop':
                getattr(hw, 'emergency_stop', hw.slow_stop)()
            elif reset_active or getattr(hw, 'armed', False):
                hw.slow_stop()
            else:
                # Manual preparation/ramping runs while the arm is unarmed.
                # Invalidate its worker ticket immediately; hold only cancels
                # pending intent and sends no joint or RTU command here.
                gripper = getattr(hw, 'gripper', None)
                if gripper is not None:
                    gripper.hold()
        self.event('info', {'connect':'连接设备', 'disconnect':'断开设备',
            'inference_start':'开启自动推理', 'inference_stop':'暂停自动推理',
            'execute_start':'用户开启动作执行', 'execute_stop':'停止动作执行',
            'emergency_stop':'用户请求急停'}[action])

    def _release(self):
        if self.client and self._session_active:
            try:
                self.transport.release(self.client.session_id)
            except Exception as exc:
                self.event('warning', '释放 Server 会话失败：' + str(exc))
            finally:
                self._session_active = False

    def _close_client(self):
        if self.client:
            self.client.stop_event.set()
            self.client.collector.close()
            self._release()
            self.client = None

    def _close_hardware(self):
        self._close_client()
        hw, self.hardware = self.hardware, None
        if hw:
            hw.close()
        with self._jpeg_lock:
            self._jpeg.clear()

    def _connect(self):
        if self.hardware_factory is not None:
            return self.hardware_factory(self.metadata)
        if self.mode == 'mock':
            if self.metadata is None:
                return None
            meta = self.metadata
            contract = meta.get('force_preprocessing')
            return MockHardware(width=meta['width'], height=meta['height'], camera_names=meta['camera_names'],
                force_preprocessing=contract,
                force_preprocessing_id=contract.get('id') if isinstance(contract, dict) else None)
        from .web_hardware import WebHardware
        return WebHardware(self.hardware_config)

    def _ready(self):
        if not self.hardware or not self.metadata:
            return False, '等待设备与 Server'
        if hasattr(self.hardware, 'can_infer'):
            return self.hardware.can_infer(self.metadata)
        state = self.hardware.status()
        return bool(state.get('connected')), state.get('error')

    def _run(self):
        health_due = connect_due = 0.0
        fresh_after, last_sequence = 0.0, -1
        try:
            while not self.stop_event.is_set():
                with self.lock:
                    preparing_reset = self._reset_preparing
                if preparing_reset:
                    self.stop_event.wait(.02)
                    continue
                with self.lock:
                    if self._reset_preparing:
                        continue
                    hardware_action, self._hardware_action = self._hardware_action, None
                    hardware_token, self._hardware_action_token = self._hardware_action_token, None
                    self._hardware_busy = hardware_action is not None
                    self._hardware_operation = hardware_action
                    sampling_action, self._sampling_action = self._sampling_action, None
                    self._sampling_busy = sampling_action is not None
                    operation_generation = self.generation
                if sampling_action:
                    try:
                        self._close_client()
                        fresh_after, last_sequence = 0.0, -1
                        with self.lock:
                            if (operation_generation != self.generation or not self.want_connect
                                    or self.stop_event.is_set()):
                                raise InterruptedError('采样操作已取消')
                            if (not self.sampling.get('supported')
                                    or sampling_action not in sampling_profile_ids(self.sampling)):
                                raise ValueError('当前 Server 已不再声明此采样档位')
                        result = self.transport.set_sampling(sampling_action)
                        sampling = result.get('sampling')
                        if not isinstance(sampling, dict) or sampling.get('sampling_profile') != sampling_action:
                            raise ValueError('Server 未确认请求的采样档位')
                        with self.lock:
                            self.sampling = dict(sampling, error=None)
                            self.inference.update(actions=[], latencies=[], inference_ms=None,
                                                  round_trip_ms=None, obs_age_ms=None, last_error=None)
                        self.event('info', '采样档位已切换至 '+sampling_action+'；请恢复自动推理')
                    except Exception as exc:
                        with self.lock:
                            self.sampling['error'] = str(exc)
                        self.event('error', '采样切换失败：'+str(exc))
                    finally:
                        with self.lock:
                            self._sampling_busy = False
                        health_due = 0.0
                if hardware_action:
                    try:
                        self._close_client()
                        fresh_after, last_sequence = 0.0, -1
                        with self.lock:
                            if (operation_generation != self.generation or not self.want_connect
                                    or self.stop_event.is_set()):
                                raise InterruptedError('夹爪操作已取消')
                        if hardware_action.startswith('gripper_'):
                            options = {'expected_token':hardware_token} if hardware_token is not None else {}
                            result = self.hardware.manual_gripper(hardware_action[len('gripper_'):], **options)
                        elif hardware_action == 'reset_to_pose':
                            result = self.hardware.reset_to_pose(deepcopy(self.saved_reset_pose),
                                                                expected_token=hardware_token)
                            with self.lock:
                                self._reset_last_result = dict(result, message='已回到记录位置；推理和执行保持暂停')
                        else:
                            result = getattr(self.hardware, hardware_action)()
                        self.event('info', ('复位结果：' if hardware_action == 'reset_to_pose' else '夹爪操作结果：')
                                   + json.dumps(result, ensure_ascii=False))
                    except Exception as exc:
                        if hardware_action == 'reset_to_pose':
                            with self.lock:
                                self._reset_last_result = dict(ok=False, message='复位未完成：'+str(exc))
                        self.event('error', ('复位未完成：' if hardware_action == 'reset_to_pose' else '夹爪操作失败：')+str(exc))
                    finally:
                        with self.lock:
                            self._hardware_busy = False
                            self._hardware_operation = None
                now = clock.now()
                if now >= health_due:
                    with self.lock:
                        health_generation = self.generation
                    try:
                        started = clock.now()
                        health = self.transport.health()
                        meta = validate_metadata(health)
                        if self.metadata is not None and meta != self.metadata:
                            with self.lock:
                                self._cancel_recovery()
                                self.generation += 1
                                self.execute = False
                            self._close_client()
                            fresh_after, last_sequence = 0.0, -1
                            self.event('warning', 'Server 模型或协议改变，已重建观测历史并暂停执行')
                        self.metadata = meta
                        with self.lock:
                            sampling = health.get('sampling', meta.get('sampling'))
                            if isinstance(sampling, dict):
                                self.sampling = dict(sampling, error=self.sampling.get('error'))
                            else:
                                # A different/older server must not inherit the previous
                                # server's profile allowlist.
                                self.sampling = dict(supported=False, profiles=[], error=None)
                            self.server.update(ready=True, backend=meta.get('backend'), error=None,
                                               ping_ms=round((clock.now()-started)*1000, 2))
                    except Exception as exc:
                        with self.lock:
                            self.server.update(ready=False, error=str(exc))
                            self.execution_available = False
                            if self.client:
                                self.client.stop_event.set()
                            was_active = self.client is not None or self.execute or self.recovery['active']
                        if was_active:
                            if not self._begin_recovery(exc, health_generation):
                                self._pause_failure(exc)
                            fresh_after, last_sequence = 0., -1
                        self.event('warning', 'Server 暂不可用：' + str(exc))
                    health_due = clock.now() + 2.0
                if not self.want_connect:
                    self._close_hardware()
                    with self.lock:
                        self.phase, self.execution_available = 'disconnected', False
                    self.stop_event.wait(.2)
                    continue
                if self.hardware is None and now >= connect_due:
                    try:
                        self.hardware = self._connect()
                        if self.hardware:
                            fresh_after, last_sequence = 0.0, -1
                            self.event('info', '设备监测已连接' if self.mode == 'realman' else '模拟数据模式：未连接真实硬件')
                    except Exception as exc:
                        self.event('error', '设备连接失败：' + str(exc))
                        self.blocked_reason = str(exc)
                    connect_due = clock.now() + 5.0
                ready, reason = self._ready()
                with self.lock:
                    ready_generation = self.generation
                    hw_can_execute = getattr(self.hardware, 'execution_available', True)
                    self.execution_available = bool(ready and self.server['ready'] and hw_can_execute)
                    self.blocked_reason = None if self.execution_available else (reason or
                        getattr(self.hardware, 'execution_reason', None) or self.server.get('error') or '设备尚未就绪')
                    lost_execution = self.execute and not self.execution_available
                if lost_execution:
                    error = RuntimeError(self.blocked_reason or '设备执行未就绪')
                    if not self._begin_recovery(error, ready_generation):
                        self._pause_failure(error)
                    fresh_after, last_sequence = 0., -1
                if self.recovery['active']:
                    fresh_after, last_sequence = 0., -1
                    if self._poll_recovery(ready):
                        self.stop_event.wait(.1)
                        continue
                if not ready or not self.server['ready']:
                    self.phase = 'monitoring'
                    self.inference['phase'] = 'waiting_sensors'
                    self.stop_event.wait(.2)
                    continue
                if not self.auto_infer:
                    self.phase = 'monitoring'
                    self.inference['phase'] = 'paused'
                    # Native telemetry stays live; resume builds a fresh model
                    # history instead of reusing a collector faulted while paused.
                    self._close_client()
                    fresh_after, last_sequence = 0.0, -1
                    self.stop_event.wait(.25)
                    continue
                if self.client is None:
                    self.client = SynchronousClient(self.transport, self.hardware, self.metadata,
                        options=self.client_options)
                    self.client.collector.start()
                try:
                    with self.lock:
                        generation = self.generation
                        execute_requested = self.execute
                        self.client.stop_event.clear()
                    cycle_started = clock.now()
                    if execute_requested:
                        # The native ACK only confirms receipt. Collect a full
                        # measured stability window after this stop before infer.
                        self.phase = self.inference['phase'] = 'settling'
                        self.hardware.slow_stop()
                        fresh_after = clock.now()
                        obs, history, video = self.client.snapshot_for_inference(fresh_after, last_sequence)
                    else:
                        obs, history, video = self.client.collector.snapshot_after(fresh_after, last_sequence, timeout_s=2.0)
                    request_started = clock.now()
                    request = self.client._request(obs, history, video)
                    request_built = clock.now()
                    with self.lock:
                        if generation != self.generation or not self.auto_infer or not self.want_connect or self.stop_event.is_set():
                            continue
                        self.phase = self.inference['phase'] = 'inferring'
                        self.client.stop_event.clear()
                        self._session_active = True
                    started = clock.now()
                    response = self.transport.infer(request)
                    end = clock.now()
                    self.client.collector.check()
                    with self.client.collector.condition:
                        if self.client.collector.history[-1].force_meta.get('epoch') != obs.force_meta.get('epoch'):
                            raise ValueError('推理期间力滤波器重置，已丢弃 action')
                    actions = validate_actions(response, request, self.metadata, now=end, max_source_age_s=120.0)
                    with self.lock:
                        valid = generation == self.generation and self.auto_infer and self.want_connect
                        if not valid:
                            self.event('info', '控制状态已改变，丢弃正在返回的 action')
                            fresh_after, last_sequence = end, obs.sequence
                            continue
                        if self._recovery_deadline is not None and clock.now() >= self._recovery_deadline:
                            raise RuntimeError('自动恢复未能在 30 秒内完成新的有效推理周期')
                        ms = response.get('inference_ms')
                        if not isinstance(ms, (int,float)) or not math.isfinite(ms) or ms < 0:
                            raise ValueError('Server 返回无效的推理延时')
                        rtt = (end-started)*1000
                        sample = dict(time=time.strftime('%H:%M:%S'), inference_ms=ms, round_trip_ms=rtt)
                        self.inference.update(requests_completed=self.inference['requests_completed']+1,
                            inference_ms=ms, round_trip_ms=rtt, obs_age_ms=(end-obs.stamp)*1000,
                            last_error=None, actions=[list(row) for row in actions],
                            latencies=(self.inference['latencies']+[sample])[-60:])
                        execute = self.execute
                    if execute:
                        self.phase = self.inference['phase'] = 'executing'
                        execution_started = clock.now()
                        fresh_after = self.client._execute(actions, source_observation=obs)
                        execution_ms = (clock.now()-execution_started)*1000
                    else:
                        execution_ms = 0.
                        fresh_after = clock.now()
                    last_sequence = obs.sequence
                    self.phase = self.inference['phase'] = 'monitoring'
                    timing = dict(source_sequence=obs.sequence,
                        sampling_profile=self.sampling.get('sampling_profile'), executed=bool(execute),
                        executed_steps=self.execute_steps if execute else 0,
                        observation_wait_ms=round((request_started-cycle_started)*1000, 2),
                        request_build_ms=round((request_built-request_started)*1000, 2),
                        server_inference_ms=round(ms, 2), round_trip_ms=round(rtt, 2),
                        execution_ms=round(execution_ms, 2),
                        cycle_ms=round((clock.now()-cycle_started)*1000, 2))
                    with self.lock:
                        self.inference['last_cycle'] = timing
                        if generation == self.generation:
                            self._recovery_deadline = None
                    LOG.info('Inference cycle %s', json.dumps(timing, allow_nan=False))
                except InterruptedError:
                    if getattr(self.hardware, 'armed', False):
                        self.hardware.slow_stop()
                    fresh_after = clock.now()
                except Exception as exc:
                    if not self._begin_recovery(exc, generation):
                        self._pause_failure(exc)
                    fresh_after, last_sequence = 0.0, -1
        except Exception as exc:
            self.event('error', '客户端已停止：' + str(exc))
            self.phase = 'error'
            self.inference.update(phase='error', last_error=str(exc))
        finally:
            with self.lock:
                self.execute = self.auto_infer = self.execution_available = False
                self._cancel_recovery()
            self._close_hardware()

    def status(self):
        with self.lock:
            hw, client = self.hardware, self.client
        sensors = dict(state=dict(hz=0, samples=0, age_ms=None, values=[], error='等待设备'),
                       force=dict(hz=0, samples=0, age_ms=None, values=[], kind='unknown', error='等待设备'), cameras=[])
        gripper = dict(ready=False, error=None, fault_code=None, clear_available=False,
                       clear_reason='夹爪通信及清错协议尚未验证')
        if hw is not None and hasattr(hw, 'gripper_diagnostics'):
            try:
                gripper = hw.gripper_diagnostics()
            except Exception as exc:
                gripper['error'] = str(exc)
        if hw is not None and hasattr(hw, 'diagnostics'):
            try:
                sensors = hw.diagnostics()
            except Exception as exc:
                sensors['state']['error'] = str(exc)
        elif hw is not None and client is not None:
            with client.collector.condition:
                history = list(client.collector.history)
            if history:
                obs = history[-1]
                age = (clock.now()-obs.stamp)*1000
                hz = (len(history)-1)/(history[-1].stamp-history[0].stamp) if len(history)>1 and age<200 else 0
                sensors = dict(state=dict(hz=hz, samples=obs.sequence+1, age_ms=age, values=list(obs.state), error=None),
                    force=dict(hz=hz, samples=obs.sequence+1, age_ms=age, values=list(obs.force), kind='模拟数据', error=None),
                    cameras=[dict(name=name, label=name, serial='mock', fps=hz, frame_number=obs.sequence,
                        age_ms=age, ready=age<200, error=None, width=hw.width, height=hw.height) for name in hw.camera_names])
        with self.lock:
            return deepcopy(dict(mode=self.mode, phase=self.phase, server=self.server, sensors=sensors,
                gripper=dict(gripper, busy=bool(self._hardware_busy or self._hardware_action),
                    operation=self._hardware_operation or self._hardware_action,
                    reconnecting=(self._hardware_operation or self._hardware_action) == 'reconnect_gripper',
                    clearing=(self._hardware_operation or self._hardware_action) in {'clear_gripper_error','diagnose_gripper'}),
                sampling=dict(self.sampling, busy=bool(self._sampling_busy or self._sampling_action)),
                inference=dict(self.inference, enabled=self.auto_infer),
                execution=dict(enabled=self.execute, available=self.execution_available, blocked_reason=self.blocked_reason),
                recovery=self.recovery,
                reset_pose=self.reset_pose_status(),
                events=list(self.events)))

    def jpeg(self, name):
        with self.lock:
            hw, client = self.hardware, self.client
        if name not in {'global','wrist'} or not hw:
            return None
        if hasattr(hw, 'frame'):
            frame = hw.frame(name)
        elif client:
            with client.collector.condition:
                obs = client.collector.history[-1] if client.collector.history else None
            frame = (obs.frames[name], hw.width, hw.height, obs.sequence) if obs else None
        else:
            frame = None
        if frame is None:
            return None
        rgb, width, height, counter = frame
        with self._jpeg_lock:
            cached = self._jpeg.get(name)
            if cached and cached[0] == counter:
                return cached[1]
            from PIL import Image
            buffer = BytesIO()
            Image.frombytes('RGB', (width,height), rgb).save(buffer, format='JPEG', quality=80)
            data = buffer.getvalue()
            self._jpeg[name] = (counter, data)
            return data

    def close(self):
        self.stop_event.set()
        with self.lock:
            self.generation += 1
            self.execute = self.auto_infer = False
            if self.client:
                self.client.stop_event.set()
        if self.hardware and getattr(self.hardware, 'armed', False):
            self.hardware.slow_stop()
        if self.thread:
            self.thread.join(timeout=self.transport.timeout_s+3)


class DashboardServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True
    def __init__(self, address, dashboard):
        if address[0] != '127.0.0.1':
            raise ValueError('Dashboard must bind to loopback only')
        self.dashboard = dashboard
        self.client_key = secrets.token_urlsafe(32)
        super().__init__(address, DashboardHandler)


class DashboardHandler(BaseHTTPRequestHandler):
    def log_message(self, *_):
        pass

    def _send(self, code, data, mime='application/json; charset=utf-8'):
        if not isinstance(data, bytes):
            data = json.dumps(data, ensure_ascii=False, allow_nan=False).encode('utf-8')
        self.send_response(code)
        self.send_header('Content-Type', mime)
        self.send_header('Content-Length', str(len(data)))
        self.send_header('Cache-Control', 'no-store')
        self.send_header('X-Content-Type-Options', 'nosniff')
        self.send_header('Content-Security-Policy', "default-src 'self'; img-src 'self' blob:; style-src 'self'; script-src 'self'; frame-ancestors 'none'")
        self.end_headers()
        try:
            self.wfile.write(data)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _local(self):
        port = self.server.server_address[1]
        return self.headers.get('Host') in {f'127.0.0.1:{port}', f'localhost:{port}'}

    def do_GET(self):
        if not self._local():
            return self._send(403, {'error':'Local host required'})
        path = urlsplit(self.path).path
        if path == '/api/status':
            return self._send(200, self.server.dashboard.status())
        if path in {'/api/camera/global.jpg', '/api/camera/wrist.jpg'}:
            data = self.server.dashboard.jpeg(path.rsplit('/',1)[1][:-4])
            return self._send(200, data, 'image/jpeg') if data else self._send(503, {'error':'等待相机画面'})
        names = {'/':('index.html','text/html; charset=utf-8'),
                 '/app.js':('app.js','text/javascript; charset=utf-8'),
                 '/style.css':('style.css','text/css; charset=utf-8')}
        if path not in names:
            return self._send(404, {'error':'Not found'})
        filename, mime = names[path]
        data = (WEB_ROOT/filename).read_bytes()
        if path == '/':
            data = data.replace(b'__CLIENT_KEY__', self.server.client_key.encode('ascii'))
        self._send(200, data, mime)

    def do_POST(self):
        origin = self.headers.get('Origin')
        if (not self._local() or (origin is not None and origin != 'http://'+self.headers.get('Host',''))
                or not secrets.compare_digest(self.headers.get('X-Client-Key',''), self.server.client_key)):
            return self._send(403, {'error':'Local dashboard authorization required'})
        if self.path != '/api/control' or self.headers.get_content_type() != 'application/json':
            return self._send(400, {'error':'Expected JSON control request'})
        try:
            length = int(self.headers.get('Content-Length','0'))
            if not 0 < length <= 4096:
                raise ValueError('Invalid request size')
            self.connection.settimeout(3)
            body = json.loads(self.rfile.read(length))
            result = self.server.dashboard.control(body['action'], sampling_profile=body.get('sampling_profile'))
            self._send(200, dict(ok=True, **(result or {})))
        except (ValueError, KeyError, TypeError, RuntimeError, OSError) as exc:
            self._send(409, {'error':str(exc)})


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--server', default='http://127.0.0.1:8766')
    parser.add_argument('--port', type=int, default=8767)
    parser.add_argument('--mode', choices=('realman','mock'), default='realman')
    parser.add_argument('--hardware-config', type=Path)
    parser.add_argument('--token-file', type=Path)
    parser.add_argument('--no-auto-infer', action='store_true')
    parser.add_argument('--log-file', type=Path, help='Local rotating UTF-8 runtime log')
    parser.add_argument('--execute-steps', type=int, default=32)
    add_motion_arguments(parser)
    args = parser.parse_args(argv)
    if args.mode == 'realman' and not args.hardware_config:
        parser.error('realman mode requires --hardware-config')
    if not 1 <= args.execute_steps <= 32:
        parser.error('--execute-steps must be in 1..32')
    handlers = [logging.StreamHandler()]
    if args.log_file:
        from logging.handlers import RotatingFileHandler
        args.log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(RotatingFileHandler(args.log_file, maxBytes=20*1024*1024,
                                           backupCount=3, encoding='utf-8'))
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s', handlers=handlers)
    token = args.token_file.read_text(encoding='utf-8').strip() if args.token_file else None
    transport = InferenceHTTPClient(args.server, timeout_s=30, token=token)
    dashboard = Dashboard(transport, mode=args.mode, hardware_config=args.hardware_config,
                          auto_infer=not args.no_auto_infer, execute_steps=args.execute_steps,
                          motion_settings=motion_options(args))
    server = DashboardServer(('127.0.0.1',args.port), dashboard)
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
    dashboard.start()
    LOG.info('Client dashboard http://127.0.0.1:%s mode=%s automatic_inference=%s execution=off',
             server.server_address[1], args.mode, not args.no_auto_infer)
    try:
        server.serve_forever(poll_interval=.1)
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        dashboard.close()


if __name__ == '__main__':
    main()
