"""Asynchronous CTAG2F120 end-effector adapter over RealMan Modbus RTU.

All RTU traffic stays on one dedicated worker thread/transport so a slow tool
bus cannot stall the 30 Hz robot motion loop or web controls.  Sensor feedback
is timestamped and kept in a bounded ring; commands are latest-only and rate
limited.  Setup never changes the gripper target or fires its trigger.
"""
from __future__ import annotations

from collections import deque
from concurrent.futures import Future, TimeoutError as FutureTimeout
import json
import logging
import math
from queue import Empty, SimpleQueue
import threading
import time
from . import clock
from typing import Any, Mapping


logger = logging.getLogger(__name__)


class FeedbackUnavailable(RuntimeError):
    """A freshness gate failed without evidence of a device/control fault."""


def _signed32(high: int, low: int) -> int:
    value = (int(high) << 16) | int(low)
    return value - (1 << 32) if value & (1 << 31) else value


class Ctag2F120Worker:
    def __init__(self, config: Mapping[str, Any]):
        self.config = dict(config)
        request_gap = self.config.get('gripper_request_gap_s', 0.)
        if (isinstance(request_gap, bool) or not isinstance(request_gap, (int, float))
                or not math.isfinite(request_gap) or not 0 <= request_gap <= .05):
            raise ValueError('gripper_request_gap_s must be finite and in [0, 0.05] seconds')
        self.config['gripper_request_gap_s'] = float(request_gap)
        for key, default in (("gripper_deadband", 0.02), ("gripper_close_deadband", 0.002),
                             ("max_gripper_step", 0.15)):
            value = self.config.get(key, default)
            if isinstance(value, bool) or not math.isfinite(float(value)) or not 0 <= float(value) <= 1:
                raise ValueError(f"{key} must be finite and in [0, 1]")
            if key == "max_gripper_step" and float(value) == 0:
                raise ValueError("max_gripper_step must be positive")
            self.config[key] = float(value)
        self.lock = threading.Lock()
        self.stop = threading.Event()
        self.wake = threading.Event()
        self.thread: threading.Thread | None = None
        self.state: dict[str, Any] = {"ready": False, "error": "夹爪初始化中"}
        self.history: deque[tuple[float, float]] = deque(maxlen=96)
        self.history_started: deque[tuple[float, float]] = deque(maxlen=96)
        self.desired: float | None = None
        self.mapping_min: int | None = None
        self.mapping_max: int | None = None
        self.requests = SimpleQueue()
        self.command_sequence = 0
        self.cancel_generation = 0
        self.command_source = None
        self.sent_target = None
        self.ramp_target: float | None = None
        self._target_inflight = None
        self.transport = None
        self.fault_generation = 0
        self.hard_fault_generation = 0
        self._confirmed_enabled = self._confirmed_alarm = None
        self.enabled_received = None
        self.enabled_value = None
        self._feedback_read_costs = deque(maxlen=64)
        self._target_write_costs = deque(maxlen=32)
        self._deadline_deferrals = 0
        self._deadline_reason = None
        self.manual_transition = None
        self._manual_queued_ticket = None
        self._manual_queued_sequence = None
        self.enable_poll_s = float(self.config.get('gripper_enable_poll_s', .25))
        if not math.isfinite(self.enable_poll_s) or not 0 < self.enable_poll_s <= .25:
            raise ValueError('gripper_enable_poll_s must be in (0, .25]')
        if self.config.get('gripper_transport', 'sdk') not in ('sdk', 'json'):
            raise ValueError('gripper_transport must be sdk or json')
        mode = self.config.get('gripper_trigger_mode', 'batch')
        for key, value in (('gripper_trigger_mode', mode),
                           ('gripper_manual_trigger_mode', self.config.get('gripper_manual_trigger_mode', mode))):
            if value not in ('batch', 'separate'):
                raise ValueError(f'{key} must be batch or separate')

    def start(self) -> None:
        if self.thread is not None:
            return
        self.stop.clear()
        self.thread = threading.Thread(target=self._run, name="ctag2f120-modbus", daemon=True)
        self.thread.start()

    def wait_ready(self, timeout_s: float = 8.0) -> dict[str, Any]:
        deadline = clock.now() + timeout_s
        while clock.now() < deadline:
            state = self.status()
            if state.get("ready"):
                return state
            if self.thread is not None and not self.thread.is_alive():
                raise RuntimeError(state.get("error") or "CTAG2F120 worker exited")
            time.sleep(0.02)
        state = self.status()
        raise TimeoutError(state.get("error") or "CTAG2F120 initialization timed out")

    def wait_initialized(self, timeout_s: float = 8.0) -> dict[str, Any]:
        deadline = clock.now() + timeout_s
        while clock.now() < deadline:
            state = self.status()
            if state.get("initialized"):
                return state
            if self.thread is not None and not self.thread.is_alive():
                raise RuntimeError(state.get("error") or "夹爪连接失败")
            time.sleep(0.02)
        raise RuntimeError(self.status().get("error") or "夹爪初始化超时")

    def _params(self, factory, address: int, count: int = 1):
        return factory(1, address, int(self.config.get("gripper_device", 1)), count)

    @staticmethod
    def _code(result: Any, method: str) -> Any:
        if isinstance(result, dict):
            code = result.get("return_code")
            payload = result
        elif isinstance(result, tuple):
            code = result[0]
            payload = result[1] if len(result) > 1 else None
        else:
            code, payload = result, None
        if type(code) is not int or code != 0:
            raise RuntimeError(f"{method} 失败: {result!r}")
        return payload

    def _read(self, sdk, factory, address: int, count: int) -> list[int]:
        if count == 1:
            result = sdk.rm_read_holding_registers(self._params(factory, address, count))
            payload = self._code(result, f"读取夹爪寄存器 0x{address:04x}")
            if type(payload) is not int:
                raise RuntimeError(f"夹爪寄存器 0x{address:04x} 返回类型错误")
            return [payload & 0xffff]
        payload = self._code(sdk.rm_read_multiple_holding_registers(
            self._params(factory, address, count)), f"读取夹爪寄存器 0x{address:04x}")
        if not isinstance(payload, list) or len(payload) != count * 2:
            raise RuntimeError(f"夹爪寄存器 0x{address:04x} 返回长度错误")
        raw = [int(value) & 0xff for value in payload]
        return [raw[i] << 8 | raw[i + 1] for i in range(0, len(raw), 2)]

    def _write(self, sdk, factory, address: int, values: list[int]) -> None:
        if not values or any(type(value) is not int or not 0 <= value <= 65535 for value in values):
            raise ValueError("夹爪寄存器写入值无效")
        raw = [byte for value in values for byte in value.to_bytes(2, "big")]
        params = self._params(factory, address, len(values))
        self._code(sdk.rm_write_registers(params, raw), f"写入夹爪寄存器 0x{address:04x}")

    def _setup(self, sdk, factory) -> tuple[int, int, float]:
        verify_arm_state(sdk)
        voltage = sdk.rm_get_tool_voltage()
        current_voltage = self._code(voltage, "读取末端供电")
        if current_voltage != 3:
            self._code(sdk.rm_set_tool_voltage(3), "设置末端 24V")
            time.sleep(0.5)
        bus = self._code(sdk.rm_get_tool_rs485_mode(), "读取末端 RS485 模式")
        if bus.get("mode") != 1 or bus.get("baudrate") != 115200 or bus.get("timeout") != 2:
            self._code(sdk.rm_set_modbus_mode(1, 115200, 2), "设置末端 Modbus RTU")
            time.sleep(0.3)

        alarm = self._read(sdk, factory, 0x0610, 3)[2]
        if alarm != 0:
            raise RuntimeError(f"夹爪报警 0x{alarm:04x}；未使能或发送目标")
        mapping = self._read(sdk, factory, 0x0404, 4)
        actuator = self._read(sdk, factory, 0x0301, 4)
        lower, upper = _signed32(mapping[0], mapping[1]), _signed32(mapping[2], mapping[3])
        actuator_lower, actuator_upper = _signed32(actuator[0], actuator[1]), _signed32(actuator[2], actuator[3])
        if lower >= upper or actuator_lower >= actuator_upper:
            raise RuntimeError("夹爪实机行程映射无效")

        speed = int(self.config.get("gripper_speed", 30))
        torque = int(self.config.get("gripper_torque", 50))
        if not 1 <= speed <= 100 or not 1 <= torque <= 100:
            raise ValueError("gripper_speed / gripper_torque 必须在 1..100")
        params = self._read(sdk, factory, 0x0104, 4)
        expected = [speed, torque, 100, 100]
        if params != expected:
            self._write(sdk, factory, 0x0104, expected)
            if self._read(sdk, factory, 0x0104, 4) != expected:
                raise RuntimeError("夹爪运动参数回读不一致")
        position = self._read(sdk, factory, 0x060D, 3)
        raw_position = _signed32(position[0], position[1])
        opening = 1.0 - (raw_position - lower) / (upper - lower)
        if not math.isfinite(opening) or not -0.02 <= opening <= 1.02:
            raise RuntimeError("夹爪反馈位置超出已保存的开合映射")
        opening = max(0.0, min(1.0, opening))
        return lower, upper, opening

    def _set_enabled(self, sdk, factory, enabled: bool, ticket=None) -> None:
        if ticket is None:
            # Compatibility for direct/internal calls. Public requests reserve
            # this ticket atomically when queued, before any blocking work.
            self.hold()
            with self.lock:
                ticket = self._control_ticket_locked()
        if enabled and self.config.get('gripper_manual_release', False):
            self._safe_manual_enable(sdk, factory, ticket)
            return
        # Enable is a single register; never touch the target/trigger block.
        old_guard = self.transport.control_guard if self.transport is not None else None
        try:
            if self.transport is not None:
                self.transport.control_guard = lambda: self._control_valid(ticket)
            self._require_control(ticket)
            if enabled:
                verify_arm_state(sdk)
                self._require_control(ticket)
                if self._read(sdk, factory, 0x0610, 3)[2] != 0:
                    raise RuntimeError("夹爪有报警，禁止使能")
            self._require_control(ticket)
            self._code(sdk.rm_write_single_register(
                self._params(factory, 0x0100), int(enabled)), "设置夹爪使能")
            read_started = clock.now()
            if self._read(sdk, factory, 0x0100, 1)[0] != int(enabled):
                raise RuntimeError("夹爪使能状态回读不一致")
            with self.lock:
                self._observe_device_locked(enabled=enabled)
                self.enabled_received, self.enabled_value = read_started, enabled
                fresh = (0 <= clock.now()-read_started <= .5 and
                         0 <= clock.now()-self.state.get('measurement_started', self.state.get('received', 0)) <= .5)
                ready = bool(enabled and fresh and self.state.get('alarm') == 0
                             and not self.state.get('communication_fault'))
                self.state.update(enabled=enabled, enabled_received=read_started, ready=ready,
                    error=None if ready else '夹爪未使能' if not enabled else '等待新鲜位置反馈')
        finally:
            if self.transport is not None:
                self.transport.control_guard = old_guard

    def _feedback_failed(self, error):
        """A successful later read restores monitoring, never an old target."""
        from .gripper_transport import transport_interruption
        with self.lock:
            self.desired = self.ramp_target = None
            self.command_sequence += 1
            self.fault_generation += 1
            if not (isinstance(error, (TimeoutError, FeedbackUnavailable)) or transport_interruption(error)):
                # Unknown parser, mapping, setup, device and control errors
                # cannot become an automatic-recovery permit after a good read.
                self.hard_fault_generation += 1
            self.state.update(ready=False, error=str(error), communication_fault=True)
            self.enabled_received = self.enabled_value = None

    def _observe_device_locked(self, *, enabled=None, alarm=None):
        # Keep these confirmed facts across communication cache invalidation.
        # A disable/alarm seen even once invalidates the execution session.
        if enabled is not None:
            if self._confirmed_enabled is True and enabled is False:
                self.hard_fault_generation += 1
            self._confirmed_enabled = enabled
        if alarm is not None:
            if alarm and alarm != self._confirmed_alarm:
                self.hard_fault_generation += 1
            self._confirmed_alarm = alarm

    def set_enabled(self, enabled: bool, timeout_s: float = 3.0, *, expected_cancel_generation=None) -> dict[str, Any]:
        if type(enabled) is not bool:
            raise ValueError("enabled 必须是布尔值")
        return self._queue_manual_control(enabled, timeout_s,
            expected_cancel_generation=expected_cancel_generation)

    def _require_expected_cancel_locked(self, expected):
        if expected is not None and (type(expected) is not int or expected != self.cancel_generation):
            from .gripper_transport import MotionCancelled
            raise MotionCancelled('夹爪手动请求已被停止或更新操作取消；未入队')

    def _control_ticket_locked(self):
        return self.command_sequence, self.cancel_generation, self.fault_generation

    def _control_valid(self, ticket):
        with self.lock:
            return (not self.stop.is_set() and self.requests.empty()
                    and self._control_ticket_locked() == ticket)

    def _control_fresh(self, enabled=None):
        with self.lock:
            now = clock.now()
            return (self.state.get('alarm') == 0 and not self.state.get('communication_fault')
                and self.enabled_received is not None
                and 0 <= now-self.enabled_received <= .5
                and 0 <= now-self.state.get('measurement_started', -math.inf) <= .5
                and (enabled is None or self.enabled_value is enabled))

    def _require_control(self, ticket, *, fresh=False, enabled=None):
        from .gripper_transport import MotionCancelled
        if not self._control_valid(ticket):
            raise MotionCancelled('夹爪手动准备已取消或被新操作替换')
        if fresh and not self._control_fresh(enabled):
            raise RuntimeError('夹爪手动准备需要500ms内真实位置、使能反馈且无报警')

    def _transition_enabled_read(self, sdk, factory):
        started = clock.now()
        word = self._read(sdk, factory, 0x0100, 1)[0]
        if word not in (0, 1):
            raise RuntimeError('夹爪使能反馈不是0或1')
        with self.lock:
            self._observe_device_locked(enabled=bool(word))
            self.enabled_received, self.enabled_value = started, bool(word)
            self.state.update(enabled=bool(word), enabled_received=started, ready=False)
        return bool(word)

    def _transition_position_read(self, sdk, factory):
        """Publish only a real reply; intentional disable is not a new fault."""
        started = clock.now()
        block = self._read(sdk, factory, 0x060D, 6)
        raw = _signed32(*block[:2])
        opening = 1-(raw-self.mapping_min)/(self.mapping_max-self.mapping_min)
        if not math.isfinite(opening) or not -.02 <= opening <= 1.02:
            raise RuntimeError('夹爪手动准备的位置反馈越过行程映射')
        opening = min(1., max(0., opening))
        completed = clock.now()
        with self.lock:
            self._observe_device_locked(alarm=block[5])
            self.history.append((completed, opening))
            self.history_started.append((completed, started))
            fresh = (0 <= completed-started <= .5 and self.enabled_received is not None
                     and 0 <= completed-self.enabled_received <= .5)
            self.state.update(initialized=True, ready=False, enabled=self.enabled_value,
                opening=opening, position=raw, alarm=block[5], received=completed,
                response_received=completed, measurement_started=started, read_started=started,
                enabled_received=self.enabled_received, communication_fault=not fresh,
                mapping_min=self.mapping_min, mapping_max=self.mapping_max,
                read_duration_ms=round((completed-started)*1000, 2))
        if block[5]:
            raise RuntimeError(f'夹爪报警 0x{block[5]:04x}；手动准备已停止')
        return raw, opening

    def _transition_enable_write(self, sdk, factory, enabled, ticket):
        self._require_control(ticket, fresh=True, enabled=not enabled)
        self._code(sdk.rm_write_single_register(self._params(factory, 0x0100), int(enabled)),
                   '设置夹爪手动准备使能')
        # Always observe a possibly applied write before evaluating cancellation.
        if self._transition_enabled_read(sdk, factory) is not enabled:
            raise RuntimeError('夹爪手动准备使能状态回读不一致')
        self._require_control(ticket)

    def _neutralize_target(self, sdk, factory, ticket):
        raw, _ = self._transition_position_read(sdk, factory)
        self._require_control(ticket, fresh=True, enabled=False)
        bits = raw & 0xffffffff
        self._write(sdk, factory, 0x0102, [bits >> 16, bits & 65535])
        self._require_control(ticket)
        # RealMan's multi-read accepts 3..12 registers; compare only target words.
        target = self._read(sdk, factory, 0x0102, 3)
        if _signed32(*target[:2]) != raw:
            raise RuntimeError('夹爪遗留目标中和读回不一致；保持失能')
        self._require_control(ticket, fresh=True, enabled=False)

    def _cleanup_transition_disable(self, sdk, factory):
        """A cancellation after enable dispatch must not leave it active.

        Only the idempotent disable is attempted; never repeat an enable or
        target with an uncertain acknowledgement. Read first to fence late I/O.
        """
        if self.transport is not None:
            self.transport.control_guard = None
        try:
            if self._transition_enabled_read(sdk, factory) is False:
                return
        except Exception:
            pass
        self._code(sdk.rm_write_single_register(self._params(factory, 0x0100), 0),
                   '取消手动准备后失能')
        if self._transition_enabled_read(sdk, factory) is not False:
            raise RuntimeError('手动准备取消后未确认失能')

    def _manual_prepare(self, sdk, factory, ticket, *, opening):
        """Explicit manual operation only; policy target dispatch never enters here."""
        self._require_control(ticket)
        with self.lock:
            self.manual_transition = '检查力矩保持' if opening else '准备安全使能'
        old_guard = self.transport.control_guard if self.transport is not None else None
        needs_stop, released = False, False
        try:
            if self.transport is not None:
                self.transport.control_guard = lambda: (
                    self._control_valid(ticket) and self._control_fresh())
            verify_arm_state(sdk)
            self._require_control(ticket)
            enabled = self._transition_enabled_read(sdk, factory)
            self._require_control(ticket)
            self._transition_position_read(sdk, factory)
            self._require_control(ticket, fresh=True, enabled=enabled)
            if opening:
                if not enabled:
                    raise RuntimeError('手动张开需要先明确使能夹爪')
                reached = self._read(sdk, factory, 0x0601, 1)[0]
                if reached not in (0, 1):
                    raise RuntimeError('夹爪力矩到达标志无效')
                self._require_control(ticket, fresh=True, enabled=True)
                with self.lock:
                    self.state['torque_reached'] = bool(reached)
                if reached:
                    with self.lock:
                        self.manual_transition = '确认退出力矩保持'
                    needs_stop = True
                    self._transition_enable_write(sdk, factory, False, ticket)
                    enabled, released = False, True
            if not enabled:
                with self.lock:
                    self.manual_transition = '实测位置中和遗留目标'
                self._neutralize_target(sdk, factory, ticket)
                with self.lock:
                    self.manual_transition = '确认安全使能'
                needs_stop = True
                self._transition_enable_write(sdk, factory, True, ticket)
            self._require_control(ticket)
            # Refresh both measurements after the transition. A write ACK is
            # never a sensor sample, and all history retains reply causality.
            self._transition_enabled_read(sdk, factory)
            self._require_control(ticket)
            self._transition_position_read(sdk, factory)
            self._require_control(ticket, fresh=True, enabled=True)
            with self.lock:
                if self._control_ticket_locked() != ticket or self.stop.is_set() or not self.requests.empty():
                    from .gripper_transport import MotionCancelled
                    raise MotionCancelled('夹爪手动准备完成前已取消')
                self.state.update(ready=True, error=None, communication_fault=False)
                if opening:
                    # Start a NEW intent from measured position. Never restore
                    # a target cancelled by hold() when this request was queued.
                    self.desired, self.ramp_target = 1., None
                    self.command_sequence += 1
                    self.command_source = 'manual_open'
                    self._manual_queued_ticket = ticket
                    self._manual_queued_sequence = self.command_sequence
            needs_stop = False
        except Exception as exc:
            if needs_stop:
                try:
                    self._cleanup_transition_disable(sdk, factory)
                except Exception as cleanup:
                    raise RuntimeError(f'{exc}；停止未确认：{cleanup}') from exc
            raise
        finally:
            if self.transport is not None:
                self.transport.control_guard = old_guard
            with self.lock:
                self.manual_transition = None
        result = self.status()
        if opening:
            result.update(manual_open_prepared=True, manual_release_performed=released,
                          queued=True, acknowledged=False, moving=False, completed=False)
            self.wake.set()
        return result

    def _safe_manual_enable(self, sdk, factory, ticket):
        return self._manual_prepare(sdk, factory, ticket, opening=False)

    def _manual_open(self, sdk, factory, ticket):
        return self._manual_prepare(sdk, factory, ticket, opening=True)

    def _queue_manual_control(self, operation, timeout_s, *, expected_cancel_generation=None):
        future = Future()
        with self.lock:
            self._require_expected_cancel_locked(expected_cancel_generation)
            if self.stop.is_set() or not self.thread or not self.thread.is_alive():
                raise RuntimeError('夹爪通信线程未连接')
            # Cancel old policy/manual targets atomically with reserving the
            # request. New user intent must invalidate this ticket, not revive it.
            self.desired = self.ramp_target = None
            self.command_sequence += 1
            self.cancel_generation += 1
            ticket = self._control_ticket_locked()
            self.requests.put((operation, future, ticket))
        self.wake.set()
        try:
            return future.result(timeout=timeout_s)
        except FutureTimeout as exc:
            future.cancel()
            with self.lock:
                own = (self._control_ticket_locked() == ticket or
                    (self._manual_queued_ticket == ticket
                     and self.command_sequence == self._manual_queued_sequence
                     and self.cancel_generation == ticket[1]))
                if own:
                    self.desired = self.ramp_target = None
                    self.command_sequence += 1
                    self.cancel_generation += 1
            self.wake.set()
            raise RuntimeError('夹爪手动准备超时并已取消；请查看实际使能状态') from exc

    def manual_open(self, timeout_s: float = 6.0, *, expected_cancel_generation=None) -> dict[str, Any]:
        if not self.config.get('gripper_manual_release', False):
            raise RuntimeError('未启用夹爪手动力矩保持释放')
        if not self.status().get('ready'):
            raise RuntimeError('夹爪未就绪，不能准备手动张开')
        return self._queue_manual_control('manual_open', timeout_s,
            expected_cancel_generation=expected_cancel_generation)

    def _raw_target(self, opening: float) -> int:
        lower, upper = self.mapping_min, self.mapping_max
        return round(lower + (1.0 - opening) * (upper - lower))

    def _next_target(self, commanded: float):
        """Filter against the last sent target, never the contact position.

        Closing corrections use a smaller deadband. Exact endpoints and the
        remaining part of an accepted ramp must reach their requested target;
        register equality prevents redundant writes. No endpoint snapping.
        """
        with self.lock:
            requested = self.desired
            if requested is None:
                return None
            delta = requested - commanded
            deadband = self.config["gripper_deadband"]
            if delta < 0:
                deadband = min(deadband, self.config["gripper_close_deadband"])
            continuing = self.ramp_target == requested
            if abs(delta) < deadband and requested not in (0.0, 1.0) and not continuing:
                return None
            step = self.config["max_gripper_step"]
            bounded = requested if abs(delta) <= step else commanded + math.copysign(step, delta)
            if self._raw_target(bounded) == self._raw_target(commanded):
                return None
            return requested, bounded, self.command_sequence, self.command_source

    def _send_target(self, sdk, factory, requested, bounded, previous, sequence, source):
        mode = self.config.get('gripper_trigger_mode', 'batch')
        if mode not in ('batch', 'separate'):
            raise ValueError('gripper_trigger_mode must be batch or separate')
        manual_mode = self.config.get('gripper_manual_trigger_mode', mode)
        if manual_mode not in ('batch', 'separate'):
            raise ValueError('gripper_manual_trigger_mode must be batch or separate')
        if source in ('manual', 'manual_open'):
            mode = manual_mode
        raw_target = self._raw_target(bounded)
        payload = list(raw_target.to_bytes(4, "big", signed=True))
        payload.extend([byte for item in (int(self.config.get("gripper_speed", 30)),
            int(self.config.get("gripper_torque", 50)), 100, 100, 1)
            for byte in item.to_bytes(2, "big")])
        params = self._params(factory, 0x0102, 6 if mode == 'separate' else 7)
        with self.lock:
            if (self.stop.is_set() or not self.requests.empty() or self.desired != requested
                    or self.command_sequence != sequence):
                return None
            # This short gate is the dispatch boundary. hold/close/enable
            # queueing use the same lock. Once marked in-flight the SDK call
            # cannot be retracted; never hold the lock across that blocking I/O.
            self._target_inflight = dict(sequence=sequence, opening=bounded, source=source)
        try:
            def still_wanted():
                healthy = self.status().get('ready', False)
                with self.lock:
                    return (healthy and not self.stop.is_set() and self.requests.empty()
                        and self.desired == requested and self.command_sequence == sequence)

            def require_current():
                if not still_wanted():
                    from .gripper_transport import MotionCancelled
                    raise MotionCancelled('Gripper target cancelled between separate trigger stages')

            if self.transport is not None:
                # The transport may drain a late response before sending. It
                # rechecks intent and BOTH real measurement ages after that.
                self.transport.motion_guard = still_wanted
            if mode == 'separate':
                # Temporary target and parameters only: 0x0108 is deliberately
                # excluded. Neither uncertain ACKs nor failed verification may
                # advance to the trigger, and no stage is retried/fallen back.
                prepared = payload[:-2]
                require_current()
                self._code(sdk.rm_write_registers(params, prepared), '写入夹爪目标和临时参数')
                require_current()
                readback = self._read(sdk, factory, 0x0102, 6)
                require_current()
                expected = [prepared[i] << 8 | prepared[i+1] for i in range(0, len(prepared), 2)]
                if readback != expected:
                    raise RuntimeError('夹爪目标或临时参数回读不一致；未触发运动')
                self._code(sdk.rm_write_single_register(self._params(factory, 0x0108), 1),
                           '触发夹爪位置运动')
            else:
                self._code(sdk.rm_write_registers(params, payload), "下发夹爪位置")
            completed = clock.now()
            with self.lock:
                measured = self.state.get('opening')
                # A hold or a newer target while the SDK write is in flight
                # must not be revived by its acknowledgement.
                self.ramp_target = (requested if self.desired == requested
                    and self.command_sequence == sequence and bounded != requested else None)
                self.sent_target = dict(sequence=sequence, source=source, requested_opening=requested,
                    opening=bounded, previous_opening=previous, raw_target=raw_target,
                    acknowledged_at=completed, direction='open' if bounded > previous else 'close',
                    trigger_mode=mode)
                trace = dict(self.sent_target, measured_opening=measured)
        except Exception as exc:
            from .gripper_transport import MotionCancelled
            if isinstance(exc, MotionCancelled):
                stale = not self.status().get('ready', False)
                with self.lock:
                    # Cancel this generation only. A replacement target must
                    # survive cancellation of an older dispatch attempt.
                    if self.command_sequence == sequence:
                        self.desired = self.ramp_target = None
                        self.command_sequence += 1
                        if stale:
                            self.fault_generation += 1
                return None
            raise
        finally:
            if self.transport is not None:
                self.transport.motion_guard = None
            with self.lock:
                self._target_inflight = None
        logger.info('Gripper target %s', json.dumps(trace, allow_nan=False))
        return completed

    def _read_budget(self):
        # A planning estimate, never permission to ignore actual measurement
        # ages. P80 avoids a single slow reply starving all later targets.
        with self.lock:
            values = sorted(self._feedback_read_costs)
        # Measured call costs already include pacing; only its floor needs the
        # configured gap added, or repeated measurements double-count it.
        return max(.075+self._request_gap(), values[min(len(values)-1, int(.8*len(values)))] if values else 0.)

    def _request_gap(self):
        return self.config['gripper_request_gap_s'] if self.config.get('gripper_transport', 'sdk') == 'json' else 0.

    def _enable_refresh_interval(self):
        if self.config.get('gripper_transport', 'sdk') != 'json':
            return self.enable_poll_s
        # Leave room for one bounded position request and the following enable
        # read. Waiting until 250 ms at poll entry left no such reserve.
        timeout = float(self.config.get('gripper_timeout_s', .3))
        return min(self.enable_poll_s, max(.001, .5-timeout-self._request_gap()-self._read_budget()-.020))

    def _motion_has_feedback_budget(self, source):
        """Defer a write if it would crowd out the next real feedback reads."""
        if self.config.get('gripper_transport', 'sdk') != 'json':
            return True
        read = self._read_budget()
        with self.lock:
            writes = sorted(self._target_write_costs)
        write = max(.085+self._request_gap(), writes[min(len(writes)-1, int(.8*len(writes)))] if writes else 0.)
        mode = self.config.get('gripper_trigger_mode', 'batch')
        if source in ('manual', 'manual_open'):
            mode = self.config.get('gripper_manual_trigger_mode', mode)
        # Separate mode has target write + verification read + trigger write.
        dispatch = write if mode == 'batch' else 2*write+read
        now = clock.now()
        with self.lock:
            position_age = now-self.state.get('measurement_started', self.state.get('received', 0.))
            enable_age = now-self.enabled_received if self.enabled_received is not None else math.inf
            fits = (position_age+dispatch+2*read+.020 < .5
                    and enable_age+dispatch+read+.020 < .5)
            self._deadline_reason = None if fits else '先刷新反馈：目标写入会挤占下一次位置或使能读取'
            if not fits:
                self._deadline_deferrals += 1
        return fits

    def _poll_feedback(self, sdk, factory, lower, upper, motion_due=False):
        """Read position/alarm fast and enable on its own bounded slower clock.

        0x060D/0x060E contain signed position; 0x0612 is the alarm at
        offset 5. Only a successful position read supplies a history sample;
        its request start bounds age, since there is no device timestamp.
        """
        read_started = clock.now()
        if (self.enabled_received is None or
                read_started-self.enabled_received >= self._enable_refresh_interval() or
                (motion_due and self.config.get('gripper_transport', 'sdk') == 'json')):
            enabled_started = clock.now()
            enabled_word = self._read(sdk, factory, 0x0100, 1)[0]
            with self.lock:
                self._feedback_read_costs.append(clock.now()-enabled_started)
            if self.stop.is_set():
                return None
            if enabled_word not in (0, 1):
                raise RuntimeError('夹爪使能反馈不是 0 或 1')
            with self.lock:
                self._observe_device_locked(enabled=enabled_word == 1)
                self.enabled_received, self.enabled_value = enabled_started, enabled_word == 1
            if not self.enabled_value:
                self.hold()
        position_started = clock.now()
        block = self._read(sdk, factory, 0x060D, 6)
        with self.lock:
            self._feedback_read_costs.append(clock.now()-position_started)
        if self.stop.is_set():
            return None
        enabled, alarm = self.enabled_value, block[5]
        raw_position = _signed32(block[0], block[1])
        opening = 1.0 - (raw_position - lower) / (upper - lower)
        if not math.isfinite(opening) or not -0.02 <= opening <= 1.02:
            raise RuntimeError('夹爪位置反馈越过行程映射')
        opening = max(0.0, min(1.0, opening))
        completed = clock.now()
        # The tool has no hardware timestamp. Request start bounds sample age
        # conservatively: a delayed response must never look age zero.
        received = position_started
        with self.lock:
            self._observe_device_locked(alarm=alarm)
            # Arrival time is the causal observation boundary. The separate
            # request time is an age upper bound, never permission to backdate
            # a newly received measurement into an earlier observation anchor.
            self.history.append((completed, opening))
            self.history_started.append((completed, received))
            fresh = (0 <= completed-received <= .5 and
                     self.enabled_received is not None and 0 <= completed-self.enabled_received <= .5)
            ready = enabled and alarm == 0 and fresh
            if ((not fresh and not self.state.get('communication_fault')) or
                    (self.state.get('enabled') is True and not enabled) or
                    (alarm and not self.state.get('alarm'))):
                self.fault_generation += 1
                self.command_sequence += 1
            self.state = dict(ready=ready, initialized=True, enabled=enabled, opening=opening,
                position=raw_position, mapping_min=lower, mapping_max=upper,
                received=completed, measurement_started=received, read_started=read_started,
                response_received=completed, enabled_received=self.enabled_received,
                read_duration_ms=round((completed-position_started)*1000, 2), alarm=alarm,
                communication_fault=not fresh,
                error=None if ready else (f'夹爪报警 0x{alarm:04x}' if alarm else
                    '夹爪反馈过期' if not fresh else '夹爪未使能，请点击夹爪使能'))
            if not ready:
                self.desired = self.ramp_target = None
            trace = dict(stamp=received, read_started=read_started, opening=opening,
                position=raw_position, enabled=enabled, alarm=alarm,
                requested_opening=self.desired, last_sent=self.sent_target)
        if self.config.get('log_gripper_feedback', False):
            logger.info('Gripper feedback %s', json.dumps(trace, allow_nan=False))
        return opening, ready, read_started

    def _run(self) -> None:
        sdk = None
        try:
            from Robotic_Arm.rm_robot_interface import (
                rm_peripheral_read_write_params_t,
            )
            from .sdk import create_sdk
            sdk = create_sdk()
            # rm_set_timeout is process-global, not scoped to this handle.
            # Runtime JSON I/O below uses its own bounded socket deadline.
            handle = sdk.rm_create_robot_arm(str(self.config["robot_ip"]),
                                             int(self.config["robot_port"]), 3)
            if handle is None or int(getattr(handle, "id", 0)) <= 0:
                raise RuntimeError("CTAG2F120 独立 SDK 连接失败")
            lower, upper, _ = self._setup(sdk, rm_peripheral_read_write_params_t)
            self.mapping_min, self.mapping_max = lower, upper
            io = sdk
            if self.config.get('gripper_transport', 'sdk') == 'json':
                from .gripper_transport import JsonModbusArm
                self.transport = JsonModbusArm(str(self.config['robot_ip']),
                    int(self.config['robot_port']),
                    timeout_s=float(self.config.get('gripper_timeout_s', .3)),
                    request_gap_s=self.config['gripper_request_gap_s'],
                    setup_sdk=sdk, trace_path=self.config.get('gripper_trace_path'))
                io = self.transport
            period = float(self.config.get("gripper_poll_s", 0.10))
            min_command_interval = float(self.config.get("gripper_command_interval_s", 0.10))
            if (not math.isfinite(period) or period <= 0 or
                    not math.isfinite(min_command_interval) or min_command_interval <= 0):
                raise ValueError('夹爪轮询与命令间隔必须是正有限数')
            startup_deadline = clock.now()+float(self.config.get('gripper_start_timeout_s', 8.))
            while not self.stop.is_set():
                try:
                    initial = self._poll_feedback(io, rm_peripheral_read_write_params_t, lower, upper)
                    break
                except Exception as exc:
                    from .gripper_transport import ModbusTransportError
                    self._feedback_failed(exc)
                    if (self.transport is None or not isinstance(exc, ModbusTransportError)
                            or self.transport.status()['failed'] or clock.now() >= startup_deadline):
                        raise
                    self.stop.wait(.1)
            else:
                return
            if initial is None:
                return
            commanded, _, read_started = initial
            command_reference_valid = True
            last_command = -math.inf
            next_poll = read_started + period
            while not self.stop.is_set():
                deadline = next_poll
                if self._next_target(commanded) is not None:
                    deadline = min(deadline, last_command + min_command_interval)
                if self.requests.empty():
                    self.wake.wait(max(0.0, deadline - clock.now()))
                self.wake.clear()
                if self.stop.is_set():
                    break
                request = None
                try:
                    request = self.requests.get_nowait()
                except Empty:
                    pass
                if request is not None:
                    enabled_request, future = request[:2]
                    ticket = request[2] if len(request) > 2 else None
                    if future.set_running_or_notify_cancel():
                        try:
                            if self.transport is not None:
                                # User-requested disable always takes priority;
                                # only an internal transition has a cancellable disable.
                                self.transport.control_guard = (None if enabled_request is False else
                                    lambda: not self.stop.is_set() and self.requests.empty())
                            if enabled_request == 'manual_open':
                                result = self._manual_open(io, rm_peripheral_read_write_params_t, ticket)
                            else:
                                self._set_enabled(io, rm_peripheral_read_write_params_t,
                                                  enabled_request, ticket=ticket)
                                result = self.status()
                            commanded = self.status()["opening"]
                            command_reference_valid = True
                            future.set_result(result)
                        except Exception as exc:
                            from .gripper_transport import MotionCancelled
                            if isinstance(exc, MotionCancelled):
                                # A replacement user intent is not a transport
                                # fault. The next real poll decides readiness;
                                # never erase its target because an old ticket ended.
                                with self.lock:
                                    self.state.update(ready=False, error=str(exc))
                            else:
                                self._feedback_failed(exc)
                            command_reference_valid = False
                            future.set_exception(exc)
                        finally:
                            if self.transport is not None:
                                self.transport.control_guard = None
                if self.stop.is_set():
                    break
                now = clock.now()
                command_due = (self._next_target(commanded) is not None and
                               now-last_command >= min_command_interval)
                refreshed = False
                if now >= next_poll or command_due or request is not None:
                    # Refresh safety state and position together BEFORE the
                    # target. Do not repeat alarm/enable reads in this cycle.
                    try:
                        sample = self._poll_feedback(io, rm_peripheral_read_write_params_t, lower, upper,
                                                     command_due)
                    except Exception as exc:
                        from .gripper_transport import ModbusTransportError
                        self._feedback_failed(exc)
                        command_reference_valid = False
                        if (self.transport is None or not isinstance(exc, ModbusTransportError)
                                or self.transport.status()['failed']):
                            raise
                        # No retries of targets/enables. Only a later scheduled
                        # read may recover monitoring after transport fencing.
                        next_poll = clock.now()+max(period, .1)
                        continue
                    if sample is None:
                        break
                    opening, ready, read_started = sample
                    refreshed = True
                    next_poll = read_started + period
                    if not ready or not command_reference_valid:
                        commanded = opening
                        command_reference_valid = True
                # A queued enable/disable or stop outranks a pending target.
                if self.stop.is_set():
                    break
                if not self.requests.empty():
                    continue
                # A hold or newer target may have arrived during the two reads.
                target = self._next_target(commanded)
                if refreshed and target is not None and clock.now()-last_command >= min_command_interval:
                    requested, bounded, sequence, source = target
                    if not self._motion_has_feedback_budget(source):
                        # Keep only the current desired value. No socket write,
                        # no invented refresh, and no uncertain target retry.
                        continue
                    try:
                        write_started = clock.now()
                        completed = self._send_target(io, rm_peripheral_read_write_params_t,
                            requested, bounded, commanded, sequence, source)
                    except Exception as exc:
                        from .gripper_transport import ModbusTransportError
                        self._feedback_failed(exc)
                        command_reference_valid = False
                        if (self.transport is None or not isinstance(exc, (ModbusTransportError, FeedbackUnavailable))
                                or self.transport.status()['failed']):
                            raise
                        next_poll = clock.now()+max(period, .1)
                        continue
                    if completed is not None:
                        # Separate mode includes three transactions; retain a
                        # per-write estimate rather than multiplying that total.
                        mode = self.config.get('gripper_trigger_mode', 'batch')
                        if source in ('manual', 'manual_open'):
                            mode = self.config.get('gripper_manual_trigger_mode', mode)
                        with self.lock:
                            self._target_write_costs.append((completed-write_started)/(3 if mode == 'separate' else 1))
                        last_command, commanded = completed, bounded
        except Exception as exc:
            with self.lock:
                self.state = {"ready": False, "error": str(exc), "received": clock.now()}
        finally:
            if self.transport is not None:
                self.transport.close()
            while not self.requests.empty():
                future = self.requests.get_nowait()[1]
                if not future.done():
                    future.set_exception(RuntimeError("夹爪通信已关闭"))
            if sdk is not None:
                try:
                    sdk.rm_delete_robot_arm()
                except Exception:
                    pass

    def status(self, max_age_s: float = 0.5) -> dict[str, Any]:
        with self.lock:
            result = dict(self.state)
            result.update(requested_opening=self.desired, command_sequence=self.command_sequence,
                          command_source=self.command_source,
                          fault_generation=self.fault_generation,
                          hard_fault_generation=self.hard_fault_generation,
                          cancel_generation=self.cancel_generation,
                          last_sent=dict(self.sent_target) if self.sent_target else None)
            enable_stamp = self.enabled_received
            transition = self.manual_transition
        result['manual_transition'] = transition
        if transition is not None:
            result.update(ready=False, error='夹爪手动准备：'+transition)
        result['enable_poll_s'] = self.enable_poll_s
        result['feedback_schedule'] = dict(enable_refresh_s=self._enable_refresh_interval(),
            read_budget_ms=round(self._read_budget()*1000, 2),
            request_gap_ms=self._request_gap()*1000,
            target_deferrals=self._deadline_deferrals, deferred_reason=self._deadline_reason,
            control_max_age_ms=500)
        result['transport'] = (self.transport.status() if self.transport is not None else dict(backend='sdk'))
        received = float(result.get('measurement_started', result.get("received", 0.0)))
        age = clock.now() - received if received else math.inf
        result["age_ms"] = round(age * 1000.0, 2) if math.isfinite(age) else None
        if age > max_age_s:
            result.update(ready=False, error="夹爪反馈过期")
        enable_age = clock.now()-enable_stamp if enable_stamp is not None else math.inf
        result['enabled_age_ms'] = round(enable_age*1000, 2) if math.isfinite(enable_age) else None
        if result.get('initialized') and (not math.isfinite(enable_age) or not 0 <= enable_age <= .5):
            result.update(ready=False, error='夹爪使能反馈过期')
        return result

    def snapshot(self, anchor: float, max_age_s: float = 0.5) -> float:
        now = clock.now()
        with self.lock:
            eligible = [(stamp, opening) for stamp, opening in self.history if stamp <= anchor]
            state = dict(self.state)
            started = dict(self.history_started)
        if not eligible:
            raise RuntimeError("没有机械臂状态锚点之前的夹爪反馈")
        stamp, opening = eligible[-1]
        source_start = started.get(stamp, stamp)
        if anchor - source_start > max_age_s or now - source_start > max_age_s or not state.get("ready"):
            raise RuntimeError(state.get("error") or "夹爪反馈与机械臂状态不同步")
        return opening

    def command(self, opening: float, *, source: str = 'policy', expected_cancel_generation=None) -> None:
        value = float(opening)
        if not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError("夹爪开度必须在 [0, 1]")
        observed = self.status()
        if not observed.get("ready"):
            raise RuntimeError("夹爪尚未就绪")
        with self.lock:
            self._require_expected_cancel_locked(expected_cancel_generation)
            now = clock.now()
            if (observed.get('fault_generation') != self.fault_generation
                    or observed.get('cancel_generation') != self.cancel_generation or not self.state.get('ready')
                    or not 0 <= now-self.state.get('measurement_started', self.state.get('received', 0)) <= .5
                    or self.enabled_received is None or not 0 <= now-self.enabled_received <= .5
                    or self.stop.is_set()):
                raise RuntimeError('夹爪反馈在目标入队前已改变或过期')
            self.desired = value
            self.command_sequence += 1
            self.command_source = source
        self.wake.set()

    def hold(self) -> None:
        """Cancel pending movement without issuing a new RTU target."""
        with self.lock:
            trace = dict(stamp=clock.now(), cancelled_opening=self.desired,
                measured_opening=self.state.get('opening'), last_sent=self.sent_target)
            self.cancel_generation += 1
            if self.desired is not None:
                self.command_sequence += 1
            self.desired = None
            self.ramp_target = None
        if trace['cancelled_opening'] is not None:
            logger.info('Gripper cancel pending %s', json.dumps(trace, allow_nan=False))

    def close(self) -> None:
        with self.lock:
            # Even restarting this same worker object requires a new explicit
            # execution session; an old recovery token cannot cross closure.
            self.hard_fault_generation += 1
            self.stop.set()
        self.wake.set()
        thread = self.thread
        if thread and thread is not threading.current_thread():
            thread.join(timeout=2.0)
        # Keep the reference while alive; reconnection must not create a
        # second RTU owner when an old native call is still winding down.
        if thread is None or not thread.is_alive():
            self.thread = None


def verify_arm_state(sdk: Any) -> None:
    result = sdk.rm_get_current_arm_state()
    if not isinstance(result, tuple) or len(result) < 2 or result[0] != 0:
        raise RuntimeError("读取机械臂状态失败，夹爪未初始化")
    state = result[1]
    errors = state.get("err", {}).get("err", []) if isinstance(state, dict) else []
    if errors and any(str(code) not in ("0", "0x0000", "") for code in errors):
        raise RuntimeError(f"机械臂控制器报告错误: {errors}")
