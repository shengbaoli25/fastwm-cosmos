"""Asynchronous CTAG2F120 end-effector adapter over RealMan Modbus RTU.

All RTU traffic stays on one dedicated worker thread/SDK handle so a slow tool
bus cannot stall the 30 Hz robot motion loop or web controls.  Sensor feedback
is timestamped and kept in a bounded ring; commands are latest-only and rate
limited.  Setup never changes the gripper target or fires its trigger.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
import math
import threading
import time
from typing import Any, Mapping


def _signed32(high: int, low: int) -> int:
    value = (int(high) << 16) | int(low)
    return value - (1 << 32) if value & (1 << 31) else value


# Bit definitions are from the supplied Changingtek_rtu_psdk.py REG_ALARM.
_ALARMS = {1: "过温", 2: "堵转", 4: "超速", 8: "初始化故障", 16: "限位", 32: "掉电"}


@dataclass
class _DebugRequest:
    action: str
    opening: float | None
    speed: int
    torque: int
    deadline: float
    epoch: int
    cancel_event: threading.Event | None = None
    cancelled: threading.Event = field(default_factory=threading.Event)
    done: threading.Event = field(default_factory=threading.Event)
    result: dict[str, Any] | None = None
    error: Exception | None = None
    actuator_write_started: bool = False


class Ctag2F120Worker:
    def __init__(self, config: Mapping[str, Any], *, diagnostic: bool = False):
        self.config = dict(config)
        self.diagnostic = bool(diagnostic)
        self.lock = threading.Lock()
        self.stop = threading.Event()
        self.wake = threading.Event()
        self.thread: threading.Thread | None = None
        self.state: dict[str, Any] = {
            "connected": False, "ready": False, "enabled": None, "alarm": None,
            "alarm_hex": None, "alarm_labels": [], "opening": None, "position": None,
            "error": "夹爪初始化中", "busy": False, "operation": None, "moving": False,
        }
        self._epoch = 0
        self._mapping_cache = None
        self._controls_cache = None
        self._controls_at = -math.inf
        self._stop_error: str | None = None
        self._pending_debug: _DebugRequest | None = None
        self._active_debug: _DebugRequest | None = None
        self.history: deque[tuple[float, float]] = deque(maxlen=96)
        self.desired: float | None = None
        self.mapping_min: int | None = None
        self.mapping_max: int | None = None

    def start(self) -> None:
        with self.lock:
            if self.thread is not None and self.thread.is_alive():
                return
            if self.stop.is_set():
                raise RuntimeError("夹爪连接已关闭；请新建连接")
            self.thread = threading.Thread(target=self._run, name="ctag2f120-modbus", daemon=True)
            self.thread.start()

    def wait_connected(self, timeout_s: float = 8.0) -> dict[str, Any]:
        """RTU communication can be healthy while an actuator is disabled."""
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            state = self.status()
            if state.get("connected"):
                return state
            if self.thread is not None and not self.thread.is_alive():
                raise RuntimeError(state.get("error") or "CTAG2F120 worker exited")
            time.sleep(.02)
        raise TimeoutError(self.status().get("error") or "CTAG2F120 connection timed out")

    def wait_ready(self, timeout_s: float = 8.0) -> dict[str, Any]:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            state = self.status()
            if state.get("ready"):
                return state
            if self.thread is not None and not self.thread.is_alive():
                raise RuntimeError(state.get("error") or "CTAG2F120 worker exited")
            time.sleep(0.02)
        state = self.status()
        raise TimeoutError(state.get("error") or "CTAG2F120 initialization timed out")

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
        params = self._params(factory, address, len(values))
        if len(values) == 1:
            # Match the vendor and verified collection driver: control words
            # (enable, reset, trigger) use the single-register 0x06 command.
            self._code(sdk.rm_write_single_register(params, values[0]),
                       f"写入夹爪寄存器 0x{address:04x}")
            return
        raw = [byte for value in values for byte in value.to_bytes(2, "big")]
        self._code(sdk.rm_write_registers(params, raw), f"写入夹爪寄存器 0x{address:04x}")

    def _configure_bus(self, sdk) -> None:
        verify_arm_state(sdk)
        voltage = sdk.rm_get_tool_voltage()
        current_voltage = self._code(voltage, "读取末端供电")
        if current_voltage != 3:
            self._code(sdk.rm_set_tool_voltage(3), "设置末端 24V")
            time.sleep(0.5)
        self._code(sdk.rm_get_tool_rs485_mode(), "读取末端 RS485 模式")
        # After controller restart, the reported settings can match while the
        # end-port service rejects every RTU read. Reinitialize once per worker
        # connection, as in the verified teleop setup. This never enables the
        # gripper, writes a target, or retries a movement command.
        self._code(sdk.rm_set_modbus_mode(1, 115200, 2), "初始化末端 Modbus RTU")
        time.sleep(0.3)

    def _setup(self, sdk, factory) -> tuple[int, int, float]:
        # Preserve the inference startup contract and existing bus fix.
        self._configure_bus(sdk)
        alarm = self._read(sdk, factory, 0x0610, 3)[2]
        if alarm != 0:
            raise RuntimeError(f"夹爪报警 0x{alarm:04x}；未使能或发送目标")
        mapping = self._read(sdk, factory, 0x0404, 4)
        actuator = self._read(sdk, factory, 0x0301, 4)
        lower, upper = _signed32(mapping[0], mapping[1]), _signed32(mapping[2], mapping[3])
        actuator_lower, actuator_upper = _signed32(actuator[0], actuator[1]), _signed32(actuator[2], actuator[3])
        if lower >= upper or actuator_lower >= actuator_upper:
            raise RuntimeError("夹爪实机行程映射无效")

        enabled = self._read(sdk, factory, 0x0100, 3)[0]
        if enabled != 1:
            raise RuntimeError("CTAG2F120 未使能；先在机械臂本地工具界面检查并使能")
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

    def _read_state(self, sdk, factory, *, refresh_mapping: bool = False,
                    refresh_controls: bool = True) -> dict[str, Any]:
        # The fixed scale is cached. Routine 100 ms polling needs only one
        # position/alarm transfer; enable/arrival registers refresh every 400 ms.
        # Explicit debug actions and streaming writes request fresh controls.
        if self._mapping_cache is None or refresh_mapping:
            mapping = self._read(sdk, factory, 0x0404, 4)
            actuator = self._read(sdk, factory, 0x0301, 4)
            self._mapping_cache = (_signed32(*mapping[:2]), _signed32(*mapping[2:]),
                                   _signed32(*actuator[:2]), _signed32(*actuator[2:]))
        lower, upper, actuator_lower, actuator_upper = self._mapping_cache
        refresh_s = min(.4, max(.05, float(self.config.get("gripper_control_poll_s", .4))))
        if self._controls_cache is None or refresh_controls or time.monotonic()-self._controls_at >= refresh_s:
            enabled = self._read(sdk, factory, 0x0100, 1)[0]
            flags = self._read(sdk, factory, 0x0601, 2)
            self._controls_cache = (enabled, flags)
            self._controls_at = time.monotonic()
        else:
            enabled, flags = self._controls_cache
        feedback = self._read(sdk, factory, 0x060D, 6)
        position, alarm = _signed32(*feedback[:2]), feedback[5]
        mapping_valid = lower < upper and actuator_lower < actuator_upper
        opening = 1-(position-lower)/(upper-lower) if mapping_valid else None
        valid_position = opening is not None and math.isfinite(opening) and -.02 <= opening <= 1.02
        labels = [name for bit, name in _ALARMS.items() if alarm & bit]
        if alarm & ~63:
            labels.append(f"未知报警位 0x{alarm & ~63:04x}")
        error = (f"夹爪报警 0x{alarm:04x}：{'、'.join(labels)}" if alarm else
                 "夹爪使能反馈无效" if enabled not in (0, 1) else
                 "夹爪实机行程映射无效" if not mapping_valid else
                 "夹爪反馈位置超出已保存的开合映射" if not valid_position else
                 "CTAG2F120 未使能" if enabled != 1 else
                 "夹爪到位标志无效" if any(x not in (0, 1) for x in flags) else None)
        self.mapping_min, self.mapping_max = lower, upper
        received = time.monotonic()
        state = dict(connected=True, ready=error is None, enabled=enabled == 1,
                     alarm=alarm, alarm_hex=f"0x{alarm:04x}", alarm_labels=labels,
                     opening=max(0., min(1., opening)) if valid_position else None,
                     position=position, mapping_min=lower, mapping_max=upper,
                     mapping_valid=mapping_valid, actuator_min=actuator_lower,
                     actuator_max=actuator_upper, enabled_valid=enabled in (0, 1),
                     arrival_flags_valid=all(x in (0, 1) for x in flags),
                     position_arrived=flags[1] == 1,
                     torque_arrived=flags[0] == 1,
                     moving=enabled == 1 and not bool(flags[0] or flags[1]),
                     stop_confirmed=enabled == 0, received=received, error=error)
        with self.lock:
            if self._stop_error is not None:
                state.update(ready=False, error=self._stop_error, stop_confirmed=False,
                             recovery_required=True)
            else:
                state["recovery_required"] = False
            self.state.update(state)
            if not state["ready"]:
                self.desired = None
            if valid_position:
                self.history.append((received, state["opening"]))
            return dict(self.state)

    def _record_error(self, exc: Exception, *, disconnected: bool = False) -> None:
        with self.lock:
            self.desired = None
            self.state.update(ready=False, error=str(exc), received=time.monotonic())
            if disconnected:
                self.state["connected"] = False

    def _check_request(self, request: _DebugRequest) -> None:
        if (self.stop.is_set() or request.cancelled.is_set() or
                request.cancel_event is not None and request.cancel_event.is_set() or
                request.epoch != self._epoch):
            raise RuntimeError("夹爪调试操作已取消；未执行的后续目标已丢弃")
        if time.monotonic() >= request.deadline:
            request.cancelled.set()
            raise TimeoutError("夹爪调试操作超时；未执行的后续目标已丢弃")

    def _debug_write(self, sdk, factory, request: _DebugRequest,
                     address: int, values: list[int]) -> None:
        # No shared lock spans SDK IO, so emergency stop is never blocked here.
        # Already submitted packets cannot be recalled; failure cleanup disables.
        self._check_request(request)
        request.actuator_write_started = True
        if (address == 0x0100 and values == [1]) or (address == 0x0108 and values == [1]):
            with self.lock:
                self.state["stop_confirmed"] = False
        self._write(sdk, factory, address, values)

    def _confirm_disabled(self, sdk, factory, *, deadline: float,
                          request: _DebugRequest | None = None) -> None:
        """Only DISABLE may repeat: its ACK can still leave 0x0100=1."""
        next_write, actual, last_error = -math.inf, None, None
        while True:
            if request is not None:
                self._check_request(request)
            if time.monotonic() >= next_write:
                try:
                    if request is None:
                        self._write(sdk, factory, 0x0100, [0])
                    else:
                        self._debug_write(sdk, factory, request, 0x0100, [0])
                except Exception as exc:
                    if request is not None:
                        self._check_request(request)
                    last_error = exc
                next_write = time.monotonic()+.2
            try:
                actual = self._read(sdk, factory, 0x0100, 1)[0]
                if actual == 0:
                    with self.lock:
                        self._stop_error = None
                        self._controls_cache = None
                        self.state.update(enabled=False, ready=False, moving=False, stop_confirmed=True,
                                          recovery_required=False)
                    return
            except Exception as exc:
                last_error = exc
            if time.monotonic() >= deadline:
                raise RuntimeError(f"夹爪失能未确认：反馈 {actual}" +
                                   (f"；{last_error}" if last_error else ""))
            time.sleep(.02)

    @staticmethod
    def _require_motion_state(state: Mapping[str, Any], *, enabled: bool = True) -> None:
        if not state.get("connected") or not state.get("mapping_valid") or state.get("opening") is None:
            raise RuntimeError(state.get("error") or "夹爪反馈或行程映射无效")
        if state.get("recovery_required"):
            raise RuntimeError(state.get("error") or "先前夹爪停止未确认；需要明确失能或重新连接")
        if not state.get("enabled_valid") or not state.get("arrival_flags_valid"):
            raise RuntimeError(state.get("error") or "夹爪使能/到位反馈无效")
        if state.get("alarm") != 0:
            raise RuntimeError(state.get("error") or "夹爪存在报警；未下发目标")
        if enabled and not state.get("enabled"):
            raise RuntimeError("CTAG2F120 未使能；未下发目标")

    def _enable_at_position(self, sdk, factory, request: _DebugRequest,
                            state: Mapping[str, Any]) -> None:
        self._require_motion_state(state, enabled=False)
        if state["enabled"]:
            return
        # 0x0108 is an action command, not a readable idle/ready latch. This
        # unit retains 1 after arrival/disable; writing 0 means no action and
        # need not clear that value. Replace and verify the retained target
        # while disabled, as in the verified collection driver, before enable.
        raw = int(state["position"]) & 0xffffffff
        self._debug_write(sdk, factory, request, 0x0102, [raw >> 16, raw & 0xffff])
        if _signed32(*self._read(sdk, factory, 0x0102, 2)) != state["position"]:
            raise RuntimeError("使能前保持位置回读不一致；未使能")
        if self._read(sdk, factory, 0x0612, 1)[0] != 0:
            raise RuntimeError("使能前夹爪出现报警；未使能")
        self._debug_write(sdk, factory, request, 0x0100, [1])
        while True:
            self._check_request(request)
            if self._read(sdk, factory, 0x0100, 1)[0] == 1:
                return
            time.sleep(.02)

    def _execute_debug(self, sdk, factory, request: _DebugRequest) -> dict[str, Any]:
        self._check_request(request)
        before = self._read_state(sdk, factory, refresh_mapping=True)
        self._check_request(request)
        action, detail = request.action, {}
        if action in {"enable", "open", "close", "move"} and before["moving"]:
            raise RuntimeError("夹爪当前仍在运动；先明确失能停止再调试")
        if action == "status":
            pass
        elif action == "disable":
            self._confirm_disabled(sdk, factory, deadline=min(request.deadline, time.monotonic()+3), request=request)
            detail["stop_confirmed"] = True
        elif action == "clear_error":
            self._confirm_disabled(sdk, factory, deadline=min(request.deadline, time.monotonic()+3), request=request)
            alarm_before = self._read(sdk, factory, 0x0612, 1)[0]
            if alarm_before:
                self._debug_write(sdk, factory, request, 0x0403, [1])
            while True:
                self._check_request(request)
                state = self._read_state(sdk, factory)
                if state["enabled"] or not state["enabled_valid"]:
                    raise RuntimeError("夹爪清错后未保持有效失能反馈")
                if state["alarm"] == 0:
                    break
                time.sleep(.05)
            detail.update(alarm_before=alarm_before, alarm_after=0,
                          reset_written=bool(alarm_before), stop_confirmed=True)
        elif action == "enable":
            self._enable_at_position(sdk, factory, request, before)
            detail["enable_confirmed"] = True
        else:
            self._require_motion_state(before)
            target = float(request.opening)
            max_step = float(self.config.get("max_gripper_step", .15))
            tolerance = int(self.config.get("gripper_position_tolerance", 10))
            if not math.isfinite(max_step) or not 0 < max_step <= 1 or not 0 <= tolerance <= 50:
                raise ValueError("夹爪调试步长/位置容差配置无效")
            final_position = round(before["mapping_min"]+(1-target)*(before["mapping_max"]-before["mapping_min"]))
            # Verified on this unit: a force-holding close can ignore an open
            # goal until disabled. Release once, preserve measured position,
            # enable once, then start bounded opening steps.
            if before["torque_arrived"] and final_position < before["position"]-tolerance:
                self._confirm_disabled(sdk, factory, deadline=min(request.deadline, time.monotonic()+3),
                                       request=request)
                stationary = self._read_state(sdk, factory)
                self._enable_at_position(sdk, factory, request, stationary)
                detail["force_hold_released"] = True
            steps = 0
            while True:
                self._check_request(request)
                state = self._read_state(sdk, factory)
                self._require_motion_state(state)
                if abs(state["position"]-final_position) <= tolerance and state["position_arrived"]:
                    break
                delta = target-state["opening"]
                bounded = state["opening"]+max(-max_step, min(max_step, delta))
                raw_target = round(state["mapping_min"]+(1-bounded)*(state["mapping_max"]-state["mapping_min"]))
                if raw_target == state["position"] and abs(raw_target-final_position) > tolerance:
                    raise RuntimeError("夹爪调试步长小于设备位置分辨率")
                unsigned = raw_target & 0xffffffff
                self._debug_write(sdk, factory, request, 0x0102,
                                  [unsigned >> 16, unsigned & 0xffff, request.speed, request.torque, 100, 100])
                if _signed32(*self._read(sdk, factory, 0x0102, 2)) != raw_target:
                    raise RuntimeError("夹爪目标回读不一致；未触发")
                self._debug_write(sdk, factory, request, 0x0108, [1])
                steps += 1
                while True:
                    self._check_request(request)
                    state = self._read_state(sdk, factory)
                    self._require_motion_state(state)
                    if state["position_arrived"] and abs(state["position"]-raw_target) <= tolerance:
                        break
                    # Torque flags may be retained from an old goal. Require
                    # position arrival plus measured position for debug success.
                    time.sleep(.03)
            detail.update(requested_opening=target, requested_position=final_position,
                          controller_reports="position_arrived", steps=steps,
                          position_tolerance=tolerance, physical_motion_verified=False)
        self._check_request(request)
        result = self._read_state(sdk, factory)
        self._check_request(request)
        if action in {"open", "close", "move"}:
            self._require_motion_state(result)
            if not result["ready"] or not result["position_arrived"] or abs(result["position"]-final_position) > tolerance:
                raise RuntimeError("夹爪最终到位复核失败；未确认完成")
        if action == "enable" and not result["ready"]:
            raise RuntimeError(result.get("error") or "夹爪使能后未就绪")
        if action in {"disable", "clear_error"} and (result["enabled"] or not result["enabled_valid"]):
            raise RuntimeError("夹爪失能未确认")
        if action == "clear_error" and result["alarm"] != 0:
            raise RuntimeError("夹爪报警复核未清除")
        result.update(command_completed=True, operation=action, operation_result=detail)
        return result

    def _finish_debug(self, sdk, factory, request: _DebugRequest) -> None:
        result, error = None, None
        try:
            result = self._execute_debug(sdk, factory, request)
        except Exception as exc:
            error = exc
            if request.actuator_write_started:
                with self.lock:
                    self.state.update(operation=request.action, stop_confirmed=False)
                try:
                    self._confirm_disabled(sdk, factory, deadline=time.monotonic()+3)
                    self._read_state(sdk, factory)
                except Exception as stop_error:
                    error = RuntimeError(f"{exc}；停止未确认：{stop_error}")
                    with self.lock:
                        self._stop_error = str(error)
                        self.state.update(recovery_required=True, stop_confirmed=False)
            with self.lock:
                self.state.update(last_command_error=str(error), error=str(error), ready=False)
        finally:
            with self.lock:
                self.desired = None
                self._active_debug = None
                self.state.update(busy=False, operation=request.action,
                                  operation_result=result.get("operation_result") if result else None,
                                  command_completed=error is None)
                if error is None:
                    self.state["last_command_error"] = None
                request.result, request.error = result, error
                request.done.set()

    def _stream_command(self, sdk, factory, requested: float, epoch: int) -> None:
        state = self._read_state(sdk, factory)
        self._require_motion_state(state)
        max_step = float(self.config.get("max_gripper_step", .15))
        if not math.isfinite(max_step) or not 0 < max_step <= 1:
            raise ValueError("max_gripper_step 必须在 (0, 1]")
        bounded = state["opening"]+max(-max_step, min(max_step, requested-state["opening"]))
        if abs(bounded-state["opening"]) < float(self.config.get("gripper_deadband", .02)):
            return
        raw_target = round(state["mapping_min"]+(1-bounded)*(state["mapping_max"]-state["mapping_min"]))
        with self.lock:
            allowed = not self.stop.is_set() and epoch == self._epoch and self.desired is not None
        if not allowed:
            return
        payload = list(raw_target.to_bytes(4, "big", signed=True))
        payload.extend(byte for item in (int(self.config.get("gripper_speed", 30)),
                       int(self.config.get("gripper_torque", 50)), 100, 100, 1)
                       for byte in item.to_bytes(2, "big"))
        try:
            self._code(sdk.rm_write_registers(self._params(factory, 0x0102, 7), payload),
                       "下发夹爪位置")
        except Exception as exc:
            # The target and trigger may have reached the actuator despite a
            # missing ACK. Never replay them; stop using the same sole RTU
            # worker, and retain the failure across subsequent healthy polls.
            error = exc
            with self.lock:
                self.desired = None
                self.state.update(ready=False, error=str(exc), last_command_error=str(exc),
                                  stop_confirmed=False)
            try:
                self._confirm_disabled(sdk, factory, deadline=time.monotonic()+3)
                self._read_state(sdk, factory)
            except Exception as stop_error:
                error = RuntimeError(f"{exc}；停止未确认：{stop_error}")
                with self.lock:
                    self._stop_error = str(error)
                    self.state.update(recovery_required=True, stop_confirmed=False)
            with self.lock:
                self.desired = None
                self.state.update(ready=False, error=str(error), last_command_error=str(error))
            if error is exc:
                raise
            raise error from exc

    def _serve(self, sdk, factory) -> None:
        if self.diagnostic:
            self._configure_bus(sdk)
        else:
            self._setup(sdk, factory)
        self._read_state(sdk, factory)
        period = float(self.config.get("gripper_poll_s", .10))
        interval = float(self.config.get("gripper_command_interval_s", .10))
        next_poll, last_command = time.monotonic(), -math.inf
        while not self.stop.is_set():
            self.wake.wait(max(0., next_poll-time.monotonic()))
            self.wake.clear()
            if self.stop.is_set():
                break
            with self.lock:
                request = self._pending_debug
                self._pending_debug = None
                if request is not None:
                    self._active_debug = request
                requested, epoch = self.desired, self._epoch
            if request is not None:
                self._finish_debug(sdk, factory, request)
                next_poll = time.monotonic()+period
                continue
            try:
                if requested is not None and time.monotonic()-last_command >= interval:
                    self._stream_command(sdk, factory, requested, epoch)
                    last_command = time.monotonic()
                if time.monotonic() >= next_poll:
                    self._read_state(sdk, factory, refresh_controls=False)
                    next_poll = time.monotonic()+period
            except Exception as exc:
                # Keep communication alive for explicit recovery, clear goals.
                self._record_error(exc, disconnected=True)
                next_poll = time.monotonic()+period

    def _run(self) -> None:
        sdk = None
        try:
            from Robotic_Arm.rm_robot_interface import (
                RoboticArm, rm_peripheral_read_write_params_t, rm_thread_mode_e,
            )
            sdk = RoboticArm(rm_thread_mode_e.RM_DUAL_MODE_E)
            sdk.rm_set_timeout(int(self.config.get("gripper_timeout_ms", 700)))
            handle = sdk.rm_create_robot_arm(str(self.config["robot_ip"]), int(self.config["robot_port"]), 3)
            if handle is None or int(getattr(handle, "id", 0)) <= 0:
                raise RuntimeError("CTAG2F120 独立 SDK 连接失败")
            self._serve(sdk, rm_peripheral_read_write_params_t)
        except Exception as exc:
            self._record_error(exc, disconnected=True)
        finally:
            with self.lock:
                pending = self._pending_debug
                self._pending_debug = None
                self.state.update(connected=False, ready=False, busy=False)
                if pending is not None:
                    pending.error = RuntimeError(self.state.get("error") or "夹爪连接已关闭")
                    pending.done.set()
            if sdk is not None:
                try:
                    sdk.rm_delete_robot_arm()
                except Exception:
                    pass

    def debug_command(self, action: str, *, opening: float | None = None,
                      speed: int | None = None, torque: int | None = None,
                      timeout_s: float = 12.,
                      cancel_event: threading.Event | None = None) -> dict[str, Any]:
        """Serialize explicit diagnostics on the sole RTU worker."""
        if action not in {"status", "enable", "disable", "clear_error", "open", "close", "move"}:
            raise ValueError("不支持的夹爪调试操作")
        if type(timeout_s) not in (int, float) or not math.isfinite(timeout_s) or not 0 < timeout_s <= 60:
            raise ValueError("夹爪调试超时必须在 (0, 60] 秒")
        speed = self.config.get("gripper_speed", 30) if speed is None else speed
        torque = self.config.get("gripper_torque", 50) if torque is None else torque
        if type(speed) is not int or type(torque) is not int or not 1 <= speed <= 100 or not 1 <= torque <= 100:
            raise ValueError("夹爪速度和力矩必须是 1..100 的整数")
        if action in {"open", "close"}:
            opening = 1. if action == "open" else 0.
        if action == "move" and (type(opening) not in (int, float) or not math.isfinite(opening) or not 0 <= opening <= 1):
            raise ValueError("夹爪开度必须在 [0, 1]")
        with self.lock:
            if self.stop.is_set() or self.thread is None or not self.thread.is_alive():
                raise RuntimeError("夹爪通信未连接")
            if cancel_event is not None and cancel_event.is_set():
                raise RuntimeError("夹爪调试操作已取消")
            if self._pending_debug is not None or self._active_debug is not None:
                raise RuntimeError("夹爪调试操作仍在执行")
            self.desired = None
            request = _DebugRequest(action, opening, speed, torque, time.monotonic()+timeout_s,
                                    self._epoch, cancel_event)
            self._pending_debug = request
            self.state.update(busy=True, operation=action, operation_result=None,
                              command_completed=False, last_command_error=None)
        self.wake.set()
        if not request.done.wait(timeout_s):
            request.cancelled.set()
            self.wake.set()
            raise TimeoutError("夹爪调试等待超时；已取消未执行目标，停止确认仍在工作线程进行")
        if request.error is not None:
            raise request.error
        result = dict(request.result or {})
        result["busy"] = False
        return result

    def status(self, max_age_s: float = .5) -> dict[str, Any]:
        with self.lock:
            result = dict(self.state)
        received = float(result.get("received", 0.))
        age = time.monotonic()-received if received else math.inf
        result["age_ms"] = round(age*1000, 2) if math.isfinite(age) else None
        if age > max_age_s:
            result.update(ready=False, feedback_stale=True)
            if not result.get("error"):
                result["error"] = "夹爪反馈过期"
        else:
            result["feedback_stale"] = False
        return result

    def snapshot(self, anchor: float, max_age_s: float = 0.5) -> float:
        now = time.monotonic()
        with self.lock:
            eligible = [(stamp, opening) for stamp, opening in self.history if stamp <= anchor]
            state = dict(self.state)
        if not eligible:
            raise RuntimeError("没有机械臂状态锚点之前的夹爪反馈")
        stamp, opening = eligible[-1]
        if anchor - stamp > max_age_s or now - stamp > max_age_s or not state.get("ready"):
            raise RuntimeError(state.get("error") or "夹爪反馈与机械臂状态不同步")
        return opening

    def command(self, opening: float) -> None:
        value = float(opening)
        if not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError("夹爪开度必须在 [0, 1]")
        if not self.status().get("ready"):
            raise RuntimeError("夹爪尚未就绪")
        with self.lock:
            if self.stop.is_set() or self._pending_debug is not None or self._active_debug is not None:
                raise RuntimeError("夹爪正在调试或连接已关闭")
            self.desired = value
        self.wake.set()

    def hold(self) -> None:
        """Invalidate old work promptly; normal future inference may resume."""
        with self.lock:
            self.desired = None
            self._epoch += 1
            for request in (self._pending_debug, self._active_debug):
                if request is not None:
                    request.cancelled.set()
        self.wake.set()

    def close(self) -> None:
        self.stop.set()
        self.hold()
        thread = self.thread
        if thread and thread is not threading.current_thread():
            thread.join(timeout=2.)
        # A still-running thread retains its SDK handle: do not lose ownership.
        if thread is not None and not thread.is_alive():
            self.thread = None
        with self.lock:
            self.state.update(ready=False, connected=False)
            if thread is not None and thread.is_alive():
                self.state["error"] = "夹爪连接正在停止，后台线程尚未退出"
        if thread is not None and thread.is_alive():
            raise RuntimeError("夹爪连接关闭未完成：后台线程尚未退出，不能建立重叠连接")


def verify_arm_state(sdk: Any) -> None:
    result = sdk.rm_get_current_arm_state()
    if not isinstance(result, tuple) or len(result) < 2 or result[0] != 0:
        raise RuntimeError("读取机械臂状态失败，夹爪未初始化")
    state = result[1]
    errors = state.get("err", {}).get("err", []) if isinstance(state, dict) else []
    if errors and any(str(code) not in ("0", "0x0000", "") for code in errors):
        raise RuntimeError(f"机械臂控制器报告错误: {errors}")
