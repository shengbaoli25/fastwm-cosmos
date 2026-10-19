"""Asynchronous CTAG2F120 end-effector adapter over RealMan Modbus RTU.

All RTU traffic stays on one dedicated worker thread/SDK handle so a slow tool
bus cannot stall the 30 Hz robot motion loop or web controls.  Sensor feedback
is timestamped and kept in a bounded ring; commands are latest-only and rate
limited.  Setup never changes the gripper target or fires its trigger.
"""
from __future__ import annotations

from collections import deque
import math
import threading
import time
from typing import Any, Mapping


def _signed32(high: int, low: int) -> int:
    value = (int(high) << 16) | int(low)
    return value - (1 << 32) if value & (1 << 31) else value


class Ctag2F120Worker:
    def __init__(self, config: Mapping[str, Any]):
        self.config = dict(config)
        self.lock = threading.Lock()
        self.stop = threading.Event()
        self.wake = threading.Event()
        self.thread: threading.Thread | None = None
        self.state: dict[str, Any] = {"ready": False, "error": "夹爪初始化中"}
        self.history: deque[tuple[float, float]] = deque(maxlen=96)
        self.desired: float | None = None
        self.mapping_min: int | None = None
        self.mapping_max: int | None = None

    def start(self) -> None:
        if self.thread is not None:
            return
        self.stop.clear()
        self.thread = threading.Thread(target=self._run, name="ctag2f120-modbus", daemon=True)
        self.thread.start()

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

    def _run(self) -> None:
        sdk = None
        try:
            from Robotic_Arm.rm_robot_interface import (
                RoboticArm, rm_peripheral_read_write_params_t, rm_thread_mode_e,
            )
            sdk = RoboticArm(rm_thread_mode_e.RM_DUAL_MODE_E)
            sdk.rm_set_timeout(int(self.config.get("gripper_timeout_ms", 700)))
            handle = sdk.rm_create_robot_arm(str(self.config["robot_ip"]),
                                             int(self.config["robot_port"]), 3)
            if handle is None or int(getattr(handle, "id", 0)) <= 0:
                raise RuntimeError("CTAG2F120 独立 SDK 连接失败")
            lower, upper, opening = self._setup(sdk, rm_peripheral_read_write_params_t)
            self.mapping_min, self.mapping_max = lower, upper
            now = time.monotonic()
            with self.lock:
                self.history.append((now, opening))
                self.state = {"ready": True, "enabled": True, "opening": opening,
                              "position": _signed32(*self._read(sdk, rm_peripheral_read_write_params_t,
                                                                0x060D, 3)[:2]),
                              "mapping_min": lower, "mapping_max": upper,
                              "received": now, "error": None}

            period = float(self.config.get("gripper_poll_s", 0.10))
            min_command_interval = float(self.config.get("gripper_command_interval_s", 0.10))
            max_opening_step = float(self.config.get("max_gripper_step", 0.15))
            last_command = -math.inf
            commanded = opening
            next_poll = time.monotonic()
            while not self.stop.is_set():
                self.wake.wait(max(0.0, next_poll - time.monotonic()))
                self.wake.clear()
                if self.stop.is_set():
                    break
                with self.lock:
                    requested = self.desired
                now = time.monotonic()
                if requested is not None and now - last_command >= min_command_interval:
                    delta = requested - commanded
                    bounded = commanded + max(-max_opening_step, min(max_opening_step, delta))
                    if abs(bounded - commanded) >= float(self.config.get("gripper_deadband", 0.02)):
                        if self._read(sdk, rm_peripheral_read_write_params_t, 0x0610, 3)[2] != 0:
                            raise RuntimeError("夹爪反馈出现报警")
                        if self._read(sdk, rm_peripheral_read_write_params_t, 0x0100, 3)[0] != 1:
                            raise RuntimeError("夹爪意外失能")
                        raw_target = round(lower + (1.0 - bounded) * (upper - lower))
                        # One contiguous RTU transfer, as used by the verified
                        # teleop gripper worker. Last register fires the target.
                        payload = list(raw_target.to_bytes(4, "big", signed=True))
                        payload.extend([byte for item in (
                            int(self.config.get("gripper_speed", 30)),
                            int(self.config.get("gripper_torque", 50)), 100, 100, 1,
                        ) for byte in item.to_bytes(2, "big")])
                        self._code(sdk.rm_write_registers(
                            self._params(rm_peripheral_read_write_params_t, 0x0102, 7), payload),
                            "下发夹爪位置")
                        commanded = bounded
                        last_command = time.monotonic()

                if time.monotonic() < next_poll:
                    continue
                # The force/tcp/action stream is 30 Hz; read user position
                # every 100 ms on its own worker to avoid blocking arm motion.
                position = self._read(sdk, rm_peripheral_read_write_params_t, 0x060D, 3)
                raw_position = _signed32(position[0], position[1])
                opening = 1.0 - (raw_position - lower) / (upper - lower)
                if not math.isfinite(opening) or not -0.02 <= opening <= 1.02:
                    raise RuntimeError("夹爪位置反馈越过行程映射")
                opening = max(0.0, min(1.0, opening))
                received = time.monotonic()
                with self.lock:
                    self.history.append((received, opening))
                    self.state = {"ready": True, "enabled": True, "opening": opening,
                                  "position": raw_position, "mapping_min": lower,
                                  "mapping_max": upper, "received": received, "error": None}
                next_poll = time.monotonic() + period
        except Exception as exc:
            with self.lock:
                self.state = {"ready": False, "error": str(exc), "received": time.monotonic()}
        finally:
            if sdk is not None:
                try:
                    sdk.rm_delete_robot_arm()
                except Exception:
                    pass

    def status(self, max_age_s: float = 0.5) -> dict[str, Any]:
        with self.lock:
            result = dict(self.state)
        received = float(result.get("received", 0.0))
        age = time.monotonic() - received if received else math.inf
        result["age_ms"] = round(age * 1000.0, 2) if math.isfinite(age) else None
        if age > max_age_s:
            result.update(ready=False, error="夹爪反馈过期")
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
            self.desired = value
        self.wake.set()

    def hold(self) -> None:
        """Cancel pending movement without issuing a new RTU target."""
        with self.lock:
            self.desired = None

    def close(self) -> None:
        self.stop.set()
        self.wake.set()
        if self.thread and self.thread is not threading.current_thread():
            self.thread.join(timeout=2.0)
        self.thread = None


def verify_arm_state(sdk: Any) -> None:
    result = sdk.rm_get_current_arm_state()
    if not isinstance(result, tuple) or len(result) < 2 or result[0] != 0:
        raise RuntimeError("读取机械臂状态失败，夹爪未初始化")
    state = result[1]
    errors = state.get("err", {}).get("err", []) if isinstance(state, dict) else []
    if errors and any(str(code) not in ("0", "0x0000", "") for code in errors):
        raise RuntimeError(f"机械臂控制器报告错误: {errors}")
