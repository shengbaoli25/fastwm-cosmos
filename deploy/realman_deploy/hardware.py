"""RealMan RM75 + two RGB cameras deployment adapter.

The adapter is deliberately lazy about optional dependencies: importing this
module does not open a camera, a robot socket, or a UDP listener.  ``connect``
is the only operation that creates those resources.  The implementation keeps
the hardware boundary in this training repository so the deployment code can
later be moved to the teleoperation project without changing the runtime
contract.

The model action is an absolute EEF ``[x, y, z, rx, ry, rz, gripper]`` row in
the same units as the RealMan dataset.  Every command is rate limited against
the measured pose before it reaches ``rm_movep_canfd``.  The controller's
slow-stop and emergency-stop calls use a separate bounded executor and never
depend on the model or camera threads.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout
from collections import deque
import json
import logging
import math
from pathlib import Path
import socket
import threading
import time
from typing import Any, Mapping, Sequence

from .contracts import Observation
from .gripper import Ctag2F120Worker


logger = logging.getLogger(__name__)


def _finite(values: Sequence[float], size: int, name: str) -> tuple[float, ...]:
    if len(values) != size:
        raise ValueError(f"{name} must contain {size} values")
    result = tuple(float(v) for v in values)
    if not all(math.isfinite(v) for v in result):
        raise ValueError(f"{name} contains a non-finite value")
    return result


def _angle_delta(target: float, current: float) -> float:
    """Shortest Euler-component delta, used only for the small safety step."""
    return (target - current + math.pi) % (2.0 * math.pi) - math.pi


class _Camera:
    """One serial-pinned RealSense color stream with a latest-frame cache."""

    def __init__(self, spec: Mapping[str, Any], *, width: int, height: int, fps: int,
                 max_age_s: float):
        self.spec = dict(spec)
        self.width, self.height, self.fps = int(width), int(height), int(fps)
        self.max_age_s = float(max_age_s)
        self.lock = threading.Lock()
        self.stop = threading.Event()
        self.thread: threading.Thread | None = None
        self.pipeline = None
        self.rgb: bytes | None = None
        self.received = 0.0
        self.meta: dict[str, Any] = {}
        self.frames: deque[tuple[float, bytes, dict[str, Any]]] = deque(maxlen=8)
        self.error: str | None = None

    def start(self) -> None:
        if self.thread is not None:
            return
        self.stop.clear()
        self.thread = threading.Thread(target=self._run, name=f"realman-camera-{self.spec['name']}",
                                       daemon=True)
        self.thread.start()

    def _run(self) -> None:
        serial = str(self.spec["serial"])
        while not self.stop.is_set():
            pipeline = None
            try:
                import numpy as np
                import pyrealsense2 as rs
                context = rs.context()
                devices = [d for d in context.query_devices()
                           if d.get_info(rs.camera_info.serial_number) == serial]
                if len(devices) != 1:
                    raise RuntimeError(f"未找到相机 {self.spec.get('name', serial)} ({serial})")
                pipeline = rs.pipeline(context)
                config = rs.config()
                config.enable_device(serial)
                config.enable_stream(rs.stream.color, self.width, self.height,
                                     rs.format.rgb8, self.fps)
                pipeline.start(config)
                self.pipeline = pipeline
                warmup = int(self.spec.get("warmup_frames", 15))
                while not self.stop.is_set():
                    frames = pipeline.wait_for_frames(1000)
                    received = time.monotonic()
                    color = frames.get_color_frame()
                    if not color:
                        continue
                    image = np.asanyarray(color.get_data())
                    if image.shape != (self.height, self.width, 3) or image.dtype != np.uint8:
                        raise RuntimeError(f"相机 {serial} 返回尺寸/dtype 不匹配")
                    if warmup:
                        warmup -= 1
                        continue
                    meta = {
                        "name": self.spec["name"], "serial": serial, "received": received,
                        "device_timestamp_ms": float(color.get_timestamp()),
                        "frame_number": int(color.get_frame_number()), "color_order": "RGB",
                        "width": self.width, "height": self.height,
                    }
                    with self.lock:
                        self.rgb = image.tobytes()
                        self.received = received
                        self.meta = meta
                        self.frames.append((received, self.rgb, meta))
                        self.error = None
            except Exception as exc:
                with self.lock:
                    self.error = str(exc)
            finally:
                if pipeline is not None:
                    try:
                        pipeline.stop()
                    except Exception:
                        pass
                self.pipeline = None
            self.stop.wait(float(self.spec.get("retry_delay_s", 2.0)))

    def snapshot(self, anchor: float, now: float | None = None) -> tuple[bytes, dict[str, Any]]:
        now = time.monotonic() if now is None else now
        with self.lock:
            eligible = [(stamp, rgb, dict(meta)) for stamp, rgb, meta in self.frames if stamp <= anchor]
            error = self.error
        if not eligible:
            raise RuntimeError(error or f"相机 {self.spec['name']} 缺少状态锚点之前的画面")
        received, rgb, meta = eligible[-1]
        age = anchor - received
        if not 0 <= age <= self.max_age_s or now - received > self.max_age_s:
            raise RuntimeError(error or f"相机 {self.spec['name']} 画面过期 ({max(age, 0) * 1000:.1f} ms)")
        meta["age_ms"] = round(age * 1000.0, 2)
        return rgb, meta

    def status(self) -> dict[str, Any]:
        now = time.monotonic()
        with self.lock:
            received, error, frame_number = self.received, self.error, self.meta.get("frame_number")
        age = now - received if received else None
        ready = received > 0 and age is not None and 0 <= age <= self.max_age_s and error is None
        return {
            "name": self.spec.get("name"), "serial": self.spec.get("serial"),
            "ready": ready, "age_ms": round(age * 1000.0, 2) if age is not None else None,
            "frame_number": frame_number, "error": None if ready else (error or "等待相机首帧"),
        }

    def close(self) -> None:
        self.stop.set()
        if self.thread and self.thread is not threading.current_thread():
            self.thread.join(timeout=2.0)
        self.thread = None
        self.pipeline = None


class _RobotFeedback:
    """Callback-owned RM75 state and six-axis force snapshot."""

    def __init__(self, sdk: Any, config: Mapping[str, Any]):
        self.sdk = sdk
        self.config = config
        self.lock = threading.Lock()
        self.received = 0.0
        self.q: tuple[float, ...] | None = None
        self.pose: tuple[float, ...] | None = None
        self.force: tuple[float, ...] | None = None
        self.force_coordinate: int | None = None
        self.error: str | None = "等待 RM75 UDP 反馈"
        self.force_error: str | None = "等待六轴力反馈"
        self.original_config: dict[str, Any] | None = None
        self.changed = False
        self.callback = None
        self.processed = deque(maxlen=256)
        self.force_filter = None
        self.calibration = config.get('_force_calibration')
        self.force_contract = None
        if self.calibration is not None:
            from realman_force.force_filter import ForceFilter, make_manifest
            from realman_force.contract import input_contract
            self.force_filter = ForceFilter(config['_force_filter'])
            self.force_contract = input_contract(self.calibration,
                make_manifest(config['_force_filter'], self.calibration))

    def ingest(self, state: Any, now: float) -> None:
        try:
            arm_ip = state.arm_ip.decode() if isinstance(state.arm_ip, bytes) else str(state.arm_ip)
            if arm_ip and arm_ip != str(self.config["robot_ip"]):
                return
            joint = state.joint_status
            if int(state.errCode) != 0 or any(int(x) != 0 for x in joint.joint_err_code):
                raise ValueError("RM75 控制器或关节反馈报错")
            if not all(bool(x) for x in joint.joint_en_flag):
                raise ValueError("RM75 存在未使能关节")
            waypoint = state.waypoint
            pose = _finite((waypoint.position.x, waypoint.position.y, waypoint.position.z,
                            waypoint.euler.rx, waypoint.euler.ry, waypoint.euler.rz), 6, "TCP pose")
            q = _finite(list(joint.joint_position), 7, "joint feedback")
            sensor = state.force_sensor
            force = _finite(list(sensor.zero_force), 6, "force feedback")
            coordinate = int(sensor.coordinate)
            if coordinate != int(self.config.get("force_coordinate", 1)):
                raise ValueError(f"力反馈坐标系为 {coordinate}，期望 {self.config.get('force_coordinate', 1)}")
            with self.lock:
                self.q, self.pose, self.force = q, pose, force
                self.force_coordinate, self.received = coordinate, now
                self.error = None
                self.force_error = None
                if self.force_filter is not None:
                    from realman_force.force_filter import FilterWarmup
                    point = dict(time=now, force=list(force), pose=list(pose), joints_deg=list(q))
                    sample = dict(force_time=now, force=list(force), pose=list(pose), force_history=[point])
                    try:
                        value = self.force_filter.apply(sample, self.calibration)
                    except FilterWarmup as exc:
                        self.force_error = str(exc)
                    else:
                        self.processed.append(dict(time=now, q=q, pose=pose,
                            force=tuple(value['force_filtered']), epoch=value['force_filter_epoch']))
        except Exception as exc:
            with self.lock:
                self.error = str(exc)
                self.q = self.pose = self.force = None
                self.force_error = str(exc)
                if self.force_filter is not None:
                    self.force_filter.reset(str(exc))
                    self.processed.clear()

    def model_snapshot(self, anchor):
        from realman_force.contract import source_at
        with self.lock:
            if self.force_error or self.error:
                raise RuntimeError(self.force_error or self.error)
            point, _ = source_at(list(self.processed), anchor)
            meta = dict(source_time=point['time'], grid_time=anchor,
                        source_age_s=anchor-point['time'], epoch=point['epoch'],
                        preprocessing_id=self.force_contract['id'])
            return point['q'], point['pose'], point['force'], point['time'], meta

    def start(self) -> None:
        from Robotic_Arm.rm_robot_interface import rm_realtime_arm_state_callback_ptr

        code, original = self.sdk.rm_get_realtime_push()
        if code != 0:
            raise RuntimeError(f"读取 RM75 UDP 配置失败: {code}")
        self.original_config = original
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as route:
            route.connect((str(self.config["robot_ip"]), int(self.config["robot_port"])))
            local_ip = route.getsockname()[0]
        self.callback = rm_realtime_arm_state_callback_ptr(
            lambda state: self.ingest(state, time.monotonic()))
        self.sdk.rm_realtime_arm_state_call_back(self.callback)
        target = dict(original, ip=local_ip,
                      port=int(self.config.get("feedback_port", 8090)),
                      cycle=int(self.config.get("feedback_cycle", 1)), enable=True,
                      force_coordinate=int(self.config.get("force_coordinate", 1)))
        from Robotic_Arm.rm_robot_interface import rm_realtime_push_config_t, rm_udp_custom_config_t
        custom = target.get("custom_config", {})
        if not isinstance(custom, dict):
            custom = {}
        request = rm_realtime_push_config_t(**dict(target, custom_config=rm_udp_custom_config_t(**custom)))
        code = self.sdk.rm_set_realtime_push(request)
        if code != 0:
            raise RuntimeError(f"开启 RM75 UDP 反馈失败: {code}")
        self.changed = True
        deadline = time.monotonic() + float(self.config.get("feedback_start_timeout_s", 2.0))
        while time.monotonic() < deadline:
            with self.lock:
                if self.q is not None and self.pose is not None and self.force is not None:
                    return
            time.sleep(0.01)
        raise RuntimeError("未收到有效 RM75 状态/力反馈")

    def snapshot(self, max_age_s: float) -> tuple[tuple[float, ...], tuple[float, ...], tuple[float, ...], float]:
        with self.lock:
            q, pose, force, received = self.q, self.pose, self.force, self.received
            error = self.error or self.force_error
        age = time.monotonic() - received
        if q is None or pose is None or force is None or error or not 0 <= age <= max_age_s:
            raise RuntimeError(error or f"RM75 状态/力反馈过期 ({max(age, 0) * 1000:.1f} ms)")
        return q, pose, force, received

    def status(self, max_age_s: float) -> dict[str, Any]:
        with self.lock:
            received, error, force_error, coordinate = self.received, self.error, self.force_error, self.force_coordinate
        age = time.monotonic() - received if received else None
        ready = (received > 0 and age is not None and 0 <= age <= max_age_s
                 and error is None and force_error is None)
        return {"ready": ready, "age_ms": round(age * 1000.0, 2) if age is not None else None,
                "force_coordinate": coordinate,
                "error": None if ready else (error or force_error or "等待 RM75 反馈")}

    def restore(self) -> None:
        if not self.changed or self.original_config is None:
            return
        from Robotic_Arm.rm_robot_interface import rm_realtime_push_config_t, rm_udp_custom_config_t
        original = dict(self.original_config)
        custom = original.get("custom_config", {})
        if not isinstance(custom, dict):
            custom = {}
        self.sdk.rm_set_timeout(2000)
        code = self.sdk.rm_set_realtime_push(
            rm_realtime_push_config_t(**dict(original, custom_config=rm_udp_custom_config_t(**custom))))
        self.changed = False
        if code != 0:
            raise RuntimeError(f"恢复 RM75 UDP 配置失败: {code}")


class RealManHardwareAdapter:
    """Concrete HardwareAdapter for one RM75 and two RealSense RGB cameras."""

    def __init__(self, config_path: str | Path | Mapping[str, Any] | None = None):
        if config_path is None:
            config_path = Path(__file__).resolve().parents[1] / "realman_hardware.example.json"
        if isinstance(config_path, Mapping):
            config = dict(config_path)
            config_root = Path(__file__).resolve().parents[1]
        else:
            config = json.loads(Path(config_path).expanduser().read_text())
            config_root = Path(config_path).expanduser().resolve().parent
        if config.get('force_preprocessing_required'):
            from realman_force.force_calibration import validate_profile
            from realman_force.force_filter import validate_config
            def read_config(key):
                path = Path(config[key]).expanduser()
                return json.loads((path if path.is_absolute() else config_root/path).read_text())
            config['_force_calibration'] = validate_profile(read_config('force_calibration_file'))
            config['_force_filter'] = validate_config(read_config('force_filter_file'))
        self.config = self._validate_config(config)
        self.lock = threading.RLock()
        self.sdk = None
        self.handle = None
        self.feedback: _RobotFeedback | None = None
        self.gripper: Ctag2F120Worker | None = None
        self.cameras: list[_Camera] = []
        self.connected = False
        self.closed = False
        self.last_command: tuple[float, ...] | None = None
        self.last_gripper: float | None = None
        self.gripper_feedback: dict[str, Any] | None = None
        self._gripper_stop = threading.Event()
        self._gripper_thread: threading.Thread | None = None
        self.sequence = 0
        self.workspace_center: tuple[float, ...] | None = None
        self.armed = False
        self.estop_latched = False
        self.joint_min: tuple[float, ...] | None = None
        self.joint_max: tuple[float, ...] | None = None
        self.ik_params_type = None
        self._stop_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="realman-stop")
        self._connect()

    @staticmethod
    def _validate_config(config: Mapping[str, Any]) -> dict[str, Any]:
        result = dict(config)
        for key in ("robot_ip", "robot_port", "expected_model", "expected_force_type"):
            if key not in result:
                raise ValueError(f"缺少硬件配置: {key}")
        cameras = result.get("cameras")
        if not isinstance(cameras, list) or len(cameras) != 2:
            raise ValueError("硬件配置必须固定两路相机")
        if [c.get("name") for c in cameras] != ["global", "wrist"]:
            raise ValueError("相机顺序必须是 global、wrist")
        if len({str(c.get("serial")) for c in cameras}) != 2:
            raise ValueError("两路相机序列号必须不同")
        result.setdefault("camera_width", 640); result.setdefault("camera_height", 480)
        result.setdefault("camera_fps", 30); result.setdefault("camera_max_age_s", 0.12)
        result.setdefault("camera_pair_skew_s", 0.05); result.setdefault("feedback_max_age_s", 0.05)
        result.setdefault("feedback_port", 8090); result.setdefault("feedback_cycle", 1)
        result.setdefault("force_coordinate", 1); result.setdefault("feedback_start_timeout_s", 2.0)
        result.setdefault("command_period_s", 1.0 / 30.0); result.setdefault("max_tcp_speed_m_s", 0.10)
        result.setdefault("max_angular_speed_rad_s", 0.8); result.setdefault("max_gripper_step", 0.15)
        result.setdefault("max_joint_step_deg", 8.0); result.setdefault("follow", False)
        result.setdefault("workspace_radius_m", 0.15)
        result.setdefault("max_orientation_deviation_rad", 0.8)
        result.setdefault("workspace_center_pose", None)
        result.setdefault("trajectory_mode", 0); result.setdefault("trajectory_radio", 0)
        result.setdefault("gripper_max_width_m", 0.07); result.setdefault("gripper_inverted", False)
        result.setdefault("gripper_deadband", 0.02)
        result.setdefault("gripper_required", True)
        result.setdefault("initial_gripper", 0.5); result.setdefault("model_arm_dof", 7)
        if not 0 < float(result["camera_max_age_s"]) <= 0.5:
            raise ValueError("camera_max_age_s must be in (0, 0.5]")
        if not 0 < float(result["camera_pair_skew_s"]) <= 0.5:
            raise ValueError("camera_pair_skew_s must be in (0, 0.5]")
        # Explicit null disables the corresponding deployment check. Omission
        # keeps its original default; invalid numbers must not disable it.
        for key in ("max_joint_step_deg", "workspace_radius_m"):
            limit = result[key]
            if limit is not None:
                if isinstance(limit, bool) or not math.isfinite(float(limit)) or float(limit) <= 0:
                    raise ValueError(f"{key} must be positive and finite, or null")
                result[key] = float(limit)
        return result

    def _connect(self) -> None:
        try:
            from Robotic_Arm.rm_robot_interface import (
                RoboticArm, rm_thread_mode_e, rm_inverse_kinematics_params_t,
            )
            self.ik_params_type = rm_inverse_kinematics_params_t
            self.sdk = RoboticArm(rm_thread_mode_e.RM_TRIPLE_MODE_E)
            self.sdk.rm_set_timeout(int(self.config.get("connect_timeout_ms", 1000)))
            self.handle = self.sdk.rm_create_robot_arm(str(self.config["robot_ip"]),
                                                        int(self.config["robot_port"]), 3)
            if self.handle is None or int(getattr(self.handle, "id", 0)) <= 0:
                raise RuntimeError("RM75 连接失败")
            info_code, info = self.sdk.rm_get_robot_info()
            if info_code != 0:
                raise RuntimeError(f"读取 RM75 信息失败: {info_code}")
            for key, info_key in (("expected_model", "arm_model"),
                                  ("expected_force_type", "force_type"),
                                  ("expected_controller_version", "robot_controller_version")):
                expected = self.config.get(key)
                actual = info.get(info_key)
                if expected is not None and actual != expected:
                    raise RuntimeError(f"RM75 {key} 不匹配: {actual} != {expected}")
            if int(info.get("arm_dof", 0)) != 7:
                raise RuntimeError(f"模型要求 RM75 七轴，控制器报告 {info.get('arm_dof')} 轴")
            min_code, joint_min = self.sdk.rm_get_joint_min_pos()
            max_code, joint_max = self.sdk.rm_get_joint_max_pos()
            if min_code != 0 or max_code != 0:
                raise RuntimeError(f"读取 RM75 关节软限位失败: min={min_code}, max={max_code}")
            self.joint_min = _finite(joint_min, 7, "joint minimum")
            self.joint_max = _finite(joint_max, 7, "joint maximum")
            if any(lo >= hi for lo, hi in zip(self.joint_min, self.joint_max)):
                raise RuntimeError("RM75 关节软限位无效")
            self.feedback = _RobotFeedback(self.sdk, self.config)
            self.feedback.start()
            _, current_pose, _, _ = self.feedback.snapshot(float(self.config["feedback_max_age_s"]))
            configured_center = self.config.get("workspace_center_pose")
            self.workspace_center = (_finite(configured_center, 6, "workspace_center_pose")
                                     if configured_center is not None else current_pose)
            self.gripper = Ctag2F120Worker(self.config)
            self.gripper.start()
            self.gripper_feedback = self.gripper.wait_ready(
                float(self.config.get("gripper_start_timeout_s", 8.0)))
            self.cameras = [_Camera(spec, width=self.config["camera_width"],
                                     height=self.config["camera_height"], fps=self.config["camera_fps"],
                                     max_age_s=self.config["camera_max_age_s"])
                            for spec in self.config["cameras"]]
            for camera in self.cameras:
                camera.start()
            self.connected = True
        except Exception:
            self.close()
            raise

    def status(self) -> dict[str, Any]:
        with self.lock:
            feedback = self.feedback
            connected = self.connected and not self.closed
        robot = feedback.status(float(self.config["feedback_max_age_s"])) if feedback else {
            "ready": False, "error": "RM75 未连接"
        }
        gripper = (self.gripper.status(float(self.config.get("gripper_feedback_max_age_s", 0.5)))
                   if self.gripper else {"ready": False, "error": "夹爪未连接"})
        gripper_required = bool(self.config.get("gripper_required", True))
        cameras = [camera.status() for camera in self.cameras]
        cameras_ready = bool(cameras) and all(item["ready"] for item in cameras)
        return {
            "connected": connected,
            "robot_ready": bool(robot.get("ready")),
            "force_ready": bool(robot.get("ready")),
            "force_preprocessing": feedback.force_contract if feedback else None,
            "gripper_ready": bool(gripper.get("ready")) if gripper_required else True,
            "armed": self.armed,
            "estop_latched": self.estop_latched,
            "joint_limits_ready": self.joint_min is not None and self.joint_max is not None,
            "max_joint_step_deg": self.config["max_joint_step_deg"],
            "workspace_radius_m": self.config["workspace_radius_m"],
            "cameras_ready": cameras_ready,
            "camera_order": [item["name"] for item in cameras],
            "cameras": cameras,
            "robot": robot,
            "gripper": gripper,
            "error": None if (connected and robot.get("ready") and cameras_ready
                               and (gripper.get("ready") or not gripper_required))
                      else (robot.get("error") or gripper.get("error")
                            or next((c["error"] for c in cameras if not c["ready"]), "硬件未就绪")),
        }

    def arm(self) -> None:
        if self.estop_latched:
            raise RuntimeError("硬件急停已锁存；检查机械臂并重启部署服务")
        if self.closed or not self.connected:
            raise RuntimeError("硬件适配器未连接")
        status = self.status()
        if not status.get("robot_ready") or not status.get("cameras_ready") or not status.get("force_ready"):
            raise RuntimeError("RM75、双相机或力反馈尚未就绪")
        if not status.get("gripper_ready"):
            raise RuntimeError("夹爪尚未就绪")
        self.armed = True

    def observe(self) -> Observation | None:
        return self.observe_at(time.monotonic())

    def observe_at(self, anchor: float) -> Observation | None:
        if not self.connected or self.feedback is None:
            return None
        force_meta = {}
        if self.feedback.force_filter is not None:
            q, pose, force, feedback_stamp, force_meta = self.feedback.model_snapshot(anchor)
        else:
            q, pose, force, feedback_stamp = self.feedback.snapshot(float(self.config["feedback_max_age_s"]))
        now = time.monotonic()
        frames: dict[str, bytes] = {}
        metadata: dict[str, Mapping[str, Any]] = {}
        for camera in self.cameras:
            rgb, meta = camera.snapshot(feedback_stamp, now)
            frames[str(meta["name"])] = rgb
            metadata[str(meta["name"])] = meta
        stamps = [float(meta["received"]) for meta in metadata.values()]
        if max(stamps) - min(stamps) > float(self.config["camera_pair_skew_s"]):
            raise RuntimeError("两路相机时间差超限")
        if any(abs(stamp - feedback_stamp) > float(self.config["camera_pair_skew_s"])
               for stamp in stamps):
            raise RuntimeError("相机与机械臂状态时间差超限")
        gripper = self.gripper.snapshot(
            feedback_stamp, float(self.config.get("gripper_feedback_max_age_s", 0.5))) if self.gripper else None
        if gripper is None:
            raise RuntimeError("CTAG2F120 夹爪反馈未连接")
        state = tuple(pose) + (gripper,)
        with self.lock:
            sequence = self.sequence
            self.sequence += 1
        return Observation(anchor, sequence, state, force, frames, metadata, force_meta)

    def _limited_pose(self, target: Sequence[float], current: Sequence[float]) -> tuple[float, ...]:
        target = _finite(target, 6, "action pose")
        current = _finite(current, 6, "measured pose")
        dt = float(self.config["command_period_s"])
        max_linear = float(self.config["max_tcp_speed_m_s"]) * dt
        delta = [target[i] - current[i] for i in range(3)]
        norm = math.sqrt(sum(x * x for x in delta))
        if norm > max_linear > 0:
            scale = max_linear / norm
            delta = [x * scale for x in delta]
        max_angular = float(self.config["max_angular_speed_rad_s"]) * dt
        angles = [_angle_delta(target[i + 3], current[i + 3]) for i in range(3)]
        angle_norm = math.sqrt(sum(x * x for x in angles))
        if angle_norm > max_angular > 0:
            scale = max_angular / angle_norm
            angles = [x * scale for x in angles]
        limited = tuple(current[i] + delta[i] for i in range(3)) + tuple(
            current[i + 3] + angles[i] for i in range(3))
        if self.workspace_center is not None:
            radius = self.config["workspace_radius_m"]
            if radius is not None:
                distance = math.sqrt(sum((limited[i] - self.workspace_center[i]) ** 2 for i in range(3)))
                if distance > radius:
                    raise ValueError(f"目标超出安全工作半径 ({distance:.3f} m)")
            orientation = math.sqrt(sum(_angle_delta(limited[i], self.workspace_center[i]) ** 2
                                        for i in range(3, 6)))
            if orientation > float(self.config["max_orientation_deviation_rad"]):
                raise ValueError(f"目标姿态偏离启动姿态过大 ({orientation:.3f} rad)")
        return limited

    def command(self, action: tuple[float, ...]) -> None:
        action = _finite(action, 7, "action")
        if not 0 <= action[6] <= 1:
            raise ValueError("gripper opening must be in [0, 1]")
        if not self.connected or self.sdk is None or self.feedback is None:
            raise RuntimeError("RM75 未连接")
        if not self.armed:
            raise RuntimeError("RM75 尚未 arm")
        if self.estop_latched:
            raise RuntimeError("硬件急停已锁存")
        q, current, _, feedback_stamp = self.feedback.snapshot(float(self.config["feedback_max_age_s"]))
        pose = self._limited_pose(action[:6], current)
        if self.joint_min is None or self.joint_max is None or self.ik_params_type is None:
            raise RuntimeError("RM75 关节软限位/逆运动学尚未验证")
        ik_code, solution = self.sdk.rm_algo_inverse_kinematics(
            self.ik_params_type(list(q), list(pose), 1))
        if ik_code != 0:
            raise RuntimeError(f"RM75 目标无逆运动学解: {ik_code}")
        solution = _finite(solution, 7, "IK joint solution")
        margin = float(self.config.get("joint_limit_margin_deg", 2.0))
        if any(x < lo + margin or x > hi - margin
               for x, lo, hi in zip(solution, self.joint_min, self.joint_max)):
            raise ValueError("RM75 目标触及关节软限位保护区")
        jump = max(abs(x - old) for x, old in zip(solution, q))
        joint_step_limit = self.config["max_joint_step_deg"]
        if joint_step_limit is not None and jump > joint_step_limit:
            raise ValueError(f"RM75 IK 关节跳变过大 ({jump:.2f}°)")
        if self.closed or not self.armed or self.estop_latched:
            raise RuntimeError("硬件已停机或未 arm")
        # Do not hold the Python state lock across an SDK motion call: a stop
        # request must be able to reach the controller on its own worker.
        code = self.sdk.rm_movep_canfd(list(pose), bool(self.config["follow"]),
                                       int(self.config["trajectory_mode"]),
                                       int(self.config["trajectory_radio"]))
        if code != 0:
            raise RuntimeError(f"RM75 TCP 指令失败: {code}")
        with self.lock:
            self.last_command = tuple(pose) + (action[6],)
        if self.config.get("log_motion_trace", False):
            logger.info("Motion trace %s", json.dumps({
                "stamp": time.monotonic(), "feedback_stamp": feedback_stamp,
                "measured_pose": list(current), "requested_pose": list(action[:6]),
                "sent_pose": list(pose), "gripper": action[6], "ik_jump_deg": jump,
            }, allow_nan=False, separators=(",", ":")))
        if self.gripper is None:
            raise RuntimeError("CTAG2F120 夹爪 worker 未连接")
        self.gripper.command(action[6])

    def _stop(self, emergency: bool) -> None:
        if emergency:
            self.estop_latched = True
            self.armed = False
        else:
            self.armed = False
        if self.sdk is None or self.closed:
            return
        if self.gripper is not None:
            self.gripper.hold()
        method_name = "rm_set_arm_stop" if emergency else "rm_set_arm_slow_stop"
        def call() -> Any:
            sdk = self.sdk
            if sdk is None or self.closed:
                return 0
            sdk.rm_set_timeout(int(self.config.get("stop_timeout_ms", 1000)))
            return getattr(sdk, method_name)()
        future = self._stop_pool.submit(call)
        try:
            code = future.result(timeout=float(self.config.get("stop_timeout_s", 1.5)))
        except FutureTimeout as exc:
            raise RuntimeError(f"RM75 {'急停' if emergency else '缓停'}超时；请使用物理急停") from exc
        if isinstance(code, int) and code != 0:
            raise RuntimeError(f"RM75 {'急停' if emergency else '缓停'}失败: {code}")

    def slow_stop(self) -> None:
        self._stop(False)

    def emergency_stop(self) -> None:
        self._stop(True)

    def close(self) -> None:
        with self.lock:
            if self.closed:
                return
            self.closed = True
            self.armed = False
            cameras, feedback, sdk = list(self.cameras), self.feedback, self.sdk
            self.connected = False
        for camera in cameras:
            camera.close()
        if self.gripper:
            self.gripper.close()
        if feedback is not None and sdk is not None:
            try:
                feedback.restore()
            except Exception:
                pass
        if sdk is not None:
            try:
                sdk.rm_delete_robot_arm()
            except Exception:
                pass
        self._stop_pool.shutdown(wait=False, cancel_futures=True)


def make_hardware(config_path: str | Path | None = None, **_: Any) -> RealManHardwareAdapter:
    """Factory for ``FASTWAM_HARDWARE_FACTORY=realman_deploy.hardware:make_hardware``."""
    return RealManHardwareAdapter(config_path)
