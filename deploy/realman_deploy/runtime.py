"""Independent sensor, inference, control and web loops with bounded buffers."""
from __future__ import annotations

import logging
import math
import threading
import time
from typing import Callable

from .adapters import HardwareAdapter, PolicyAdapter
from .contracts import ActionChunk, RuntimeStatus
from .cuda_status import CudaStatusProbe
from .normalization import DatasetNormalizer, OnlineHistory
from .queues import ActionTimeline, ChunkActionQueue, EpochGate, LatestObservation
from .reference import RollingActionReference


logger = logging.getLogger(__name__)


class DeploymentRuntime:
    SENSOR_HZ = 30.0
    CONTROL_HZ = 30.0
    MAX_OBSERVATION_AGE_S = 0.20
    MIN_FRESH_ACTIONS = 3
    FIRST_RESULT_TIMEOUT_S = 60.0

    def __init__(self, normalizer: DatasetNormalizer, *, reference=None,
                 clock: Callable[[], float] = time.monotonic, require_cuda: bool = True,
                 execution_mode: str = "timestamped", cuda_device: str = "cuda",
                 sampling_profile: str | None = None,
                 inference_device: str = "local",
                 max_result_age_s: float | None = None,
                 action_smoothing_alpha: float = 1.0):
        if execution_mode not in {"timestamped", "async_relaxed", "synchronous"}:
            raise ValueError("execution_mode must be timestamped, async_relaxed or synchronous")
        self.execution_mode = execution_mode
        self.sampling_profile = sampling_profile
        if inference_device not in {"local", "qizhi_h200"}:
            raise ValueError("inference_device must be local or qizhi_h200")
        if max_result_age_s is not None and (type(max_result_age_s) not in (int, float)
                                            or not math.isfinite(max_result_age_s) or max_result_age_s <= 0):
            raise ValueError("max_result_age_s must be positive or None")
        if (type(action_smoothing_alpha) not in (int, float) or not math.isfinite(action_smoothing_alpha)
                or not 0 < action_smoothing_alpha <= 1):
            raise ValueError("action_smoothing_alpha must be in (0, 1]")
        self.inference_device = inference_device
        # Remote calls need an explicit bounded freshness window. Local keeps
        # the historical behaviour unless a caller opts in to a deadline.
        self.max_result_age_s = (0.6 if inference_device == "qizhi_h200" and max_result_age_s is None
                                 else max_result_age_s)
        self.action_smoothing_alpha = float(action_smoothing_alpha)
        self.cuda_probe = CudaStatusProbe(device=cuda_device)
        self.normalizer = normalizer
        self.require_cuda = require_cuda
        self.reference = reference or RollingActionReference(self.CONTROL_HZ)
        self.clock = clock
        self.gate = EpochGate()
        self.observations = LatestObservation()
        queue_type = ActionTimeline if execution_mode == "timestamped" else ChunkActionQueue
        self.timeline = queue_type(self.CONTROL_HZ, max_actions=32)
        self.lock = threading.RLock()
        self._command_lock = threading.Lock()
        self.wake = threading.Event()
        self.stop_event = threading.Event()
        self.policy: PolicyAdapter | None = None
        self.hardware: HardwareAdapter | None = None
        self.state = "idle"
        self.error: str | None = None
        self.inference_count = self.discarded_results = self._run_inference_count = 0
        self.latency_ms: float | None = None
        self._history = OnlineHistory(normalizer, 32)
        self._input_history = None
        self._running_since: float | None = None
        self._threads: list[threading.Thread] = []
        self._load_thread: threading.Thread | None = None
        self._infer_active = False
        self._estop_latched = False
        self._estop_reset_busy = False
        self._estop_inflight = 0
        self._relaxed_holding = False
        self._relaxed_end_at = 0.0
        self._pending_chunk: ActionChunk | None = None
        self._unstarted_chunk: ActionChunk | None = None
        self._smoothed_action: tuple[float, ...] | None = None
        self._remote_latency_estimate_s: float | None = None
        self._synchronous_ready_at: float | None = None
        self.input_age_at_result_ms: float | None = None
        self.input_age_at_first_action_ms: float | None = None
        self._debug_gripper = None
        self._gripper_debug_busy = False
        self._gripper_debug_operation = None
        self._gripper_debug_error = None
        self._gripper_debug_cancel = None
        self._hardware_connecting = False

    def _require_inactive(self) -> None:
        if self._estop_reset_busy:
            raise RuntimeError("急停恢复正在进行，请等待检查完成")
        if self._estop_latched:
            raise RuntimeError("急停已锁存；解除物理急停并检查现场后，点击恢复急停")
        if self._hardware_connecting or self.gripper_debug_status().get('busy'):
            raise RuntimeError("硬件连接或夹爪调试正在进行，请等待完成")
        if self.state in ("running", "loading", "stopping", "stopped") or self._infer_active:
            raise RuntimeError("先暂停并等待当前推理/加载结束，再替换适配器")
        if self._load_thread and self._load_thread.is_alive():
            raise RuntimeError("上一模型仍在加载；请等待完成")

    def _gripper_worker(self):
        return getattr(self.hardware, 'gripper', None) if self.hardware else self._debug_gripper

    def gripper_debug_status(self) -> dict:
        """Cached feedback only: HTTP polling never performs Modbus I/O."""
        with self.lock:
            worker = self._gripper_worker()
            state = worker.status() if worker is not None else dict(
                connected=False, ready=False, enabled=None, alarm=None, opening=None, error=None)
            result = dict(state)
            result.update(supported=True, busy=self._gripper_debug_busy or bool(state.get('busy')),
                          operation=self._gripper_debug_operation or state.get('operation'),
                          command_error=self._gripper_debug_error)
            return result

    def debug_gripper(self, action: str, *, factory=None, opening=None, speed=None, torque=None) -> dict:
        """Reserve the actuator without holding the runtime lock during RTU I/O."""
        if not isinstance(action, str) or action not in {'connect', 'status', 'enable', 'disable', 'clear_error', 'open', 'close', 'move'}:
            raise ValueError('未知夹爪调试操作')
        if action == 'move':
            if isinstance(opening, bool) or not isinstance(opening, (int, float)) or not math.isfinite(opening) or not 0 <= opening <= 1:
                raise ValueError('夹爪开度必须在 [0, 1]')
        elif opening is not None:
            raise ValueError('只有 move 操作接受 opening')
        for name, value in (('speed', speed), ('torque', torque)):
            if value is not None and (type(value) is not int or not 1 <= value <= 100):
                raise ValueError(name + ' 必须为 1..100 的整数')
        if action not in {'move', 'open', 'close'} and (speed is not None or torque is not None):
            raise ValueError('速度和力度只用于夹爪移动')
        with self.lock:
            self._require_inactive()
            hardware = self.hardware
            if hardware is not None:
                hs = hardware.status()
                if hs.get('armed') or hs.get('estop_latched'):
                    raise RuntimeError('请先暂停机械臂；急停锁定时不可调试')
            worker = self._gripper_worker()
            if action != 'connect' and worker is None:
                raise RuntimeError('请先连接夹爪')
            if action == 'connect' and factory is None:
                raise ValueError('缺少夹爪连接配置')
            epoch = self.gate.current()
            self._gripper_debug_busy = True
            self._gripper_debug_operation = action
            self._gripper_debug_error = None
            cancel = self._gripper_debug_cancel = threading.Event()
        try:
            if action == 'connect':
                if worker is not None:
                    worker.close()
                with self.lock:
                    if not self.gate.accepts(epoch) or self._estop_latched:
                        raise RuntimeError('夹爪连接已取消')
                new_worker = factory()
                with self.lock:
                    cancelled = not self.gate.accepts(epoch) or self._estop_latched
                    if not cancelled:
                        worker = new_worker
                        if hardware is not None:
                            hardware.gripper = worker
                        else:
                            self._debug_gripper = worker
                        worker.start()
                if cancelled:
                    new_worker.close()
                    raise RuntimeError('夹爪连接已取消')
                result = worker.wait_connected(timeout_s=8.0)
            else:
                result = worker.debug_command(action, opening=opening, speed=speed, torque=torque,
                                              timeout_s=12.0, cancel_event=cancel)
            with self.lock:
                if not self.gate.accepts(epoch) or self._estop_latched:
                    raise RuntimeError('夹爪操作已被暂停或急停取消')
            return {**result, 'supported': True}
        except Exception as exc:
            with self.lock:
                self._gripper_debug_error = str(exc)
            raise
        finally:
            with self.lock:
                self._gripper_debug_busy = False
                self._gripper_debug_operation = None
                self._gripper_debug_cancel = None

    def connect_hardware(self, factory) -> dict:
        """Hand standalone gripper ownership to the full adapter, never overlap."""
        with self.lock:
            self._require_inactive()
            if self.hardware is not None:
                return self.hardware.status()
            self._hardware_connecting = True
            standalone = self._debug_gripper
            epoch = self.gate.current()
        adapter = None
        try:
            if standalone is not None:
                standalone.close()
                with self.lock:
                    self._debug_gripper = None
            with self.lock:
                if not self.gate.accepts(epoch) or self._estop_latched:
                    raise RuntimeError('硬件连接已取消')
            adapter = factory()
            with self.lock:
                if not self.gate.accepts(epoch) or self._estop_latched:
                    raise RuntimeError('硬件连接已取消')
                status = adapter.status()
                self._validate_hardware_space(status)
                if status.get('force_preprocessing') != self.normalizer.force_preprocessing:
                    raise ValueError('模型归一化文件与硬件力预处理不一致')
                self.hardware, adapter = adapter, None
                self._invalidate('ready')
                return self.hardware.status()
        finally:
            try:
                if adapter is not None:
                    adapter.close()
            finally:
                with self.lock:
                    self._hardware_connecting = False

    def _invalidate(self, state: str, error: str | None = None) -> int:
        """Caller holds lock; publication and invalidation share this lock."""
        epoch = self.gate.advance()
        self.timeline.clear()
        self.observations.clear()
        self.reference.reset()
        self._input_history = None
        self._relaxed_holding = False
        self._relaxed_end_at = 0.0
        self._pending_chunk = None
        self._unstarted_chunk = None
        self._smoothed_action = None
        self.input_age_at_first_action_ms = None
        self._synchronous_ready_at = None
        self.state, self.error = state, error
        return epoch

    def attach(self, *, policy=None, hardware=None) -> None:
        with self.lock:
            self._require_inactive()
            if policy is not None:
                self._validate_policy_normalizer(policy)
            if hardware is not None:
                self._validate_hardware_space(hardware.status())
                actual = hardware.status().get('force_preprocessing')
                expected = self.normalizer.force_preprocessing
                if actual != expected:
                    raise ValueError('模型归一化文件与硬件力预处理不一致；请使用同一新数据集导出的模型和 normalization.json')
            old_policy = self.policy if policy is not None else None
            old_hardware = self.hardware if hardware is not None else None
            if policy is not None:
                self.policy = policy
            if hardware is not None:
                self.hardware = hardware
            self._invalidate("ready")
        if old_policy and old_policy is not policy:
            old_policy.close()
        if old_hardware and old_hardware is not hardware:
            old_hardware.close()

    def _validate_policy_normalizer(self, policy) -> None:
        normalizer = getattr(policy, 'normalizer', self.normalizer)
        if (normalizer.stats != self.normalizer.stats
                or normalizer.action_space != self.normalizer.action_space
                or normalizer.force_preprocessing != self.normalizer.force_preprocessing):
            raise ValueError('模型与运行时 normalization.json 不一致；请使用同一训练导出文件重启服务')

    def configure_sampling(self, profile: str, mode: str, *, persist=None,
                           inference_device: str | None = None,
                           max_result_age_s: float | None = None,
                           action_smoothing_alpha: float | None = None) -> None:
        """Change an inactive session atomically; new settings require model reload."""
        from .sampling import updated_config
        updated_config({}, profile, mode)
        next_device = self.inference_device if inference_device is None else inference_device
        next_age = self.max_result_age_s if max_result_age_s is None else max_result_age_s
        if next_device != self.inference_device and max_result_age_s is None:
            next_age = 0.6 if next_device == "qizhi_h200" else None
        next_alpha = self.action_smoothing_alpha if action_smoothing_alpha is None else action_smoothing_alpha
        if next_device not in {"local", "qizhi_h200"}:
            raise ValueError("inference_device must be local or qizhi_h200")
        if next_age is not None and (type(next_age) not in (int, float)
                                     or not math.isfinite(next_age) or next_age <= 0):
            raise ValueError("max_result_age_s must be positive or None")
        if type(next_alpha) not in (int, float) or not math.isfinite(next_alpha) or not 0 < next_alpha <= 1:
            raise ValueError("action_smoothing_alpha must be in (0, 1]")
        with self.lock:
            self._require_inactive()
            if self.normalizer.action_space != "joint8":
                raise ValueError("三专家采样设置仅适用于 Joint8 模型")
            # Persist before invalidating; an unwritable config must leave the
            # existing model/settings usable. start/load cannot race this lock.
            if persist is not None:
                persist()
            old, self.policy = self.policy, None
            self._invalidate("idle")
            self.execution_mode, self.sampling_profile = mode, profile
            self.inference_device = next_device
            self.max_result_age_s = (0.6 if next_device == "qizhi_h200" and next_age is None
                                     else next_age)
            self.action_smoothing_alpha = float(next_alpha)
            self.timeline = (ActionTimeline if mode == "timestamped" else ChunkActionQueue)(
                self.CONTROL_HZ, max_actions=32)
            self.latency_ms = self.input_age_at_result_ms = None
            self._remote_latency_estimate_s = None
        if old:
            old.close()

    def _validate_hardware_space(self,status):
        if status.get('action_space','eef_pose7') != self.normalizer.action_space:
            raise ValueError('Hardware and policy action space differ; joint8 requires joint commands in radians')

    def load_policy_async(self, factory: Callable[[str], PolicyAdapter], checkpoint: str) -> None:
        with self.lock:
            self._require_inactive()
            epoch = self._invalidate("loading")
            old, self.policy = self.policy, None

            def load():
                new = None
                try:
                    if old:
                        old.close()
                    new = factory(checkpoint)
                    with self.lock:
                        self._validate_policy_normalizer(new)
                        if self.gate.accepts(epoch) and self.state == "loading":
                            self.policy, new = new, None
                            self.state = "ready"
                except Exception as exc:
                    with self.lock:
                        if self.gate.accepts(epoch):
                            self.state, self.error = "error", f"load: {exc}"
                finally:
                    if new:
                        new.close()
            self._load_thread = threading.Thread(target=load, name="deploy-model-load", daemon=True)
            self._load_thread.start()

    def start(self) -> None:
        with self.lock:
            self._require_inactive()
            if self.state not in ("ready", "paused", "idle"):
                raise RuntimeError(f"cannot start while runtime is {self.state}")
            if self.policy is None or self.hardware is None:
                raise RuntimeError("模型和硬件必须就绪")
            requires_local_cuda = getattr(self.policy, "requires_local_cuda", True) is not False
            if self.require_cuda and requires_local_cuda and not self._cuda_status().get("available"):
                raise RuntimeError("CUDA 未就绪；禁止在无 GPU 环境下启动闭环控制")
            status = self.hardware.status()
            self._validate_hardware_space(status)
            self._validate_policy_normalizer(self.policy)
            if status.get('force_preprocessing') != self.normalizer.force_preprocessing:
                raise ValueError('硬件力预处理与模型不一致，请重新连接匹配的配置')
            if not all(status.get(k) for k in ("robot_ready", "cameras_ready", "force_ready", "gripper_ready")):
                raise RuntimeError("RM75、双相机、六轴力和夹爪反馈必须就绪")
            self.hardware.arm()
            self._invalidate("running")
            self._history = OnlineHistory(self.normalizer, 32)
            self._run_inference_count = 0
            self._running_since = self.clock()
            if not self._threads:
                self._threads = [threading.Thread(target=fn, name=name, daemon=True) for name, fn in (
                    ("realman-sensor", self._sensor_loop), ("cosmos-inference", self._inference_loop),
                    ("realman-control", self._control_loop))]
                for thread in self._threads:
                    thread.start()
        self.wake.set()

    def _stop_hardware(self, hardware, *, emergency: bool) -> None:
        if hardware is None:
            return
        if emergency:
            hardware.emergency_stop()
            return
        # A dispatched command completes before a slow-stop acknowledgement.
        # Emergency stop is independent of this serialization lock.
        if not self._command_lock.acquire(timeout=1.0):
            hardware.emergency_stop()
            raise RuntimeError("动作接口阻塞，已请求急停")
        try:
            hardware.slow_stop()
        finally:
            self._command_lock.release()

    def pause(self) -> None:
        with self.lock:
            if self.state in ("estop", "stopped", "stopping"):
                return
            if self.state in ("idle", "ready", "paused") and not (self.gripper_debug_status()['busy'] or self._hardware_connecting):
                return
            epoch = self._invalidate("stopping")
            hardware = self.hardware
            gripper = self._gripper_worker()
            if self._gripper_debug_cancel is not None:
                self._gripper_debug_cancel.set()
        self.wake.set()
        try:
            try:
                if gripper is not None:
                    gripper.hold()
            finally:
                self._stop_hardware(hardware, emergency=False)
        except Exception as exc:
            with self.lock:
                if self.gate.accepts(epoch):
                    self.state, self.error = "error", f"暂停停机失败: {exc}"
            raise
        with self.lock:
            if self.gate.accepts(epoch):
                self.state = "paused"

    def emergency_stop(self) -> None:
        with self.lock:
            self._estop_latched = True
            self._estop_inflight += 1
            epoch = self._invalidate("estop", "急停已锁存；解除物理急停并检查现场后，点击恢复急停")
            hardware = self.hardware
            gripper = self._gripper_worker()
            if self._gripper_debug_cancel is not None:
                self._gripper_debug_cancel.set()
        self.wake.set()
        try:
            try:
                if gripper is not None:
                    gripper.hold()
            finally:
                self._stop_hardware(hardware, emergency=True)
        except Exception as exc:
            with self.lock:
                if self.gate.accepts(epoch):
                    self.error += f"；硬件急停未确认: {exc}"
            raise
        finally:
            with self.lock:
                self._estop_inflight -= 1

    def reset_emergency_stop(self, *, confirmed: bool = False) -> dict:
        """Release the software latch only; never arm or resume old actions."""
        if confirmed is not True:
            raise ValueError("请先确认已解除物理急停并检查现场安全")
        with self.lock:
            if self.state in ("stopping", "stopped") or self.stop_event.is_set():
                raise RuntimeError("服务正在停止，不能恢复急停")
            if (self._estop_reset_busy or self._estop_inflight or self._infer_active
                    or self._hardware_connecting or self.gripper_debug_status().get("busy")
                    or (self._load_thread and self._load_thread.is_alive())):
                raise RuntimeError("停止、推理、加载或硬件操作仍在进行，请等待后重试恢复")
            hardware = self.hardware
            hs = hardware.status() if hardware is not None else {}
            if hs.get("armed"):
                raise RuntimeError("硬件仍处于控制使能状态，不能恢复急停")
            # A successful reset response can be lost. A repeated confirmed
            # request may acknowledge only an already paused, disarmed runtime.
            if not self._estop_latched:
                if self.state == "paused" and not hs.get("estop_latched"):
                    return self.status()
                raise RuntimeError("当前未处于软件急停锁定状态")
            if self.state != "estop":
                raise RuntimeError("当前状态不允许恢复急停")
            reset = getattr(hardware, "reset_emergency_stop", None) if hardware is not None else None
            if hardware is not None and not callable(reset):
                raise RuntimeError("当前硬件适配器不支持安全恢复急停")
            self._estop_reset_busy = True
            epoch = self._invalidate("estop", "正在检查急停恢复；机械臂保持停止")
            gripper = self._gripper_worker()
        acquired = False
        try:
            # Emergency stop deliberately bypasses this lock. The epoch below
            # prevents a late reset from overriding a newer stop request.
            acquired = self._command_lock.acquire(blocking=False)
            if not acquired:
                raise RuntimeError("动作接口仍未退出，请等待后重试恢复")
            if gripper is not None:
                gripper.hold()
            with self.lock:
                if not self.gate.accepts(epoch):
                    raise RuntimeError("恢复已被新的停止操作取消")
            if reset is not None:
                reset()
                hs = hardware.status()
                if hs.get("estop_latched") is not False or hs.get("armed") is not False:
                    raise RuntimeError("硬件未确认解除软件锁存或仍处于控制使能状态")
            with self.lock:
                if not self.gate.accepts(epoch) or hardware is not self.hardware:
                    raise RuntimeError("恢复已被新的停止操作取消")
                self._estop_latched = False
                self._invalidate("paused")
                self._history = OnlineHistory(self.normalizer, 32)
                self._running_since = None
                self._run_inference_count = 0
        except Exception as exc:
            with self.lock:
                if self.gate.accepts(epoch):
                    self.state = "estop"
                    self.error = f"急停恢复失败，保持锁定: {exc}"
            raise
        finally:
            if acquired:
                self._command_lock.release()
            with self.lock:
                self._estop_reset_busy = False
        return self.status()

    def _sensor_loop(self) -> None:
        deadline = self.clock()
        while not self.stop_event.is_set():
            with self.lock:
                hardware, running, epoch = self.hardware, self.state == "running", self.gate.current()
            if running and hardware:
                try:
                    observation = (hardware.observe_at(deadline) if hasattr(hardware, 'observe_at')
                                   else hardware.observe())
                    if observation is not None:
                        if not 0 <= self.clock() - observation.stamp <= self.MAX_OBSERVATION_AGE_S:
                            raise ValueError("robot/camera observation is stale")
                        with self.lock:
                            if self.state == "running" and self.gate.accepts(epoch):
                                expected = self.normalizer.force_preprocessing
                                if observation.action_space != self.normalizer.action_space:
                                    raise ValueError('Observation action space differs from training')
                                if expected is not None and observation.force_meta.get('preprocessing_id') != expected['id']:
                                    raise ValueError('观测力预处理版本与训练版本不一致')
                                self._history.append(observation.state, observation.force)
                                # Publish image/pose and its matching history atomically.
                                self._input_history = self._history.tensors()
                                self.observations.put(observation)
                        self.wake.set()
                except Exception as exc:
                    self._fail("sensor", exc, epoch)
            deadline += 1.0 / self.SENSOR_HZ
            if deadline < self.clock() - 1.0 / self.SENSOR_HZ:
                if running:
                    self._fail('sensor', RuntimeError('30 Hz observation grid overrun'), epoch)
                deadline = self.clock()
            self.stop_event.wait(max(0.0, deadline - self.clock()))

    def _publish(self, normalized_actions, observation, epoch: int, started: float) -> bool:
        # Shape/finite checks precede decoding; no NaN can be hidden by clipping.
        if len(normalized_actions) != 32 or any(len(row) != self.normalizer.action_dim for row in normalized_actions):
            raise ValueError(f"AE output must be Action32x{self.normalizer.action_dim}")
        predicted = self.normalizer.decode("action", normalized_actions)
        execution = [row[:-1] + [max(0.0, min(1.0, row[-1]))] for row in predicted]
        finished = self.clock()
        chunk = ActionChunk(observation.sequence, observation.stamp, epoch,
                            tuple(tuple(row) for row in execution), finished, self.normalizer.action_space)
        with self.lock:
            # A stopped run still needs its actual latency for diagnosis.
            self.latency_ms = round((finished - started) * 1000.0, 1)
            self.input_age_at_result_ms = round((finished - observation.stamp) * 1000.0, 1)
            if self.state != "running" or not self.gate.accepts(epoch):
                self.discarded_results += 1
                return False
            self._check_result_age(chunk, finished, "返回")
            if self.inference_device == "qizhi_h200":
                elapsed = max(0.0, finished - started)
                self._remote_latency_estimate_s = (elapsed if self._remote_latency_estimate_s is None
                                                  else max(elapsed, self._remote_latency_estimate_s * 0.8))
            # Relaxed async mode computes exactly one next chunk while the
            # current chunk executes. Hold that result until the chunk boundary
            # so a fast model cannot repeatedly erase most of its own horizon.
            queue_active = len(self.timeline) > 0 or finished < self._relaxed_end_at
            if (self.execution_mode == "async_relaxed" and not self._relaxed_holding
                    and queue_active):
                if self._pending_chunk is not None:
                    self.discarded_results += 1
                    return False
                self._pending_chunk = chunk
                self.inference_count += 1
                self._run_inference_count += 1
                logger.info("async_relaxed next chunk ready: source_sequence=%s input_age_ms=%.1f "
                            "inference_ms=%.1f; waiting for current chunk boundary",
                            observation.sequence, self.input_age_at_result_ms, self.latency_ms)
                return True
            _, kept = self.timeline.replace(chunk, finished)
            self._unstarted_chunk = chunk
            if kept < self.MIN_FRESH_ACTIONS:
                self.timeline.clear()
                self.discarded_results += 1
                raise RuntimeError(f"推理耗时 {self.latency_ms:.1f} ms，观测至返回 "
                                   f"{(finished - observation.stamp) * 1000:.1f} ms；"
                                   f"仅剩 {kept} 个动作，需要至少 {self.MIN_FRESH_ACTIONS} 个")
            # Same transaction as timeline publication; pause cannot be followed
            # by a late reference write from an old epoch.
            reference_stamp = observation.stamp
            if self.execution_mode != "timestamped":
                reference_stamp = self.timeline.start_stamp
                logger.info("%s chunk accepted: source_sequence=%s input_age_ms=%.1f "
                            "inference_ms=%.1f actions=%s execution_start=%.6f",
                            self.execution_mode, observation.sequence, self.input_age_at_result_ms,
                            self.latency_ms, kept, reference_stamp)
            if getattr(self.policy, "reference_mode", "rolling_ae") != "none":
                self.reference.update(execution, source_stamp=reference_stamp)
            self.inference_count += 1
            self._run_inference_count += 1
            return True

    def _check_result_age(self, chunk: ActionChunk, now: float, phase: str) -> None:
        """Bound observation age only at receipt and before a chunk's first command."""
        if self.max_result_age_s is None:
            return
        age = now - chunk.source_stamp
        if not 0 <= age <= self.max_result_age_s:
            self.discarded_results += 1
            raise RuntimeError(f"推理结果在{phase}时已过期：观测年龄 {age * 1000:.1f} ms，"
                               f"上限 {self.max_result_age_s * 1000:.0f} ms；请暂停后检查远端时延")

    def _remote_prefetch_lead_s(self) -> float:
        estimate = self._remote_latency_estimate_s or 0.3
        # Include a sensor/control scheduling margin; cap the waiting horizon so
        # a very fast response cannot age out while the previous chunk executes.
        lead = estimate * 1.15 + 2.0 / self.CONTROL_HZ
        return min(lead, self.max_result_age_s * 0.8) if self.max_result_age_s else lead

    def _remote_prefetch_ready(self, now: float) -> bool:
        remaining = len(self.timeline) / self.CONTROL_HZ + max(0.0, self._relaxed_end_at - now)
        return remaining <= self._remote_prefetch_lead_s()

    def _inference_loop(self) -> None:
        last_version = -1
        while not self.stop_event.is_set():
            self.wake.wait(0.05)
            self.wake.clear()
            with self.lock:
                observation, version = self.observations.get()
                if (self.state != "running" or self.policy is None or observation is None
                        or version == last_version or self._input_history is None):
                    continue
                if not 0 <= self.clock() - observation.stamp <= self.MAX_OBSERVATION_AGE_S:
                    continue
                if self.execution_mode == "synchronous":
                    # A full action chunk and its final control slot must finish
                    # before requesting a stop. Sample only AFTER its acknowledgement.
                    if (not self._relaxed_holding or self._synchronous_ready_at is None
                            or len(self.timeline) or self.clock() < self._relaxed_end_at
                            or observation.stamp <= self._synchronous_ready_at):
                        continue
                    self._synchronous_ready_at = None
                elif self.execution_mode == "async_relaxed" and self._pending_chunk is not None:
                    # Keep one inference in flight per action chunk. The latest
                    # sensor frame is used when the next chunk starts computing.
                    continue
                elif (self.execution_mode == "async_relaxed" and self.inference_device == "qizhi_h200"
                      and not self._remote_prefetch_ready(self.clock())):
                    continue
                last_version = version
                policy, epoch = self.policy, self.gate.current()
                reference = (None if getattr(policy, "reference_mode", "rolling_ae") == "none"
                             else self.reference.get(observation, self.normalizer))
                history = self._input_history
                self._infer_active = True
            started = self.clock()
            try:
                output = policy.infer(observation, reference, history)
                self._publish(output, observation, epoch, started)
            except Exception as exc:
                self._fail("inference", exc, epoch)
            finally:
                with self.lock:
                    self._infer_active = False
                    retired = self.state == "stopped"
                if retired:
                    policy.close()

    def _control_tick(self) -> None:
        with self._command_lock:
            stop_between_chunks = False
            arm_for_chunk = False
            first_chunk = None
            with self.lock:
                if self.state != "running" or self.hardware is None:
                    return
                hardware, epoch, now = self.hardware, self.gate.current(), self.clock()
                observation, _ = self.observations.get()
                waiting_sensor = observation is None and now - self._running_since < self.MAX_OBSERVATION_AGE_S
                if waiting_sensor:
                    return
                if observation is None or not 0 <= now - observation.stamp <= self.MAX_OBSERVATION_AGE_S:
                    detail = ("未收到观测" if observation is None else
                              f"序号 {observation.sequence} 已过 {(now - observation.stamp) * 1000:.1f} ms")
                    raise RuntimeError(f"观测中断/过期，停止动作（{detail}；"
                                       f"上限 {self.MAX_OBSERVATION_AGE_S * 1000:.0f} ms）")
                if self.execution_mode != "timestamped":
                    action = self.timeline.take(now, epoch)
                    if (action is None and self.execution_mode == "async_relaxed"
                            and self._pending_chunk is not None
                            and (self._relaxed_holding or (self.inference_device == "qizhi_h200"
                                 and not len(self.timeline) and now >= self._relaxed_end_at))):
                        chunk, self._pending_chunk = self._pending_chunk, None
                        self._check_result_age(chunk, now, "等待结束")
                        self.timeline.replace(chunk, now)
                        self._unstarted_chunk = chunk
                        if getattr(self.policy, "reference_mode", "rolling_ae") != "none":
                            self.reference.update(chunk.actions, source_stamp=self.timeline.start_stamp)
                        action = self.timeline.take(now, epoch)
                    if action is None:
                        if len(self.timeline) or now < self._relaxed_end_at or self._relaxed_holding:
                            return
                        # An empty horizon is an expected waiting phase for a
                        # slow model. Stop once; keep sensors/inference running.
                        self._relaxed_holding = True
                        stop_between_chunks = True
                    arm_for_chunk = action is not None and self._relaxed_holding
                else:
                    action = self.timeline.take(now, epoch)
                    waiting_first = (not self._run_inference_count and
                                     now - self._running_since < self.FIRST_RESULT_TIMEOUT_S)
                    if action is None:
                        # A newly published chunk may begin at the next control slot.
                        if len(self.timeline) or waiting_first:
                            return
                        raise RuntimeError("动作缓冲耗尽；停止动作")
                if (action is not None and self._unstarted_chunk is not None
                        and self._unstarted_chunk.source_sequence == action.source_sequence):
                    first_chunk = self._unstarted_chunk
            if stop_between_chunks:
                # Already serialized with commands. Do not hold the state lock
                # over the SDK call: pause/estop can still invalidate this epoch.
                hardware.slow_stop()
                with self.lock:
                    if (self.execution_mode == "synchronous" and self.state == "running"
                            and self.gate.accepts(epoch)):
                        self._synchronous_ready_at = self.clock()
                logger.info("%s waiting for next action chunk in epoch %s", self.execution_mode, epoch)
                self.wake.set()
                return
            if arm_for_chunk:
                hardware.arm()
                with self.lock:
                    if self.state != "running" or not self.gate.accepts(epoch):
                        return
                    self._relaxed_holding = False
                logger.info("%s resumed action execution in epoch %s", self.execution_mode, epoch)
            if action is None:
                return
            with self.lock:
                if self.state != "running" or not self.gate.accepts(epoch):
                    return
                if first_chunk is not None:
                    self._check_result_age(first_chunk, self.clock(), "首次执行")
                    self.input_age_at_first_action_ms = round(
                        (self.clock() - first_chunk.source_stamp) * 1000.0, 1)
                    if self._unstarted_chunk is first_chunk:
                        self._unstarted_chunk = None
                command = self._smooth_command(action.action, observation)
            hardware.command(command)
            if self.execution_mode != "timestamped":
                with self.lock:
                    if self.state == "running" and self.gate.accepts(epoch):
                        self._relaxed_end_at = self.clock() + self.timeline.dt

    def _smooth_command(self, action, observation) -> tuple[float, ...]:
        """EMA of physical Joint8 joint targets; SDK limiters remain authoritative."""
        command = tuple(action)
        if self.normalizer.action_space != "joint8" or self.action_smoothing_alpha == 1.0:
            return command
        previous = self._smoothed_action or tuple(observation.state)
        alpha = self.action_smoothing_alpha
        command = tuple(old + alpha * (target - old) for old, target in zip(previous[:7], command[:7])) + (command[-1],)
        self._smoothed_action = command
        return command

    def _control_loop(self) -> None:
        deadline = self.clock()
        while not self.stop_event.is_set():
            epoch = self.gate.current()
            try:
                self._control_tick()
            except Exception as exc:
                self._fail("control", exc, epoch)
            deadline = max(deadline + 1.0 / self.CONTROL_HZ, self.clock())
            self.stop_event.wait(max(0.0, deadline - self.clock()))

    def _fail(self, stage: str, exc: Exception, epoch: int) -> None:
        logger.error("%s failed in epoch %s: %s", stage, epoch, exc,
                     exc_info=(type(exc), exc, exc.__traceback__))
        with self.lock:
            if self.state != "running" or not self.gate.accepts(epoch):
                return
            failed_epoch = self._invalidate("error", f"{stage}: {exc}")
            hardware = self.hardware
        try:
            self._stop_hardware(hardware, emergency=False)
        except Exception as stop_exc:
            with self.lock:
                if self.gate.accepts(failed_epoch):
                    self.error += f"；停机确认失败: {stop_exc}"

    def status(self) -> dict:
        with self.lock:
            hardware = self.hardware.status() if self.hardware else {}
            observation, _ = self.observations.get()
            result = RuntimeStatus(state=self.state, epoch=self.gate.current(),
                observation_sequence=observation.sequence if observation else -1,
                action_queue=len(self.timeline), inference_count=self.inference_count,
                discarded_results=self.discarded_results, latency_ms=self.latency_ms,
                error=self.error, reference_ready=self.reference.ready(), model_ready=self.policy is not None,
                robot_ready=bool(hardware.get("robot_ready")), cameras_ready=bool(hardware.get("cameras_ready")),
                force_ready=bool(hardware.get("force_ready")), gripper_ready=bool(hardware.get("gripper_ready"))).as_dict()
            phase = self.state
            if self.execution_mode != "timestamped" and self.state == "running":
                executing = bool(len(self.timeline)) or self.clock() < self._relaxed_end_at
                phase = ("executing_and_inferring" if self._infer_active else "executing") if executing else "waiting"
                if self.execution_mode == "synchronous" and not executing:
                    phase = "inferring" if self._infer_active else "waiting_for_observation"
            no_reference = getattr(self.policy, "reference_mode", "rolling_ae") == "none"
            if no_reference:
                result["reference_ready"] = False
            result.update(reference_mode="none" if no_reference else "rolling_ae",
                          bootstrap=None if no_reference else self.reference.bootstrap(),
                          inference_busy=self._infer_active, hardware=hardware,
                          execution_mode=self.execution_mode,
                          sampling_profile=self.sampling_profile,
                          inference_device=self.inference_device,
                          max_result_age_s=self.max_result_age_s,
                          action_smoothing_alpha=self.action_smoothing_alpha,
                          remote_prefetch_lead_ms=(round(self._remote_prefetch_lead_s() * 1000.0, 1)
                                                   if self.inference_device == "qizhi_h200" else None),
                          execution_phase=phase,
                          input_age_at_result_ms=self.input_age_at_result_ms,
                          input_age_at_first_action_ms=self.input_age_at_first_action_ms,
                          pending_chunk=self._pending_chunk is not None,
                          gripper_debug=self.gripper_debug_status(),
                          gripper_debug_busy=self.gripper_debug_status()['busy'],
                          hardware_connecting=self._hardware_connecting,
                          estop_latched=self._estop_latched,
                          estop_reset_busy=self._estop_reset_busy,
                          estop_reset_supported=self.hardware is None or callable(
                              getattr(self.hardware, "reset_emergency_stop", None)),
                          model_diagnostics=dict(getattr(self.policy, "diagnostics", {})))
        result["cuda"] = self._cuda_status()
        return result

    def _cuda_status(self) -> dict:
        return self.cuda_probe.status()

    def close(self) -> None:
        try:
            self.pause()
        finally:
            with self.lock:
                self._invalidate("stopped")
                policy, hardware = self.policy, self.hardware
                debug_gripper, self._debug_gripper = self._debug_gripper, None
                active = self._infer_active
                self.policy = self.hardware = None
            self.stop_event.set()
            self.wake.set()
            deadline = time.monotonic() + 2.0
            for thread in self._threads:
                if thread is not threading.current_thread():
                    thread.join(timeout=max(0.0, deadline - time.monotonic()))
            # A CUDA call cannot be safely killed inside a Python thread.
            # An active inference owns its policy until it returns.
            if policy and not active:
                policy.close()
            try:
                if hardware:
                    hardware.close()
            finally:
                try:
                    if debug_gripper:
                        debug_gripper.close()
                finally:
                    self.cuda_probe.close()
