"""Independent sensor, inference, control and web loops with bounded buffers."""
from __future__ import annotations

import logging
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
                 sampling_profile: str | None = None):
        if execution_mode not in {"timestamped", "async_relaxed", "synchronous"}:
            raise ValueError("execution_mode must be timestamped, async_relaxed or synchronous")
        self.execution_mode = execution_mode
        self.sampling_profile = sampling_profile
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
        self._relaxed_holding = False
        self._relaxed_end_at = 0.0
        self._pending_chunk: ActionChunk | None = None
        self._synchronous_ready_at: float | None = None
        self.input_age_at_result_ms: float | None = None

    def _require_inactive(self) -> None:
        if self._estop_latched:
            raise RuntimeError("急停已锁存；检查机械臂并在本地复位后重启部署服务")
        if self.state in ("running", "loading", "stopping", "stopped") or self._infer_active:
            raise RuntimeError("先暂停并等待当前推理/加载结束，再替换适配器")
        if self._load_thread and self._load_thread.is_alive():
            raise RuntimeError("上一模型仍在加载；请等待完成")

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

    def configure_sampling(self, profile: str, mode: str, *, persist=None) -> None:
        """Change an inactive session atomically; new settings require model reload."""
        from .sampling import updated_config
        updated_config({}, profile, mode)
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
            self.timeline = ChunkActionQueue(self.CONTROL_HZ, max_actions=32)
            self.latency_ms = self.input_age_at_result_ms = None
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
            if self.require_cuda and not self._cuda_status().get("available"):
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
            if self.state in ("idle", "ready", "paused", "estop", "stopped", "stopping"):
                return
            epoch = self._invalidate("stopping")
            hardware = self.hardware
        self.wake.set()
        try:
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
            epoch = self._invalidate("estop", "急停已锁存；检查机械臂并在本地复位后重启部署服务")
            hardware = self.hardware
        self.wake.set()
        try:
            self._stop_hardware(hardware, emergency=True)
        except Exception as exc:
            with self.lock:
                if self.gate.accepts(epoch):
                    self.error += f"；硬件急停未确认: {exc}"
            raise

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
                            and self._relaxed_holding and self._pending_chunk is not None):
                        chunk, self._pending_chunk = self._pending_chunk, None
                        self.timeline.replace(chunk, now)
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
            hardware.command(action.action)
            if self.execution_mode != "timestamped":
                with self.lock:
                    if self.state == "running" and self.gate.accepts(epoch):
                        self._relaxed_end_at = self.clock() + self.timeline.dt

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
                          execution_phase=phase,
                          input_age_at_result_ms=self.input_age_at_result_ms,
                          pending_chunk=self._pending_chunk is not None,
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
            if hardware:
                hardware.close()
            self.cuda_probe.close()
