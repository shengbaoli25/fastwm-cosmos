"""Synchronous obs -> infer -> execute -> confirmed hold -> fresh obs loop."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import logging
import math
import threading
import time
import uuid

from .wire import PROTOCOL, encode_observation, validate_response

logger = logging.getLogger(__name__)


def _positive(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a positive finite number")
    return float(value)


def validate_metadata(health):
    if not isinstance(health, dict) or health.get("protocol") != PROTOCOL:
        raise ValueError("server protocol mismatch")
    meta = health.get("metadata")
    if not isinstance(meta, dict):
        raise ValueError("server health must contain metadata")
    result = dict(meta)
    if result.get("action_space") != "joint8":
        raise ValueError("client requires joint8 absolute joint radians + gripper openness")
    for key in ("width", "height", "action_horizon", "history_length", "video_context_frames"):
        if key == "video_context_frames":
            result.setdefault(key, 1)
        if type(result.get(key)) is not int or result[key] <= 0:
            raise ValueError(f"metadata {key} must be a positive integer")
    if result["action_horizon"] != 32 or result["history_length"] != 32:
        raise ValueError("client requires the trained 32-step action and observation history contract")
    if result["video_context_frames"] > 32 or result["width"] * result["height"] > 4096 * 4096:
        raise ValueError("unsupported video context or image size")
    result.setdefault("video_context_stride", 1)
    if type(result["video_context_stride"]) is not int or not 1 <= result["video_context_stride"] <= 32:
        raise ValueError("video_context_stride must be an integer in [1, 32]")
    if 1 + (result["video_context_frames"] - 1) * result["video_context_stride"] > 32:
        raise ValueError("video context exceeds the 32 observation history window")
    if _positive(result.get("control_hz"), "control_hz") != 30.0:
        raise ValueError("client requires the trained 30 Hz control/observation grid")
    names = result.get("camera_names")
    if not isinstance(names, list) or names != ["global", "wrist"]:
        raise ValueError("camera_names must be [global, wrist] in training order")
    return result


def history_row(observation):
    return dict(sequence=observation.sequence, stamp=observation.stamp,
                state=list(observation.state), force=list(observation.force),
                force_meta=dict(observation.force_meta))


def observation_row(observation, metadata):
    row = history_row(observation)
    row.update(action_space=observation.action_space, frame_meta=dict(observation.frame_meta),
               frames={name: dict(width=metadata["width"], height=metadata["height"],
                                  data=observation.frames[name])
                       for name in metadata["camera_names"]})
    return encode_observation(row)


class ObservationCollector:
    """Continuously sample measured history, including while HTTP is blocked."""
    def __init__(self, hardware, metadata, *, max_age_s=.12, startup_timeout_s=10.0):
        self.hardware = hardware
        self.metadata = metadata
        self.period_s = 1.0 / metadata["control_hz"]
        self.max_age_s = _positive(max_age_s, "max_observation_age_s")
        self.startup_timeout_s = _positive(startup_timeout_s, "startup_timeout_s")
        self.condition = threading.Condition()
        self.history = deque(maxlen=metadata["history_length"])
        self.context_window = 1 + (metadata["video_context_frames"] - 1) * metadata["video_context_stride"]
        self.video = deque(maxlen=self.context_window)
        self.error = None
        self.started = False
        self.stop_event = threading.Event()
        self.thread = None

    def start(self):
        self.thread = threading.Thread(target=self._run, name="realman-observation-30hz", daemon=True)
        self.thread.start()

    def _append(self, observation, now):
        if not 0 <= now - observation.stamp <= self.max_age_s:
            raise ValueError("robot/camera observation is stale")
        if observation.action_space != self.metadata["action_space"]:
            raise ValueError("hardware observation action space differs from server")
        expected = self.metadata.get("force_preprocessing")
        expected_id = self.metadata.get("force_preprocessing_id") or (
            expected.get("id") if isinstance(expected, dict) else None)
        if expected_id and observation.force_meta.get("preprocessing_id") != expected_id:
            raise ValueError("hardware force preprocessing differs from trained model")
        names = self.metadata["camera_names"]
        if set(observation.frames) != set(names):
            raise ValueError("hardware camera identities differ from trained model")
        for name in names:
            if len(observation.frames[name]) != self.metadata["width"] * self.metadata["height"] * 3:
                raise ValueError("hardware RGB resolution differs from server")
        if self.history:
            previous = self.history[-1]
            if observation.sequence <= previous.sequence or observation.stamp <= previous.stamp:
                raise ValueError("observation sequence/timestamp did not advance")
            if observation.stamp - previous.stamp > self.period_s * 1.5:
                raise ValueError("30 Hz observation history has a gap")
            if observation.force_meta.get("epoch") != previous.force_meta.get("epoch"):
                self.history.clear()
                self.video.clear()
        self.history.append(observation)
        self.video.append(observation)
        self.started = True

    def _run(self):
        deadline = time.monotonic()
        startup_deadline = deadline + self.startup_timeout_s
        while not self.stop_event.is_set():
            try:
                observation = self.hardware.observe_at(deadline)
                if observation is None:
                    raise RuntimeError("hardware returned no observation")
                with self.condition:
                    self._append(observation, time.monotonic())
                    self.condition.notify_all()
            except Exception as exc:
                with self.condition:
                    if self.started or time.monotonic() >= startup_deadline:
                        self.error = exc
                        self.condition.notify_all()
                        return
            deadline += self.period_s
            now = time.monotonic()
            if deadline < now:
                with self.condition:
                    if self.started:
                        self.error = RuntimeError("30 Hz observation grid overrun")
                        self.condition.notify_all()
                        return
                deadline = now + self.period_s
            self.stop_event.wait(max(0, deadline - time.monotonic()))

    def check(self):
        with self.condition:
            if self.error is not None:
                raise RuntimeError(f"sensor collection failed: {self.error}") from self.error
            if not self.history or not 0 <= time.monotonic() - self.history[-1].stamp <= self.max_age_s:
                raise RuntimeError("latest observation is stale")

    def snapshot_after(self, stamp, sequence, *, timeout_s):
        deadline = time.monotonic() + timeout_s
        with self.condition:
            while True:
                if self.error is not None:
                    raise RuntimeError(f"sensor collection failed: {self.error}") from self.error
                if (len(self.history) == self.metadata["history_length"]
                        and len(self.video) == self.context_window
                        and self.history[-1].stamp > stamp and self.history[-1].sequence > sequence):
                    self.check()
                    video = tuple(self.video)[::self.metadata["video_context_stride"]]
                    return self.history[-1], tuple(self.history), video
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("timed out waiting for fresh 30 Hz observation history")
                self.condition.wait(min(.1, remaining))

    def close(self):
        self.stop_event.set()
        if self.thread:
            self.thread.join(timeout=2.0)


def validate_actions(response, request, metadata, *, now, max_source_age_s):
    actions = validate_response(response, request, metadata)
    # Both values are from this robot host. Server monotonic clocks are unrelated.
    if not 0 <= now - request["observation"]["stamp"] <= max_source_age_s:
        raise ValueError("action result expired on the client clock")
    if len(actions) != metadata["action_horizon"]:
        raise ValueError("response must contain the full trained action horizon")
    return tuple(tuple(row) for row in actions)


def validate_hardware_chunk(hardware, actions, measured_state):
    """Preflight all joint targets before arm; native command checks remain active."""
    lower, upper = getattr(hardware, "joint_min", None), getattr(hardware, "joint_max", None)
    if lower is None or upper is None:
        return
    config = hardware.config
    margin = float(config.get("joint_limit_margin_deg", 2.0))
    max_step = config.get("max_joint_step_deg")
    previous = tuple(math.degrees(x) for x in measured_state[:7])
    for row in actions:
        target = tuple(math.degrees(x) for x in row[:7])
        if any(x < lo + margin or x > hi - margin for x, lo, hi in zip(target, lower, upper)):
            raise ValueError("action chunk enters native joint soft-limit protection zone")
        if max_step is not None and max(abs(x - old) for x, old in zip(target, previous)) > max_step:
            raise ValueError("action chunk exceeds configured joint step limit")
        previous = target


@dataclass(frozen=True)
class ClientOptions:
    execute_steps: int = 32
    max_result_age_s: float = 120.0
    sensor_timeout_s: float = 10.0
    max_observation_age_s: float = .12

    def __post_init__(self):
        if type(self.execute_steps) is not int or not 1 <= self.execute_steps <= 32:
            raise ValueError("execute_steps must be an integer in [1, 32]")
        _positive(self.max_result_age_s, "max_result_age_s")
        _positive(self.sensor_timeout_s, "sensor_timeout_s")
        _positive(self.max_observation_age_s, "max_observation_age_s")


class SynchronousClient:
    def __init__(self, transport, hardware, metadata, *, options=None):
        self.transport, self.hardware, self.metadata = transport, hardware, metadata
        self.options = options or ClientOptions()
        if type(self.options.execute_steps) is not int or not 1 <= self.options.execute_steps <= metadata["action_horizon"]:
            raise ValueError("execute_steps must be within the model action horizon")
        _positive(self.options.max_result_age_s, "max_result_age_s")
        self.session_id = str(uuid.uuid4())
        self.completed_cycles = 0
        self.stop_event = threading.Event()
        self.collector = ObservationCollector(hardware, metadata,
            max_age_s=self.options.max_observation_age_s, startup_timeout_s=self.options.sensor_timeout_s)

    def _request(self, observation, history, video):
        request = dict(protocol=PROTOCOL, session_id=self.session_id, request_id=str(uuid.uuid4()),
                       observation=observation_row(observation, self.metadata),
                       history=[history_row(item) for item in history],
                       force_preprocessing=self.hardware.status().get("force_preprocessing"))
        if self.metadata["video_context_frames"] > 1:
            request["video_context"] = [observation_row(item, self.metadata) for item in video]
        return request

    def _wait_control(self, duration):
        deadline = time.monotonic() + duration
        while True:
            self.collector.check()
            if self.stop_event.is_set():
                raise InterruptedError("client stop requested")
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return
            self.stop_event.wait(min(.01, remaining))

    def _execute(self, actions):
        self.collector.check()
        with self.collector.condition:
            current = self.collector.history[-1]
        validate_hardware_chunk(self.hardware, actions, current.state)
        self.hardware.arm()
        period = 1.0 / self.metadata["control_hz"]
        for action in actions[:self.options.execute_steps]:
            self.collector.check()
            if self.stop_event.is_set():
                raise InterruptedError("client stop requested")
            started = time.monotonic()
            self.hardware.command(action)
            remaining = period - (time.monotonic() - started)
            if remaining <= 0:
                raise RuntimeError("30 Hz hardware command period overrun; refusing a catch-up burst")
            # Account for SDK latency inside the 30 Hz period. The next slot is
            # anchored to this command's actual start, never a past deadline.
            self._wait_control(remaining)
        self.hardware.slow_stop()  # Returns only after native stop acknowledgement.
        return time.monotonic()

    def run(self, *, cycles=0):
        if type(cycles) is not int or cycles < 0:
            raise ValueError("cycles must be a non-negative integer (0 means continuous)")
        last_sequence = -1
        pending_error = None
        try:
            self.hardware.slow_stop()
            fresh_after = time.monotonic()
            self.collector.start()
            while not self.stop_event.is_set() and (cycles == 0 or self.completed_cycles < cycles):
                observation, history, video = self.collector.snapshot_after(fresh_after, last_sequence,
                    timeout_s=self.options.sensor_timeout_s)
                request = self._request(observation, history, video)
                self.collector.check()
                started = time.monotonic()
                response = self.transport.infer(request)
                self.collector.check()
                with self.collector.condition:
                    if self.collector.history[-1].force_meta.get("epoch") != observation.force_meta.get("epoch"):
                        raise ValueError("force filter epoch changed while awaiting this action result")
                actions = validate_actions(response, request, self.metadata, now=time.monotonic(),
                                           max_source_age_s=self.options.max_result_age_s)
                if self.stop_event.is_set():
                    raise InterruptedError("client stop requested")
                inference_s = time.monotonic() - started
                fresh_after = self._execute(actions)
                last_sequence = observation.sequence
                self.completed_cycles += 1
                logger.info("cycle=%d sequence=%d infer_s=%.3f executed=%d",
                            self.completed_cycles, last_sequence, inference_s, self.options.execute_steps)
            return self.completed_cycles
        except BaseException as exc:
            pending_error = exc
            raise
        finally:
            self.stop_event.set()
            cleanup_errors = []
            try:
                self.hardware.slow_stop()
            except Exception as exc:
                cleanup_errors.append(f"slow-stop acknowledgement failed: {exc}")
                logger.error("native slow-stop acknowledgement failed: %s", exc)
            self.collector.close()
            try:
                self.hardware.close()
            except Exception as exc:
                cleanup_errors.append(f"hardware close failed: {exc}")
            if hasattr(self.transport, "release"):
                try:
                    self.transport.release(self.session_id)
                except Exception as exc:
                    logger.warning("server session release failed: %s", exc)
            if cleanup_errors:
                raise RuntimeError("; ".join(cleanup_errors)) from pending_error
