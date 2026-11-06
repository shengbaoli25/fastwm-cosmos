"""Synchronous obs -> infer -> execute -> confirmed hold -> fresh obs loop."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import logging
import math
import json
import threading
import time
from . import clock
import uuid

from .wire import PROTOCOL, encode_observation, validate_response, temporal_tolerances

logger = logging.getLogger(__name__)


class ObservationDriftError(RuntimeError):
    """The inference source no longer describes the arm; require a new run."""

    recoverable = False


class MotionNotSettledError(RuntimeError):
    """Physical settling could not be confirmed; require a new run."""

    recoverable = False


def _positive(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a positive finite number")
    return float(value)


def validate_metadata(health):
    if not isinstance(health, dict) or health.get("protocol") != PROTOCOL:
        raise ValueError("server protocol mismatch")
    if health.get("ready") is not True:
        raise ValueError("server is not ready for inference")
    meta = health.get("metadata")
    if not isinstance(meta, dict):
        raise ValueError("server health must contain metadata")
    result = dict(meta)
    temporal_tolerances(result)
    if result.get("action_space") != "joint8":
        raise ValueError("client requires joint8 absolute joint radians + gripper openness")
    # Legacy joint8 servers already imply these conventions. Explicit new
    # declarations must agree; never silently accumulate an incremental chunk.
    for key, expected in (("action_semantics", "absolute"), ("joint_unit", "radian"),
                          ("gripper_semantics", "absolute_openness_0_1")):
        if key in result and result[key] != expected:
            raise ValueError(f"server {key} must be {expected}")
    for key in ("width", "height", "action_horizon", "history_length", "video_context_frames"):
        if key == "video_context_frames":
            result.setdefault(key, 1)
        if type(result.get(key)) is not int or result[key] <= 0:
            raise ValueError(f"metadata {key} must be a positive integer")
    if result["action_horizon"] > 32 or result["history_length"] != 32:
        raise ValueError("client requires action_horizon in [1, 32] and 32-step observation history")
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
    def __init__(self, hardware, metadata, *, max_age_s=.12, startup_timeout_s=10.0,
                 stability_window_s=.2):
        self.hardware = hardware
        self.metadata = metadata
        self.period_s = 1.0 / metadata["control_hz"]
        self.max_age_s = _positive(max_age_s, "max_observation_age_s")
        self.startup_timeout_s = _positive(startup_timeout_s, "startup_timeout_s")
        self.condition = threading.Condition()
        self.history = deque(maxlen=metadata["history_length"])
        # Keep the model's history length unchanged, even for a longer physical
        # settling window. Entries are the same immutable measured observations.
        self.stability_history = deque(maxlen=max(metadata["history_length"],
            math.ceil(_positive(stability_window_s, "stability_window_s") / self.period_s) + 2))
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
                self.stability_history.clear()
        self.history.append(observation)
        self.stability_history.append(observation)
        self.video.append(observation)
        self.started = True

    def _run(self):
        deadline = clock.now()
        startup_deadline = deadline + self.startup_timeout_s
        while not self.stop_event.is_set():
            # Recheck the actual high-resolution deadline after every sleep.
            # Keep the fixed 30 Hz grid and freshness checks unchanged.
            if not clock.wait_until(deadline, self.stop_event):
                return
            try:
                observation = self.hardware.observe_at(deadline)
                if observation is None:
                    raise RuntimeError("hardware returned no observation")
                with self.condition:
                    self._append(observation, clock.now())
                    self.condition.notify_all()
            except Exception as exc:
                with self.condition:
                    if self.started or clock.now() >= startup_deadline:
                        self.error = exc
                        self.condition.notify_all()
                        return
            deadline += self.period_s
            now = clock.now()
            if deadline < now:
                with self.condition:
                    if self.started:
                        self.error = RuntimeError("30 Hz observation grid overrun")
                        self.condition.notify_all()
                        return
                deadline = now + self.period_s

    def check(self):
        with self.condition:
            if self.error is not None:
                raise RuntimeError(f"sensor collection failed: {self.error}") from self.error
            if not self.history or not 0 <= clock.now() - self.history[-1].stamp <= self.max_age_s:
                raise RuntimeError("latest observation is stale")

    def snapshot_after(self, stamp, sequence, *, timeout_s):
        deadline = clock.now() + timeout_s
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
                remaining = deadline - clock.now()
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
    settle_window_s: float = .2
    settle_tolerance_deg: float = .1
    settle_velocity_deg_s: float = .5
    settle_timeout_s: float = 5.0
    max_inference_drift_deg: float = .5

    def __post_init__(self):
        if type(self.execute_steps) is not int or not 1 <= self.execute_steps <= 32:
            raise ValueError("execute_steps must be an integer in [1, 32]")
        _positive(self.max_result_age_s, "max_result_age_s")
        _positive(self.sensor_timeout_s, "sensor_timeout_s")
        _positive(self.max_observation_age_s, "max_observation_age_s")
        for name in ("settle_window_s", "settle_tolerance_deg", "settle_velocity_deg_s",
                     "settle_timeout_s", "max_inference_drift_deg"):
            _positive(getattr(self, name), name)
        if self.settle_window_s > 5.0:
            raise ValueError("settle_window_s must not exceed 5 seconds")


class SynchronousClient:
    def __init__(self, transport, hardware, metadata, *, options=None):
        self.transport, self.hardware, self.metadata = transport, hardware, metadata
        self.options = options or ClientOptions(execute_steps=metadata["action_horizon"])
        if type(self.options.execute_steps) is not int or not 1 <= self.options.execute_steps <= metadata["action_horizon"]:
            raise ValueError("execute_steps must be within the model action horizon")
        _positive(self.options.max_result_age_s, "max_result_age_s")
        self.session_id = str(uuid.uuid4())
        self.completed_cycles = 0
        self.stop_event = threading.Event()
        self._source_observation = None
        self.last_settle_metrics = None
        self.last_execution_guard = None
        self.collector = ObservationCollector(hardware, metadata,
            max_age_s=self.options.max_observation_age_s, startup_timeout_s=self.options.sensor_timeout_s,
            stability_window_s=self.options.settle_window_s)

    def snapshot_for_inference(self, after, sequence, *, timeout_s=None):
        """Require a complete measured stationary window after the stop ACK.

        This confirms physical settling independently of whether the controller
        reached the final target. No targets or synthetic history are produced.
        """
        started = clock.now()
        # Sensor startup has its own budget. Only a complete, fresh model
        # history starts the physical-settling deadline. An explicit timeout
        # additionally caps the entire operation (useful to bounded callers).
        overall_deadline = None if timeout_s is None else started + _positive(timeout_s, "timeout_s")
        startup_deadline = started + self.options.sensor_timeout_s
        settling_deadline = None
        collector = self.collector
        self.last_settle_metrics = None
        with collector.condition:
            while True:
                if self.stop_event.is_set():
                    raise InterruptedError("client stop requested while waiting for physical settling")
                deadline = startup_deadline if settling_deadline is None else settling_deadline
                if overall_deadline is not None:
                    deadline = min(deadline, overall_deadline)
                remaining = deadline - clock.now()
                if remaining <= 0:
                    if settling_deadline is None:
                        raise TimeoutError("timed out waiting for fresh 30 Hz observation history before settling")
                    logger.error("Physical settling failed %s", json.dumps(dict(
                        after=after, timeout_s=self.options.settle_timeout_s,
                        last_measurement=self.last_settle_metrics),
                        allow_nan=False))
                    raise MotionNotSettledError(
                        "physical settling timed out; no inference or motion was started; explicit restart required")
                if collector.error is not None:
                    raise RuntimeError(f"sensor collection failed: {collector.error}") from collector.error
                if (len(collector.history) == self.metadata["history_length"]
                        and len(collector.video) == collector.context_window):
                    collector.check()
                    latest = collector.history[-1]
                    if latest.stamp > after and latest.sequence > sequence:
                        if settling_deadline is None:
                            settling_deadline = clock.now() + self.options.settle_timeout_s
                        # Include the sample immediately preceding the cutoff so
                        # the measured window covers at least the full duration.
                        cutoff = latest.stamp - self.options.settle_window_s
                        rows = tuple(collector.stability_history)
                        begin = next((i for i in range(len(rows) - 1, -1, -1)
                                      if rows[i].stamp <= cutoff), None)
                        if begin is not None:
                            window = rows[begin:]
                            if window[0].stamp > after and len(window) >= 2:
                                spans = [b.stamp - a.stamp for a, b in zip(window, window[1:])]
                                if any(dt <= 0 or dt > collector.period_s * 1.5
                                       for dt in spans):
                                    raise RuntimeError("30 Hz physical settling history has a gap")
                                q = [tuple(math.degrees(x) for x in row.state[:7]) for row in window]
                                peak = [max(row[j] for row in q) - min(row[j] for row in q)
                                        for j in range(7)]
                                velocity = [max(abs(b[j] - a[j]) / dt
                                                for a, b, dt in zip(q, q[1:], spans))
                                            for j in range(7)]
                                metrics = dict(source_sequence=latest.sequence,
                                    observation_stamp=latest.stamp, after=after,
                                    window_s=window[-1].stamp-window[0].stamp,
                                    samples=len(window), peak_to_peak_deg=peak,
                                    max_velocity_deg_s=velocity,
                                    tolerance_deg=self.options.settle_tolerance_deg,
                                    velocity_limit_deg_s=self.options.settle_velocity_deg_s,
                                    wait_ms=(clock.now()-started)*1000)
                                self.last_settle_metrics = metrics
                                if (max(peak) <= self.options.settle_tolerance_deg
                                        and max(velocity) <= self.options.settle_velocity_deg_s):
                                    logger.info("Physical settling %s", json.dumps(metrics, allow_nan=False))
                                    video = tuple(collector.video)[::self.metadata["video_context_stride"]]
                                    return latest, tuple(collector.history), video
                collector.condition.wait(min(.05, remaining))

    def _request(self, observation, history, video):
        request = dict(protocol=PROTOCOL, session_id=self.session_id, request_id=str(uuid.uuid4()),
                       observation=observation_row(observation, self.metadata),
                       history=[history_row(item) for item in history],
                       force_preprocessing=self.hardware.status().get("force_preprocessing"))
        if self.metadata["video_context_frames"] > 1:
            # Collector video ends with the very same immutable Observation.
            # Keep both wire entries but avoid encoding its two RGB views twice.
            request["video_context"] = [request["observation"] if item is observation
                                        else observation_row(item, self.metadata) for item in video]
        self._source_observation = observation
        return request

    def _wait_control(self, duration):
        deadline = clock.now() + duration
        if not clock.wait_until(deadline, self.stop_event, check=self.collector.check):
            raise InterruptedError("client stop requested")

    def _check_inference_drift(self, observation, first_action, *, phase):
        if self.stop_event.is_set():
            raise InterruptedError("client stop requested")
        if observation is None or observation.action_space != "joint8":
            raise ValueError("joint8 inference source observation is required before execution")
        measured, stamp = self.hardware.joint_feedback()
        measured = tuple(measured)
        if (len(measured) != 7 or any(isinstance(x, bool) or not isinstance(x, (int, float))
                                     or not math.isfinite(x) for x in measured)):
            raise ValueError("joint_feedback must contain seven finite measured radians")
        now = clock.now()
        if (isinstance(stamp, bool) or not isinstance(stamp, (int, float))
                or not math.isfinite(stamp)
                or not 0 <= now-stamp <= self.options.max_observation_age_s):
            raise RuntimeError("joint feedback is stale before action dispatch")
        drift = [abs(math.degrees(new-old)) for new, old in zip(measured, observation.state[:7])]
        metrics = dict(source_sequence=observation.sequence, observation_stamp=observation.stamp,
            phase=phase, feedback_stamp=stamp, feedback_age_ms=(now-stamp)*1000,
            q_obs_deg=[math.degrees(x) for x in observation.state[:7]],
            q_exec_deg=[math.degrees(x) for x in measured],
            first_target_deg=[math.degrees(x) for x in first_action[:7]],
            drift_deg=drift, max_drift_deg=max(drift),
            drift_limit_deg=self.options.max_inference_drift_deg,
            accepted=max(drift) <= self.options.max_inference_drift_deg,
            settling=self.last_settle_metrics)
        self.last_execution_guard = metrics
        logger.info("Inference motion guard %s", json.dumps(metrics, allow_nan=False))
        if not metrics["accepted"]:
            raise ObservationDriftError(
                f"inference observation drifted {max(drift):.3f} deg "
                f"(limit {self.options.max_inference_drift_deg:.3f} deg); "
                "action discarded; explicit restart required")
        if self.stop_event.is_set():
            raise InterruptedError("client stop requested")
        return measured

    def _execute(self, actions, *, source_observation=None):
        if self.stop_event.is_set():
            raise InterruptedError("client stop requested")
        if not actions:
            raise ValueError("cannot execute an empty action chunk")
        self.collector.check()
        source = source_observation if source_observation is not None else self._source_observation
        measured = self._check_inference_drift(source, actions[0], phase="before_arm")
        validate_hardware_chunk(self.hardware, actions, measured)
        if self.stop_event.is_set():
            raise InterruptedError("client stop requested")
        self.hardware.arm()
        period = 1.0 / self.metadata["control_hz"]
        for index, action in enumerate(actions[:self.options.execute_steps]):
            self.collector.check()
            if self.stop_event.is_set():
                raise InterruptedError("client stop requested")
            if index == 0:
                self._check_inference_drift(source, action, phase="before_first_command")
            started = clock.now()
            self.hardware.command(action)
            remaining = period - (clock.now() - started)
            if remaining <= 0:
                raise RuntimeError("30 Hz hardware command period overrun; refusing a catch-up burst")
            # Account for SDK latency inside the 30 Hz period. The next slot is
            # anchored to this command's actual start, never a past deadline.
            self._wait_control(remaining)
        self.hardware.slow_stop()  # Returns only after native stop acknowledgement.
        return clock.now()

    def run(self, *, cycles=0):
        if type(cycles) is not int or cycles < 0:
            raise ValueError("cycles must be a non-negative integer (0 means continuous)")
        last_sequence = -1
        pending_error = None
        try:
            self.hardware.slow_stop()
            fresh_after = clock.now()
            self.collector.start()
            while not self.stop_event.is_set() and (cycles == 0 or self.completed_cycles < cycles):
                observation, history, video = self.snapshot_for_inference(fresh_after, last_sequence)
                if self.stop_event.is_set():
                    raise InterruptedError("client stop requested")
                request = self._request(observation, history, video)
                self.collector.check()
                if self.stop_event.is_set():
                    raise InterruptedError("client stop requested")
                started = clock.now()
                response = self.transport.infer(request)
                self.collector.check()
                with self.collector.condition:
                    if self.collector.history[-1].force_meta.get("epoch") != observation.force_meta.get("epoch"):
                        raise ValueError("force filter epoch changed while awaiting this action result")
                actions = validate_actions(response, request, self.metadata, now=clock.now(),
                                           max_source_age_s=self.options.max_result_age_s)
                if self.stop_event.is_set():
                    raise InterruptedError("client stop requested")
                inference_s = clock.now() - started
                fresh_after = self._execute(actions, source_observation=observation)
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
