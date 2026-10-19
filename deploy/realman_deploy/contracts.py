"""Small, dependency-free contracts shared by sensors, inference and control.

The runtime uses monotonic host timestamps for scheduling.  Device timestamps
are retained by camera metadata but are never used as a control deadline.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math
import time
from typing import Any, Mapping, Sequence


def action_width(action_space: str) -> int:
    if action_space not in ('eef_pose7', 'joint8'):
        raise ValueError('Explicit eef_pose7 or joint8 action space required')
    return 8 if action_space == 'joint8' else 7


def _finite_vector(value: Sequence[float], size: int, name: str) -> tuple[float, ...]:
    if len(value) != size:
        raise ValueError(f"{name} must contain {size} values")
    result = tuple(float(x) for x in value)
    if not all(math.isfinite(x) for x in result):
        raise ValueError(f"{name} contains a non-finite value")
    return result


@dataclass(frozen=True)
class Observation:
    """One synchronized robot/camera observation.

    ``state`` and ``force`` are raw dataset units: EEF xyz in metres, Euler
    angles in radians, gripper in [0, 1], and wrench in N/Nm.  The model
    adapter owns normalization.  RGB bytes must be contiguous RGB images with
    the configured width/height; no JPEG is accepted as a model input.
    """

    stamp: float
    sequence: int
    state: tuple[float, ...]
    force: tuple[float, ...]
    frames: Mapping[str, bytes]
    frame_meta: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
    force_meta: Mapping[str, Any] = field(default_factory=dict)
    action_space: str = 'eef_pose7'

    def __post_init__(self) -> None:
        if not math.isfinite(self.stamp) or self.stamp <= 0:
            raise ValueError("observation stamp must be a positive monotonic time")
        if type(self.sequence) is not int or self.sequence < 0:
            raise ValueError("observation sequence must be a non-negative integer")
        object.__setattr__(self, "state", _finite_vector(self.state, action_width(self.action_space), "state"))
        force = _finite_vector(self.force, 6, "force")
        if not 0 <= self.state[-1] <= 1:
            raise ValueError("gripper opening must be in [0, 1]")
        object.__setattr__(self, "force", force)
        if not self.frames or any(not isinstance(k, str) or not isinstance(v, bytes)
                                  or not v for k, v in self.frames.items()):
            raise ValueError("observation must contain non-empty RGB frame bytes")

    @classmethod
    def now(cls, sequence: int, state: Sequence[float], force: Sequence[float],
            frames: Mapping[str, bytes], frame_meta: Mapping[str, Mapping[str, Any]] | None = None,
            *, action_space: str = 'eef_pose7') -> "Observation":
        return cls(time.monotonic(), sequence, tuple(state), tuple(force), dict(frames), frame_meta or {}, action_space=action_space)


@dataclass(frozen=True)
class ActionChunk:
    """An absolute EEF action chunk anchored to its source observation."""

    source_sequence: int
    source_stamp: float
    epoch: int
    actions: tuple[tuple[float, ...], ...]
    created_stamp: float = field(default_factory=time.monotonic)
    action_space: str = 'eef_pose7'

    def __post_init__(self) -> None:
        if type(self.source_sequence) is not int or self.source_sequence < 0:
            raise ValueError("source_sequence must be a non-negative integer")
        if type(self.epoch) is not int or self.epoch < 0:
            raise ValueError("epoch must be a non-negative integer")
        if not math.isfinite(self.source_stamp) or not math.isfinite(self.created_stamp):
            raise ValueError("action timestamps must be finite")
        if not 1 <= len(self.actions) <= 32:
            raise ValueError("an action chunk must contain 1..32 actions")
        normalized = tuple(_finite_vector(action, action_width(self.action_space), "action") for action in self.actions)
        if any(not 0 <= action[-1] <= 1 for action in normalized):
            raise ValueError("gripper opening must be in [0, 1]")
        object.__setattr__(self, "actions", normalized)


@dataclass(frozen=True)
class RuntimeStatus:
    state: str
    epoch: int
    observation_sequence: int = -1
    action_queue: int = 0
    inference_count: int = 0
    discarded_results: int = 0
    latency_ms: float | None = None
    error: str | None = None
    reference_ready: bool = False
    model_ready: bool = False
    robot_ready: bool = False
    cameras_ready: bool = False
    force_ready: bool = False
    gripper_ready: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "state": self.state, "epoch": self.epoch,
            "observation_sequence": self.observation_sequence,
            "action_queue": self.action_queue, "inference_count": self.inference_count,
            "discarded_results": self.discarded_results, "latency_ms": self.latency_ms,
            "error": self.error, "reference_ready": self.reference_ready,
            "model_ready": self.model_ready, "robot_ready": self.robot_ready,
            "cameras_ready": self.cameras_ready, "force_ready": self.force_ready,
            "gripper_ready": self.gripper_ready,
        }
