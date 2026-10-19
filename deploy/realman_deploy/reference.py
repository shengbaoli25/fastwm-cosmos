"""Time-aligned clean Action32 reference shared by the two world models."""
from __future__ import annotations

import math
import threading
from typing import Sequence

from .contracts import Observation
from .normalization import DatasetNormalizer


class RollingActionReference:
    """Bootstrap with measured EEF hold, then roll the last accepted AE output.

    Actions are absolute EEF poses at 30 Hz. Drop elapsed slots relative to the
    horizon's supplied anchor, and repeat its last pose to fill 32 slots. The
    timestamped runtime uses observation time; relaxed async uses the new
    execution start. This is a deployment condition; training used recorded actions.
    """
    mode = "rolling"

    def __init__(self, fps: float = 30.0) -> None:
        if not math.isfinite(fps) or fps <= 0:
            raise ValueError("reference fps must be positive")
        self.fps = fps
        self._lock = threading.Lock()
        self._raw: list[list[float]] | None = None
        self._stamp: float | None = None

    def ready(self) -> bool:
        return True

    def bootstrap(self) -> bool:
        with self._lock:
            return self._raw is None

    def get(self, observation: Observation, normalizer: DatasetNormalizer) -> list[list[float]]:
        with self._lock:
            raw = None if self._raw is None else [row[:] for row in self._raw]
            stamp = self._stamp
        if raw is None:
            raw = [list(observation.state) for _ in range(32)]
        else:
            elapsed = max(0.0, observation.stamp - stamp)
            offset = min(31, math.floor(elapsed * self.fps + 1e-6))
            raw = raw[offset:]
            raw += [raw[-1][:] for _ in range(32 - len(raw))]
        return normalizer.encode("action", raw)

    def update(self, raw_actions: Sequence[Sequence[float]], *, source_stamp: float) -> None:
        if len(raw_actions) != 32 or any(len(row) != 7 for row in raw_actions):
            raise ValueError("rolling reference must be Action32x7")
        horizon = [[float(v) for v in row] for row in raw_actions]
        if not math.isfinite(source_stamp) or not all(math.isfinite(v) for row in horizon for v in row):
            raise ValueError("rolling reference and timestamp must be finite")
        with self._lock:
            self._raw, self._stamp = horizon, source_stamp

    def last_raw(self) -> list[list[float]] | None:
        with self._lock:
            return None if self._raw is None else [row[:] for row in self._raw]

    def reset(self) -> None:
        with self._lock:
            self._raw = self._stamp = None
