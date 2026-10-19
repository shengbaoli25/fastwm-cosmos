"""Bounded, epoch-aware queues used by the deployment loops."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, replace
import math
import threading
import time
from typing import Callable, Generic, TypeVar

from .contracts import ActionChunk, Observation

T = TypeVar("T")


class LatestObservation:
    """Keep one newest observation; a slow GPU never backlogs camera frames."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._item: Observation | None = None
        self._version = 0

    def put(self, item: Observation) -> int:
        with self._lock:
            if self._item is not None and item.sequence <= self._item.sequence:
                raise ValueError("observation sequence must increase")
            self._item = item
            self._version += 1
            return self._version

    def get(self) -> tuple[Observation | None, int]:
        with self._lock:
            return self._item, self._version

    def clear(self) -> None:
        with self._lock:
            self._item = None
            self._version += 1


@dataclass(frozen=True)
class ScheduledAction:
    due_at: float
    expires_at: float
    action: tuple[float, ...]
    source_sequence: int
    epoch: int


class ActionTimeline:
    """Replaceable action horizon with explicit stale/epoch invalidation."""

    def __init__(self, fps: float = 30.0, max_actions: int = 64) -> None:
        if not math.isfinite(fps) or fps <= 0:
            raise ValueError("fps must be positive")
        self.dt = 1.0 / fps
        self.max_actions = max_actions
        self._items: deque[ScheduledAction] = deque(maxlen=max_actions)

    def clear(self) -> None:
        self._items.clear()

    def replace(self, chunk: ActionChunk, now: float) -> tuple[int, int]:
        if chunk.epoch < 0:
            raise ValueError("invalid action epoch")
        # A slot that started before inference finished is already stale.  Ceil
        # is conservative: it never executes a late action for an old timestamp.
        skipped = min(len(chunk.actions), max(0, math.ceil((now - chunk.source_stamp) / self.dt)))
        fresh = [ScheduledAction(chunk.source_stamp + i * self.dt,
                                 chunk.source_stamp + (i + 1) * self.dt,
                                 action, chunk.source_sequence, chunk.epoch)
                 for i, action in enumerate(chunk.actions[skipped:], start=skipped)]
        # Preserve the still-valid old prefix until the new chunk starts.
        # A fast policy otherwise continually replaces the next slot before
        # control can consume it (30 Hz refresh/control starvation).
        prefix = []
        if fresh:
            cutoff = fresh[0].due_at
            prefix = [replace(item, expires_at=min(item.expires_at, cutoff))
                      for item in self._items if item.epoch == chunk.epoch
                      and item.expires_at > now and item.due_at < cutoff]
        self._items.clear()
        self._items.extend((prefix + fresh)[:self.max_actions])
        return skipped, len(fresh)

    def take(self, now: float, epoch: int) -> ScheduledAction | None:
        result: ScheduledAction | None = None
        while self._items and self._items[0].expires_at <= now:
            self._items.popleft()
        while self._items and self._items[0].due_at <= now:
            item = self._items.popleft()
            if item.epoch == epoch and now < item.expires_at:
                result = item
        return result

    def __len__(self) -> int:
        return len(self._items)


class ChunkActionQueue:
    """One delayed chunk, paced from publication instead of observation time.

    This is for asynchronous deployment with relaxed result timing. It deliberately
    does not claim that a delayed result is a time-aligned prediction. Control
    ticks consume at most one action and never burst to catch up after a delay.
    """

    def __init__(self, fps: float = 30.0, max_actions: int = 32) -> None:
        if not math.isfinite(fps) or fps <= 0:
            raise ValueError("fps must be positive")
        self.dt = 1.0 / fps
        self.max_actions = max_actions
        self._items: deque[tuple[float, ...]] = deque()
        self._epoch = -1
        self._source_sequence = -1
        self.start_stamp = self._next_due = 0.0

    def clear(self) -> None:
        self._items.clear()
        self._epoch = -1

    def replace(self, chunk: ActionChunk, now: float) -> tuple[int, int]:
        if len(chunk.actions) > self.max_actions:
            raise ValueError("chunk exceeds queue capacity")
        # Keep the control cadence if a newer result arrives before the next
        # slot. Repeated fast publications must not postpone execution forever.
        next_due = max(now, self._next_due) if self._epoch == chunk.epoch else now
        self._items.clear()
        self._items.extend(chunk.actions)
        self._epoch, self._source_sequence = chunk.epoch, chunk.source_sequence
        self.start_stamp = self._next_due = next_due
        return 0, len(self._items)

    def take(self, now: float, epoch: int) -> ScheduledAction | None:
        if epoch != self._epoch:
            self.clear()
            return None
        # Allow sub-millisecond control-thread jitter without skipping a whole
        # 30 Hz slot. Large stalls still consume only one action, never a burst.
        if not self._items or now < self._next_due - min(0.001, self.dt / 20):
            return None
        action = self._items.popleft()
        self._next_due = now + self.dt
        return ScheduledAction(now, self._next_due, action, self._source_sequence, epoch)

    def __len__(self) -> int:
        return len(self._items)


class EpochGate:
    """Invalidates in-flight inference after pause, stop, or emergency stop."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._epoch = 0

    def current(self) -> int:
        with self._lock:
            return self._epoch

    def advance(self) -> int:
        with self._lock:
            self._epoch += 1
            return self._epoch

    def accepts(self, epoch: int) -> bool:
        with self._lock:
            return epoch == self._epoch


def wait_or_stop(event: threading.Event, timeout: float) -> bool:
    """Wait using an interruptible event instead of an unbounded sleep."""
    return event.wait(max(0.0, timeout))
