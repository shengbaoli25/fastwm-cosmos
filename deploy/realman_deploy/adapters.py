"""Runtime adapter interfaces and explicit plugin loading."""
from __future__ import annotations

import importlib
from typing import Any, Protocol

from .contracts import Observation


class PolicyAdapter(Protocol):
    def infer(self, observation: Observation, normalized_reference: list[list[float]],
              history: tuple[list[list[float]], list[bool], list[list[bool]]]) -> list[list[float]]: ...
    def close(self) -> None: ...


class HardwareAdapter(Protocol):
    """A sensor/control bridge. Command/stop methods must have bounded latency."""

    def status(self) -> dict[str, Any]: ...
    def arm(self) -> None: ...
    def observe(self) -> Observation | None: ...
    def command(self, action: tuple[float, ...]) -> None: ...
    def slow_stop(self) -> None: ...
    def emergency_stop(self) -> None: ...
    def reset_emergency_stop(self) -> dict[str, Any]: ...
    def close(self) -> None: ...


def load_factory(spec: str, **kwargs: Any) -> Any:
    """Load ``module:callable`` without importing hardware/model at UI startup."""
    if not isinstance(spec, str) or ":" not in spec:
        raise ValueError("adapter factory must use module:callable syntax")
    module_name, attr = spec.split(":", 1)
    if not module_name or not attr:
        raise ValueError("adapter factory must use module:callable syntax")
    factory = getattr(importlib.import_module(module_name), attr)
    if not callable(factory):
        raise TypeError(f"adapter factory is not callable: {spec}")
    return factory(**kwargs)
