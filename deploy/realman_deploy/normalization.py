"""RealMan 1018 normalization and online history assembly."""
from __future__ import annotations

from collections import deque
import json
import math
from pathlib import Path
from typing import Sequence
from struct import pack, unpack


def _f32(value: float) -> float:
    return unpack("f", pack("f", float(value)))[0]


class DatasetNormalizer:
    """Matches the recorded FP32 population z-score and [-5, 5] clip."""

    def __init__(self, stats: dict, force_preprocessing=None):
        self.stats = stats
        self.force_preprocessing = force_preprocessing
        if force_preprocessing is not None:
            from realman_force.contract import validate_contract_identity
            validate_contract_identity(force_preprocessing)
        for key, width in (("state", 7), ("action", 7), ("force", 6)):
            item = stats.get(key)
            if not isinstance(item, dict) or len(item.get("mean", [])) != width or len(item.get("std", [])) != width:
                raise ValueError(f"missing {key} normalization statistics")
            if any(not math.isfinite(float(x)) or float(x) <= 0 for x in item["std"]):
                raise ValueError(f"{key} standard deviation must be positive")
            if any(not math.isfinite(float(x)) for x in item["mean"]):
                raise ValueError(f"{key} mean must be finite")

    @classmethod
    def from_json(cls, path: str | Path) -> "DatasetNormalizer":
        payload = json.loads(Path(path).read_text())
        return cls(payload.get("statistics", payload), payload.get('force_preprocessing'))

    def _transform(self, name, values, *, inverse=False):
        item = self.stats[name]
        mean, std = item["mean"], item["std"]
        if not len(values):
            raise ValueError(f"{name} must not be empty")
        matrix = isinstance(values[0], (list, tuple))
        rows = values if matrix else [values]
        result = []
        for row in rows:
            if len(row) != len(mean) or not all(math.isfinite(float(v)) for v in row):
                raise ValueError(f"{name} requires {len(mean)} finite values per row")
            output = []
            for v, m, st in zip(row, mean, std):
                # Match stock FP32 scale/offset arithmetic, including rounding.
                denominator = _f32(_f32(st) + _f32(1e-8))
                scale, offset = _f32(1.0 / denominator), _f32(-_f32(m) / denominator)
                value = (_f32(_f32(_f32(v) - offset) / scale) if inverse else
                         _f32(_f32(_f32(v) * scale) + offset))
                if not math.isfinite(value):
                    raise ValueError(f"{name} normalization produced non-finite values")
                output.append(value if inverse else max(-5.0, min(5.0, value)))
            result.append(output)
        return result if matrix else result[0]

    def encode(self, name: str, values: Sequence) -> list:
        return self._transform(name, values)

    def decode(self, name: str, values: Sequence) -> list:
        return self._transform(name, values, inverse=True)


class OnlineHistory:
    """Builds the exact [state7, force6] history expected by Force WM."""

    def __init__(self, normalizer: DatasetNormalizer, length: int = 32):
        self.normalizer = normalizer
        self.length = length
        self._state: deque[list[float]] = deque(maxlen=length)
        self._force: deque[list[float]] = deque(maxlen=length)

    def reset(self, state: Sequence[float], force: Sequence[float]) -> None:
        self._state.clear(); self._force.clear()
        self.append(state, force)

    def append(self, state: Sequence[float], force: Sequence[float]) -> None:
        self._state.append(self.normalizer.encode("state", state))
        self._force.append(self.normalizer.encode("force", force))

    def tensors(self) -> tuple[list[list[float]], list[bool], list[list[bool]]]:
        state = list(self._state); force = list(self._force)
        pad = max(0, self.length - len(state))
        rows = [[0.0] * 13 for _ in range(pad)] + [s + f for s, f in zip(state, force)]
        return rows[-self.length:], [True] * pad + [False] * len(state), [[not x] * 13 for x in ([True] * pad + [False] * len(state))][-self.length:]
