#!/usr/bin/env python3
"""Measure client RGB/state/force -> remote model -> CPU action, without hardware.

Blank images measure the compressible lower bound. Seeded random RGB images
measure an incompressible payload at the same resolution. Both exercise the real
model unless the configured service explicitly identifies itself as a mock.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import json
import math
from pathlib import Path
import random
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "deploy"), str(ROOT / "src")]

from realman_deploy.contracts import Observation
from realman_deploy.normalization import OnlineHistory
from realman_deploy.remote_transport import RemoteJointPolicy


def summarize(values):
    if not values:
        return {name: None for name in ("mean", "p50", "p95", "p99", "minimum", "maximum", "std")}
    ordered = sorted(values)
    def quantile(q):
        offset = (len(ordered) - 1) * q
        low = int(math.floor(offset))
        high = int(math.ceil(offset))
        return ordered[low] + (ordered[high] - ordered[low]) * (offset - low)
    return dict(mean=statistics.mean(values), p50=quantile(.50), p95=quantile(.95),
                p99=quantile(.99), minimum=min(values), maximum=max(values),
                std=statistics.pstdev(values))


def _timeout_error(exc):
    current = exc
    seen = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        text = str(current).lower()
        if isinstance(current, TimeoutError) or any(word in text for word in (
                "timed out", "timeout", "deadline", "http 504")):
            return True
        current = current.__cause__ or current.__context__
    return False


def _pattern_summary(rows):
    successes = [row for row in rows if row["success"]]
    total = len(rows)
    timeouts = sum(row.get("timeout", False) for row in rows)
    failures = total - len(successes)
    overhead_passes = sum(row["transport_overhead_ms"] < 100 for row in successes)
    non_model_passes = sum(row["non_model_overhead_ms"] < 100 for row in successes)
    return dict(
        attempts=total, successes=len(successes), failures=failures, timeouts=timeouts,
        success_rate=len(successes) / total if total else None,
        failure_rate=failures / total if total else None,
        timeout_rate=timeouts / total if total else None,
        error_counts=dict(Counter(row["error_type"] for row in rows if not row["success"])),
        total_client_ms=summarize([row["total_client_ms"] for row in successes]),
        transport_overhead_ms=summarize([row["transport_overhead_ms"] for row in successes]),
        non_model_overhead_ms=summarize([row["non_model_overhead_ms"] for row in successes]),
        server_inference_ms=summarize([row["server_timing_ms"]["inference"] for row in successes]),
        upload_bytes=successes[0]["upload_bytes"] if successes else None,
        # Failed responses never count as having met the latency requirement.
        overhead_under_100ms_fraction=overhead_passes / total if total else None,
        non_model_under_100ms_fraction=non_model_passes / total if total else None,
        overhead_under_100ms_fraction_among_successes=(overhead_passes / len(successes) if successes else None),
        non_model_under_100ms_fraction_among_successes=(non_model_passes / len(successes) if successes else None),
    )


def _save_progress(result, output):
    if output:
        target = Path(output).expanduser()
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(target.name + ".tmp")
        temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
        temporary.replace(target)


def run(config_path, *, checkpoint=None, runs=50, warmup=5,
        patterns=("blank", "random"), output=None):
    patterns = tuple(patterns)
    if type(runs) is not int or runs < 1 or type(warmup) is not int or warmup < 0:
        raise ValueError("runs must be positive and warmup non-negative")
    if not patterns or len(set(patterns)) != len(patterns) or any(pattern not in {"blank", "random"} for pattern in patterns):
        raise ValueError("patterns must be unique blank/random values")
    path = Path(config_path).expanduser().resolve()
    config = json.loads(path.read_text())
    policy = RemoteJointPolicy(checkpoint, config, config_path=path)
    samples, warmup_samples = [], []
    result = dict(schema="realman-remote-benchmark/1", runs_per_pattern=runs, warmup_per_pattern=warmup,
                  server=policy.diagnostics["remote"], image_encoding=policy.image_encoding,
                  camera_width=policy.width, camera_height=policy.height, completed=False,
                  summary={}, warmup_summary={}, samples=samples, warmup_samples=warmup_samples,
                  timing_note="Client encoding + HTTP upload/response + server decode/inference. "
                              "Overhead subtracts server decode+inference; it also includes response serialization/decoding. "
                              "non_model_overhead subtracts inference only and includes server decode. "
                              "Blank compression is a channel lower bound, not representative camera throughput. "
                              "Latency percentiles include successful measured requests only; timeout/failure rates count every measured request. "
                              "The under-100ms fractions use all measured attempts; failed requests do not pass. "
                              "Every index is one new observation request, with no automatic inference retries.")
    def save_progress():
        result["summary"] = {pattern: _pattern_summary([row for row in samples if row["pattern"] == pattern])
                             for pattern in patterns}
        result["warmup_summary"] = {pattern: _pattern_summary([row for row in warmup_samples if row["pattern"] == pattern])
                                    for pattern in patterns}
        _save_progress(result, output)
    try:
        save_progress()
        n = policy.normalizer
        state, force = tuple(n.stats["state"]["mean"]), tuple(n.stats["force"]["mean"])
        # Dataset means can be marginally outside the physical gripper domain.
        state = state[:-1] + (max(0., min(1., state[-1])),)
        history = OnlineHistory(n, 32)
        history.append(state, force)
        history_input = history.tensors()
        reference = None if policy.reference_mode == "none" else [n.encode("action", state)] * 32
        sequence = 0
        for pattern in patterns:
            rng = random.Random(20261025)
            count = policy.width * policy.height * 3
            frames = {name: bytes(count) if pattern == "blank" else rng.randbytes(count)
                      for name in policy.camera_order}
            for index in range(warmup + runs):
                obs = Observation.now(sequence, state, force, frames, action_space=n.action_space)
                if n.force_preprocessing:
                    from dataclasses import replace
                    obs = replace(obs, force_meta={"preprocessing_id": n.force_preprocessing["id"]})
                sequence += 1
                started = time.monotonic()
                try:
                    actions = policy.infer(obs, reference, history_input)
                    elapsed = (time.monotonic() - started) * 1000
                    row = dict(pattern=pattern, run=index - warmup, total_client_ms=elapsed,
                               **deepcopy(policy.diagnostics["last_inference"]),
                               success=True, timeout=False, action_shape=[len(actions), len(actions[0])])
                    row["non_model_overhead_ms"] = elapsed - row["server_timing_ms"]["inference"]
                    row["first_action_normalized"] = actions[0]
                    progress = dict(pattern=pattern, run=index - warmup, success=True, total_ms=round(elapsed, 3),
                        overhead_ms=round(row["transport_overhead_ms"], 3),
                        server_ms=round(row["server_timing_ms"]["inference"], 3),
                        upload_bytes=row["upload_bytes"], warmup=index < warmup)
                except Exception as exc:
                    # Only this request failed. Start the next scheduled index
                    # with its own observation; never retry a timed-out request.
                    elapsed = (time.monotonic() - started) * 1000
                    row = dict(pattern=pattern, run=index - warmup, total_client_ms=elapsed,
                               source_sequence=obs.sequence, success=False, timeout=_timeout_error(exc),
                               error_type=type(exc).__name__, error=str(exc))
                    progress = {**row, "warmup": index < warmup}
                (samples if index >= warmup else warmup_samples).append(row)
                save_progress()
                print(json.dumps(progress, ensure_ascii=False, allow_nan=False), flush=True)
        result["completed"] = True
        save_progress()
        return result
    finally:
        policy.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", help="Selected local checkpoint for SHA verification; otherwise config must declare SHA")
    parser.add_argument("--runs", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--patterns", nargs="+", choices=("blank", "random"), default=["blank", "random"])
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    if args.runs < 1 or args.warmup < 0:
        parser.error("runs must be positive and warmup non-negative")
    result = run(args.config, checkpoint=args.checkpoint, runs=args.runs, warmup=args.warmup,
                 patterns=args.patterns, output=args.output)
    print(json.dumps(result["summary"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
