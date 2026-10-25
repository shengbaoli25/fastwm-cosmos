#!/usr/bin/env python3
"""Offline synthetic-input inference timing. Never opens hardware or executes actions."""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "deploy"), str(ROOT / "src")]


def percentile(values, percent):
    ordered = sorted(values)
    position = (len(ordered) - 1) * percent / 100
    left = int(position)
    right = min(left + 1, len(ordered) - 1)
    return ordered[left] + (ordered[right] - ordered[left]) * (position - left)


def summary(values):
    return dict(count=len(values), p50=percentile(values, 50), p95=percentile(values, 95),
                mean=statistics.mean(values), min=min(values), max=max(values))


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def benchmark_profile(config, config_path, checkpoint, profile, runs, *, optimization=None,
                      policy_factory=None, synchronize=None):
    from realman_deploy.contracts import Observation
    from realman_deploy.normalization import OnlineHistory
    from realman_deploy.sampling import updated_config, profile_steps
    if runs < 1:
        raise ValueError("runs must be positive")
    selected = updated_config(deepcopy(config), profile, "synchronous")
    if optimization is not None:
        selected["inference_optimization"]["mode"] = optimization
    if policy_factory is None:
        from realman_deploy.model_adapter import CosmosJointPolicy
        policy_factory = CosmosJointPolicy
    if synchronize is None:
        import torch
        synchronize = torch.cuda.synchronize
    policy = None
    started = time.perf_counter()
    try:
        # Each profile gets a fresh model/runner and its own two warmup calls.
        policy = policy_factory(checkpoint, selected, config_path=config_path)
        synchronize(policy.device)
        load_and_warmup_ms = (time.perf_counter() - started) * 1000
        warmup = deepcopy(policy.diagnostics.get("warmup", []))
        width = int(selected.get("camera_width", 640))
        height = int(selected.get("camera_height", 480))
        camera_order = selected.get("camera_order", ["global", "wrist"])
        if camera_order != ["global", "wrist"] or width <= 0 or height <= 0:
            raise ValueError("benchmark requires positive RGB dimensions and global/wrist camera order")
        # Uniform distinct RGB images make reproducible, compressibility-independent
        # model input. There is no encoding/network path in this measurement.
        colors = [bytes((32, 64, 96)), bytes((96, 64, 32))]
        frames = {name: colors[i] * (width * height) for i, name in enumerate(camera_order)}
        state = tuple(policy.normalizer.stats["state"]["mean"])
        force = tuple(policy.normalizer.stats["force"]["mean"])
        history = OnlineHistory(policy.normalizer, 32)
        history.append(state, force)
        history_input = history.tensors()
        rows = []
        for index in range(runs):
            observation = Observation.now(index, state, force, frames,
                                           action_space=policy.normalizer.action_space)
            policy.inference_index = 0
            synchronize(policy.device)
            tick = time.perf_counter()
            actions = policy.infer(observation, None, history_input)
            synchronize(policy.device)
            duration = (time.perf_counter() - tick) * 1000
            rows.append(dict(index=index, wall_ms=duration,
                             timings=deepcopy(policy.diagnostics.get("last_inference", {})),
                             action_shape=[len(actions), len(actions[0]) if actions else 0]))
        return dict(profile=profile, steps=profile_steps(profile),
                    requested_device=selected.get("device", "cuda"),
                    optimization=selected.get("inference_optimization", {}).get("mode", "auto"),
                    input=dict(camera_order=camera_order, width=width, height=height,
                               rgb="fixed uniform bytes; camera-specific colors", state_force="normalization means",
                               history="one observation plus normal deployment padding"),
                    seeds=dict(action=int(selected.get("action_seed", 42)),
                               video=int(selected.get("video_seed", 44)), force=int(selected.get("force_seed", 43))),
                    load_and_warmup_ms=load_and_warmup_ms, warmup=warmup,
                    wall_ms=summary([row["wall_ms"] for row in rows]), runs=rows,
                    diagnostics=deepcopy(policy.diagnostics))
    finally:
        if policy is not None:
            policy.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--profiles", nargs="+", choices=["161616", "4416"], default=["161616", "4416"])
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--optimization", choices=["auto", "eager", "compile", "cuda_graph", "compile_cuda_graph"])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.runs < 1:
        parser.error("--runs must be positive")
    if len(args.profiles) != len(set(args.profiles)):
        parser.error("--profiles must be unique")
    if args.output.exists():
        parser.error("--output already exists; use a new report path")
    config_path = args.config.expanduser().resolve()
    checkpoint = args.checkpoint.expanduser().resolve()
    config = json.loads(config_path.read_text())
    report = dict(schema="realman_offline_latency_v1", created_at=datetime.now(timezone.utc).isoformat(),
                  config=str(config_path), checkpoint=str(checkpoint), checkpoint_sha256=file_sha256(checkpoint),
                  requested_device=config.get("device", "cuda"),
                  machine=platform.machine(), python=sys.version.split()[0],
                  scope="synthetic input -> image resize/H2D/VAE -> joint sampling -> CPU normalized actions",
                  excluded=["camera/robot I/O", "image upload/network", "physical action execution", "quality evaluation"],
                  hardware_connected=False, action_execution=False,
                  note="Fixed input/seeds. Two model-loader warmups are separate from timed runs. Subsecond latency is hardware-dependent.",
                  profiles=[])
    for profile in args.profiles:
        print(f"Loading profile {profile}; two initialization warmups, then {args.runs} timed calls", file=sys.stderr, flush=True)
        result = benchmark_profile(config, config_path, checkpoint, profile, args.runs, optimization=args.optimization)
        report["profiles"].append(result)
        print(f"{profile}: p50={result['wall_ms']['p50']:.2f} ms, p95={result['wall_ms']['p95']:.2f} ms", file=sys.stderr, flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    print(args.output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
