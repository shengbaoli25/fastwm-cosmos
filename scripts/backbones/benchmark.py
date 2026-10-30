#!/usr/bin/env python3
"""Measure a trained backbone variant at batch=1, BF16 and 16 inference steps.

Use the saved training/config.yaml (with bound cache and teacher identities),
not an unbound experiment template. RGB timing includes host-to-device transfer,
observed-only visual encoding, prediction and Action/Force inference. Cached
timing includes feature transfer and policy inference but excludes the encoder.
Both exclude file reads, episode decoding, host capture and model loading.
Results describe repeated inference of one fixed test window, not an episode
average, closed-loop control frequency or prediction-quality measurement.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys
import time


ROOT = Path(__file__).resolve().parents[2]
OBSERVED_OFFSETS = (-4, 0)
INFERENCE_STEPS = 16
CONDITION_KEYS = ("proprio", "history", "history_channel_valid", "history_is_pad")


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_measurement(warmup, iterations, sample_index):
    for name, value, minimum in (("warmup", warmup, 1), ("iterations", iterations, 1),
                                 ("sample_index", sample_index, 0)):
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}")


def latency_statistics(milliseconds):
    values = sorted(float(value) for value in milliseconds)
    if not values or any(not math.isfinite(value) or value <= 0 for value in values):
        raise ValueError("Latency samples must be nonempty, positive and finite")

    def percentile(fraction):
        position = (len(values) - 1) * fraction
        lo, hi = math.floor(position), math.ceil(position)
        return values[lo] + (values[hi] - values[lo]) * (position - lo)

    mean = statistics.fmean(values)
    return dict(iterations=len(values), mean_ms=mean, p50_ms=percentile(.5),
                p95_ms=percentile(.95), min_ms=values[0], max_ms=values[-1],
                hz=1000. / mean, percentile_method="linear interpolation, (n-1)*p")


def observed_clip(dataset, sample_index):
    """Decode outside timing, selecting only frames at -4 and 0 in one segment."""
    import torch

    if not 0 <= sample_index < len(dataset.windows):
        raise IndexError("sample_index is outside the test split")
    source_index, episode, frame = dataset.windows[sample_index]
    source = dataset.source
    row_episode, row_frame, _, _ = source.rows[source_index]
    lo, hi = source.bounds[source_index]
    indexes = [frame + offset for offset in OBSERVED_OFFSETS]
    if (row_episode != episode or row_frame != frame
            or any(index < lo or index >= hi for index in indexes)):
        raise ValueError("Observed window does not belong to one valid source segment")
    frames = source.decode_episode(episode)
    if frames.ndim != 4 or frames.shape[1] != 3 or indexes[-1] >= len(frames):
        raise ValueError("Decoded episode must contain the two observed RGB frames")
    # Advanced indexing creates an independent two-frame tensor. Future frames
    # from the decoded episode are never passed to the teacher or policy.
    return frames[torch.tensor(indexes)].permute(1, 0, 2, 3).unsqueeze(0).contiguous()


def measure(run_once, synchronize, *, warmup, iterations, before_measure=None,
            clock=time.perf_counter):
    validate_measurement(warmup, iterations, 0)
    output = None
    for _ in range(warmup):
        output = None
        output = run_once()
        synchronize()
    output = None
    synchronize()
    if before_measure is not None:
        before_measure()
    milliseconds = []
    for _ in range(iterations):
        output = None
        synchronize()
        started = clock()
        output = run_once()
        synchronize()
        milliseconds.append((clock() - started) * 1000.)
    return milliseconds, output


def validate_bound_config(cfg, source_sha):
    """Reject experiment templates or inputs not bound to the saved training run."""
    train = cfg["data"]["train"]
    manifest = cfg["model"]["manifest"]
    teacher = manifest.get("teacher", {})
    if (train["source_sha256"] != source_sha
            or manifest.get("dataset_source_sha256") != source_sha
            or manifest.get("cache_identity_sha256") != train["expected_cache_identity"]
            or teacher.get("name") != cfg["teacher"]
            or train["expected_teacher"] != cfg["teacher"]
            or teacher.get("weight_sha256") != train["expected_weight_sha256"]
            or manifest.get("observation_offsets") != list(OBSERVED_OFFSETS)):
        raise ValueError("Use the saved training config bound to this dataset, cache and teacher")
    for digest in (source_sha, train["expected_cache_identity"], train["expected_weight_sha256"]):
        if (not isinstance(digest, str) or len(digest) != 64
                or any(character not in "0123456789abcdef" for character in digest)):
            raise ValueError("Bound identities must be lowercase SHA256 values")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--source-sha", required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--teacher-source", type=Path,
                        help="RGB mode: pinned teacher source (defaults to saved config)")
    parser.add_argument("--weights", type=Path,
                        help="RGB mode: teacher checkpoint (defaults to saved config)")
    parser.add_argument("--mode", choices=("cached", "rgb"), default="rgb")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    validate_measurement(args.warmup, args.iterations, args.sample_index)
    sys.path[:0] = [str(ROOT / "src"), str(ROOT)]
    import torch
    from hydra.utils import instantiate
    from omegaconf import OmegaConf
    from torch.utils.data._utils.collate import default_collate
    from fastwam.cosmos.dense_encoder import load_dense_teacher, teacher_spec
    from fastwam.cosmos.jepa_video_cache import JepaVideoCachedDataset

    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("This benchmark requires a CUDA GPU; CPU timings are not comparable")
    torch.cuda.set_device(device)
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("BF16-capable CUDA GPU required")
    cfg = OmegaConf.to_container(OmegaConf.load(args.config), resolve=True, throw_on_missing=True)
    validate_bound_config(cfg, args.source_sha)
    if cfg["model"]["_target_"] != "fastwam.cosmos.dense_features.create_dense_feature_joint":
        raise ValueError("Only the shared DenseFeatureJointModel is supported")
    train = cfg["data"]["train"]
    spec = teacher_spec(cfg["teacher"])
    if cfg["model"]["feature_dim"] != spec["feature_dim"]:
        raise ValueError("Model and teacher feature widths differ")
    dataset = JepaVideoCachedDataset(dataset_root=args.dataset, source_sha256=args.source_sha,
        feature_root=args.cache_root, expected_cache_identity=train["expected_cache_identity"],
        expected_teacher=cfg["teacher"], expected_weight_sha256=train["expected_weight_sha256"],
        split="test", cache_capacity=1)
    if args.sample_index >= len(dataset):
        raise IndexError("sample_index is outside the test split")
    sample = dataset[args.sample_index]
    # Remove all future labels before batching or transferring inference inputs.
    inputs = {key: sample[key] for key in CONDITION_KEYS}
    if args.mode == "cached":
        inputs["visual_features"] = sample["visual_features"][:1].clone()
    inputs = default_collate([inputs])
    inputs = {key: value.to(dtype=torch.bfloat16 if value.is_floating_point() else value.dtype)
              .contiguous().pin_memory() for key, value in inputs.items()}
    clip = None
    if args.mode == "rgb":
        clip = observed_clip(dataset, args.sample_index)
        if clip.shape != (1, 3, 2, 224, 448) or not bool(torch.isfinite(clip).all()):
            raise ValueError("Observed RGB input must be finite [1,3,2,224,448]")
        clip = clip.float().contiguous().pin_memory()
    model = instantiate(cfg["model"], device=str(device), model_dtype=torch.bfloat16)
    checkpoint_step = model.load_checkpoint(args.checkpoint)
    model.eval().requires_grad_(False)
    teacher = None
    if args.mode == "rgb":
        teacher_source = args.teacher_source or cfg.get("teacher_source")
        teacher_weights = args.weights or cfg.get("teacher_weights")
        if not teacher_source or not teacher_weights:
            raise ValueError("RGB mode requires --teacher-source and --weights or saved paths")
        teacher = load_dense_teacher(cfg["teacher"], teacher_source, teacher_weights,
            train["expected_weight_sha256"], device=str(device))
        saved_teacher = cfg["model"]["manifest"]["teacher"]
        if any(saved_teacher.get(key) != value for key, value in teacher.manifest().items()):
            raise ValueError("Loaded online teacher differs from the training cache teacher")
    checkpoint_sha = file_sha256(args.checkpoint)
    memory = {}

    def before_measure():
        memory.update(baseline_allocated_bytes=torch.cuda.memory_allocated(device),
                      baseline_reserved_bytes=torch.cuda.memory_reserved(device))
        torch.cuda.reset_peak_memory_stats(device)

    def run_once():
        current = {key: value.to(device, non_blocking=True) for key, value in inputs.items()}
        if teacher is not None:
            current["visual_features"] = teacher.encode_video(
                clip.to(device, non_blocking=True), context_only=True)
        return model.infer_joint(current, num_inference_steps=INFERENCE_STEPS,
                                 seed=42, force_seed=43)

    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        timings, prediction = measure(run_once, lambda: torch.cuda.synchronize(device),
            warmup=args.warmup, iterations=args.iterations, before_measure=before_measure)
    # Capture peak before finite-output checks, which are outside the timed call.
    memory.update(peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
                  peak_reserved_bytes=torch.cuda.max_memory_reserved(device))
    memory["incremental_peak_allocated_bytes"] = (
        memory["peak_allocated_bytes"] - memory["baseline_allocated_bytes"])
    for name, shape in (("action", (1, 32, 8)), ("force", (1, 32, 6)),
                        ("future_features", (1, 4, 392, spec["feature_dim"]))):
        if tuple(prediction[name].shape) != shape or not bool(torch.isfinite(prediction[name]).all()):
            raise ValueError(f"Invalid inference output: {name}")
    _, episode, frame = dataset.windows[args.sample_index]
    report = dict(
        schema="fastwam-backbone-inference-benchmark-v1",
        measured_at_utc=datetime.now(timezone.utc).isoformat(),
        mode=args.mode, batch_size=1, dtype="bfloat16", inference_steps=INFERENCE_STEPS,
        action_seed=42, force_seed=43, warmup=args.warmup,
        latency=latency_statistics(timings), latency_samples_ms=timings, memory=memory,
        device=dict(name=torch.cuda.get_device_name(device), index=torch.cuda.current_device(),
                    total_memory_bytes=torch.cuda.get_device_properties(device).total_memory,
                    torch_version=torch.__version__, cuda_version=torch.version.cuda,
                    cudnn_version=torch.backends.cudnn.version()),
        config=dict(path=str(args.config.resolve()), sha256=file_sha256(args.config)),
        checkpoint=dict(path=str(args.checkpoint.resolve()), sha256=checkpoint_sha,
                        step=checkpoint_step, schema=model.checkpoint_schema),
        source=dict(sha256=args.source_sha, identity_sha256=dataset.source.identity_sha256,
                    split="test", sample_index=args.sample_index, episode=episode, frame=frame,
                    observation_offsets=list(OBSERVED_OFFSETS)),
        cache_identity_sha256=train["expected_cache_identity"],
        teacher=cfg["model"]["manifest"]["teacher"],
        model_manifest=model.manifest,
        benchmark_script_sha256=file_sha256(__file__),
        timing_scope=dict(
            includes=["pinned host input to GPU transfer", "visual predictor", "Action/Force 16-step inference"]
                     + (["observed-only RGB teacher encoding"] if teacher else []),
            excludes=["disk/cache reads", "episode decode", "host capture", "model/checkpoint loading", "output transfer to CPU"]
                     + (["visual teacher encoding"] if teacher is None else []),
            notes=["Repeated single fixed test window; not an episode-wide latency distribution.",
                   "Hz means sequential full inference calls/second, not generated action frames/second.",
                   "Peak memory includes loaded model weights; reserved memory uses the warmed CUDA allocator.",
                   "CPU wall clock is synchronized before and after every measured CUDA call."]),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    print(json.dumps(dict(output=str(args.output), latency=report["latency"], memory=memory), indent=2))


if __name__ == "__main__":
    main()
