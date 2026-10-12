#!/usr/bin/env python3
"""Verify complete frozen features on CPU; optionally measure VAE agreement.

The default path never constructs a model or queries a CUDA device. Explicit
--check-vae loads only the pinned frozen VAE for nine batch-one comparisons.
Neither path constructs a policy, optimizer, nor starts training.
"""
from __future__ import annotations

import argparse
import importlib.metadata
import importlib.util
import json
import os
import platform
import sys
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SELECTED_EPISODES = (0, 12, 24, 36, 49)
VAE_EPISODES = (0, 24, 49)


def _precompute_helpers():
    spec = importlib.util.spec_from_file_location("feature_verifier_precompute", REPO_ROOT / "scripts/forcevla/precompute_vae_cache.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def runtime_versions(*, check_vae=False, device="cuda:0"):
    import torch

    versions = {name: importlib.metadata.version(name)
                for name in ("torch", "torchvision", "datasets", "pyarrow", "pillow", "safetensors")}
    result = {"python": platform.python_version(), "packages": versions,
              "torch_cuda_build": torch.version.cuda, "cuda_device_queried": False,
              "device": None}
    if check_vae:
        if not device.startswith("cuda"):
            raise ValueError("Explicit VAE comparison requires a CUDA device")
        properties = torch.cuda.get_device_properties(device)
        result.update(cuda_device_queried=True,
                      device={"requested": device, "name": properties.name,
                              "total_memory_bytes": properties.total_memory,
                              "compute_capability": [properties.major, properties.minor]})
    return result


def episode_windows(lengths, episodes):
    """Return the first, middle and last original anchor, preserving short ends."""
    offsets, offset = [], 0
    for length in lengths:
        offsets.append(offset)
        offset += length
    result = []
    for episode in episodes:
        if type(episode) is not int or episode < 0 or episode >= len(lengths):
            raise ValueError(f"Episode is absent from the full source: {episode}")
        for position, frame in (("start", 0), ("middle", lengths[episode] // 2), ("end", lengths[episode] - 1)):
            result.append({"episode_index": episode, "frame_index": frame,
                           "sample_index": offsets[episode] + frame, "position": position})
    return result


def build_numeric_reference(raw):
    """Apply the actual stock formatter and normalization to numeric columns.

    Replication commutes with these deterministic per-row operations. This
    permits comparison of every cached label without decoding every RGB clip.
    Selected clips are also checked through the full strict stock getter.
    """
    import pyarrow.parquet as pq
    import torch
    from omegaconf import OmegaConf
    from fastwam.cache.latent import CacheIntegrityError
    from fastwam.datasets.lerobot.lerobot.datasets.utils import hf_transform_to_torch
    from fastwam.datasets.lerobot.utils.normalizer import LinearNormalizer, load_dataset_stats_from_json
    from fastwam.datasets.lerobot.transforms.action_state_merger import ConcatLeftAlign

    shape_meta = OmegaConf.to_container(raw.shape_meta, resolve=True)
    processor = raw.processor
    normalizer = LinearNormalizer(shape_meta=shape_meta,
        use_stepwise_action_norm=bool(processor.use_stepwise_action_norm),
        default_mode=str(processor.norm_default_mode), exception_mode=None,
        stats=load_dataset_stats_from_json(raw.pretrained_norm_stats))
    merger = ConcatLeftAlign(state_target_dim=8)
    merger.set_shape_meta(shape_meta)
    parts = {key: [] for key in ("action", "proprio", "raw_current_state", "raw_current_wrench", "task_index")}
    lengths = []
    for episode, path in enumerate(sorted(Path(raw.dataset_dirs[0]).glob("data/chunk-*/episode_*.parquet"))):
        table = pq.read_table(path, columns=["action", "observation.state", "observation.wrench",
                                            "index", "episode_index", "frame_index", "task_index"])
        columns = {key: table[key].to_pylist() for key in table.column_names}
        length, offset = len(table), sum(lengths)
        if (columns["index"] != list(range(offset, offset + length)) or
                columns["frame_index"] != list(range(length)) or
                columns["episode_index"] != [episode] * length):
            raise CacheIntegrityError("Numeric reference source indices changed")
        formatted = hf_transform_to_torch({key: columns[key] for key in ("action", "observation.state")})
        batch = {"action": {"default": torch.stack(formatted["action"])},
                 "state": {"default": torch.stack(formatted["observation.state"])}}
        batch = merger.forward(normalizer.forward(batch))
        parts["action"].append(batch["action"])
        parts["proprio"].append(batch["state"])
        parts["raw_current_state"].append(torch.tensor(columns["observation.state"], dtype=torch.float64))
        parts["raw_current_wrench"].append(torch.tensor(columns["observation.wrench"], dtype=torch.float64))
        parts["task_index"].append(torch.tensor(columns["task_index"], dtype=torch.int64))
        lengths.append(length)
    if not lengths:
        raise CacheIntegrityError("Numeric reference source is empty")
    return {key: torch.cat(values) for key, values in parts.items()}, lengths


def _exact(actual, expected, label):
    import torch
    from fastwam.cache.latent import CacheIntegrityError

    if actual.shape != expected.shape or actual.dtype != expected.dtype or not torch.equal(actual, expected):
        raise CacheIntegrityError(f"Exact stock feature comparison failed: {label}")


def task_prompts_from_source(dataset_root):
    from fastwam.cache.latent import CacheIntegrityError
    from fastwam.datasets.lerobot.robot_video_dataset import DEFAULT_PROMPT

    prompts = {}
    for line in (Path(dataset_root) / "meta/tasks.jsonl").read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        index, task = row.get("task_index"), row.get("task")
        if type(index) is not int or index < 0 or index in prompts or not isinstance(task, str) or not task.strip():
            raise CacheIntegrityError("Source task inventory has an invalid or duplicate task")
        prompts[index] = DEFAULT_PROMPT.format(task=task)
    if not prompts:
        raise CacheIntegrityError("Source task inventory is empty")
    return prompts


def verify_cpu_payloads(cache, reference, source_texts, source_text_hashes, getter, *, task_prompts,
                        selected_episodes=SELECTED_EPISODES):
    """Audit all labels, padding, text associations and selected stock clips."""
    import torch
    from safetensors.torch import load_file
    from fastwam.cache.latent import CacheIntegrityError, NUMERIC_SHAPES, file_sha256

    manifest, root = cache.manifest, cache.root
    identity = manifest["identity"]
    lengths = identity["episode_lengths"]
    count = identity["sample_count"]
    if source_text_hashes != identity["text_payload_sha256"] or set(source_texts) != set(source_text_hashes):
        raise CacheIntegrityError("Original T5 payload identity differs from the cache")
    for entry in manifest["texts"]:
        payload = load_file(str(root / entry["path"]))
        for key in ("context", "context_mask"):
            _exact(payload[key], source_texts[entry["prompt"]][key], f"T5 {key}: {entry['prompt']}")
    if any(len(tensor) != count for tensor in reference.values()):
        raise CacheIntegrityError("Numeric reference does not cover every original anchor")
    prompt_indices = {entry["prompt"]: index for index, entry in enumerate(manifest["texts"])}
    try:
        expected_text_indices = torch.tensor([prompt_indices[task_prompts[int(index)]]
                                              for index in reference["task_index"]], dtype=torch.int64)
    except KeyError as error:
        raise CacheIntegrityError("Source task has no corresponding cached text") from error
    episode_ends = torch.tensor([sum(lengths[:episode + 1]) for episode in range(len(lengths))], dtype=torch.int64)
    labels_checked = 0
    for entry in manifest["shards"]:
        path = root / entry["path"]
        if file_sha256(path) != entry["sha256"]:
            raise CacheIntegrityError("Cache shard changed during verification")
        payload = load_file(str(path))
        anchors, episodes = payload["sample_index"], payload["episode_index"]
        ends = episode_ends[episodes]
        indices = torch.minimum(anchors[:, None] + torch.arange(32), ends[:, None] - 1)
        _exact(payload["action"], reference["action"][indices], f"action shard {entry['start']}")
        _exact(payload["proprio"], reference["proprio"][indices], f"proprio shard {entry['start']}")
        for key in ("raw_current_state", "raw_current_wrench", "task_index"):
            if key not in payload:
                raise CacheIntegrityError(f"Production cache lost source metadata: {key}")
            _exact(payload[key], reference[key][anchors], f"{key} shard {entry['start']}")
        _exact(payload["text_index"], expected_text_indices[anchors], f"task text association shard {entry['start']}")
        for key, offsets in (("image_is_pad", torch.arange(0, 33, 4)),
                             ("action_is_pad", torch.arange(32)), ("proprio_is_pad", torch.arange(33))):
            _exact(payload[key], anchors[:, None] + offsets >= ends[:, None], f"{key} shard {entry['start']}")
        labels_checked += len(anchors)
    windows = episode_windows(lengths, selected_episodes)
    raw_checks = []
    for window in windows:
        index = window["sample_index"]
        actual, expected = cache[index], getter(index)
        if expected.get("sample_index") != index or actual["prompt"] != expected["prompt"]:
            raise CacheIntegrityError("Full stock getter substituted an index or changed its task")
        for key in (*NUMERIC_SHAPES, "raw_current_state", "raw_current_wrench"):
            _exact(actual[key], expected[key], f"full stock getter {index}: {key}")
        for key in ("context", "context_mask"):
            _exact(actual[key], source_texts[expected["prompt"]][key], f"full stock text {index}: {key}")
        raw_checks.append({**window, "exact": True})
    boundaries = []
    offset = 0
    for episode, length in enumerate(lengths):
        boundaries.append({"episode_index": episode, "start_index": offset,
                           "end_index": offset + length - 1, "length": length,
                           "labels_and_padding_exact": True})
        offset += length
    return {"payload_samples_checked": labels_checked, "all_labels_exact": True,
            "all_padding_exact": True, "all_texts_exact": True, "texts_checked": len(source_texts),
            "all_task_text_associations_exact": True,
            "episode_boundaries": boundaries, "strict_stock_windows": raw_checks}


def vae_measurements(cache, getter, encoder, *, episodes=VAE_EPISODES):
    """Measure agreement; hardware/batch-one equality is not a batch-16 gate."""
    import torch
    from fastwam.cache.latent import CacheIntegrityError, LATENT_SHAPE

    measurements = []
    for window in episode_windows(cache.manifest["identity"]["episode_lengths"], episodes):
        sample = getter(window["sample_index"])
        with torch.inference_mode():
            actual = encoder(sample["video"].unsqueeze(0)).detach().cpu()
        expected = cache[window["sample_index"]]["video_latents"].unsqueeze(0)
        if tuple(actual.shape) != (1, *LATENT_SHAPE) or actual.dtype != torch.bfloat16 or not bool(torch.isfinite(actual).all()):
            raise CacheIntegrityError("VAE comparison output changed shape/dtype or is nonfinite")
        difference = actual.float() - expected.float()
        measurements.append({**window, "torch_equal": torch.equal(actual, expected),
                             "max_abs_error": difference.abs().max().item(),
                             "mean_abs_error": difference.abs().mean().item(),
                             "rmse": difference.square().mean().sqrt().item()})
    return {"status": "measured", "batch_size": 1, "sample_count": len(measurements),
            "all_torch_equal": all(row["torch_equal"] for row in measurements),
            "bitwise_batch16_hardware_gate": False, "numeric_tolerance_gate": None,
            "comparisons": measurements}


def verify(args, receipt):
    from omegaconf import OmegaConf
    from fastwam.cache import LatentCacheDataset
    from fastwam.cache.latent import CacheIntegrityError, file_sha256
    from fastwam.cache.source import build_index_rows, build_strict_source

    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise ValueError("Feature verification is a single-process audit, not a distributed training command")
    if args.check_vae and not args.vae_weights:
        raise ValueError("--check-vae requires explicit local --vae-weights")

    helper = _precompute_helpers()
    _, raw = helper.load_config(args)
    cache = LatentCacheDataset(args.cache_root, expected_identity_sha256=args.expected_identity)
    manifest, identity = cache.manifest, cache.manifest["identity"]
    receipt.update(cache_identity_sha256=manifest["identity_sha256"],
                   cache_manifest_sha256=file_sha256(cache.root / "manifest.json"),
                   encoder_source_code_sha256=identity["source_code_sha256"],
                   samples=identity["sample_count"], episodes=len(identity["episode_lengths"]))
    pins = {"vae_weights_sha256": helper.EXPECTED_VAE_SHA256,
            "text_encoder_weights_sha256": helper.EXPECTED_T5_SHA256,
            "dataset_stats_sha256": helper.EXPECTED_STATS_SHA256}
    if any(identity[key] != value for key, value in pins.items()):
        raise CacheIntegrityError("Cache identity differs from the pinned VAE/T5/full50 statistics")
    asset_checks = {"identity_pins": pins, "local_weights": {}}
    for key, path in (("vae_weights_sha256", args.vae_weights), ("text_encoder_weights_sha256", args.t5_weights)):
        if path:
            digest = file_sha256(Path(path).resolve())
            if digest != pins[key]:
                raise CacheIntegrityError(f"Local official component SHA256 differs: {key}")
            asset_checks["local_weights"][key] = {"path": str(Path(path).resolve()), "sha256": digest}
    if file_sha256(raw.pretrained_norm_stats) != pins["dataset_stats_sha256"]:
        raise CacheIntegrityError("Raw source statistics changed")
    spec = importlib.util.spec_from_file_location("feature_verify_conversion", REPO_ROOT / "scripts/forcevla/prepare_forcevla.py")
    converter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(converter)
    dataset_root = Path(raw.dataset_dirs[0]).resolve()
    converter.validate_converted_libero2cam(dataset_root)
    if file_sha256(dataset_root / "conversion_manifest.json") != identity["data_conversion_sha256"]:
        raise CacheIntegrityError("Converted source identity differs from the cached source")
    for relative, expected in identity["source_code_sha256"].items():
        path = (REPO_ROOT / relative).resolve()
        if not path.is_relative_to(REPO_ROOT) or file_sha256(path) != expected:
            raise CacheIntegrityError(f"Encoder source changed: {relative}")
    resolved = OmegaConf.to_container(raw, resolve=True)
    preprocessing = {key: value for key, value in resolved.items()
                     if key not in ("dataset_dirs", "pretrained_norm_stats", "text_embedding_cache_dir", "use_text_embed_cache")}
    receipt["runtime"] = runtime_versions()
    preprocessing["runtime_versions"] = receipt["runtime"]["packages"]
    if preprocessing != identity["preprocessing"]:
        raise CacheIntegrityError("Stock preprocessing configuration or runtime package versions changed")
    texts, hashes = helper._task_texts(raw)
    reference, lengths = build_numeric_reference(raw)
    if lengths != identity["episode_lengths"]:
        raise CacheIntegrityError("Source episode lengths differ from the complete cached source")
    rows = build_index_rows(dataset_root)
    # The stock reader consults PartialState for its stats output. Initialize
    # that shared state explicitly on CPU, including when CUDA is visible.
    from accelerate import PartialState
    PartialState(cpu=True)
    getter = build_strict_source(raw, rows, Path(args.output_receipt).resolve().parent / "feature_verifier_source_runtime")
    receipt["assets"] = asset_checks
    receipt["cpu"] = verify_cpu_payloads(cache, reference, texts, hashes, getter,
                                          task_prompts=task_prompts_from_source(dataset_root))
    receipt["vae"] = {"status": "not_requested", "loaded": False}
    if args.check_vae:
        if identity["encoding_backend"] != "cudagraphs":
            raise CacheIntegrityError("Explicit VAE check requires the same cached cudagraphs encoding backend")
        import torch
        from fastwam.models.wan22.helpers.loader import _load_registered_model

        receipt["runtime"] = runtime_versions(check_vae=True, device=args.device)
        vae = _load_registered_model(str(Path(args.vae_weights).resolve()), "wan_video_vae",
                                     torch_dtype=torch.bfloat16, device=args.device)
        vae.eval().requires_grad_(False)
        encode = torch.compile(vae.model.encode, backend="cudagraphs", fullgraph=True)
        def encoder(video):
            with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                return encode(video.to(device=args.device, dtype=torch.bfloat16), vae.scale).clone()
        receipt["vae"] = {**vae_measurements(cache, getter, encoder), "loaded": True,
                          "input_dtype": "bfloat16", "compute_dtype": "bfloat16", "backend": "cudagraphs"}
    receipt["status"] = "verified_cpu_with_vae_measurements" if args.check_vae else "verified_cpu"
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--config")
    group.add_argument("--task", default="forcevla_libero_2cam_cached_4gpu_30k")
    parser.add_argument("--raw-config")
    parser.add_argument("--dataset")
    parser.add_argument("--stats")
    parser.add_argument("--text-cache")
    parser.add_argument("--cache-root", required=True)
    parser.add_argument("--expected-identity")
    parser.add_argument("--vae-weights")
    parser.add_argument("--t5-weights")
    parser.add_argument("--output-receipt", required=True)
    parser.add_argument("--check-vae", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    output = Path(args.output_receipt).resolve()
    if output.exists() or output.is_relative_to(Path(args.cache_root).resolve()):
        parser.error("Receipt must be a new file outside the strict cache directory")
    started = time.monotonic()
    receipt = {"schema": "fastwam-feature-verification-v1", "status": "checking",
               "training_started": False, "optimizer_constructed": False,
               "check_vae_requested": args.check_vae, "cache_root": str(Path(args.cache_root).resolve())}
    exit_code = 0
    try:
        verify(args, receipt)
    except Exception as error:
        receipt.update(status="failed", error_type=type(error).__name__, error=str(error))
        exit_code = 1
    from fastwam.cache.latent import file_sha256
    receipt.update(elapsed_seconds=time.monotonic() - started,
                   verifier_sha256=file_sha256(Path(__file__).resolve()))
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + f".tmp-{os.getpid()}")
    temporary.write_text(json.dumps(receipt, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(output)
    print(json.dumps({"status": receipt["status"], "receipt": str(output),
                      "training_started": False, "vae_check_requested": args.check_vae}, ensure_ascii=False))
    if exit_code:
        print(receipt["error"], file=sys.stderr)
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
