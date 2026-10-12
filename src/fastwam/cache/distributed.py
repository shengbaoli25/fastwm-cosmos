"""Independent nine-frame workers and an audited, immutable cache merge.

Each rank owns complete episodes. Workers never share a writable cache or a
VAE temporal state. Numeric references are CPU tensors produced by the stock
formatter/normalizer; this layer independently reconstructs every window.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import time

import torch
from safetensors.torch import load_file

from .latent import (CacheIntegrityError, INDEX_KEYS, LATENT_SHAPE, NUMERIC_SHAPES,
                     OPTIONAL_SHAPES, VIDEO_SHAPE, _atomic_json, _expected_rows,
                     _path, _save_tensors, _sha, _validate_identity, _validate_shard,
                     _validate_text, cache_identity_sha256, file_sha256, validate_cache)


PLAN_SCHEMA = "fastwam-pixel-window-distributed-plan-v1"
WORKER_SCHEMA = "fastwam-pixel-window-worker-v1"
REFERENCE_SHAPES = {"action": (7,), "proprio": (8,), "raw_action": (7,),
                    "raw_state": (7,), "raw_wrench": (6,), "task_index": ()}


def _require(condition, message):
    if not condition:
        raise CacheIntegrityError(message)


def tensor_bits_sha256(tensor):
    """Hash values as bytes, including signed zero, plus dtype and shape."""
    _require(isinstance(tensor, torch.Tensor) and tensor.device.type == "cpu", "References must be CPU tensors")
    import hashlib
    digest = hashlib.sha256()
    digest.update(json.dumps([str(tensor.dtype), list(tensor.shape)], separators=(",", ":")).encode())
    digest.update(tensor.detach().contiguous().view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _bits_equal(left, right):
    return (left.dtype == right.dtype and left.shape == right.shape and
            torch.equal(left.detach().cpu().contiguous().view(torch.uint8),
                        right.detach().cpu().contiguous().view(torch.uint8)))


def episode_partition(identity, world_size):
    """Longest episode first, then least loaded rank, with explicit tie rules."""
    _validate_identity(identity)
    _require(type(world_size) is int and 0 < world_size <= len(identity["episode_lengths"]), "Invalid worker world size")
    if identity["scope"] == "production":
        _require(world_size == 4, "Production pixel cache requires four independent workers")
    lengths = identity["episode_lengths"]
    loads, ranks = [0] * world_size, [[] for _ in range(world_size)]
    for episode in sorted(range(len(lengths)), key=lambda number: (-lengths[number], number)):
        rank = min(range(world_size), key=lambda number: (loads[number], number))
        ranks[rank].append(episode)
        loads[rank] += lengths[episode]
    return [sorted(episodes) for episodes in ranks]


def _reference_inventory(identity, index_rows, reference):
    _require(set(reference) == set(REFERENCE_SHAPES), "Numeric reference tensor inventory differs")
    count = identity["sample_count"]
    inventory = {}
    for key, shape in REFERENCE_SHAPES.items():
        value = reference[key]
        dtype = torch.int64 if key == "task_index" else torch.float64 if key.startswith("raw_") else torch.float32
        _require(isinstance(value, torch.Tensor) and value.device.type == "cpu" and
                 tuple(value.shape) == (count, *shape) and value.dtype == dtype,
                 "Numeric reference shape/dtype/device differs: " + key)
        _require(not value.is_floating_point() or bool(torch.isfinite(value).all()), "Nonfinite numeric reference: " + key)
        inventory[key] = {"shape": list(value.shape), "dtype": str(dtype), "sha256": tensor_bits_sha256(value)}
    expected = _expected_rows(identity)
    _require(len(index_rows) == len(expected), "Source row count differs")
    rows = []
    for index, (row, coordinates) in enumerate(zip(index_rows, expected)):
        _require(all(type(row.get(key)) is int for key in ("sample_index", "episode_index", "frame_index", "task_index")), "Source row indices must be exact integers")
        _require(tuple(row[key] for key in ("sample_index", "episode_index", "frame_index")) == coordinates and
                 row["task_index"] == int(reference["task_index"][index]) and row["task_index"] >= 0,
                 "Source row coordinate/task identity differs")
        for key, column in (("raw_current_state", "raw_state"), ("raw_current_wrench", "raw_wrench")):
            if key in row:
                _require(_bits_equal(row[key], reference[column][index]), "Source raw labels differ: " + key)
        rows.append({key: row[key] for key in ("sample_index", "episode_index", "frame_index", "task_index")})
    return inventory, cache_identity_sha256(rows)


def make_distributed_plan(identity, pixel_identity_sha256, index_rows, numeric_reference,
                          task_prompts, *, world_size, device_bindings, source_provenance):
    _validate_identity(identity)
    _require(_sha(pixel_identity_sha256), "Exact pixel cache identity is required")
    partition = episode_partition(identity, world_size)
    _require(isinstance(source_provenance, dict) and bool(source_provenance), "Source provenance is required")
    inventory, rows_sha = _reference_inventory(identity, index_rows, numeric_reference)
    _require(all(type(key) is int and key >= 0 and isinstance(value, str) and value for key, value in task_prompts.items()), "Invalid task-to-prompt mapping")
    _require(set(task_prompts.values()) == set(identity["text_payload_sha256"]) and
             all(row["task_index"] in task_prompts for row in index_rows), "Task indices/prompts differ from frozen T5 bindings")
    _require(isinstance(device_bindings, list) and len(device_bindings) == world_size and
             all(isinstance(value, dict) and isinstance(value.get("device"), str) and value["device"] and
                 isinstance(value.get("physical_device"), str) and value["physical_device"] for value in device_bindings), "Each rank requires explicit logical/physical device binding")
    _require(len({value["physical_device"] for value in device_bindings}) == world_size, "Worker physical devices overlap")
    if identity["scope"] == "production":
        _require(all(str(value["device"]).startswith("cuda:") for value in device_bindings), "Production workers require explicit CUDA devices")
        relative = "src/fastwam/cache/distributed.py"
        _require(identity["source_code_sha256"].get(relative) == file_sha256(__file__), "New distributed source is absent or changed in cache identity")
    plan = {"schema": PLAN_SCHEMA, "identity": identity, "identity_sha256": cache_identity_sha256(identity),
            "pixel_identity_sha256": pixel_identity_sha256, "world_size": world_size,
            "partition": partition, "partition_rule": "length_desc_episode_asc_then_load_asc_rank_asc",
            "device_bindings": device_bindings, "source_rows_sha256": rows_sha,
            "numeric_reference": inventory, "task_prompts": {str(key): value for key, value in sorted(task_prompts.items())},
            "source_provenance": source_provenance,
            "encoding": "one independent complete nine-frame batch-one stock VAE call per original anchor",
            "formal_training_started": False}
    plan["sha256"] = cache_identity_sha256(plan)
    return plan


def validate_distributed_plan(plan, index_rows, numeric_reference):
    _require(isinstance(plan, dict) and plan.get("schema") == PLAN_SCHEMA, "Unknown distributed plan")
    _require(plan.get("sha256") == cache_identity_sha256({key: value for key, value in plan.items() if key != "sha256"}), "Distributed plan hash differs")
    try:
        tasks = {int(key): value for key, value in plan["task_prompts"].items()}
        rebuilt = make_distributed_plan(plan["identity"], plan["pixel_identity_sha256"], index_rows, numeric_reference,
                                        tasks, world_size=plan["world_size"], device_bindings=plan["device_bindings"],
                                        source_provenance=plan["source_provenance"])
    except (KeyError, TypeError, ValueError) as error:
        raise CacheIntegrityError("Invalid distributed plan: " + str(error)) from error
    _require(rebuilt == plan, "Distributed plan/reference/source inventory differs")
    return plan


def expected_numeric_window(identity, row, reference):
    index, episode, frame = (row[key] for key in ("sample_index", "episode_index", "frame_index"))
    length = identity["episode_lengths"][episode]
    start = index - frame
    indices = start + torch.clamp(frame + torch.arange(32), max=length - 1)
    return {"action": reference["action"][indices], "proprio": reference["proprio"][indices],
            "raw_action": reference["raw_action"][indices], "raw_wrench_future": reference["raw_wrench"][indices],
            "raw_current_state": reference["raw_state"][index], "raw_current_wrench": reference["raw_wrench"][index],
            "image_is_pad": frame + torch.arange(0, 33, 4) >= length,
            "action_is_pad": frame + torch.arange(32) >= length,
            "proprio_is_pad": frame + torch.arange(33) >= length}


def _assigned_indices(plan, rank):
    _require(type(rank) is int and 0 <= rank < plan["world_size"], "Invalid worker rank")
    episodes = set(plan["partition"][rank])
    return [index for index, episode, _ in _expected_rows(plan["identity"]) if episode in episodes]


def worker_identity(plan, rank):
    _assigned_indices(plan, rank)
    value = {"schema": WORKER_SCHEMA, "plan_sha256": plan["sha256"], "rank": rank,
             "device_binding": plan["device_bindings"][rank], "assigned_episodes": plan["partition"][rank],
             "cache_identity_sha256": plan["identity_sha256"], "pixel_identity_sha256": plan["pixel_identity_sha256"],
             "vae_weights_sha256": plan["identity"]["vae_weights_sha256"],
             "text_encoder_weights_sha256": plan["identity"]["text_encoder_weights_sha256"],
             "source_rows_sha256": plan["source_rows_sha256"]}
    value["sha256"] = cache_identity_sha256(value)
    return value


def _audit_payload(payload, entry, plan, index_rows, reference, texts):
    identity = plan["identity"]
    _validate_shard(payload, entry, identity, texts)
    _require(set(payload) == {"video_latents", *NUMERIC_SHAPES, *OPTIONAL_SHAPES, *INDEX_KEYS}, "Worker must retain all original raw/numeric labels")
    prompts = [row["prompt"] for row in texts]
    for offset, index in enumerate(range(entry["start"], entry["stop"])):
        row = index_rows[index]
        _require(int(payload["task_index"][offset]) == row["task_index"] and
                 prompts[int(payload["text_index"][offset])] == plan["task_prompts"][str(row["task_index"])], "Window task/T5 association differs")
        expected = expected_numeric_window(identity, row, reference)
        for key, value in expected.items():
            _require(_bits_equal(payload[key][offset], value), "Original numeric reference bytes differ: " + key)


def _save_immutable(root, relative, tensors):
    target = root / relative
    _require(not target.exists() and not target.with_name(target.name + ".tmp").exists(), "Committed worker payload cannot be replaced")
    return _save_tensors(root, relative, tensors)


def validate_worker_cache(worker_root, plan, index_rows, numeric_reference, *, require_complete=True,
                          validate_plan=True, expected_manifest_sha256=None):
    if validate_plan:
        validate_distributed_plan(plan, index_rows, numeric_reference)
    root = Path(worker_root).resolve()
    raw_manifest = (root / "manifest.json").read_bytes()
    read_manifest_sha = hashlib.sha256(raw_manifest).hexdigest()
    _require(expected_manifest_sha256 is None or expected_manifest_sha256 == read_manifest_sha, "Worker manifest drift before validation read")
    manifest = json.loads(raw_manifest)
    _require(manifest.get("schema") == WORKER_SCHEMA and manifest.get("plan") == plan and
             manifest.get("plan_sha256") == plan["sha256"], "Worker plan/source/pixel identity differs")
    rank = manifest.get("rank")
    indices = _assigned_indices(plan, rank)
    _require(manifest.get("worker_identity") == worker_identity(plan, rank), "Worker identity hash/source/device/weights binding differs")
    _require(manifest.get("device_binding") == plan["device_bindings"][rank] and
             manifest.get("assigned_episodes") == plan["partition"][rank] and
             manifest.get("assigned_samples") == len(indices), "Worker rank/device/partition binding differs")
    _require(manifest.get("status") in ("partial", "complete") and not manifest.get("error"), "Worker failed or has invalid status")
    _require(not require_complete or manifest["status"] == "complete", "Partial worker set cannot be merged")
    texts = manifest.get("texts", [])
    _require([entry.get("prompt") for entry in texts] == sorted(plan["identity"]["text_payload_sha256"]), "Worker task text inventory/order differs")
    for entry in texts:
        _require(entry.get("source_payload_sha256") == plan["identity"]["text_payload_sha256"][entry["prompt"]], "Worker source T5 hash differs")
        path = _path(root, entry["path"])
        _require(file_sha256(path) == entry.get("sha256"), "Worker text hash differs")
        _validate_text(load_file(str(path)))
    observed = []
    expected_files = {"manifest.json", *[entry["path"] for entry in texts]}
    for entry in manifest.get("shards", []):
        start, stop = entry.get("start"), entry.get("stop")
        _require(type(start) is int and type(stop) is int and 0 <= start < stop <= plan["identity"]["sample_count"], "Invalid worker shard range")
        shard_indices = list(range(start, stop))
        _require(indices[len(observed):len(observed) + len(shard_indices)] == shard_indices, "Worker shard gaps/duplicates/foreign indices")
        _require(index_rows[start]["episode_index"] == index_rows[stop - 1]["episode_index"], "Worker shard spans episodes")
        path = _path(root, entry["path"])
        _require(file_sha256(path) == entry.get("sha256"), "Worker shard hash differs")
        _audit_payload(load_file(str(path)), entry, plan, index_rows, numeric_reference, texts)
        observed.extend(shard_indices)
        expected_files.add(entry["path"])
    _require(type(manifest.get("completed_samples")) is int and manifest["completed_samples"] == len(observed) and
             (manifest["status"] != "complete" or observed == indices), "Worker completion is inaccurate")
    _require({path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file()} == expected_files, "Worker contains missing/untracked files")
    _require(file_sha256(root / "manifest.json") == read_manifest_sha, "Worker manifest changed while being validated")
    return manifest


def _write_worker_cache(worker_root, plan, rank, sample_getter, index_rows, encoder, texts,
                       numeric_reference, *, shard_size=128, resume=False,
                       max_new_samples=None, max_seconds=None, on_shard_commit=None):
    validate_distributed_plan(plan, index_rows, numeric_reference)
    _require(type(shard_size) is int and shard_size > 0, "Positive integer shard size required")
    _require(max_new_samples is None or type(max_new_samples) is int and max_new_samples > 0, "Invalid worker sample bound")
    _require(max_seconds is None or isinstance(max_seconds, (int, float)) and not isinstance(max_seconds, bool) and math.isfinite(max_seconds) and max_seconds > 0, "Invalid worker time bound")
    indices = _assigned_indices(plan, rank)
    _require(set(texts) == set(plan["identity"]["text_payload_sha256"]), "Worker T5 prompts differ")
    for value in texts.values():
        _validate_text(value)
    root = Path(worker_root).resolve()
    if root.exists():
        _require(resume, "Worker output exists; explicit resume required")
        manifest = validate_worker_cache(root, plan, index_rows, numeric_reference, require_complete=False, validate_plan=False)
        _require(manifest["rank"] == rank and manifest["shard_size"] == shard_size, "Worker resume rank/shard size changed")
        for entry in manifest["texts"]:
            previous = load_file(str(_path(root, entry["path"])))
            _require(all(_bits_equal(previous[key], texts[entry["prompt"]][key]) for key in previous), "Worker resume T5 values changed")
        if manifest["status"] == "complete":
            return manifest
    else:
        root.mkdir(parents=True)
        manifest = {"schema": WORKER_SCHEMA, "plan": plan, "plan_sha256": plan["sha256"], "rank": rank,
                    "worker_identity": worker_identity(plan, rank),
                    "device_binding": plan["device_bindings"][rank], "assigned_episodes": plan["partition"][rank],
                    "assigned_samples": len(indices), "status": "partial", "completed_samples": 0,
                    "shard_size": shard_size, "shards": [], "texts": [], "formal_training_started": False}
        for number, prompt in enumerate(sorted(texts)):
            entry = _save_immutable(root, f"texts/task_{number:04d}.safetensors", texts[prompt])
            entry.update(prompt=prompt, source_payload_sha256=plan["identity"]["text_payload_sha256"][prompt])
            manifest["texts"].append(entry)
        _atomic_json(root / "manifest.json", manifest)
    prompt_indices = {entry["prompt"]: number for number, entry in enumerate(manifest["texts"])}
    initial, started = manifest["completed_samples"], time.monotonic()
    limit = min(len(indices), initial + max_new_samples) if max_new_samples is not None else len(indices)
    elapsed_before = manifest.get("preprocessing_seconds", 0.)
    offset = initial
    try:
        while offset < limit:
            first = indices[offset]
            episode = index_rows[first]["episode_index"]
            samples = []
            while offset < limit and len(samples) < shard_size and index_rows[indices[offset]]["episode_index"] == episode:
                if max_seconds is not None and time.monotonic() - started >= max_seconds:
                    break
                index = indices[offset]
                sample = sample_getter(index)
                _require(type(sample.get("sample_index")) is int and sample["sample_index"] == index, "Worker source substituted an anchor")
                video = sample.get("video")
                _require(isinstance(video, torch.Tensor) and tuple(video.shape) == VIDEO_SHAPE and
                         video.dtype == torch.float32 and video.device.type == "cpu" and
                         video.stride() == (100352, 301056, 448, 1) and bool(torch.isfinite(video).all()),
                         "Worker requires stock FP32 normalized RGB/layout")
                prompt = plan["task_prompts"][str(index_rows[index]["task_index"])]
                _require(sample.get("prompt") == prompt, "Worker task prompt changed")
                numeric = expected_numeric_window(plan["identity"], index_rows[index], numeric_reference)
                for key, expected in numeric.items():
                    value = sample.get(key)
                    _require(isinstance(value, torch.Tensor) and _bits_equal(value, expected), "Worker sample numeric bytes differ: " + key)
                    numeric[key] = value.detach().cpu().clone()
                with torch.inference_mode():
                    latent = encoder(video.unsqueeze(0))
                _require(isinstance(latent, torch.Tensor) and tuple(latent.shape) == (1, *LATENT_SHAPE) and latent.dtype == torch.bfloat16,
                         "Worker VAE requires BF16 full-nine-frame latent")
                numeric["video_latents"] = latent[0].detach().cpu().clone()
                numeric.update({key: torch.tensor(index_rows[index][key], dtype=torch.int64) for key in INDEX_KEYS if key != "text_index"})
                numeric["text_index"] = torch.tensor(prompt_indices[prompt], dtype=torch.int64)
                samples.append(numeric)
                offset += 1
            if not samples:
                break
            stop = first + len(samples)
            payload = {key: torch.stack([sample[key] for sample in samples]) for key in samples[0]}
            entry = {"start": first, "stop": stop, "tensor_shapes": {key: list(value.shape) for key, value in payload.items()}}
            _audit_payload(payload, entry, plan, index_rows, numeric_reference, manifest["texts"])
            entry.update(_save_immutable(root, f"shards/window_{first:06d}_{stop:06d}.safetensors", payload))
            manifest["shards"].append(entry)
            manifest["completed_samples"] = offset
            manifest["status"] = "complete" if offset == len(indices) else "partial"
            manifest["preprocessing_seconds"] = elapsed_before + time.monotonic() - started
            _atomic_json(root / "manifest.json", manifest)
            if on_shard_commit is not None:
                on_shard_commit(manifest, entry)
        if manifest["status"] != "complete":
            manifest["stopped_reason"] = "max_new_samples" if offset >= limit else "max_seconds"
        else:
            manifest.pop("stopped_reason", None)
        manifest["preprocessing_seconds"] = elapsed_before + time.monotonic() - started
        _atomic_json(root / "manifest.json", manifest)
    except BaseException as error:
        manifest.update(status="error", error={"type": type(error).__name__, "message": str(error)})
        _atomic_json(root / "manifest.json", manifest)
        raise
    return validate_worker_cache(root, plan, index_rows, numeric_reference,
                                 require_complete=manifest["status"] == "complete", validate_plan=False)


def write_worker_cache(worker_root, *args, **kwargs):
    """A rank has one writer; a stale lock requires explicit external recovery."""
    root = Path(worker_root).resolve()
    lock = root.with_name(root.name + "_writer.lock")
    lock.parent.mkdir(parents=True, exist_ok=True)
    with lock.open("x") as stream:
        stream.write(json.dumps({"pid": os.getpid(), "worker_root": str(root)}) + "\n")
    try:
        return _write_worker_cache(root, *args, **kwargs)
    finally:
        lock.unlink()


def _copy_checked(source, target, expected_sha):
    before = source.stat()
    signature = lambda value: (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns)
    target.parent.mkdir(parents=True, exist_ok=True)
    with source.open("rb") as reader, target.open("xb") as writer:
        shutil.copyfileobj(reader, writer, 4 * 1024**2)
    _require(signature(source.stat()) == signature(before) and target.stat().st_size == before.st_size and
             file_sha256(target) == expected_sha, "Worker payload drift during immutable merge")


def _merge_worker_caches(worker_roots, output_root, plan, index_rows, numeric_reference, *, receipt_path=None):
    """Publish a standard complete cache only after all rank/source audits pass."""
    validate_distributed_plan(plan, index_rows, numeric_reference)
    roots = [Path(root).resolve() for root in worker_roots]
    _require(len(roots) == plan["world_size"] and len(set(roots)) == len(roots), "Missing/duplicate worker roots")
    source_snapshots = {str(root): file_sha256(root / "manifest.json") for root in roots}
    workers = [validate_worker_cache(root, plan, index_rows, numeric_reference, validate_plan=False,
                                    expected_manifest_sha256=source_snapshots[str(root)]) for root in roots]
    _require(all(file_sha256(root / "manifest.json") == source_snapshots[str(root)] for root in roots), "Worker manifest drift during validation")
    by_rank = {worker["rank"]: (root, worker) for root, worker in zip(roots, workers)}
    _require(set(by_rank) == set(range(plan["world_size"])) and len(by_rank) == len(workers), "Missing/duplicate worker ranks")
    output = Path(output_root).resolve()
    receipt = Path(receipt_path).resolve() if receipt_path else output.with_name(output.name + "_merge.json")
    _require(not output.exists() and not receipt.exists(), "Merge requires fresh cache and receipt paths")
    stage = output.with_name(output.name + f".staging-{os.getpid()}")
    protected = [output, stage, *roots]
    _require(all(not receipt.is_relative_to(root) and not root.is_relative_to(receipt) for root in protected) and
             all(not output.is_relative_to(root) and not root.is_relative_to(output) for root in roots), "Merge output/receipt overlaps worker cache or staging")
    _require(not stage.exists(), "Ambiguous merge staging directory exists")
    stage.mkdir(parents=True)
    texts = by_rank[0][1]["texts"]
    for root, worker in by_rank.values():
        _require(worker["texts"] == texts, "Workers disagree on actual T5 tensors")
    for entry in texts:
        _copy_checked(_path(by_rank[0][0], entry["path"]), stage / entry["path"], entry["sha256"])
    entries = sorted([(entry, root) for root, worker in by_rank.values() for entry in worker["shards"]], key=lambda pair: pair[0]["start"])
    stop = 0
    for entry, root in entries:
        _require(entry["start"] == stop, "Merged shards overlap, duplicate, or omit source indices")
        _copy_checked(_path(root, entry["path"]), stage / entry["path"], entry["sha256"])
        stop = entry["stop"]
    _require(stop == plan["identity"]["sample_count"], "Merged cache is missing original anchors")
    manifest = {"identity": plan["identity"], "identity_sha256": plan["identity_sha256"], "status": "complete",
                "completed_samples": stop, "shard_size": max(worker["shard_size"] for _, worker in by_rank.values()),
                "texts": texts, "shards": [entry for entry, _ in entries],
                "preprocessing_seconds": max(worker.get("preprocessing_seconds", 0.) for _, worker in by_rank.values())}
    _atomic_json(stage / "manifest.json", manifest)
    validate_cache(stage, plan["identity_sha256"], require_complete=True)
    for entry, _ in entries:
        _audit_payload(load_file(str(_path(stage, entry["path"]))), entry, plan, index_rows, numeric_reference, texts)
    for root in roots:
        validate_worker_cache(root, plan, index_rows, numeric_reference, validate_plan=False,
                              expected_manifest_sha256=source_snapshots[str(root)])
    _require(all(file_sha256(root / "manifest.json") == source_snapshots[str(root)] for root in roots), "Worker manifest drift during merge")
    result = {"schema": "fastwam-pixel-window-merge-receipt-v1", "status": "complete", "plan_sha256": plan["sha256"],
              "identity_sha256": plan["identity_sha256"], "pixel_identity_sha256": plan["pixel_identity_sha256"],
              "completed_samples": stop, "source_rows_sha256": plan["source_rows_sha256"],
              "numeric_reference": plan["numeric_reference"], "worker_manifests": source_snapshots,
              "worker_ranks": sorted(by_rank), "manifest_sha256": file_sha256(stage / "manifest.json"),
              "worker_identity_sha256": {str(rank): worker["worker_identity"]["sha256"] for rank, (_, worker) in by_rank.items()},
              "copy_policy": "fresh copies; no mutable hardlinks or relabeled stock pilot shards",
              "hardware_equivalence_verified": False, "formal_training_started": False}
    receipt.parent.mkdir(parents=True, exist_ok=True)
    _require(not output.exists(), "Merge output appeared during validation")
    stage.rename(output)
    with receipt.open("x") as stream:
        stream.write(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    return result


def merge_worker_caches(worker_roots, output_root, *args, **kwargs):
    """Serialize publication attempts; a crash leaves a lock for explicit recovery."""
    output = Path(output_root).resolve()
    roots = [Path(root).resolve() for root in worker_roots]
    receipt = Path(kwargs["receipt_path"]).resolve() if kwargs.get("receipt_path") else output.with_name(output.name + "_merge.json")
    stage = output.with_name(output.name + f".staging-{os.getpid()}")
    _require(not output.exists() and not receipt.exists(), "Merge requires fresh cache and receipt paths")
    _require(all(not output.is_relative_to(root) and not root.is_relative_to(output) for root in roots) and
             all(not receipt.is_relative_to(root) and not root.is_relative_to(receipt) for root in [output, stage, *roots]),
             "Merge output/receipt overlaps worker cache or staging")
    lock = output.with_name(output.name + "_merge.lock")
    lock.parent.mkdir(parents=True, exist_ok=True)
    with lock.open("x") as stream:
        stream.write(json.dumps({"pid": os.getpid(), "output_root": str(output)}) + "\n")
    try:
        return _merge_worker_caches(roots, output, *args, **kwargs)
    finally:
        lock.unlink()
