"""Strict, sharded BF16 VAE-window cache with original normalized labels.

Every latent is produced from a complete independently encoded nine-frame
window. Episode-wide encoding followed by slicing is not this protocol.
Training reads only safetensors, small JSON metadata and task-level text.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
import math
from collections import OrderedDict
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file


SCHEMA = "fastwam-frozen-window-cache-v1"
LATENT_SHAPE = (48, 3, 14, 28)
VIDEO_SHAPE = (3, 9, 224, 448)
NUMERIC_SHAPES = {"action": (32, 7), "proprio": (32, 8),
                  "image_is_pad": (9,), "action_is_pad": (32,), "proprio_is_pad": (33,)}
OPTIONAL_SHAPES = {"raw_current_state": (7,), "raw_current_wrench": (6,),
                   "raw_wrench_future": (32, 6), "raw_action": (32, 7)}
INDEX_KEYS = ("sample_index", "episode_index", "frame_index", "task_index", "text_index")


class CacheIntegrityError(ValueError):
    pass


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def cache_identity_sha256(identity):
    return hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(",", ":"),
                                    ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def _atomic_json(path, value):
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(path)


def _sha(value):
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def make_cache_identity(*, data_conversion_sha256, dataset_stats_sha256, vae_weights_sha256,
                        text_encoder_weights_sha256, preprocessing, source_code_sha256,
                        text_payload_sha256, episode_lengths, test_fixture=False,
                        encoding_backend="cudagraphs", numeric_dtype="float32"):
    identity = {"schema": SCHEMA, "scope": "TEST-only synthetic fixture" if test_fixture else "production",
                "data_conversion_sha256": data_conversion_sha256, "dataset_stats_sha256": dataset_stats_sha256,
                "vae_weights_sha256": vae_weights_sha256, "text_encoder_weights_sha256": text_encoder_weights_sha256,
                "preprocessing": preprocessing, "source_code_sha256": source_code_sha256,
                "text_payload_sha256": text_payload_sha256, "episode_lengths": list(episode_lengths),
                "sample_count": sum(episode_lengths), "latent_dtype": "bfloat16", "latent_shape": list(LATENT_SHAPE),
                "video_shape": list(VIDEO_SHAPE), "video_num_frames": 9, "numeric_dtype": numeric_dtype,
                "raw_numeric_dtype": "float64", "vae_input_dtype": "bfloat16", "vae_compute_dtype": "bfloat16",
                "encoding": "independent_complete_9frame_window_scaled_posterior_mean",
                "encoding_backend": encoding_backend, "vae_temporal_cache": "clear_before_and_after_each_window",
                "task_context": "zero_original_masked_rows_then_all_true_attention_mask"}
    _validate_identity(identity)
    return identity


def _validate_identity(identity):
    if identity.get("schema") != SCHEMA or identity.get("latent_shape") != list(LATENT_SHAPE):
        raise CacheIntegrityError("Cache schema/latent shape differs from the fixed LIBERO window contract")
    if (identity.get("latent_dtype") != "bfloat16" or identity.get("video_shape") != list(VIDEO_SHAPE)
            or identity.get("video_num_frames") != 9
            or identity.get("encoding") != "independent_complete_9frame_window_scaled_posterior_mean"
            or identity.get("numeric_dtype") not in ("float32", "float64")
            or identity.get("raw_numeric_dtype") != "float64" or identity.get("vae_input_dtype") != "bfloat16"
            or identity.get("vae_compute_dtype") != "bfloat16"):
        raise CacheIntegrityError("Cache encoding/dtype contract changed")
    for key in ("data_conversion_sha256", "dataset_stats_sha256", "vae_weights_sha256", "text_encoder_weights_sha256"):
        if not _sha(identity.get(key)):
            raise CacheIntegrityError(f"Cache identity needs an exact SHA256: {key}")
    for key in ("source_code_sha256", "text_payload_sha256"):
        if not identity.get(key) or not all(_sha(value) for value in identity[key].values()):
            raise CacheIntegrityError(f"Cache identity needs nonempty source/payload hashes: {key}")
    lengths = identity.get("episode_lengths", [])
    if not lengths or any(type(length) is not int or length < 2 for length in lengths) or sum(lengths) != identity.get("sample_count"):
        raise CacheIntegrityError("Cache episode lengths/sample count are inconsistent")
    scope = identity.get("scope")
    if scope == "production":
        if len(lengths) != 50 or sum(lengths) != 34808 or identity["numeric_dtype"] != "float32":
            raise CacheIntegrityError("Production cache requires all 50 episodes / 34808 anchors and original stock FP32 normalized labels")
    elif scope != "TEST-only synthetic fixture":
        raise CacheIntegrityError("Unknown cache scope")


def _path(root, relative):
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()) or not path.is_file():
        raise CacheIntegrityError(f"Cache file missing or escapes its root: {relative}")
    return path


def _validate_text(payload):
    if set(payload) != {"context", "context_mask"}:
        raise CacheIntegrityError("Task text payload must contain context and context_mask only")
    if (tuple(payload["context"].shape) != (128, 4096) or payload["context"].dtype != torch.bfloat16
            or not bool(torch.isfinite(payload["context"]).all())
            or tuple(payload["context_mask"].shape) != (128,) or payload["context_mask"].dtype != torch.bool
            or not bool(payload["context_mask"].all())):
        raise CacheIntegrityError("Task text must preserve the stock BF16 [128,4096] context / all-true mask")


def _expected_rows(identity):
    rows, offset = [], 0
    for episode, length in enumerate(identity["episode_lengths"]):
        rows.extend((offset + frame, episode, frame) for frame in range(length))
        offset += length
    return rows


def _validate_shard(payload, entry, identity, texts):
    start, stop = entry["start"], entry["stop"]
    n = stop - start
    expected_shapes = {"video_latents": LATENT_SHAPE, **NUMERIC_SHAPES, **{key: () for key in INDEX_KEYS}}
    keys = set(payload)
    if not set(expected_shapes).issubset(keys) or keys - set(expected_shapes) - set(OPTIONAL_SHAPES):
        raise CacheIntegrityError("Cache shard is missing required features or contains unexpected tensors")
    for key, tensor in payload.items():
        shape = expected_shapes.get(key, OPTIONAL_SHAPES.get(key))
        if tuple(tensor.shape) != (n, *shape):
            raise CacheIntegrityError(f"Cache shard tensor shape mismatch: {key}")
        dtype = (torch.bfloat16 if key == "video_latents" else torch.bool if key.endswith("_is_pad")
                 else torch.int64 if key in INDEX_KEYS else torch.float64 if key in OPTIONAL_SHAPES
                 else getattr(torch, identity["numeric_dtype"]))
        if tensor.dtype != dtype or (tensor.is_floating_point() and not bool(torch.isfinite(tensor).all())):
            raise CacheIntegrityError(f"Cache shard tensor dtype/nonfinite mismatch: {key}")
    expected = _expected_rows(identity)[start:stop]
    for key, column in (("sample_index", 0), ("episode_index", 1), ("frame_index", 2)):
        if payload[key].tolist() != [row[column] for row in expected]:
            raise CacheIntegrityError(f"Cache source indices/order changed: {key}")
    for key, offsets in (("image_is_pad", torch.arange(0, 33, 4)),
                         ("action_is_pad", torch.arange(32)), ("proprio_is_pad", torch.arange(33))):
        expected_pad = torch.stack([frame + offsets >= identity["episode_lengths"][episode]
                                    for _, episode, frame in expected])
        if not torch.equal(payload[key], expected_pad):
            raise CacheIntegrityError(f"Cache original episode-end padding changed: {key}")
    if any(index < 0 or index >= len(texts) for index in payload["text_index"].tolist()):
        raise CacheIntegrityError("Cache shard references an absent task text")
    if bool((payload["task_index"] < 0).any()):
        raise CacheIntegrityError("Cache shard contains an invalid task index")
    if entry.get("tensor_shapes") != {key: list(value.shape) for key, value in payload.items()}:
        raise CacheIntegrityError("Cache shard tensor inventory differs from its manifest")


def validate_cache(cache_root, expected_identity_sha256=None, *, require_complete=True):
    root = Path(cache_root).resolve()
    manifest = json.loads((root / "manifest.json").read_text())
    identity = manifest.get("identity", {})
    _validate_identity(identity)
    digest = cache_identity_sha256(identity)
    if manifest.get("identity_sha256") != digest or (expected_identity_sha256 is not None and digest != expected_identity_sha256):
        raise CacheIntegrityError("Cache identity changed; use a separate output directory")
    if manifest.get("status") not in ("partial", "complete"):
        raise CacheIntegrityError("Unknown cache completion status")
    if require_complete and manifest["status"] != "complete":
        raise CacheIntegrityError("Incomplete latent cache cannot be used for training")
    texts = manifest.get("texts", [])
    if not texts or len({entry.get("prompt") for entry in texts}) != len(texts):
        raise CacheIntegrityError("Task texts must be nonempty and unique")
    if {entry.get("prompt") for entry in texts} != set(identity["text_payload_sha256"]):
        raise CacheIntegrityError("Cached task prompts differ from the identity-bound source T5 payloads")
    for entry in texts:
        if entry.get("source_payload_sha256") != identity["text_payload_sha256"][entry["prompt"]]:
            raise CacheIntegrityError("Cached task text source identity changed")
        path = _path(root, entry["path"])
        if file_sha256(path) != entry.get("sha256"):
            raise CacheIntegrityError(f"Task context payload SHA256 mismatch: {entry['path']}")
        _validate_text(load_file(str(path)))
    stop = 0
    for entry in manifest.get("shards", []):
        if entry.get("start") != stop or type(entry.get("stop")) is not int or entry["stop"] <= stop or entry["stop"] > identity["sample_count"]:
            raise CacheIntegrityError("Cache shard indices have gaps, duplicates, or overlap")
        path = _path(root, entry["path"])
        if file_sha256(path) != entry.get("sha256"):
            raise CacheIntegrityError(f"Latent cache payload SHA256 mismatch: {entry['path']}")
        _validate_shard(load_file(str(path)), entry, identity, texts)
        stop = entry["stop"]
    if manifest.get("completed_samples") != stop or (manifest["status"] == "complete" and stop != identity["sample_count"]):
        raise CacheIntegrityError("Cache completion must cover every original source index")
    expected_files = {"manifest.json", *[entry["path"] for entry in texts], *[entry["path"] for entry in manifest.get("shards", [])]}
    actual_files = {path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file()}
    if actual_files != expected_files:
        raise CacheIntegrityError("Cache contains untracked or missing payload files; refusing an ambiguous resume")
    return manifest


def _save_tensors(root, relative, tensors):
    target = root / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".tmp")
    save_file({key: value.detach().cpu().contiguous() for key, value in tensors.items()}, str(temporary))
    temporary.replace(target)
    return {"path": relative, "sha256": file_sha256(target)}


def write_window_cache(cache_root, identity, sample_getter, index_rows, encoder, texts, *, shard_size=128, resume=False,
                       max_new_samples=None, max_seconds=None, on_shard_commit=None):
    """Call encoder once per complete source window; persist atomic small shards.

    `sample_getter(index)` must return the successful original source index in
    sample_index. Encoder accepts [1,3,9,224,448] and returns [1,48,3,14,28].
    Resume validates all saved shards first. A failed encode before the shard
    write is re-encoded; committed windows are never encoded again. Files left
    by a process death during atomic commit require explicit recovery and are
    rejected as untracked rather than silently consumed or deleted.
    """
    _validate_identity(identity)
    if type(shard_size) is not int or shard_size < 1:
        raise ValueError("shard_size must be a positive integer")
    if max_new_samples is not None and (type(max_new_samples) is not int or max_new_samples < 1):
        raise ValueError("max_new_samples must be positive or None")
    if max_seconds is not None and (not math.isfinite(max_seconds) or max_seconds <= 0):
        raise ValueError("max_seconds must be finite and positive or None")
    started = time.monotonic()
    if [(row["sample_index"], row["episode_index"], row["frame_index"]) for row in index_rows] != _expected_rows(identity):
        raise CacheIntegrityError("Requested indices must cover every original episode/frame in order")
    if not texts or set(texts) != set(identity["text_payload_sha256"]):
        raise CacheIntegrityError("Task prompts differ from the identity-bound source T5 payloads")
    for payload in texts.values():
        _validate_text(payload)
    root = Path(cache_root).resolve()
    digest = cache_identity_sha256(identity)
    if root.exists():
        if not resume:
            raise FileExistsError(f"Cache exists; explicit --resume is required: {root}")
        manifest = validate_cache(root, digest, require_complete=False)
        if manifest["status"] == "complete":
            return manifest
        if manifest.get("shard_size") != shard_size:
            raise CacheIntegrityError("Resume shard size changed")
        for entry in manifest["texts"]:
            previous = load_file(str(root / entry["path"]))
            if any(not torch.equal(previous[key], texts[entry["prompt"]][key]) for key in previous):
                raise CacheIntegrityError("Resume task context values differ from the original frozen text payload")
    else:
        root.mkdir(parents=True)
        manifest = {"identity": identity, "identity_sha256": digest, "status": "partial", "completed_samples": 0,
                    "shard_size": shard_size, "texts": [], "shards": []}
        for number, (prompt, payload) in enumerate(sorted(texts.items())):
            _validate_text(payload)
            entry = _save_tensors(root, f"texts/task_{number:04d}.safetensors", payload)
            entry["prompt"] = prompt
            entry["source_payload_sha256"] = identity["text_payload_sha256"][prompt]
            manifest["texts"].append(entry)
        _atomic_json(root / "manifest.json", manifest)
    prompt_indices = {entry["prompt"]: number for number, entry in enumerate(manifest["texts"])}
    initial_samples = manifest["completed_samples"]
    limit = min(initial_samples + max_new_samples, identity["sample_count"]) if max_new_samples is not None else identity["sample_count"]
    previous_seconds = manifest.get("preprocessing_seconds", 0.0)

    def stopped(reason):
        manifest["stopped_reason"] = reason
        manifest["preprocessing_seconds"] = previous_seconds + time.monotonic() - started
        _atomic_json(root / "manifest.json", manifest)
        return validate_cache(root, digest, require_complete=False)

    for start in range(manifest["completed_samples"], identity["sample_count"], shard_size):
        if start >= limit:
            return stopped("max_new_samples")
        stop = min(start + shard_size, identity["sample_count"], limit)
        samples = []
        for index in range(start, stop):
            if max_seconds is not None and time.monotonic() - started >= max_seconds:
                if not samples:
                    return stopped("max_seconds")
                stop = index
                break
            sample = sample_getter(index)
            if int(sample.get("sample_index", -1)) != index:
                raise CacheIntegrityError("Raw reader substituted/retried another source index; no fallback is allowed")
            video = sample.get("video")
            if not isinstance(video, torch.Tensor) or tuple(video.shape) != VIDEO_SHAPE or not bool(torch.isfinite(video).all()):
                raise CacheIntegrityError("Encoder requires the original complete [3,9,224,448] RGB window")
            if sample.get("prompt") not in prompt_indices:
                raise CacheIntegrityError("Raw sample task has no identity-bound text context")
            numeric = {key: sample[key].detach().cpu().clone() for key in NUMERIC_SHAPES}
            numeric.update({key: sample[key].detach().cpu().clone() for key in OPTIONAL_SHAPES if key in sample})
            for key, tensor in numeric.items():
                shape = NUMERIC_SHAPES.get(key, OPTIONAL_SHAPES.get(key))
                dtype = (torch.bool if key.endswith("_is_pad") else torch.float64 if key in OPTIONAL_SHAPES
                         else getattr(torch, identity["numeric_dtype"]))
                if (tuple(tensor.shape) != shape or tensor.dtype != dtype
                        or (tensor.is_floating_point() and not bool(torch.isfinite(tensor).all()))):
                    raise CacheIntegrityError(f"Stock source numeric tensor shape/dtype/value changed: {key}")
            with torch.inference_mode():
                latent = encoder(video.unsqueeze(0))
            if not isinstance(latent, torch.Tensor) or tuple(latent.shape) != (1, *LATENT_SHAPE) or latent.dtype != torch.bfloat16:
                raise CacheIntegrityError("Frozen VAE encoder must return scaled BF16 [1,48,3,14,28] latents")
            numeric["video_latents"] = latent[0].detach().cpu().clone()
            numeric.update({key: torch.tensor(index_rows[index][key], dtype=torch.int64) for key in INDEX_KEYS if key != "text_index"})
            numeric["text_index"] = torch.tensor(prompt_indices[sample["prompt"]], dtype=torch.int64)
            samples.append(numeric)
        if any(set(sample) != set(samples[0]) for sample in samples):
            raise CacheIntegrityError("Cache optional numeric fields changed within a shard")
        payload = {key: torch.stack([sample[key] for sample in samples]) for key in samples[0]}
        entry = {"start": start, "stop": stop, "tensor_shapes": {key: list(value.shape) for key, value in payload.items()}}
        _validate_shard(payload, entry, identity, manifest["texts"])
        entry.update(_save_tensors(root, f"shards/window_{start:06d}_{stop:06d}.safetensors", payload))
        manifest["shards"].append(entry)
        manifest["completed_samples"] = stop
        manifest["status"] = "complete" if stop == identity["sample_count"] else "partial"
        manifest.pop("stopped_reason", None)
        manifest["preprocessing_seconds"] = previous_seconds + time.monotonic() - started
        _atomic_json(root / "manifest.json", manifest)
        if on_shard_commit is not None:
            on_shard_commit(manifest, entry)
    # A bounded stop inside the final planned shard can exhaust range() while
    # still leaving source indices. Persist its reason and validate as partial.
    if manifest["status"] != "complete":
        manifest["stopped_reason"] = "max_new_samples" if manifest["completed_samples"] >= limit else "max_seconds"
        _atomic_json(root / "manifest.json", manifest)
    return validate_cache(root, digest, require_complete=manifest["status"] == "complete")


class LatentCacheDataset(torch.utils.data.Dataset):
    """Feature-only reader: no raw dataset, image decoder, text encoder or VAE.

    Full payload verification happens at construction. Each shard is checked
    again on its first read, and a bounded LRU retains only numeric/features.
    Returned tensors are copies so callers cannot corrupt cached source data.
    """
    def __init__(self, cache_root, expected_identity_sha256=None, max_cached_shards=2, require_production=True):
        self.root = Path(cache_root).resolve()
        self.manifest = validate_cache(self.root, expected_identity_sha256)
        if require_production and self.manifest["identity"]["scope"] != "production":
            raise CacheIntegrityError("TEST-only latent fixture cannot be used for production training")
        if type(max_cached_shards) is not int or max_cached_shards < 1:
            raise ValueError("max_cached_shards must be positive")
        self.max_cached_shards = max_cached_shards
        self.identity_sha256 = self.manifest["identity_sha256"]
        self._shards = OrderedDict()
        self._texts = [load_file(str(self.root / entry["path"])) for entry in self.manifest["texts"]]
        self._lookup = [(number, index - entry["start"]) for number, entry in enumerate(self.manifest["shards"])
                        for index in range(entry["start"], entry["stop"])]

    def __len__(self):
        return self.manifest["identity"]["sample_count"]

    def __getitem__(self, index):
        if type(index) is not int or index < 0 or index >= len(self):
            raise IndexError(index)
        number, offset = self._lookup[index]
        if number not in self._shards:
            entry = self.manifest["shards"][number]
            path = _path(self.root, entry["path"])
            if file_sha256(path) != entry["sha256"]:
                raise CacheIntegrityError("Latent shard changed after cache validation")
            payload = load_file(str(path))
            _validate_shard(payload, entry, self.manifest["identity"], self.manifest["texts"])
            self._shards[number] = payload
            if len(self._shards) > self.max_cached_shards:
                self._shards.popitem(last=False)
        payload = self._shards[number]
        self._shards.move_to_end(number)
        sample = {key: value[offset].clone() for key, value in payload.items() if key != "text_index"}
        text_index = int(payload["text_index"][offset])
        sample.update({key: value.clone() for key, value in self._texts[text_index].items()})
        sample["prompt"] = self.manifest["texts"][text_index]["prompt"]
        sample["video_num_frames"] = 9
        return sample
