"""Exhaustive CPU pixel/anchor proof using independently invoked stock operators.

This proves RGB preprocessing and window assembly, never hardware VAE parity.
The reference does not call the candidate frame processor or use cached pixels.
"""
from __future__ import annotations

import importlib.metadata
import json
from pathlib import Path
import time

import torch

from .latent import CacheIntegrityError, file_sha256, _sha
from .pixels import CAMERAS, validate_pixel_cache
from .deduplicated_source import ROOT, build_deduplicated_source, verify_conversion


AUDIT_SCHEMA = "fastwam-exhaustive-stock-pixel-cpu-audit-v1"
AUDIT_FILES = ("src/fastwam/cache/pixel_audit.py", "scripts/forcevla/prepare_pixel_cache.py")


def _external_receipt(path, pixel_root):
    destination, root = Path(path).resolve(), Path(pixel_root).resolve()
    if destination.is_relative_to(root):
        raise CacheIntegrityError("Pixel proof receipt must remain outside immutable cache payloads")
    return destination


def _stock_reference_batches(raw, episode, *, batch_size=33):
    """Stock Base getter cast + actual processor.preprocess + final operators."""
    import pyarrow.parquet as pq
    from hydra.utils import instantiate
    from fastwam.datasets.lerobot.base_lerobot_dataset import BaseLerobotDataset
    from fastwam.datasets.lerobot.lerobot.datasets.utils import hf_transform_to_torch
    from fastwam.datasets.lerobot.utils.normalizer import load_dataset_stats_from_json
    from fastwam.datasets.dataset_utils import ResizeSmallestSideAspectPreserving, CenterCrop, Normalize

    processor = instantiate(raw.processor)
    processor.eval()
    processor.num_image_steps = batch_size
    processor.set_normalizer_from_stats(load_dataset_stats_from_json(raw.pretrained_norm_stats))
    names = [f"observation.images.{camera}" for camera in CAMERAS]
    path = Path(raw.dataset_dirs[0]) / f"data/chunk-{episode // 1000:03d}/episode_{episode:06d}.parquet"
    for batch in pq.ParquetFile(path).iter_batches(batch_size=batch_size, columns=[*names, "index", "episode_index", "frame_index"]):
        count = len(batch)
        values = {key: batch.column(batch.schema.get_field_index(key)).to_pylist() for key in batch.schema.names}
        padded = {key: values[key] + [values[key][-1]] * (batch_size - count) for key in names}
        formatted = hf_transform_to_torch(padded)
        images = {}
        for camera, key in zip(CAMERAS, names):
            formatted[key] = torch.stack(formatted[key])
            if formatted[key].dtype != torch.float32 or tuple(formatted[key].shape) != (batch_size, 3, 480, 640):
                raise CacheIntegrityError("Stock reference raw recorded camera shape/dtype changed")
            images[camera] = BaseLerobotDataset._get_image(None, {"key": camera, "lerobot_key": key,
                                                               "raw_shape": [3, 480, 640]}, formatted)
        processed = processor.preprocess({"task": "CPU-only stock image proof", "idx": 0, "images": images,
            "action": {"default": torch.zeros(32, 7)}, "state": {"default": torch.zeros(33, 7)},
            "image_is_pad": torch.zeros(batch_size, dtype=torch.bool), "action_is_pad": torch.zeros(32, dtype=torch.bool),
            "state_is_pad": torch.zeros(33, dtype=torch.bool)})
        packed = torch.cat([processed["pixel_values"][0], processed["pixel_values"][1]], dim=-1)
        dimensions = {"img_w": 448, "img_h": 224}
        packed = ResizeSmallestSideAspectPreserving(dimensions)(packed)
        packed = CenterCrop(dimensions)(packed)
        packed = Normalize({"mean": .5, "std": .5})(packed)
        yield values, packed[:count]


def _numeric_oracle(source, index, end):
    reference = source.numeric_reference
    future = torch.tensor([min(index + step, end - 1) for step in range(32)])
    return {"action": reference["action"][future], "proprio": reference["proprio"][future],
            "raw_action": reference["raw_action"][future], "raw_wrench_future": reference["raw_wrench"][future],
            "raw_current_state": reference["raw_state"][index], "raw_current_wrench": reference["raw_wrench"][index],
            "image_is_pad": torch.tensor([index + step >= end for step in range(0, 33, 4)]),
            "action_is_pad": torch.tensor([index + step >= end for step in range(32)]),
            "proprio_is_pad": torch.tensor([index + step >= end for step in range(33)])}


def _stock_prompt_oracle(raw):
    from hydra.utils import instantiate
    from fastwam.datasets.lerobot.robot_video_dataset import DEFAULT_PROMPT
    processor = instantiate(raw.processor)
    prompts = {}
    for line in (Path(raw.dataset_dirs[0]) / "meta/tasks.jsonl").read_text().splitlines():
        if line.strip():
            task = json.loads(line)
            index = task["task_index"]
            if type(index) is not int or index in prompts:
                raise CacheIntegrityError("Independent stock task inventory changed")
            instruction = processor.augment_instruction({"task": task["task"]})
            prompts[index] = DEFAULT_PROMPT.format(task=instruction)
    return prompts


def audit_pixel_cache(raw, pixel_cache_root, receipt_path, *, test_fixture=False, on_episode=None):
    """Prove every unique frame and every anchor with bounded stock lookahead."""
    destination = _external_receipt(receipt_path, pixel_cache_root)
    if destination.exists():
        raise FileExistsError("Use a fresh external pixel proof receipt path")
    started = time.monotonic()
    audit_hashes = {relative: file_sha256(ROOT / relative) for relative in AUDIT_FILES}
    source = build_deduplicated_source(raw, pixel_cache_root, test_fixture=test_fixture)
    stock_prompts = _stock_prompt_oracle(raw)
    if stock_prompts != source.task_prompts:
        raise CacheIntegrityError("Independent stock task instruction/prompt transformation differs")
    initial_manifest_sha = file_sha256(Path(pixel_cache_root) / "manifest.json")
    frames_checked, anchors_checked, start = 0, 0, 0
    for episode, length in enumerate(source.episode_lengths):
        end, anchor, pending, loaded = start + length, start, {}, start
        for values, reference_frames in _stock_reference_batches(raw, episode):
            n = len(reference_frames)
            if (values["index"] != list(range(loaded, loaded + n)) or values["episode_index"] != [episode] * n
                    or values["frame_index"] != list(range(loaded - start, loaded - start + n))):
                raise CacheIntegrityError("Stock RGB reference row correspondence changed")
            for local, frame in enumerate(reference_frames):
                index = loaded + local
                candidate = source.pixels[index]
                if (candidate.dtype != frame.dtype or tuple(candidate.shape) != tuple(frame.shape)
                        or not torch.equal(candidate.view(torch.uint8), frame.view(torch.uint8))
                        or not torch.equal(candidate.to(torch.bfloat16), frame.to(torch.bfloat16))):
                    raise CacheIntegrityError(f"Stock unique recorded frame FP32/BF16 mismatch at {index}")
                pending[index] = frame.clone()
                frames_checked += 1
            loaded += n
            while anchor < end and (anchor + 32 < loaded or loaded == end):
                indices = [min(anchor + step, end - 1) for step in range(0, 33, 4)]
                expected_video = torch.stack([pending[index] for index in indices]).permute(1, 0, 2, 3)
                candidate = source(anchor)
                if (candidate["sample_index"] != anchor or candidate["video"].stride() != expected_video.stride()
                        or not torch.equal(candidate["video"].view(torch.uint8), expected_video.view(torch.uint8))):
                    raise CacheIntegrityError(f"Stock full nine-frame anchor mismatch at {anchor}")
                for key, expected in _numeric_oracle(source, anchor, end).items():
                    actual = candidate[key]
                    if actual.dtype != expected.dtype or actual.shape != expected.shape or not torch.equal(actual.view(torch.uint8), expected.view(torch.uint8)):
                        raise CacheIntegrityError(f"Stock independent numeric/mask oracle mismatch for {key} at {anchor}")
                task = int(source.numeric_reference["task_index"][anchor])
                if candidate["prompt"] != stock_prompts[task]:
                    raise CacheIntegrityError(f"Stock task prompt association mismatch at {anchor}")
                pending.pop(anchor, None)
                anchor += 1
                anchors_checked += 1
        if loaded != end or anchor != end:
            raise CacheIntegrityError("Exhaustive stock pixel/anchor proof omitted source rows")
        if on_episode is not None:
            on_episode({"episode_index": episode, "unique_frames_checked": frames_checked,
                        "anchors_checked": anchors_checked, "elapsed_seconds": time.monotonic() - started})
        start = end
    verify_conversion(raw.dataset_dirs[0], test_fixture=test_fixture)
    if (file_sha256(Path(raw.dataset_dirs[0]) / "conversion_manifest.json") != source.source_provenance["conversion_manifest_sha256"]
            or file_sha256(raw.pretrained_norm_stats) != source.source_provenance["dataset_stats_sha256"]):
        raise CacheIntegrityError("Original source/stats changed during exhaustive stock proof")
    validate_pixel_cache(pixel_cache_root, source.pixels.identity_sha256)
    if file_sha256(Path(pixel_cache_root) / "manifest.json") != initial_manifest_sha:
        raise CacheIntegrityError("Pixel manifest changed during its exhaustive CPU proof")
    files = {**source.pixel_manifest["identity"]["source_code_sha256"], **audit_hashes}
    if any(file_sha256(ROOT / relative) != digest for relative, digest in files.items()):
        raise CacheIntegrityError("Candidate or stock code changed during CPU proof")
    receipt = {"schema": AUDIT_SCHEMA, "status": "complete", "scope": source.pixel_manifest["identity"]["scope"],
        "pixel_identity_sha256": source.pixels.identity_sha256, "pixel_manifest_sha256": initial_manifest_sha,
        "sample_count": len(source), "unique_frames_checked": frames_checked, "anchors_checked": anchors_checked,
        "episode_count": len(source.episode_lengths), "episode_lengths": source.episode_lengths,
        "source_provenance": source.source_provenance, "source_code_sha256": files,
        "runtime_versions": source.pixel_manifest["identity"]["runtime_versions"],
        "all_unique_frames_fp32_bits_exact": True, "all_unique_frames_bf16_bits_exact": True,
        "all_full9_anchor_bits_and_strides_exact": True, "all_numeric_raw_labels_and_masks_exact": True,
        "all_task_prompt_associations_exact": True, "mismatch_count": 0,
        "reference_method": "actual stock HF/Base casts and processor.preprocess in batch33; original final operators; bounded independent anchor gather",
        "hardware_vae_parity_checked": False, "training_started": False, "cuda_queried": False,
        "elapsed_seconds": time.monotonic() - started}
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x") as stream:
        stream.write(json.dumps(receipt, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    return receipt


def validate_pixel_audit_receipt(receipt_path, pixel_cache_root, expected_pixel_identity_sha256):
    """Fresh integrity/code/runtime binding gate before any VAE construction."""
    path = _external_receipt(receipt_path, pixel_cache_root)
    receipt = json.loads(path.read_text())
    manifest = validate_pixel_cache(pixel_cache_root, expected_pixel_identity_sha256)
    count, identity = manifest["identity"]["sample_count"], manifest["identity"]
    gates = ("all_unique_frames_fp32_bits_exact", "all_unique_frames_bf16_bits_exact",
             "all_full9_anchor_bits_and_strides_exact", "all_numeric_raw_labels_and_masks_exact",
             "all_task_prompt_associations_exact")
    if (receipt.get("schema") != AUDIT_SCHEMA or receipt.get("status") != "complete"
            or receipt.get("scope") != identity["scope"] or receipt.get("pixel_identity_sha256") != expected_pixel_identity_sha256
            or receipt.get("pixel_manifest_sha256") != file_sha256(Path(pixel_cache_root) / "manifest.json")
            or any(receipt.get(key) is not True for key in gates)
            or receipt.get("mismatch_count") != 0 or any(receipt.get(key) != count for key in ("sample_count", "unique_frames_checked", "anchors_checked"))
            or receipt.get("episode_count") != len(identity["episode_lengths"]) or receipt.get("episode_lengths") != identity["episode_lengths"]
            or any(receipt.get(key) is not False for key in ("hardware_vae_parity_checked", "training_started", "cuda_queried"))):
        raise CacheIntegrityError("Pixel CPU proof is missing/incomplete or belongs to different payloads")
    expected_files = {**identity["source_code_sha256"], **{relative: file_sha256(ROOT / relative) for relative in AUDIT_FILES}}
    if receipt.get("source_code_sha256") != expected_files or any(file_sha256(ROOT / relative) != digest for relative, digest in expected_files.items()):
        raise CacheIntegrityError("Pixel CPU proof stock/candidate source code changed")
    versions = {name: importlib.metadata.version(name) for name in identity["runtime_versions"]}
    if receipt.get("runtime_versions") != identity["runtime_versions"] or versions != identity["runtime_versions"]:
        raise CacheIntegrityError("Pixel CPU proof runtime versions changed")
    provenance = receipt.get("source_provenance", {})
    if (provenance.get("schema") != "deduplicated-stock-source-v2"
            or not _sha(provenance.get("numeric_reference_sha256")) or not _sha(provenance.get("dataset_stats_sha256"))
            or provenance.get("pixel_identity_sha256") != expected_pixel_identity_sha256
            or provenance.get("pixel_manifest_sha256") != receipt["pixel_manifest_sha256"]
            or provenance.get("conversion_manifest_sha256") != identity["data_conversion_sha256"]
            or provenance.get("source_code_sha256") != identity["source_code_sha256"]):
        raise CacheIntegrityError("Pixel CPU proof source provenance differs from its recorded-frame identity")
    return receipt
