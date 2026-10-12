"""New identity-bound FP32 cache of independently processed recorded frames.

Only RGB preprocessing is deduplicated. A consumer must gather nine original
frames and invoke the unchanged VAE once for that complete anchor window.
"""
from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
import json
import os

import torch
from safetensors.torch import load_file, save_file

from .latent import CacheIntegrityError, cache_identity_sha256, file_sha256, _sha


PIXEL_SCHEMA = "fastwam-stock-fp32-frame-cache-v1"
FRAME_SHAPE = (3, 224, 448)
CAMERAS = ("cam_main", "cam_wrist")
PIXEL_PIPELINE = "PIL_ToTensor_times255_uint8_FP32_div255_Resize224_concat_horizontal_final_resize_crop_normalize"


def preprocess_rgb_frame(images):
    """Use the original operators/casts in their original order, batch one.

Embedded bytes and PIL RGB images are accepted through the original HF
formatter; no direct uint8 shortcut, early BF16 cast, or latent slicing.
"""
    from torchvision.transforms import Resize
    from fastwam.datasets.lerobot.lerobot.datasets.utils import hf_transform_to_torch
    from fastwam.datasets.lerobot.transforms.image import ToTensor
    from fastwam.datasets.dataset_utils import ResizeSmallestSideAspectPreserving, CenterCrop, Normalize

    if set(images) != set(CAMERAS) or torch.get_default_dtype() != torch.float32:
        raise CacheIntegrityError("Pixel preprocessing requires both ordered RGB cameras and stock default FP32")
    cameras = []
    for camera in CAMERAS:
        formatted = hf_transform_to_torch({camera: [images[camera]]})[camera][0]
        if not isinstance(formatted, torch.Tensor) or formatted.dtype != torch.float32 or tuple(formatted.shape) != (3, 480, 640):
            raise CacheIntegrityError(f"Original recorded camera shape/dtype changed: {camera}")
        original_uint8 = (formatted.unsqueeze(0) * 255).to(torch.uint8)
        cameras.append(Resize([224, 224])(ToTensor()(original_uint8)))
    video = torch.cat(cameras, dim=-1)
    dimensions = {"img_w": 448, "img_h": 224}
    video = ResizeSmallestSideAspectPreserving(dimensions)(video)
    video = CenterCrop(dimensions)(video)
    video = Normalize({"mean": .5, "std": .5})(video)
    if tuple(video.shape) != (1, *FRAME_SHAPE) or video.dtype != torch.float32 or not bool(torch.isfinite(video).all()):
        raise CacheIntegrityError("Stock processed RGB frame changed shape/dtype or became nonfinite")
    return video[0]


def make_pixel_identity(*, data_conversion_sha256, source_code_sha256, runtime_versions, episode_lengths,
                        preprocessing, test_fixture=False):
    identity = {"schema": PIXEL_SCHEMA, "scope": "TEST-only synthetic fixture" if test_fixture else "production",
                "data_conversion_sha256": data_conversion_sha256, "source_code_sha256": source_code_sha256,
                "runtime_versions": runtime_versions, "preprocessing": preprocessing,
                "episode_lengths": list(episode_lengths), "sample_count": sum(episode_lengths),
                "frame_shape": list(FRAME_SHAPE), "pixel_dtype": "float32", "torch_default_dtype": "torch.float32",
                "camera_order": list(CAMERAS), "pipeline": PIXEL_PIPELINE,
                "vae_input_policy": "gather_full9_original_frames_then_original_device_BF16_cast",
                "latent_slicing_allowed": False}
    validate_pixel_identity(identity)
    return identity


def validate_pixel_identity(identity):
    if (not isinstance(identity, dict) or identity.get("schema") != PIXEL_SCHEMA
            or identity.get("frame_shape") != list(FRAME_SHAPE) or identity.get("pixel_dtype") != "float32"
            or identity.get("torch_default_dtype") != "torch.float32" or identity.get("camera_order") != list(CAMERAS)
            or identity.get("pipeline") != PIXEL_PIPELINE or identity.get("latent_slicing_allowed") is not False
            or identity.get("vae_input_policy") != "gather_full9_original_frames_then_original_device_BF16_cast"):
        raise CacheIntegrityError("Pixel identity differs from the stock FP32 recorded-frame contract")
    if (not _sha(identity.get("data_conversion_sha256")) or not isinstance(identity.get("source_code_sha256"), dict)
            or not identity["source_code_sha256"]
            or not all(_sha(value) for value in identity["source_code_sha256"].values())
            or not isinstance(identity.get("runtime_versions"), dict) or not identity["runtime_versions"]
            or not isinstance(identity.get("preprocessing"), dict)):
        raise CacheIntegrityError("Pixel identity requires exact conversion/preprocessor provenance")
    lengths = identity.get("episode_lengths")
    if (not isinstance(lengths, list) or not lengths or any(type(length) is not int or length < 2 for length in lengths)
            or type(identity.get("sample_count")) is not int or sum(lengths) != identity["sample_count"]):
        raise CacheIntegrityError("Pixel episode/global frame inventory is inconsistent")
    if identity.get("scope") == "production":
        if len(lengths) != 50 or sum(lengths) != 34808:
            raise CacheIntegrityError("Production pixel cache must cover all50 episodes /34808 recorded frames")
    elif identity.get("scope") != "TEST-only synthetic fixture":
        raise CacheIntegrityError("Unknown pixel cache scope")


def pixel_rows(identity):
    rows, offset = [], 0
    for episode, length in enumerate(identity["episode_lengths"]):
        rows.extend((offset + frame, episode, frame) for frame in range(length))
        offset += length
    return rows


def _relative_file(root, relative):
    path = (root / relative).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise CacheIntegrityError("Pixel payload is absent or escapes its cache root")
    return path


def _validate_payload(payload, entry, expected):
    n = entry["stop"] - entry["start"]
    if set(payload) != {"pixel_frame", "sample_index", "episode_index", "frame_index"}:
        raise CacheIntegrityError("Pixel shard tensor inventory changed")
    pixels = payload["pixel_frame"]
    if tuple(pixels.shape) != (n, *FRAME_SHAPE) or pixels.dtype != torch.float32 or not bool(torch.isfinite(pixels).all()):
        raise CacheIntegrityError("Pixel shard dtype/shape/finite values changed")
    for column, key in enumerate(("sample_index", "episode_index", "frame_index")):
        tensor = payload[key]
        if tensor.dtype != torch.int64 or tuple(tensor.shape) != (n,) or tensor.tolist() != [row[column] for row in expected]:
            raise CacheIntegrityError("Pixel shard global source correspondence changed")
    if entry.get("tensor_shapes") != {key: list(value.shape) for key, value in payload.items()}:
        raise CacheIntegrityError("Pixel shard manifest tensor inventory changed")


def validate_pixel_cache(cache_root, expected_identity_sha256=None, *, require_complete=True):
    root = Path(cache_root).resolve()
    manifest = json.loads((root / "manifest.json").read_text())
    identity = manifest.get("identity")
    validate_pixel_identity(identity)
    digest = cache_identity_sha256(identity)
    if manifest.get("identity_sha256") != digest or (expected_identity_sha256 is not None and digest != expected_identity_sha256):
        raise CacheIntegrityError("Pixel cache/preprocessor identity changed; use a separate root")
    if manifest.get("status") not in ("partial", "complete") or (require_complete and manifest["status"] != "complete"):
        raise CacheIntegrityError("Incomplete pixel cache cannot supply production VAE windows")
    expected, stop, files = pixel_rows(identity), 0, {"manifest.json"}
    for entry in manifest.get("shards", []):
        if (entry.get("start") != stop or type(entry.get("stop")) is not int
                or entry["stop"] <= stop or entry["stop"] > len(expected)):
            raise CacheIntegrityError("Pixel shards have missing/duplicate/overlapping indices")
        path = _relative_file(root, entry["path"])
        if file_sha256(path) != entry.get("sha256"):
            raise CacheIntegrityError("Pixel immutable shard SHA changed")
        _validate_payload(load_file(str(path)), entry, expected[stop:entry["stop"]])
        files.add(entry["path"])
        stop = entry["stop"]
    if manifest.get("completed_frames") != stop or (manifest["status"] == "complete" and stop != len(expected)):
        raise CacheIntegrityError("Pixel completion does not cover its exact global frame inventory")
    if {path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file()} != files:
        raise CacheIntegrityError("Pixel cache contains untracked/missing files; refusing ambiguous resume")
    return manifest


def _write_manifest(root, manifest):
    temporary = root / f"manifest.json.tmp-{os.getpid()}"
    with temporary.open("x") as stream:
        stream.write(json.dumps(manifest, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
        stream.flush()
    temporary.replace(root / "manifest.json")


def write_pixel_cache(cache_root, identity, frame_getter, *, shard_size=128, resume=False, max_new_frames=None, on_shard_commit=None):
    """Acquire an external exclusive lock before validating or writing a root."""
    root = Path(cache_root).resolve()
    root.parent.mkdir(parents=True, exist_ok=True)
    lock = root.parent / (root.name + "_pixel_writer.lock")
    try:
        descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as error:
        raise CacheIntegrityError("Pixel writer lock already exists; active/stale writer requires explicit recovery") from error
    try:
        with os.fdopen(descriptor, "w") as stream:
            stream.write(json.dumps({"pid": os.getpid(), "cache_root": str(root)}) + "\n")
        return _write_pixel_cache(root, identity, frame_getter, shard_size=shard_size, resume=resume,
                                  max_new_frames=max_new_frames, on_shard_commit=on_shard_commit)
    finally:
        lock.unlink()


def _write_pixel_cache(cache_root, identity, frame_getter, *, shard_size=128, resume=False, max_new_frames=None, on_shard_commit=None):
    """Process each original recorded frame once, with immutable committed shards."""
    validate_pixel_identity(identity)
    if type(shard_size) is not int or shard_size < 1 or (max_new_frames is not None and (type(max_new_frames) is not int or max_new_frames < 1)):
        raise ValueError("Pixel shard/sample bounds must be positive integers")
    root = Path(cache_root).resolve()
    digest = cache_identity_sha256(identity)
    if root.exists():
        if not resume:
            raise FileExistsError(f"Pixel cache exists; explicit resume required: {root}")
        manifest = validate_pixel_cache(root, digest, require_complete=False)
        if manifest["status"] == "complete":
            return manifest
        if manifest.get("shard_size") != shard_size:
            raise CacheIntegrityError("Pixel resume shard size changed")
    else:
        root.mkdir(parents=True)
        manifest = {"identity": identity, "identity_sha256": digest, "status": "partial",
                    "completed_frames": 0, "shard_size": shard_size, "shards": []}
        _write_manifest(root, manifest)
    expected = pixel_rows(identity)
    stop_limit = min(len(expected), manifest["completed_frames"] + max_new_frames) if max_new_frames is not None else len(expected)
    for start in range(manifest["completed_frames"], stop_limit, shard_size):
        stop = min(start + shard_size, stop_limit)
        frames = []
        for index in range(start, stop):
            frame = frame_getter(index)
            if not isinstance(frame, torch.Tensor) or frame.dtype != torch.float32 or tuple(frame.shape) != FRAME_SHAPE or not bool(torch.isfinite(frame).all()):
                raise CacheIntegrityError("Processed recorded RGB frame changed dtype/shape/finiteness")
            frames.append(frame.detach().cpu().clone())
        payload = {"pixel_frame": torch.stack(frames)}
        payload.update({key: torch.tensor([row[column] for row in expected[start:stop]], dtype=torch.int64)
                        for column, key in enumerate(("sample_index", "episode_index", "frame_index"))})
        relative = f"shards/frame_{start:06d}_{stop:06d}.safetensors"
        entry = {"start": start, "stop": stop, "path": relative,
                 "tensor_shapes": {key: list(value.shape) for key, value in payload.items()}}
        _validate_payload(payload, entry, expected[start:stop])
        target = root / relative
        target.parent.mkdir(exist_ok=True)
        temporary = target.with_name(target.name + ".tmp")
        if target.exists() or temporary.exists():
            raise CacheIntegrityError("Refusing to replace an existing immutable pixel payload")
        save_file(payload, str(temporary))
        temporary.replace(target)
        entry["sha256"] = file_sha256(target)
        manifest["shards"].append(entry)
        manifest["completed_frames"] = stop
        manifest["status"] = "complete" if stop == len(expected) else "partial"
        _write_manifest(root, manifest)
        if on_shard_commit is not None:
            on_shard_commit(manifest, entry)
    return validate_pixel_cache(root, digest, require_complete=stop_limit == len(expected))


class PixelCacheDataset:
    def __init__(self, cache_root, expected_identity_sha256=None, *, require_complete=True, max_cached_shards=2):
        if type(max_cached_shards) is not int or max_cached_shards < 1:
            raise ValueError("Pixel shard cache must retain at least one shard")
        self.root = Path(cache_root).resolve()
        self.manifest = validate_pixel_cache(self.root, expected_identity_sha256, require_complete=require_complete)
        self.identity_sha256 = self.manifest["identity_sha256"]
        self.max_cached_shards, self._loaded = max_cached_shards, OrderedDict()
        self._lookup = [None] * self.manifest["completed_frames"]
        self._episode_ends = []
        offset = 0
        for length in self.manifest["identity"]["episode_lengths"]:
            offset += length
            self._episode_ends.extend([offset] * length)
        for number, entry in enumerate(self.manifest["shards"]):
            self._lookup[entry["start"]:entry["stop"]] = [number] * (entry["stop"] - entry["start"])

    def __len__(self):
        return self.manifest["completed_frames"]

    def __getitem__(self, index):
        if type(index) is not int or not 0 <= index < len(self):
            raise IndexError("Recorded pixel source index is outside its complete inventory")
        number = self._lookup[index]
        entry = self.manifest["shards"][number]
        if number not in self._loaded:
            path = _relative_file(self.root, entry["path"])
            if file_sha256(path) != entry["sha256"]:
                raise CacheIntegrityError("Pixel shard changed after initial verification")
            self._loaded[number] = load_file(str(path))
            if len(self._loaded) > self.max_cached_shards:
                self._loaded.popitem(last=False)
        self._loaded.move_to_end(number)
        return self._loaded[number]["pixel_frame"][index - entry["start"]].clone()

    def window(self, index):
        if type(index) is not int or not 0 <= index < len(self):
            raise IndexError("Pixel anchor is outside its complete inventory")
        end = self._episode_ends[index]
        # Preserve the stock NCHW-time stack then CTHW permute strides.
        return torch.stack([self[min(index + offset, end - 1)] for offset in range(0, 33, 4)]).permute(1, 0, 2, 3)
