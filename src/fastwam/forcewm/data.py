"""Force-only numeric adaptation of the immutable, verified 2-camera cache.

Stock latents/action/proprio/text are returned unchanged. History is [-31..0]
and wrench targets are [+1..+32], both clamped within the anchor's episode only
after padding is computed. Stock ``raw_wrench_future`` is [0..31]; it is retained
as ``raw_wrench_future_stock`` and the adapter's raw target uses the +1 alignment.

All six measured wrench channels are required and finite. Temporal padding has
explicit validity and zeroed normalized inputs/targets, never invented measured
zeroes. Future targets/GT validity belong only to the loss/evaluation path;
autoregressive inference must not consume either. Teacher forcing reads only
strictly shifted previous targets; target padding remains loss-only.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

from fastwam.cache.latent import (CacheIntegrityError, LatentCacheDataset,
                                 _path, cache_identity_sha256, file_sha256,
                                 validate_cache)
from fastwam.datasets.lerobot.utils import normalizer as stock_normalizer
from fastwam.datasets.lerobot.utils.normalizer import (SingleFieldLinearNormalizer,
                                                      load_dataset_stats_from_json)


SIDECAR_SCHEMA = "fastwam-forcewm-ar30-numeric-sidecar-v1"
HISTORY_SIZE = HORIZON = 32
NORMALIZER_SOURCE = "src/fastwam/datasets/lerobot/utils/normalizer.py"


def _require(condition, message):
    if not condition:
        raise CacheIntegrityError(message)


def _bytes_sha(value):
    return hashlib.sha256(value).hexdigest()


def _tensor_sha(value):
    header = json.dumps([str(value.dtype), list(value.shape)], separators=(",", ":")).encode()
    return _bytes_sha(header + value.contiguous().view(torch.uint8).numpy().tobytes())


def _bits_equal(left, right):
    return (left.dtype == right.dtype and left.shape == right.shape
            and torch.equal(left.contiguous().view(torch.uint8), right.contiguous().view(torch.uint8)))


def _source_hashes():
    return {"adapter": file_sha256(__file__), "stock_normalizer": file_sha256(stock_normalizer.__file__)}


def _fingerprint(path):
    stat = path.stat()
    return (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)


def _normalizers(stats_path, identity):
    _require(file_sha256(stats_path) == identity["dataset_stats_sha256"], "Force stats SHA256 differs from cache identity")
    stats = load_dataset_stats_from_json(stats_path)
    result = {}
    for name, width, field in (("state", 7, stats.get("state", {}).get("default")),
                               ("force", 6, stats.get("dualwm_force"))):
        _require(isinstance(field, dict), f"Stats lack complete {name} normalization")
        means, stds = field.get("global_mean"), field.get("global_std")
        _require(isinstance(means, torch.Tensor) and isinstance(stds, torch.Tensor)
                 and means.shape == stds.shape == (width,)
                 and means.dtype == stds.dtype == torch.float32
                 and bool(torch.isfinite(means).all()) and bool(torch.isfinite(stds).all())
                 and bool((stds > 0).all()), f"{name} stats require finite {width}D mean and positive std")
        result[name] = SingleFieldLinearNormalizer({"mean": means, "std": stds}, mode="z-score")
    if identity["scope"] == "production":
        _require(stats.get("num_episodes") == 50 and stats.get("num_transition") == 34808,
                 "Force stats must describe the same production 50 episodes / 34808 frames")
    pinned = identity["source_code_sha256"].get(NORMALIZER_SOURCE)
    _require((pinned is not None or identity["scope"] != "production")
             and (pinned is None or pinned == file_sha256(stock_normalizer.__file__)),
             "Stock normalization source differs from cache provenance")
    return result


def _derive(cache_root, dataset_stats_path, expected_identity_sha256, require_production):
    root, stats_path = Path(cache_root).resolve(), Path(dataset_stats_path).resolve()
    before = (root / "manifest.json").read_bytes()
    manifest = validate_cache(root, expected_identity_sha256)
    _require(not require_production or manifest["identity"]["scope"] == "production",
             "TEST-only cache cannot supply production Force inputs")
    identity = manifest["identity"]
    _require(identity["numeric_dtype"] == "float32", "Force adapter requires stock normalized FP32 cache")
    norms = _normalizers(stats_path, identity)
    states, wrenches = [], []
    fingerprints = {}
    for entry in manifest["shards"]:
        path = _path(root, entry["path"])
        fingerprint = _fingerprint(path)
        with safe_open(str(path), framework="pt", device="cpu") as payload:
            _require({"raw_current_state", "raw_current_wrench"}.issubset(payload.keys()),
                     "Force requires complete measured raw_current_state7 / raw_current_wrench6")
            state = payload.get_tensor("raw_current_state").clone()
            wrench = payload.get_tensor("raw_current_wrench").clone()
            proprio = payload.get_tensor("proprio")[:, 0].clone()
        n = entry["stop"] - entry["start"]
        _require(state.shape == (n, 7) and wrench.shape == (n, 6)
                 and state.dtype == wrench.dtype == torch.float64
                 and bool(torch.isfinite(state).all()) and bool(torch.isfinite(wrench).all()),
                 "Force requires finite, complete raw FP64 state7 and measured wrench6")
        _require(_bits_equal(norms["state"].forward(state.float()), proprio[:, :7])
                 and bool((proprio[:, 7] == 0).all()),
                 "Raw state / pinned stock normalization differs from cached proprio7+zero")
        _require(_fingerprint(path) == fingerprint and file_sha256(path) == entry["sha256"],
                 "Cache shard changed while deriving Force inputs")
        fingerprints[entry["path"]] = fingerprint
        states.append(state)
        wrenches.append(wrench)
    for entry in manifest["texts"]:
        path = _path(root, entry["path"])
        fingerprint = _fingerprint(path)
        _require(file_sha256(path) == entry["sha256"] and _fingerprint(path) == fingerprint,
                 "Cached text changed during Force derivation")
        fingerprints[entry["path"]] = fingerprint
    arrays = {"raw_state": torch.cat(states), "raw_wrench": torch.cat(wrenches)}
    _require((root / "manifest.json").read_bytes() == before, "Cache manifest bytes changed during Force derivation")
    _require(file_sha256(stats_path) == identity["dataset_stats_sha256"], "Force stats changed during derivation")
    binding = {"cache_identity_sha256": manifest["identity_sha256"],
               "cache_manifest_sha256": _bytes_sha(before),
               "cache_shards": [{"path": entry["path"], "sha256": entry["sha256"]} for entry in manifest["shards"]],
               "cache_texts": [{"path": entry["path"], "sha256": entry["sha256"]} for entry in manifest["texts"]],
               "dataset_stats_sha256": identity["dataset_stats_sha256"],
               "data_conversion_sha256": identity["data_conversion_sha256"],
               "source_code_sha256": _source_hashes(), "episode_lengths": identity["episode_lengths"],
               "sample_count": identity["sample_count"], "history_offsets": list(range(-31, 1)),
               "action_offsets": list(range(32)), "force_offsets": list(range(1, 33)),
               "normalization": "stock_FP32_zscore_scale_offset_eps1e-8_clip[-5,5]",
               "raw_array_inventory": {key: {"dtype": str(value.dtype), "shape": list(value.shape),
                                              "bits_sha256": _tensor_sha(value)} for key, value in arrays.items()},
               "force_channels": "all_six_finite_measured_channels_required"}
    return root, stats_path, manifest, binding, arrays, norms, fingerprints


def prepare_forcewm_sidecar(cache_root, dataset_stats_path, output_root,
                            expected_identity_sha256=None, *, require_production=True):
    """Write a fresh external numeric sidecar; never alter or relabel the cache.

    Existing output is rejected. A failed preparation leaves no complete
    manifest; there is no silent resume or overwrite of a partial sidecar.
    """
    root, stats_path = Path(cache_root).resolve(), Path(dataset_stats_path).resolve()
    output = Path(output_root).resolve()
    _require(not output.is_relative_to(root) and not root.is_relative_to(output)
             and not stats_path.is_relative_to(output), "Force sidecar must be external to immutable cache and stats")
    _, _, _, binding, arrays, _, _ = _derive(root, stats_path, expected_identity_sha256, require_production)
    output.mkdir(parents=True, exist_ok=False)
    numeric_path = output / "numeric.safetensors"
    save_file(arrays, str(numeric_path))
    receipt = {"schema": SIDECAR_SCHEMA, "status": "complete", "binding": binding,
               "numeric": {"path": "numeric.safetensors", "sha256": file_sha256(numeric_path)}}
    receipt["identity_sha256"] = cache_identity_sha256(receipt)
    (output / "manifest.json").write_text(json.dumps(receipt, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    return receipt


class ForceWMCachedDataset(torch.utils.data.Dataset):
    """Stock cached samples plus same-anchor, episode-local Force observations.

    Construction audits all cache hashes and derives reference raw arrays.
    Supplied sidecars are additionally compared bitwise against those arrays.
    Every read checks exact manifest/stats/source bindings and file fingerprints
    for all contributing history/future/anchor shards. ``audit_integrity`` repeats
    full hashes, including shards not yet consumed; it is suitable for launch and
    final receipts. Stock reader additionally hashes a shard on its first load.
    """
    def __init__(self, cache_root, dataset_stats_path, expected_identity_sha256=None,
                 sidecar_root=None, require_production=True, max_cached_shards=2):
        (self.root, self.stats_path, self.manifest, self.binding, arrays,
         self.normalizers, self._fingerprints) = _derive(
            cache_root, dataset_stats_path, expected_identity_sha256, require_production)
        self.identity_sha256 = self.manifest["identity_sha256"]
        self.sidecar_root = None if sidecar_root is None else Path(sidecar_root).resolve()
        self._sidecar_manifest_sha = self._sidecar_numeric_fingerprint = None
        if self.sidecar_root is not None:
            _require(not self.sidecar_root.is_relative_to(self.root)
                     and not self.root.is_relative_to(self.sidecar_root), "Force sidecar must be external to immutable cache")
            receipt_bytes = (self.sidecar_root / "manifest.json").read_bytes()
            receipt = json.loads(receipt_bytes)
            _require(set(receipt) == {"schema", "status", "binding", "numeric", "identity_sha256"}
                     and receipt["schema"] == SIDECAR_SCHEMA and receipt["status"] == "complete"
                     and receipt["binding"] == self.binding
                     and receipt["identity_sha256"] == cache_identity_sha256(
                         {key: value for key, value in receipt.items() if key != "identity_sha256"}),
                     "Force sidecar identity/bindings differ")
            _require(receipt["numeric"].get("path") == "numeric.safetensors", "Unknown Force sidecar payload")
            path = _path(self.sidecar_root, receipt["numeric"]["path"])
            _require(file_sha256(path) == receipt["numeric"].get("sha256"), "Force sidecar numeric SHA256 differs")
            saved = load_file(str(path), device="cpu")
            _require(set(saved) == set(arrays) and all(_bits_equal(saved[key], arrays[key]) for key in arrays),
                     "Force sidecar raw values differ from independently verified cache rows")
            self._sidecar_manifest_sha = _bytes_sha(receipt_bytes)
            self._sidecar_numeric_fingerprint = _fingerprint(path)
            self.sidecar_identity_sha256 = receipt["identity_sha256"]
        else:
            self.sidecar_identity_sha256 = cache_identity_sha256({"schema": SIDECAR_SCHEMA, "binding": self.binding})
        self._raw_state, self._raw_wrench = arrays["raw_state"], arrays["raw_wrench"]
        self._state = self.normalizers["state"].forward(self._raw_state.float())
        self._wrench = self.normalizers["force"].forward(self._raw_wrench.float())
        self.base = LatentCacheDataset(self.root, self.identity_sha256, max_cached_shards, require_production)
        self._rows, self._shard_for_index, offset = [], [], 0
        for episode, length in enumerate(self.manifest["identity"]["episode_lengths"]):
            self._rows.extend((episode, frame, offset, length) for frame in range(length))
            offset += length
        for entry in self.manifest["shards"]:
            self._shard_for_index.extend([entry["path"]] * (entry["stop"] - entry["start"]))
        self._check_bindings()

    def __len__(self):
        return len(self._rows)

    def _check_bindings(self):
        _require(file_sha256(self.root / "manifest.json") == self.binding["cache_manifest_sha256"],
                 "Immutable cache manifest bytes changed")
        _require(file_sha256(self.stats_path) == self.binding["dataset_stats_sha256"], "Pinned Force stats changed")
        _require(_source_hashes() == self.binding["source_code_sha256"], "Force adapter/normalizer source changed")
        for entry in self.manifest["texts"]:
            _require(_fingerprint(_path(self.root, entry["path"])) == self._fingerprints[entry["path"]],
                     "Immutable cached text changed")
        if self.sidecar_root is not None:
            _require(file_sha256(self.sidecar_root / "manifest.json") == self._sidecar_manifest_sha,
                     "Force sidecar manifest bytes changed")
            _require(_fingerprint(self.sidecar_root / "numeric.safetensors") == self._sidecar_numeric_fingerprint,
                     "Force sidecar numeric payload changed")

    def audit_integrity(self):
        """Rehash every immutable cache/sidecar payload without GPU execution."""
        self._check_bindings()
        validate_cache(self.root, self.identity_sha256)
        for entry in self.manifest["shards"]:
            _require(_fingerprint(_path(self.root, entry["path"])) == self._fingerprints[entry["path"]],
                     "Immutable cache shard changed")
        if self.sidecar_root is not None:
            receipt = json.loads((self.sidecar_root / "manifest.json").read_text())
            _require(file_sha256(self.sidecar_root / "numeric.safetensors") == receipt["numeric"]["sha256"],
                     "Force sidecar numeric SHA256 differs")
        return {"cache_identity_sha256": self.identity_sha256,
                "force_sidecar_identity_sha256": self.sidecar_identity_sha256, "verified": True}

    def __getitem__(self, index):
        if type(index) is not int or not 0 <= index < len(self):
            raise IndexError(index)
        self._check_bindings()
        episode, frame, start, length = self._rows[index]
        history_frames = frame + torch.arange(-31, 1)
        future_frames = frame + torch.arange(1, 33)
        history_pad = (history_frames < 0) | (history_frames >= length)
        future_pad = (future_frames < 0) | (future_frames >= length)
        history_indices = start + history_frames.clamp(0, length - 1)
        future_indices = start + future_frames.clamp(0, length - 1)
        source_indices = torch.cat((history_indices, future_indices, torch.tensor([index])))
        for relative in {self._shard_for_index[int(i)] for i in source_indices}:
            _require(_fingerprint(_path(self.root, relative)) == self._fingerprints[relative],
                     "Immutable cache shard changed after Force derivation")
        sample = self.base[index]
        _require(int(sample["sample_index"]) == index and int(sample["episode_index"]) == episode
                 and int(sample["frame_index"]) == frame, "Stock cached reader changed the Force anchor")
        history_state = self._state[history_indices].clone().masked_fill(history_pad[:, None], 0)
        history_wrench = self._wrench[history_indices].clone().masked_fill(history_pad[:, None], 0)
        target = self._wrench[future_indices].clone().masked_fill(future_pad[:, None], 0)
        if "raw_wrench_future" in sample:
            sample["raw_wrench_future_stock"] = sample["raw_wrench_future"]
        raw_future = self._raw_wrench[future_indices].clone()
        sample.update(history=torch.cat((history_state, history_wrench), dim=-1),
                      history_proprio=history_state, wrench_history=history_wrench,
                      history_is_pad=history_pad,
                      history_channel_valid=(~history_pad[:, None]).expand(-1, 13).clone(),
                      history_wrench_valid=(~history_pad[:, None]).expand(-1, 6).clone(),
                      wrench_future=target, force_is_pad=future_pad,
                      wrench_valid=(~future_pad[:, None]).expand(-1, 6).clone(),
                      raw_history_proprio=self._raw_state[history_indices].clone(),
                      raw_wrench_history=self._raw_wrench[history_indices].clone(),
                      raw_wrench_future=raw_future, raw_wrench_future64=raw_future.clone())
        return sample
