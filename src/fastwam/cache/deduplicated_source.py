"""New stock-equivalent source: frame pixels once, complete windows per anchor."""
from __future__ import annotations

import hashlib
import importlib.metadata
import importlib.util
import json
from pathlib import Path

import torch

from .latent import CacheIntegrityError, file_sha256
from .pixels import PixelCacheDataset, CAMERAS, make_pixel_identity, preprocess_rgb_frame, pixel_rows


ROOT = Path(__file__).resolve().parents[3]
SOURCE_FILES = ("src/fastwam/cache/pixels.py", "src/fastwam/cache/deduplicated_source.py",
                "src/fastwam/cache/latent.py", "scripts/forcevla/prepare_forcevla.py",
                "src/fastwam/datasets/lerobot/lerobot/datasets/utils.py",
                "src/fastwam/datasets/lerobot/base_lerobot_dataset.py",
                "src/fastwam/datasets/lerobot/processors/fastwam_processor.py",
                "src/fastwam/datasets/lerobot/processors/base_processor.py",
                "src/fastwam/datasets/lerobot/transforms/image.py",
                "src/fastwam/datasets/dataset_utils.py", "src/fastwam/datasets/lerobot/robot_video_dataset.py",
                "src/fastwam/datasets/lerobot/utils/normalizer.py",
                "src/fastwam/datasets/lerobot/transforms/action_state_merger.py")


def validate_raw_config(raw):
    from omegaconf import OmegaConf

    config = OmegaConf.to_container(raw, resolve=True)
    processor = config["processor"]
    transforms = [{"_target_": "fastwam.datasets.lerobot.transforms.image.ToTensor"},
                  {"_target_": "torchvision.transforms.Resize", "size": [224, 224]}]
    if (config.get("_target_") != "fastwam.datasets.lerobot.robot_video_dataset.RobotVideoDataset"
            or processor.get("_target_") != "fastwam.datasets.lerobot.processors.fastwam_processor.FastWAMProcessor"
            or processor.get("shape_meta") != config["shape_meta"]
            or config.get("context_len") != 128
            or config.get("skip_padding_as_possible") is not False or config.get("val_set_proportion") != 0
            or config.get("global_sample_stride") != 1 or config.get("num_frames") != 33
            or config.get("action_video_freq_ratio") != 4 or config.get("video_size") != [224, 448]
            or config.get("concat_multi_camera") != "horizontal" or len(config.get("dataset_dirs", [])) != 1
            or config.get("override_instruction") is not None or config.get("camera_key") is not None
            or [image["key"] for image in config["shape_meta"]["images"]] != list(CAMERAS)
            or any(image["shape"] != [3, 224, 224] or image["raw_shape"] != [3, 480, 640]
                   for image in config["shape_meta"]["images"])
            or config["shape_meta"].get("action") != [{"key": "default", "raw_shape": 7, "shape": 7}]
            or config["shape_meta"].get("state") != [{"key": "default", "raw_shape": 7, "shape": 7}]
            or processor.get("action_output_dim") != 7 or processor.get("proprio_output_dim") != 8
            or processor.get("num_obs_steps") != 33 or processor.get("num_image_steps", 33) != 33
            or processor.get("num_output_cameras") != 2 or processor.get("action_state_transforms") is not None
            or processor.get("action_state_merger") != {"_target_": "fastwam.datasets.lerobot.transforms.action_state_merger.ConcatLeftAlign", "state_target_dim": 8}
            or processor.get("norm_default_mode") != "z-score" or processor.get("norm_exception_mode") is not None
            or processor.get("use_stepwise_action_norm") is not False
            or processor.get("train_transforms") != transforms or processor.get("val_transforms") != transforms
            or processor.get("delta_action_dim_mask") != {"default": [False] * 7}
            or processor.get("drop_high_level_prob", 1.) != 1. or processor.get("use_zh_instruction", False)
            or torch.get_default_dtype() != torch.float32):
        raise CacheIntegrityError("Deduplicated source requires the exact original deterministic LIBERO recipe/default FP32")
    return config


def verify_conversion(dataset_root, *, test_fixture=False):
    root = Path(dataset_root).resolve()
    conversion = json.loads((root / "conversion_manifest.json").read_text())
    if not test_fixture:
        spec = importlib.util.spec_from_file_location("pixel_source_converter", ROOT / "scripts/forcevla/prepare_forcevla.py")
        converter = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(converter)
        return converter.validate_converted_libero2cam(root)
    if conversion.get("test_fixture") is not True or conversion.get("usage") != "TEST-only synthetic fixture":
        raise CacheIntegrityError("Synthetic pixel tests require an explicitly marked converted fixture")
    for entry in (*conversion["metadata_files"], *conversion["output_files"]):
        path = (root / entry["path"]).resolve()
        if not path.is_relative_to(root) or file_sha256(path) != entry["sha256"]:
            raise CacheIntegrityError("Converted fixture source payload differs from its manifest")
    return conversion


def pixel_identity_from_source(raw, *, test_fixture=False):
    config = validate_raw_config(raw)
    root = Path(config["dataset_dirs"][0]).resolve()
    verify_conversion(root, test_fixture=test_fixture)
    lengths = [json.loads(line)["length"] for line in (root / "meta/episodes.jsonl").read_text().splitlines() if line.strip()]
    preprocessing = {key: value for key, value in config.items()
                     if key not in ("dataset_dirs", "pretrained_norm_stats", "text_embedding_cache_dir", "use_text_embed_cache")}
    return make_pixel_identity(data_conversion_sha256=file_sha256(root / "conversion_manifest.json"),
        source_code_sha256={path: file_sha256(ROOT / path) for path in SOURCE_FILES},
        runtime_versions={name: importlib.metadata.version(name) for name in ("torch", "torchvision", "pillow", "datasets", "pyarrow", "safetensors")},
        episode_lengths=lengths, preprocessing=preprocessing, test_fixture=test_fixture)


class RecordedFrameSource:
    """Read one episode's embedded RGB columns at a time, without HF startup."""
    def __init__(self, dataset_root, identity):
        self.root, self.identity = Path(dataset_root).resolve(), identity
        self._rows = pixel_rows(identity)
        self._episode, self._columns, self._offset = None, None, None

    def __call__(self, index):
        import pyarrow.parquet as pq

        if type(index) is not int or not 0 <= index < len(self._rows):
            raise IndexError("Recorded source frame index is absent")
        _, episode, frame = self._rows[index]
        if episode != self._episode:
            path = self.root / f"data/chunk-{episode // 1000:03d}/episode_{episode:06d}.parquet"
            names = [f"observation.images.{camera}" for camera in CAMERAS]
            table = pq.read_table(path, columns=[*names, "index", "episode_index", "frame_index"])
            columns = {key: table[key].to_pylist() for key in table.column_names}
            length = self.identity["episode_lengths"][episode]
            offset = sum(self.identity["episode_lengths"][:episode])
            if (len(table) != length or columns["index"] != list(range(offset, offset + length))
                    or columns["frame_index"] != list(range(length)) or columns["episode_index"] != [episode] * length):
                raise CacheIntegrityError("RGB parquet source indices changed")
            self._episode, self._columns, self._offset = episode, columns, offset
        images = {camera: self._columns[f"observation.images.{camera}"][frame] for camera in CAMERAS}
        if any(not isinstance(image, dict) or not isinstance(image.get("bytes"), bytes) or image.get("path") is not None for image in images.values()):
            raise CacheIntegrityError("Recorded RGB must retain original embedded image bytes; no fallback paths")
        return preprocess_rgb_frame(images)


def numeric_reference_from_source(raw):
    """Original HF FP32 formatter + original normalization/merger once per row."""
    import pyarrow.parquet as pq
    from omegaconf import OmegaConf
    from fastwam.datasets.lerobot.lerobot.datasets.utils import hf_transform_to_torch
    from fastwam.datasets.lerobot.utils.normalizer import LinearNormalizer, load_dataset_stats_from_json
    from fastwam.datasets.lerobot.transforms.action_state_merger import ConcatLeftAlign

    config = validate_raw_config(raw)
    shape_meta = OmegaConf.to_container(raw.shape_meta, resolve=True)
    normalizer = LinearNormalizer(shape_meta=shape_meta, use_stepwise_action_norm=False,
        default_mode="z-score", exception_mode=None, stats=load_dataset_stats_from_json(raw.pretrained_norm_stats))
    merger = ConcatLeftAlign(state_target_dim=8)
    merger.set_shape_meta(shape_meta)
    parts = {key: [] for key in ("action", "proprio", "raw_action", "raw_state", "raw_wrench", "task_index")}
    rows, lengths = [], []
    for episode, path in enumerate(sorted(Path(config["dataset_dirs"][0]).glob("data/chunk-*/episode_*.parquet"))):
        table = pq.read_table(path, columns=["action", "observation.state", "observation.wrench", "index", "episode_index", "frame_index", "task_index"])
        values = {key: table[key].to_pylist() for key in table.column_names}
        n, offset = len(table), sum(lengths)
        if (values["index"] != list(range(offset, offset + n)) or values["episode_index"] != [episode] * n
                or values["frame_index"] != list(range(n))):
            raise CacheIntegrityError("Numeric parquet source indices changed")
        formatted = hf_transform_to_torch({key: values[key] for key in ("action", "observation.state")})
        result = merger.forward(normalizer.forward({"action": {"default": torch.stack(formatted["action"])},
                                                   "state": {"default": torch.stack(formatted["observation.state"])}}))
        parts["action"].append(result["action"])
        parts["proprio"].append(result["state"])
        for target, source in (("raw_action", "action"), ("raw_state", "observation.state"), ("raw_wrench", "observation.wrench")):
            parts[target].append(torch.tensor(values[source], dtype=torch.float64))
        parts["task_index"].append(torch.tensor(values["task_index"], dtype=torch.int64))
        rows.extend({"sample_index": offset + frame, "episode_index": episode, "frame_index": frame,
                     "task_index": int(values["task_index"][frame])} for frame in range(n))
        lengths.append(n)
    if not lengths:
        raise CacheIntegrityError("Numeric source inventory is empty")
    return {key: torch.cat(values) for key, values in parts.items()}, rows, lengths


def tensor_inventory_sha256(reference):
    digest = hashlib.sha256()
    for key, value in sorted(reference.items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(json.dumps([key, str(tensor.dtype), list(tensor.shape)], separators=(",", ":")).encode())
        digest.update(tensor.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


class DeduplicatedSource:
    def __init__(self, pixels, numeric_reference, index_rows, episode_lengths, task_prompts, *, source_provenance=None):
        self.pixels, self.pixel_manifest = pixels, pixels.manifest
        self.numeric_reference, self.index_rows = numeric_reference, index_rows
        self.episode_lengths, self.task_prompts = list(episode_lengths), dict(task_prompts)
        self.source_provenance = source_provenance or {}
        self._ends = []
        offset = 0
        for length in self.episode_lengths:
            offset += length
            self._ends.extend([offset] * length)
        if len(pixels) != len(index_rows) or len(pixels) != offset:
            raise CacheIntegrityError("Pixel/numeric anchor inventories differ")
        expected_shapes = {"action": (offset, 7), "proprio": (offset, 8), "raw_action": (offset, 7),
                           "raw_state": (offset, 7), "raw_wrench": (offset, 6), "task_index": (offset,)}
        if set(numeric_reference) != set(expected_shapes):
            raise CacheIntegrityError("Independent numeric reference inventory changed")
        for key, shape in expected_shapes.items():
            value = numeric_reference[key]
            dtype = torch.int64 if key == "task_index" else torch.float64 if key.startswith("raw_") else torch.float32
            if (not isinstance(value, torch.Tensor) or value.device.type != "cpu" or value.dtype != dtype
                    or tuple(value.shape) != shape or not bool(torch.isfinite(value).all())):
                raise CacheIntegrityError(f"Independent numeric reference {key} shape/dtype/finite values changed")
        expected_rows = pixel_rows(pixels.manifest["identity"])
        if self.episode_lengths != pixels.manifest["identity"]["episode_lengths"]:
            raise CacheIntegrityError("Pixel/numeric episode lengths differ")
        for index, (row, expected) in enumerate(zip(index_rows, expected_rows)):
            if (any(type(row.get(key)) is not int for key in ("sample_index", "episode_index", "frame_index", "task_index"))
                    or tuple(row[key] for key in ("sample_index", "episode_index", "frame_index")) != expected
                    or row["task_index"] != int(numeric_reference["task_index"][index])):
                raise CacheIntegrityError("Numeric reference/global source correspondence changed")
        if (any(type(key) is not int or not isinstance(value, str) or not value for key, value in self.task_prompts.items())
                or set(numeric_reference["task_index"].tolist()) - set(self.task_prompts)):
            raise CacheIntegrityError("Original source task prompt inventory is incomplete or invalid")

    def __len__(self):
        return len(self.index_rows)

    def expected_numeric(self, index):
        if type(index) is not int or not 0 <= index < len(self):
            raise IndexError("Original anchor is absent")
        end, reference = self._ends[index], self.numeric_reference
        indices = torch.minimum(index + torch.arange(32), torch.tensor(end - 1))
        return {"action": reference["action"][indices].clone(), "proprio": reference["proprio"][indices].clone(),
                "raw_action": reference["raw_action"][indices].clone(), "raw_wrench_future": reference["raw_wrench"][indices].clone(),
                "raw_current_state": reference["raw_state"][index].clone(), "raw_current_wrench": reference["raw_wrench"][index].clone(),
                "image_is_pad": index + torch.arange(0, 33, 4) >= end,
                "action_is_pad": index + torch.arange(32) >= end, "proprio_is_pad": index + torch.arange(33) >= end}

    def __call__(self, index):
        import numpy as np

        values = self.expected_numeric(index)
        # Original FastWAMProcessor.augment_instruction consumes one NumPy draw
        # even with deterministic drop_high_level_prob=1. Preserve that state.
        np.random.rand()
        task_index = int(self.numeric_reference["task_index"][index])
        if task_index not in self.task_prompts:
            raise CacheIntegrityError("Original task has no source prompt")
        return {**values, "video": self.pixels.window(index), "sample_index": index,
                "prompt": self.task_prompts[task_index]}

    get = __call__


def build_deduplicated_source(raw_config, pixel_cache_root, index_rows=None, expected_pixel_identity_sha256=None, *, test_fixture=False):
    from fastwam.datasets.lerobot.robot_video_dataset import DEFAULT_PROMPT

    identity = pixel_identity_from_source(raw_config, test_fixture=test_fixture)
    from .latent import cache_identity_sha256
    expected = cache_identity_sha256(identity)
    if expected_pixel_identity_sha256 is not None and expected != expected_pixel_identity_sha256:
        raise CacheIntegrityError("Current pixel preprocessor/source differs from the pinned pixel identity")
    pixels = PixelCacheDataset(pixel_cache_root, expected)
    stats_sha = file_sha256(raw_config.pretrained_norm_stats)
    reference, rows, lengths = numeric_reference_from_source(raw_config)
    if index_rows is not None and (len(index_rows) != len(rows) or any(tuple(row[key] for key in ("sample_index", "episode_index", "frame_index", "task_index"))
                                     != tuple(expected_row[key] for key in ("sample_index", "episode_index", "frame_index", "task_index"))
                                     for row, expected_row in zip(index_rows, rows))):
        raise CacheIntegrityError("Supplied global anchor index inventory changed")
    tasks = {}
    root = Path(raw_config.dataset_dirs[0]).resolve()
    for line in (root / "meta/tasks.jsonl").read_text().splitlines():
        if line.strip():
            value = json.loads(line)
            task, instruction = value["task_index"], value["task"]
            if type(task) is not int or task in tasks or not isinstance(instruction, str):
                raise CacheIntegrityError("Source task identity is invalid")
            if "@" in instruction:
                if instruction.count("@") != 1:
                    raise CacheIntegrityError("Source task requires invalid stock language selection")
                _, instruction = instruction.split("@")
            tasks[task] = DEFAULT_PROMPT.format(task=instruction)
    provenance = {"schema": "deduplicated-stock-source-v2", "pixel_identity_sha256": expected,
                  "pixel_manifest_sha256": file_sha256(Path(pixel_cache_root) / "manifest.json"),
                  "numeric_reference_sha256": tensor_inventory_sha256(reference),
                  "dataset_stats_sha256": stats_sha,
                  "conversion_manifest_sha256": file_sha256(root / "conversion_manifest.json"),
                  "source_code_sha256": identity["source_code_sha256"]}
    if (provenance["conversion_manifest_sha256"] != identity["data_conversion_sha256"]
            or file_sha256(raw_config.pretrained_norm_stats) != stats_sha
            or any(file_sha256(ROOT / path) != digest for path, digest in identity["source_code_sha256"].items())):
        raise CacheIntegrityError("Original source/stats/code changed while constructing independent reference")
    return DeduplicatedSource(pixels, reference, rows, lengths, tasks, source_provenance=provenance)
