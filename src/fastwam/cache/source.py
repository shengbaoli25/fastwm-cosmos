"""Precomputation-only adapter around the unchanged stock reader/transforms.

This module is intentionally not imported by the feature-only training reader.
The guard bypasses upstream random retry handlers on read failure; successful
samples still use VerifiedEvaluationDataset._get and its original processor.
"""
from __future__ import annotations

from pathlib import Path

import torch

from .latent import CacheIntegrityError, NUMERIC_SHAPES


class _AbortRawRead(BaseException):
    """Escape the upstream except-Exception random-replacement loop."""


class _StrictReadProxy:
    def __init__(self, reader, index):
        self.reader, self.index = reader, index

    def __getattr__(self, key):
        return getattr(self.reader, key)

    def __getitem__(self, index):
        if index != self.index:
            raise _AbortRawRead("Raw reader attempted a replacement index")
        try:
            return self.reader[index]
        except Exception as error:
            raise _AbortRawRead(f"Source index {index} could not be read") from error


def build_index_rows(dataset_root):
    """Read only numeric parquet columns, retaining exact raw pose/wrench labels."""
    import pyarrow.parquet as pq

    rows = []
    for path in sorted(Path(dataset_root).glob("data/chunk-*/episode_*.parquet")):
        table = pq.read_table(path, columns=["index", "episode_index", "frame_index", "task_index",
                                           "observation.state", "observation.wrench", "action"])
        columns = {key: table[key].to_pylist() for key in table.column_names}
        for frame in range(len(table)):
            rows.append({"sample_index": int(columns["index"][frame]),
                         "episode_index": int(columns["episode_index"][frame]),
                         "frame_index": int(columns["frame_index"][frame]),
                         "task_index": int(columns["task_index"][frame]),
                         "raw_current_state": torch.tensor(columns["observation.state"][frame], dtype=torch.float64),
                         "raw_current_wrench": torch.tensor(columns["observation.wrench"][frame], dtype=torch.float64)})
    if not rows or [row["sample_index"] for row in rows] != list(range(len(rows))):
        raise CacheIntegrityError("Source numeric rows must retain every original contiguous global index")
    return rows


def build_strict_source(raw_config, index_rows, work_dir):
    from hydra.utils import instantiate
    from omegaconf import OmegaConf
    from fastwam.baseline.evaluation import VerifiedEvaluationDataset, _CaptureBaseDataset
    from fastwam.utils import misc

    class StrictCaptureBaseDataset(_CaptureBaseDataset):
        def __getitem__(self, index):
            original = self.multi_dataset
            self.multi_dataset = _StrictReadProxy(original, index)
            try:
                return super().__getitem__(index)
            except _AbortRawRead as error:
                raise CacheIntegrityError(str(error)) from error
            finally:
                self.multi_dataset = original

    class StrictVerifiedDataset(VerifiedEvaluationDataset):
        base_dataset_cls = StrictCaptureBaseDataset

    config = OmegaConf.create(OmegaConf.to_container(raw_config, resolve=True))
    if (config.get("skip_padding_as_possible") is not False or float(config.get("val_set_proportion", -1)) != 0
            or int(config.get("global_sample_stride", 0)) != 1 or int(config.num_frames) != 33
            or int(config.action_video_freq_ratio) != 4 or list(config.video_size) != [224, 448]
            or str(config.concat_multi_camera) != "horizontal" or len(config.dataset_dirs) != 1
            or [item.key for item in config.shape_meta.images] != ["cam_main", "cam_wrist"]):
        raise CacheIntegrityError("Precompute requires the unchanged unsplit LIBERO two-camera 33-step source configuration")
    if config.get("override_instruction") is not None:
        raise CacheIntegrityError("ForceVLA cache must preserve the recorded task instructions")
    processor_config = config.processor
    transforms = [{"_target_": "fastwam.datasets.lerobot.transforms.image.ToTensor"},
                  {"_target_": "torchvision.transforms.Resize", "size": [224, 224]}]
    if (int(processor_config.action_output_dim) != 7 or int(processor_config.proprio_output_dim) != 8
            or int(processor_config.num_output_cameras) != 2 or processor_config.action_state_transforms is not None
            or str(processor_config.norm_default_mode) != "z-score" or processor_config.norm_exception_mode is not None
            or bool(processor_config.use_stepwise_action_norm)
            or OmegaConf.to_container(processor_config.train_transforms, resolve=True) != transforms
            or OmegaConf.to_container(processor_config.val_transforms, resolve=True) != transforms
            or OmegaConf.to_container(processor_config.delta_action_dim_mask, resolve=True) != {"default": [False] * 7}
            or any(list(image.shape) != [3, 224, 224] for image in config.shape_meta.images)):
        raise CacheIntegrityError("Precompute requires original deterministic transforms and action7/proprio8 z-score processing")
    # Contexts are loaded once by the CLI, not repeatedly by the raw RGB reader.
    config.use_text_embed_cache = False
    # Keep nested DictConfig/ListConfig types expected by the original reader.
    values = {key: value for key, value in config.items()}
    values.pop("_target_", None)
    misc.register_work_dir(str(Path(work_dir).resolve()))
    processor_config = values.pop("processor")
    processor = instantiate(processor_config)
    dataset = StrictVerifiedDataset(**values, processor=processor)
    dataset.lerobot_dataset.processor.eval()
    if len(dataset) != len(index_rows):
        raise CacheIntegrityError("Raw dataset length differs from the complete source index inventory")

    def get(index):
        # Bypass RobotVideoDataset.__getitem__, which has another random fallback.
        sample = dataset._get(index)
        actual = dataset.lerobot_dataset.last_actual_index
        if actual != index:
            raise CacheIntegrityError("Upstream reader substituted a source index")
        sample["sample_index"] = actual
        for key in NUMERIC_SHAPES:
            if key not in sample:
                raise CacheIntegrityError(f"Stock source sample lost a numeric feature: {key}")
        sample["raw_current_state"] = index_rows[index]["raw_current_state"]
        sample["raw_current_wrench"] = index_rows[index]["raw_current_wrench"]
        return sample

    return get
