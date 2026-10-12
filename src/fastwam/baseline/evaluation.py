"""Evaluation-only capture of the successful unchanged upstream reader call."""
from fastwam.datasets.lerobot.base_lerobot_dataset import BaseLerobotDataset
from fastwam.datasets.lerobot.robot_video_dataset import RobotVideoDataset


class _CaptureBaseDataset(BaseLerobotDataset):
    def __getitem__(self, idx):
        sample = super().__getitem__(idx)
        self.last_actual_index = int(sample["idx"])
        self.last_raw_action = sample["gt_action"]["default"].clone()
        return sample


class VerifiedEvaluationDataset(RobotVideoDataset):
    """Stock samples plus an audit of reader retry and unclipped GT."""
    base_dataset_cls = _CaptureBaseDataset


def infer_unconditioned(model, sample, *, seed, steps):
    """Invoke original inference using only observation and task context."""
    return model.infer_joint(
        prompt=None, input_image=sample["video"][:, :, 0],
        num_video_frames=sample["video"].shape[2], action_horizon=sample["action"].shape[1],
        action=None, proprio=sample["proprio"][:, 0],
        context=sample["context"], context_mask=sample["context_mask"],
        text_cfg_scale=1.0, num_inference_steps=steps, seed=seed,
        test_action_with_infer_action=False, compile_action_infer=False,
    )
