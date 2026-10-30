"""Wrap the existing Cosmos Joint8 inference path without changing training."""
from pathlib import Path

from realman_deploy.contracts import Observation
from realman_deploy.normalization import OnlineHistory


class CosmosPolicy:
    def __init__(self, config, config_path, checkpoint):
        from realman_deploy.model_adapter import CosmosJointPolicy
        self.policy = CosmosJointPolicy(checkpoint, config, config_path=Path(config_path))
        try:
            if self.policy.normalizer.action_space != "joint8" or self.policy.reference_mode != "none":
                raise ValueError("Synchronous server requires a no-reference Joint8 checkpoint")
            self.metadata = dict(camera_names=config.get("camera_order", ["global", "wrist"]),
                                 width=int(config.get("camera_width", 640)),
                                 height=int(config.get("camera_height", 480)),
                                 action_space="joint8", action_horizon=32, control_hz=30,
                                 history_length=32, video_context_frames=1, video_context_stride=1,
                                 force_preprocessing=self.policy.normalizer.force_preprocessing,
                                 backend="cosmos")
        except Exception:
            self.policy.close()
            raise

    def predict(self, observation, history, video_context):
        normalizer = self.policy.normalizer
        measured = OnlineHistory(normalizer, 32)
        for sample in history:
            measured.append(sample["state"], sample["force"])
        obs = Observation(observation["stamp"], observation["sequence"],
                          tuple(observation["state"]), tuple(observation["force"]),
                          {name: item["data"] for name, item in observation["frames"].items()},
                          observation["frame_meta"], observation["force_meta"], "joint8")
        normalized = self.policy.infer(obs, None, measured.tensors())
        if len(normalized) != 32:
            raise ValueError("Cosmos output requires 32 actions")
        decoded = normalizer.decode("action", normalized)
        # Identical to the existing deploy runtime: only gripper is clipped.
        return [row[:-1] + [max(0.0, min(1.0, row[-1]))] for row in decoded]

    def close(self):
        self.policy.close()
