"""CPU-only hold policy for protocol testing; never loads a checkpoint."""


class MockPolicy:
    def __init__(self, *, width=64, height=48, video_context_frames=1):
        self.metadata = dict(camera_names=["global", "wrist"], width=width, height=height,
                             action_space="joint8", action_horizon=32, control_hz=30,
                             history_length=32, video_context_frames=video_context_frames,
                             video_context_stride=4 if video_context_frames > 1 else 1,
                             force_preprocessing=None, backend="mock")

    def predict(self, observation, history, video_context):
        return [list(observation["state"]) for _ in range(32)]

    def close(self):
        pass
