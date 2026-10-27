"""Independent contract tests for video conditioning and per-camera encoding."""

import pytest
import torch
from torch import nn

from fastwam.cosmos.dense_encoder import DenseTeacher


class GloballyAttendingVideoEncoder(nn.Module):
    """A teacher whose every output deliberately depends on every input frame."""

    def __init__(self):
        super().__init__()
        self.anchor = nn.Parameter(torch.ones(()))
        self.inputs = []

    def forward(self, video):
        self.inputs.append(video.detach().clone())
        # Distinguish local tubelet content and full-clip bidirectional context.
        frame_values = video.float().mean((1, 3, 4))
        tubelets = frame_values.reshape(len(video), -1, 2).mean(-1)
        global_context = frame_values.mean(-1, keepdim=True)
        values = tubelets + 3 * global_context
        return values.repeat_interleave(196, dim=1)[..., None].expand(-1, -1, 768)


@pytest.fixture
def teacher():
    old_threads = torch.get_num_threads()
    torch.set_num_threads(2)
    encoder = GloballyAttendingVideoEncoder().eval().requires_grad_(False)
    result = DenseTeacher('vjepa21', encoder, 384, 24, 14, 'test-digest')
    yield result
    torch.set_num_threads(old_threads)


def clip():
    video = torch.empty(1, 3, 10, 224, 448)
    for step in range(10):
        video[:, :, step, :, :224] = -0.8 + step * 0.1
        video[:, :, step, :, 224:] = -0.2 + step * 0.1
    return video


def test_future_changes_cannot_enter_observation_path(teacher):
    video = clip()
    changed = video.clone()
    changed[:, :, 2:] = 0.95

    observed = teacher.encode_video(video[:, :, :2], context_only=True)
    changed_observed = teacher.encode_video(changed[:, :, :2], context_only=True)
    full = teacher.encode_video(video)
    changed_full = teacher.encode_video(changed)

    assert observed.shape == (1, 1, 392, 768)
    assert full.shape == (1, 5, 392, 768)
    torch.testing.assert_close(observed, changed_observed, rtol=0, atol=0)
    # This makes the test detect the tempting but incorrect full[:, :1] path.
    assert not torch.equal(full[:, :1], changed_full[:, :1])
    assert not torch.equal(full[:, 1:], changed_full[:, 1:])
    assert not torch.equal(observed, full[:, :1])
    assert not observed.requires_grad


@pytest.mark.parametrize('context_only', [True, False])
def test_views_stay_independent_and_main_precedes_wrist(teacher, context_only):
    video = clip()[:, :, :2] if context_only else clip()
    changed = video.clone()
    changed[..., 224:] = 0.95

    features = teacher.encode_video(video, context_only=context_only)
    changed_features = teacher.encode_video(changed, context_only=context_only)
    torch.testing.assert_close(features[:, :, :196], changed_features[:, :, :196],
                               rtol=0, atol=0)
    assert not torch.equal(features[:, :, 196:], changed_features[:, :, 196:])
    assert bool((features[:, :, :196] < features[:, :, 196:]).all())
    assert teacher.encoder.inputs[0].shape[:3] == (2, 3, 2 if context_only else 10)


def test_both_observed_frames_contribute_to_the_video_tubelet(teacher):
    video = torch.zeros(1, 3, 2, 224, 448)
    first_changed, second_changed = video.clone(), video.clone()
    first_changed[:, :, 0] = 0.5
    second_changed[:, :, 1] = 0.5

    baseline = teacher.encode_video(video, context_only=True)
    first = teacher.encode_video(first_changed, context_only=True)
    second = teacher.encode_video(second_changed, context_only=True)
    assert not torch.equal(first, baseline)
    assert not torch.equal(second, baseline)
    torch.testing.assert_close(first, second, rtol=0, atol=0)


def test_video_path_rejects_incomplete_or_future_contaminated_context(teacher):
    video = clip()
    with pytest.raises(ValueError, match='2\\|10'):
        teacher.encode_video(video, context_only=True)
    with pytest.raises(ValueError, match='2\\|10'):
        teacher.encode_video(video[:, :, :2])
    with pytest.raises(ValueError, match='2\\|10'):
        teacher.encode_video(video[:, :, :9])
