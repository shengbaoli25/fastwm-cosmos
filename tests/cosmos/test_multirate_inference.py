"""Check multirate math against recomputation and count expensive evaluations."""
from unittest.mock import patch
import pytest
import torch

from conftest import sample
from test_noaction import model
from test_noaction_inference import assert_outputs
from fastwam.cosmos.multirate_inference import MultirateInferenceRunner, infer_multirate_reference


@pytest.mark.parametrize('video_steps,control_steps', [(4, 8), (4, 4), (1, 3)])
@torch.no_grad()
def test_multirate_matches_full_recomputation_and_refreshes(video_steps, control_steps):
    m = model().eval()
    runner = MultirateInferenceRunner(m, video_steps=video_steps, compile_layers=False, cuda_graph=False)
    data = sample(1)
    for request in range(2):
        expected = infer_multirate_reference(m, data, video_steps=video_steps, control_steps=control_steps)
        with patch.object(runner.layers[0], 'video_forward', wraps=runner.layers[0].video_forward) as video, \
             patch.object(runner.layers[0].control, 'forward', wraps=runner.layers[0].control.forward) as control:
            actual = runner.infer_joint(data, num_inference_steps=control_steps)
        assert video.call_count == video_steps
        assert control.call_count == control_steps
        assert_outputs(actual, expected)
        if video_steps == control_steps:
            assert_outputs(actual, m.infer_joint(data, num_inference_steps=control_steps))
        torch.testing.assert_close(actual['video_latents'][:, :, :1], data['first_frame_latents'])
        data['first_frame_latents'].add_(.4)
        data['history'].add_(.2)
        data['cosmos_context'].add_(.3)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='Requires CUDA')
@torch.no_grad()
def test_multirate_graph_refreshes_inputs_and_preserves_outputs():
    m = model().eval().to('cuda', torch.bfloat16)
    runner = MultirateInferenceRunner(m, video_steps=2, compile_layers=False, cuda_graph=True)
    data = sample(1)
    first = runner.infer_joint(data, num_inference_steps=4)
    saved = {k: v.clone() for k, v in first.items()}
    data['first_frame_latents'].add_(.5)
    data['history_is_pad'][:, :8] = True
    data['cosmos_context'].add_(.2)
    actual = runner.infer_joint(data, num_inference_steps=4, seed=47)
    expected = infer_multirate_reference(m, data, video_steps=2, control_steps=4, seed=47)
    assert_outputs(actual, expected, atol=.04, rtol=.04)
    assert_outputs(first, saved, atol=0, rtol=0)
    runner.video_steps = 4
    equal = runner.infer_joint(data, num_inference_steps=4, seed=47)
    assert_outputs(equal, m.infer_joint(data, num_inference_steps=4, seed=47), atol=.04, rtol=.04)
    assert runner.stats['graph_builds'] == 2


@torch.no_grad()
def test_invalid_step_ratio_is_rejected():
    runner = MultirateInferenceRunner(model().eval(), video_steps=3, compile_layers=False, cuda_graph=False)
    with pytest.raises(ValueError, match='divide'):
        runner.infer_joint(sample(1), num_inference_steps=8)
