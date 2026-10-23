"""Acceleration parity, cache invalidation and captured-output ownership."""
import pytest
import torch

from conftest import sample
from test_noaction import model
from fastwam.cosmos.noaction_inference import NoActionInferenceRunner


def assert_outputs(actual, expected, *, atol=4e-6, rtol=4e-5):
    for key in expected:
        torch.testing.assert_close(actual[key], expected[key], atol=atol, rtol=rtol)


@torch.no_grad()
def test_request_cache_refreshes_observation_masks_text_and_weights():
    m = model().eval()
    runner = NoActionInferenceRunner(m, compile_layers=False, cuda_graph=False)
    data = sample(1)
    for change in range(5):
        if change == 1:
            data['first_frame_latents'].add_(.5)
            data['history'].add_(.2)
            data['history_is_pad'][:, :7] = True
            data['history_channel_valid'][:, 8, -2:] = False
            data['proprio'][:, 0].add_(.3)
        elif change == 2:
            data['cosmos_context'].add_(.4)
        elif change == 3:
            m.video.net.crossattn_proj[0].weight.add_(.01)
        elif change == 4:
            data = sample(2)
        expected = m.infer_joint(data, num_inference_steps=3, seed=42 + change)
        actual = runner.infer_joint(data, num_inference_steps=3, seed=42 + change)
        assert_outputs(actual, expected)
    assert runner.stats['text_cache_hits'] >= 1
    assert runner.stats['schedule_builds'] == 3


@torch.no_grad()
def test_bank_cache_works_inside_inference_mode_and_invalidates_in_place():
    m = model().eval()
    data = sample(1)
    m.text_bank = torch.cat((data.pop('cosmos_context'), torch.randn(1, 3, 24)))
    runner = NoActionInferenceRunner(m, compile_layers=False, cuda_graph=False)
    for index in (0, 0, 1, 1):
        with torch.inference_mode():
            data['cosmos_text_index'] = torch.tensor([index])
            expected = m.infer_joint(data, num_inference_steps=2)
            actual = runner.infer_joint(data, num_inference_steps=2)
        assert_outputs(actual, expected)
    assert runner.stats['text_cache_hits'] == 2
    m.text_bank.add_(.25)
    assert_outputs(runner.infer_joint(data, num_inference_steps=2),
                   m.infer_joint(data, num_inference_steps=2))
    assert runner.stats['text_cache_builds'] == 3


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA graph needs a GPU')
@torch.no_grad()
def test_cuda_graph_refreshes_inputs_and_does_not_overwrite_returned_outputs():
    m = model().eval().to('cuda', torch.bfloat16)
    runner = NoActionInferenceRunner(m, compile_layers=False, cuda_graph=True)
    data = sample(1)
    first = runner.infer_joint(data, num_inference_steps=3)
    saved = {key: value.clone() for key, value in first.items()}
    data['first_frame_latents'].add_(.5)
    data['cosmos_context'].add_(.2)
    data['history_is_pad'][:, :8] = True
    data['history'].add_(.3)
    second = runner.infer_joint(data, num_inference_steps=3, seed=47)
    assert_outputs(first, saved, atol=0, rtol=0)
    assert_outputs(second, m.infer_joint(data, num_inference_steps=3, seed=47), atol=.04, rtol=.04)
    assert runner.stats['graph_builds'] == 1
    runner.close()
    assert runner._graph is None and runner.schedule is None
