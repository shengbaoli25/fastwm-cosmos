from itertools import product
from unittest.mock import patch
import pytest
import torch
from conftest import sample
from test_noaction import model
from test_noaction_inference import assert_outputs
from fastwam.cosmos.expert_schedule_inference import ExpertScheduleRunner, infer_expert_reference, expert_plan
from fastwam.cosmos.multirate_inference import MultirateInferenceRunner


@pytest.mark.parametrize('counts', list(product((1, 2, 4), repeat=3)))
@torch.no_grad()
def test_all_factorial_grids_match_recomputation(counts):
    m = model().eval()
    data = sample(1)
    settings = dict(zip(('video_steps','force_steps','action_steps'), counts))
    runner = ExpertScheduleRunner(m, **settings, compile_layers=False, cuda_graph=False)
    with patch.object(runner.layers[0], 'video_forward', wraps=runner.layers[0].video_forward) as video, \
         patch.object(runner.force_layers[0], 'forward', wraps=runner.force_layers[0].forward) as force, \
         patch.object(runner.action_layers[0], 'forward', wraps=runner.action_layers[0].forward) as action:
        actual = runner.infer_joint(data)
    assert video.call_count == counts[0]
    assert force.call_count == runner.stats['force_forward_evaluations']
    assert action.call_count == counts[2]
    expected = infer_expert_reference(m, data, **settings)
    assert_outputs(actual, expected)
    if len(set(counts)) == 1:
        assert_outputs(actual, m.infer_joint(data, num_inference_steps=counts[0]))
    if counts[1] == counts[2] and counts[1] >= counts[0]:
        previous = MultirateInferenceRunner(m, video_steps=counts[0], compile_layers=False, cuda_graph=False)
        assert_outputs(actual, previous.infer_joint(data, num_inference_steps=counts[1]))


@torch.no_grad()
def test_action_grid_does_not_change_video_or_force():
    m = model().eval()
    data = sample(1)
    outputs=[]
    for action_steps in (1,2,4):
        runner = ExpertScheduleRunner(m, video_steps=4, force_steps=1, action_steps=action_steps,
                                     compile_layers=False, cuda_graph=False)
        outputs.append(runner.infer_joint(data))
    for output in outputs[1:]:
        for key in ('video_latents','force'):
            torch.testing.assert_close(output[key],outputs[0][key],atol=0,rtol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
@torch.no_grad()
def test_captured_expert_cache_is_request_local():
    m = model().eval().to('cuda',torch.bfloat16)
    settings = dict(video_steps=2,force_steps=1,action_steps=4)
    runner = ExpertScheduleRunner(m,**settings,compile_layers=False,cuda_graph=True)
    data=sample(1)
    first=runner.infer_joint(data)
    saved={k:v.clone() for k,v in first.items()}
    data['first_frame_latents'].add_(.3)
    data['history'].add_(.2)
    data['history_is_pad'][:,:5]=True
    actual=runner.infer_joint(data,seed=47)
    expected=infer_expert_reference(m,data,**settings,seed=47)
    assert_outputs(actual,expected,atol=.04,rtol=.04)
    assert_outputs(first,saved,atol=0,rtol=0)


def test_grids_reject_nonpositive_counts():
    with pytest.raises(ValueError,match='positive'):
        expert_plan(0,8,16)
