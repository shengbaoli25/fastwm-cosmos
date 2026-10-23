from itertools import product
from unittest.mock import patch
import pytest
import torch
from conftest import sample
from test_noaction import model
from test_noaction_inference import assert_outputs
from fastwam.cosmos.expert_schedule_inference import ExpertScheduleRunner, infer_expert_reference, expert_plan
from fastwam.cosmos.multirate_inference import MultirateInferenceRunner
from fastwam.cosmos.multirate import MultirateRunner


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


@pytest.mark.parametrize('counts', [(16, 16, 16), (4, 4, 16)])
@torch.no_grad()
def test_deployment_profiles_match_current_multirate_runner(counts):
    m = model().eval()
    settings = dict(zip(('video_steps', 'force_steps', 'action_steps'), counts))
    runner = ExpertScheduleRunner(m, **settings, compile_layers=False, cuda_graph=False)
    previous = MultirateRunner(m, mode='eager')
    assert runner.diagnostics['effective_mode'] == 'pending'
    data = sample(1)
    for seed in (42, 43):
        expected = previous.infer(data, **settings, seed=seed)
        actual = runner.infer_joint(data, seed=seed)
        assert_outputs(actual, expected)
        data['first_frame_latents'].add_(.1)
        data['history'].add_(.1)
        data['proprio'].add_(.1)
        data['history_is_pad'][:, :5] = True
    diagnostics = runner.diagnostics
    assert diagnostics['effective_mode'] == 'eager'
    assert diagnostics['stats']['nominal_steps'] == dict(zip(('video', 'force', 'action'), counts))
    assert diagnostics['stats']['schedule_builds'] == 1
    assert diagnostics['stats']['text_cache_hits'] == 1
    runner.close()
    assert runner.diagnostics['effective_mode'] == 'pending'


def test_expert_failure_is_reported_and_not_replaced_by_fallback(monkeypatch):
    m = model().eval()
    runner = ExpertScheduleRunner(m, compile_layers=False, cuda_graph=False)
    def fail(*args):
        raise RuntimeError('capture failure')
    monkeypatch.setattr(runner, '_rollout', fail)
    with pytest.raises(RuntimeError, match='capture failure'):
        runner.infer_joint(sample(1))
    assert runner.diagnostics['effective_mode'] == 'error'
    assert 'capture failure' in runner.diagnostics['error']
    assert runner.diagnostics['fallback_reason'] is None


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
@pytest.mark.parametrize('counts', [(16, 16, 16), (4, 4, 16)])
@torch.no_grad()
def test_compiled_full_graph_matches_previous_sampler_and_refreshes_inputs(counts):
    # A tiny randomly initialized model exercises the production execution path;
    # this is a regression test, not a cup checkpoint latency/quality benchmark.
    m = model().eval().to('cuda', torch.bfloat16)
    settings = dict(zip(('video_steps', 'force_steps', 'action_steps'), counts))
    runner = ExpertScheduleRunner(m, **settings, compile_layers=True, cuda_graph=True)
    previous = MultirateRunner(m, mode='eager')
    data = sample(1)
    assert runner.diagnostics['effective_mode'] == 'pending'
    try:
        with torch.inference_mode():
            first = runner.infer_joint(data)
            expected = previous.infer(data, **settings)
        assert_outputs(first, expected, atol=.04, rtol=.04)
        saved = {key: value.clone() for key, value in first.items()}
        data['first_frame_latents'].add_(.3)
        data['history'].add_(.2)
        data['proprio'].add_(.1)
        data['history_is_pad'][:, :5] = True
        with torch.inference_mode():
            second = runner.infer_joint(data, seed=47)
            expected = previous.infer(data, **settings, seed=47)
        assert_outputs(second, expected, atol=.04, rtol=.04)
        assert_outputs(first, saved, atol=0, rtol=0)
        diag = runner.diagnostics
        assert diag['effective_mode'] == 'compiled_cuda_graph'
        assert diag['graph_status'] == 'captured' and diag['compile_status'] == 'active'
        assert diag['stats']['graph_builds'] == 1
        assert diag['stats']['schedule_builds'] == 1
    finally:
        runner.close()
        previous.close()
