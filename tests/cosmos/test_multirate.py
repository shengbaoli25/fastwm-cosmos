"""CPU math/ownership tests using native tiny networks, not latency or quality claims."""
import copy
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from conftest import sample
from test_noaction import model
from fastwam.cosmos import multirate
from fastwam.cosmos.multirate import MultirateRunner, MultirateSteps, infer_multirate


def assert_outputs(actual, expected):
    assert actual.keys() == expected.keys()
    for name in actual:
        torch.testing.assert_close(actual[name], expected[name], atol=4e-6, rtol=4e-5, msg=name)


def full_model_reference(m, sample, counts, *, seed=42, video_seed=44, force_seed=43):
    """Separate schedule implementation; no production sampler/cache helpers.

    At each fine tick all experts see the same pre-update snapshot. An expert
    records its velocity at the start of its own interval, integrates at the end,
    and keeps its state and clock fixed between those boundaries.
    """
    m.eval()
    with torch.no_grad():
        first = sample['first_frame_latents'].to(m.device, m.torch_dtype)
        shapes = [(*first.shape[:2], 3, *first.shape[3:]), (len(first), 32, 6),
                  (len(first), 32, m.action_dim)]
        states = [torch.randn(shape, generator=torch.Generator().manual_seed(seed)).to(m.device, m.torch_dtype)
                  for shape, seed in zip(shapes, (video_seed, force_seed, seed))]
        states[0][:, :, :1] = first
        schedulers = (m.v_scheduler, m.f_scheduler, m.a_scheduler)
        schedules = [s.build_inference_schedule(n, m.device, m.torch_dtype)
                     for s, n in zip(schedulers, counts)]
        strides = [max(counts) // n for n in counts]
        own_step = [0, 0, 0]
        held = [None, None, None]
        conditions = m._conditions(sample)
        for tick in range(max(counts)):
            times = [schedules[e][0][own_step[e]].expand(len(first)) for e in range(3)]
            velocities = m.joint_denoise_core(*states, *times, conditions, static_cache=None)
            for e in range(3):
                if tick % strides[e] == 0:
                    held[e] = velocities[e]
            updated = list(states)
            for e in range(3):
                if (tick + 1) % strides[e] == 0:
                    updated[e] = schedulers[e].step(held[e], schedules[e][1][own_step[e]], states[e])
                    own_step[e] += 1
            states = updated
            states[0][:, :, :1] = first
        return dict(video_latents=states[0], force=states[1], action=states[2])


@pytest.mark.parametrize('counts', [(4, 4, 16), (8, 4, 16), (4, 8, 16), (16, 4, 4)])
def test_exact_dependency_cache_matches_independent_full_model(counts):
    m = model()
    s = sample(1)
    # Nontrivial history padding exercises Force/Action validity masks.
    s['history_is_pad'][:, :3] = True
    s['history_channel_valid'][:, 7:10, :2] = False
    expected = full_model_reference(m, s, counts)
    actual = infer_multirate(m, s, **dict(zip(('video_steps', 'force_steps', 'action_steps'), counts)))
    assert_outputs(actual, expected)
    torch.testing.assert_close(actual['video_latents'][:, :, :1], s['first_frame_latents'], rtol=0, atol=0)


def test_equal_16_steps_matches_existing_joint_sampler():
    m = model()
    s = sample(1)
    expected = m.infer_joint(s, num_inference_steps=16, use_cache=False)
    actual = infer_multirate(m, s, video_steps=16, force_steps=16, action_steps=16)
    assert_outputs(actual, expected)


def test_force_features_refresh_on_video_changes_without_replacing_held_velocity():
    m = model()
    s = sample(1)
    runner = MultirateRunner(m)
    with patch.object(runner, '_call', wraps=runner._call) as call:
        result = runner.infer(s, video_steps=8, force_steps=4, action_steps=16)
    calls = {branch: [entry for entry in call.call_args_list if entry.args[0] == branch]
             for branch in ('video', 'force', 'action')}
    assert {branch: len(entries) for branch, entries in calls.items()} == dict(video=8, force=8, action=16)
    # Second Force feature pass has new Video time but the same Force time and state.
    a, b = calls['force'][:2]
    torch.testing.assert_close(a.args[1], b.args[1], rtol=0, atol=0)
    torch.testing.assert_close(a.args[2], b.args[2], rtol=0, atol=0)
    assert not torch.equal(a.args[3], b.args[3])
    assert_outputs(result, full_model_reference(m, s, (8, 4, 16)))


def test_request_conditions_and_weights_invalidate_reusable_execution_state():
    m = model()
    s = sample(1)
    runner = MultirateRunner(m)
    baseline = runner.infer(s)
    changed = copy.deepcopy(s)
    changed['history'] += .7
    changed['first_frame_latents'] += .2
    changed['cosmos_context'] += .3
    result = runner.infer(changed)
    assert_outputs(result, full_model_reference(m, changed, (4, 4, 16)))
    assert not torch.allclose(result['action'], baseline['action'])
    runner._disabled.add('action')
    runner._fallback_reasons['action'] = 'old device failure'
    with torch.no_grad():
        next(m.parameters()).add_(.001)
    runner.infer(changed)
    assert not runner._disabled and not runner.diagnostics['fallback_reasons']
    runner.close()
    assert not runner._functions and not runner._graphs and runner._signature is None


def test_no_future_labels_rng_or_module_mode_changes():
    m = model()
    m.dit['action'].eval()
    modes = [module.training for module in m.modules()]
    s = sample(1)
    changed = copy.deepcopy(s)
    for name in ('action', 'wrench_future', 'video_latents'):
        changed[name].fill_(float('nan'))
    rng = torch.get_rng_state().clone()
    runner = MultirateRunner(m)
    a = runner.infer(s)
    b = runner.infer(changed)
    for name in a:
        torch.testing.assert_close(a[name], b[name], rtol=0, atol=0)
    assert torch.equal(rng, torch.get_rng_state())
    assert modes == [module.training for module in m.modules()]


def test_compile_results_own_storage_even_when_compiler_reuses_output_buffers(monkeypatch):
    # Simulate reduce-overhead CUDA Graph storage reuse on CPU, including nested K/V.
    buffer = torch.zeros(2)
    def compile_stub(fn, **kwargs):
        assert kwargs == dict(mode='reduce-overhead', fullgraph=True)
        def compiled(value):
            buffer.copy_(value * 2)
            return buffer, ((buffer, buffer),)
        return compiled
    monkeypatch.setattr(torch, 'compile', compile_stub)
    runner = MultirateRunner(SimpleNamespace(device=torch.device('cpu')), mode='compile')
    first = runner._call('force', torch.tensor([1., 2.]))
    second = runner._call('force', torch.tensor([3., 4.]))
    torch.testing.assert_close(first[0], torch.tensor([2., 4.]))
    torch.testing.assert_close(first[1][0][0], torch.tensor([2., 4.]))
    torch.testing.assert_close(second[0], torch.tensor([6., 8.]))
    assert first[0].data_ptr() != buffer.data_ptr()
    assert runner.diagnostics['branch_modes']['force'] == 'compile'


@pytest.mark.parametrize('mode', ['cuda_graph', 'compile_cuda_graph'])
def test_cuda_graph_request_on_cpu_has_observable_eager_fallback(mode):
    m = model()
    runner = MultirateRunner(m, mode=mode)
    with pytest.warns(RuntimeWarning, match='requires a CUDA device'):
        actual = runner.infer(sample(1))
    assert_outputs(actual, full_model_reference(m, sample(1), (4, 4, 16)))
    assert runner.diagnostics['requested_mode'] == mode
    assert runner.diagnostics['effective_mode'] == 'eager'
    assert 'device' in runner.diagnostics['fallback_reasons']


@pytest.mark.parametrize('mode', ['compile', 'compile_cuda_graph'])
def test_unsupported_accelerator_falls_back_once_per_branch(monkeypatch, mode):
    def unsupported(*args, **kwargs):
        raise RuntimeError('capture/compile unsupported in this runtime')
    monkeypatch.setattr(torch, 'compile', unsupported)
    monkeypatch.setattr(multirate, '_CudaGraphCall', unsupported)
    monkeypatch.setattr(torch.cuda, 'is_current_stream_capturing', lambda: False)
    monkeypatch.setitem(multirate._BRANCHES, 'action', lambda m, value: value + 1)
    # No actual CUDA calls; exercise a non-default device index through fallback.
    runner = MultirateRunner(SimpleNamespace(device=torch.device('cuda:2')), mode=mode)
    with pytest.warns(RuntimeWarning, match='using eager'):
        output = runner._call('action', torch.tensor([2.]))
    torch.testing.assert_close(output, torch.tensor([3.]))
    torch.testing.assert_close(runner._call('action', torch.tensor([4.])), torch.tensor([5.]))
    status = runner.diagnostics
    assert status['branch_modes']['action'] == 'eager'
    assert status['effective_mode'] == 'mixed'
    assert 'unsupported' in status['fallback_reasons']['action']
    assert not runner._graphs and not runner._functions


@pytest.mark.parametrize('error', [torch.OutOfMemoryError('out of memory'), RuntimeError('CUDA illegal memory access')])
def test_fatal_device_errors_are_not_hidden_as_optimization_fallback(monkeypatch, error):
    def broken(*args, **kwargs):
        raise error
    monkeypatch.setattr(torch, 'compile', broken)
    runner = MultirateRunner(SimpleNamespace(device=torch.device('cuda:1')), mode='compile')
    with pytest.raises(type(error), match=str(error)):
        runner._call('video', torch.tensor([2.]))
    assert not runner._disabled and runner.fallback_reason is None


@pytest.mark.parametrize('counts', [(0, 4, 16), (-1, 4, 16), (True, 4, 16), (4., 4, 16), (3, 4, 16)])
def test_invalid_sampling_grids_are_rejected(counts):
    with pytest.raises(ValueError):
        MultirateSteps(*counts)


def test_runner_rejects_a_different_model():
    m = model()
    with pytest.raises(ValueError, match='different model'):
        infer_multirate(m, sample(1), runner=MultirateRunner(model()))
