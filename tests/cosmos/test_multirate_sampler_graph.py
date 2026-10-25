"""Whole-sampler graph correctness on tiny native networks, never live hardware.

CUDA coverage verifies real capture/replay, request refresh, and storage lifetime.
These tests make no trained-model quality or production latency claims.
"""
import copy
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from conftest import sample
from test_noaction import model
from test_multirate import full_model_reference
from fastwam.cosmos import multirate
from fastwam.cosmos.multirate import MultirateRunner, MultirateSteps


CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA device required for real graph replay')


def assert_cuda_outputs(actual, expected):
    assert actual.keys() == expected.keys()
    for name in actual:
        torch.testing.assert_close(actual[name], expected[name], atol=5e-6, rtol=5e-5, msg=name)


@CUDA
@pytest.mark.parametrize('counts', [(4, 4, 16), (8, 4, 16), (16, 16, 16)])
def test_whole_sampler_graph_matches_reference_and_refreshes_request(counts):
    m = model().cuda()
    s = sample(1)
    s['history_is_pad'][:, :3] = True
    s['history_channel_valid'][:, 7:10, :2] = False
    options = dict(zip(('video_steps', 'force_steps', 'action_steps'), counts))
    runner = MultirateRunner(m, mode='cuda_graph')
    try:
        result = runner.infer(s, **options)
        assert_cuda_outputs(result, full_model_reference(m, s, counts))
        saved = {name: value.clone() for name, value in result.items()}
        status = runner.diagnostics
        assert status['effective_mode'] == 'cuda_graph'
        assert status['graph_scope'] == 'sampler'
        assert status['sampler_captures'] == 1
        assert status['sampler_replays'] == 1
        assert set(runner._graphs) == {'sampler'}

        changed = copy.deepcopy(s)
        changed['first_frame_latents'] += .2
        changed['history'] += .7
        changed['cosmos_context'] += .3
        changed['proprio'][:, 0] += .5
        changed['history_is_pad'][:, :5] = True
        changed['history_channel_valid'][:, 9:13, :3] = False
        seeds = dict(seed=103, video_seed=104, force_seed=105)
        expected = full_model_reference(m, changed, counts, **seeds)
        # A cache hit must replay the captured sampler, without invoking any
        # Python expert loop or request-specific prefill again.
        with patch.object(multirate, '_sample_core', side_effect=AssertionError('Python sampler on replay')):
            refreshed = runner.infer(changed, **options, **seeds)
        assert_cuda_outputs(refreshed, expected)
        assert not torch.allclose(refreshed['action'], result['action'])
        for name in result:
            torch.testing.assert_close(result[name], saved[name], rtol=0, atol=0)
            assert result[name].data_ptr() != refreshed[name].data_ptr()
        torch.testing.assert_close(refreshed['video_latents'][:, :, :1],
                                   changed['first_frame_latents'].cuda(), rtol=0, atol=0)
        assert runner.diagnostics['sampler_captures'] == 1
        assert runner.diagnostics['sampler_replays'] == 2
    finally:
        runner.close()


@CUDA
def test_whole_sampler_graph_invalidates_for_profile_shape_and_weight_changes():
    m = model().cuda()
    runner = MultirateRunner(m, mode='cuda_graph')
    try:
        runner.infer(sample(1))
        first_graph = runner._graphs['sampler'][1]
        result = runner.infer(sample(1), video_steps=8, force_steps=4, action_steps=16)
        second_graph = runner._graphs['sampler'][1]
        assert second_graph is not first_graph
        assert_cuda_outputs(result, full_model_reference(m, sample(1), (8, 4, 16)))
        assert runner.diagnostics['sampler_captures'] == 2

        result = runner.infer(sample(2), video_steps=8, force_steps=4, action_steps=16)
        third_graph = runner._graphs['sampler'][1]
        assert third_graph is not second_graph
        assert_cuda_outputs(result, full_model_reference(m, sample(2), (8, 4, 16)))
        assert runner.diagnostics['sampler_captures'] == 3
        assert set(runner._graphs) == {'sampler'}

        with torch.no_grad():
            next(m.parameters()).add_(.001)
        result = runner.infer(sample(2), video_steps=8, force_steps=4, action_steps=16)
        assert runner._graphs['sampler'][1] is not third_graph
        assert_cuda_outputs(result, full_model_reference(m, sample(2), (8, 4, 16)))
        assert not runner.diagnostics['fallback_reasons']
    finally:
        runner.close()


@CUDA
def test_compiled_experts_inside_whole_sampler_graph_match_reference(monkeypatch):
    # Exercise actual Inductor kernels inside the outer graph, with inner
    # cudagraphs disabled. Keep compilation CPU and GPU footprint bounded.
    import torch._inductor.config
    monkeypatch.setattr(torch._inductor.config, 'compile_threads', 2)
    m = model().cuda()
    s = sample(1)
    m.dit['action'].eval()
    original_modes = [module.training for module in m.modules()]
    cpu_rng = torch.get_rng_state().clone()
    cuda_rng = torch.cuda.get_rng_state().clone()
    runner = MultirateRunner(m, mode='compile_cuda_graph')
    try:
        result = runner.infer(s)
        assert original_modes == [module.training for module in m.modules()]
        assert torch.equal(cpu_rng, torch.get_rng_state())
        assert torch.equal(cuda_rng, torch.cuda.get_rng_state())
        assert_cuda_outputs(result, full_model_reference(m, s, (4, 4, 16)))
        assert runner.diagnostics['effective_mode'] == 'compile_cuda_graph'
        assert runner.diagnostics['branch_modes'] == dict.fromkeys(('video', 'force', 'action'), 'compile')
        assert runner.diagnostics['graph_scope'] == 'sampler'
        assert not runner.diagnostics['fallback_reasons']
        saved = {name: value.clone() for name, value in result.items()}
        changed = copy.deepcopy(s)
        changed['first_frame_latents'] += .5
        changed['history'] += .3
        with patch.object(multirate, '_sample_core', side_effect=AssertionError('Python sampler on replay')):
            refreshed = runner.infer(changed, seed=143)
        assert_cuda_outputs(refreshed, full_model_reference(m, changed, (4, 4, 16), seed=143))
        for name in result:
            torch.testing.assert_close(result[name], saved[name], rtol=0, atol=0)
        assert runner.diagnostics['sampler_captures'] == 1
        assert runner.diagnostics['sampler_replays'] == 2
    finally:
        runner.close()


def test_combined_mode_disables_inner_compiler_cuda_graphs(monkeypatch):
    compiled_options = []
    def compile_stub(function, **options):
        compiled_options.append(options)
        return function
    monkeypatch.setattr(torch, 'compile', compile_stub)
    monkeypatch.setitem(multirate._BRANCHES, 'action', lambda m, value: value + 1)
    runner = MultirateRunner(SimpleNamespace(device=torch.device('cuda:2')), mode='compile_cuda_graph')
    first = runner._call('action', torch.tensor([2.]))
    second = runner._call('action', torch.tensor([4.]))
    torch.testing.assert_close(first, torch.tensor([3.]))
    torch.testing.assert_close(second, torch.tensor([5.]))
    assert len(compiled_options) == 1
    assert compiled_options[0]['fullgraph'] is True
    assert compiled_options[0]['options']['triton.cudagraphs'] is False


def test_cuda_graph_mode_has_no_per_expert_capture(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError('Graph capture belongs at the complete sampler boundary')
    monkeypatch.setattr(multirate, '_CudaGraphCall', forbidden)
    monkeypatch.setitem(multirate._BRANCHES, 'action', lambda m, value: value + 1)
    runner = MultirateRunner(SimpleNamespace(device=torch.device('cuda:2')), mode='cuda_graph')
    torch.testing.assert_close(runner._call('action', torch.tensor([2.])), torch.tensor([3.]))
    assert not runner._graphs


@pytest.mark.parametrize('mode', ['cuda_graph', 'compile_cuda_graph'])
def test_whole_sampler_capture_failure_falls_back_once(monkeypatch, mode):
    attempts = []
    def unsupported(*args, **kwargs):
        attempts.append(1)
        raise RuntimeError('capture unsupported in this runtime')
    monkeypatch.setattr(multirate, '_CudaGraphCall', unsupported)
    monkeypatch.setattr(multirate, '_sample_core', lambda *args, **kwargs: {'action': torch.tensor([7.])})
    runner = MultirateRunner(SimpleNamespace(device=torch.device('cuda:2')), mode=mode)
    args = (torch.ones(1), (), (), (), MultirateSteps())
    with pytest.warns(RuntimeWarning, match='using uncaptured sampling'):
        result = runner._sample(*args)
    torch.testing.assert_close(result['action'], torch.tensor([7.]))
    torch.testing.assert_close(runner._sample(*args)['action'], torch.tensor([7.]))
    assert len(attempts) == 1
    assert not runner._graphs
    assert runner.diagnostics['graph_scope'] == 'none'
    assert 'unsupported' in runner.diagnostics['fallback_reasons']['sampler']


@pytest.mark.parametrize('error', [torch.OutOfMemoryError('out of memory'),
                                  RuntimeError('CUDA illegal memory access'),
                                  RuntimeError('CUDA invalid device ordinal'),
                                  RuntimeError('GPU has fallen off the bus')])
def test_whole_sampler_fatal_device_error_does_not_fallback(monkeypatch, error):
    def broken(*args, **kwargs):
        raise error
    monkeypatch.setattr(multirate, '_CudaGraphCall', broken)
    runner = MultirateRunner(SimpleNamespace(device=torch.device('cuda:2')), mode='cuda_graph')
    with pytest.raises(type(error), match=str(error)):
        runner._sample(torch.ones(1), (), (), (), MultirateSteps())
    assert not runner._disabled
    assert not runner.diagnostics['fallback_reasons']


def test_graph_replay_orders_stream_event_copies_and_output_clones_on_cpu(monkeypatch):
    # No CUDA runtime is used: model the asynchronous submission boundary with
    # CPU tensors so a second caller's stream cannot overwrite pending outputs.
    events = []
    entered = []
    device = torch.device('cuda:2')
    streams = [SimpleNamespace(name='first'), SimpleNamespace(name='second')]
    selected = [streams[0]]
    current = lambda: selected[0].name
    event = SimpleNamespace()

    @contextmanager
    def device_context(requested):
        assert requested == device
        entered.append(requested)
        events.append(('enter', current()))
        try:
            yield
        finally:
            events.append(('exit', current()))
            entered.pop()

    def current_stream(requested):
        assert requested == device and entered == [device]
        events.append(('current', current()))
        return selected[0]

    def wait_event(received):
        assert entered == [device] and received is event
        events.append(('wait', current()))

    def record(stream):
        assert entered == [device] and stream is selected[0]
        events.append(('record', current()))

    for stream in streams:
        stream.wait_event = wait_event
    event.record = record
    original_copy, original_clone = torch.Tensor.copy_, torch.Tensor.clone
    inputs = [torch.zeros(2), torch.zeros(2)]
    output_tensors = [torch.zeros(2), torch.zeros(2)]
    def copy_(target, source, *args, **kwargs):
        assert entered == [device]
        events.append(('copy', current()))
        return original_copy(target, source, *args, **kwargs)
    def clone(value, *args, **kwargs):
        assert entered == [device]
        events.append(('clone', current()))
        return original_clone(value, *args, **kwargs)
    def replay():
        assert entered == [device]
        events.append(('replay', current()))
        original_copy(output_tensors[0], inputs[0] + inputs[1])
        original_copy(output_tensors[1], inputs[0] - inputs[1])

    graph = multirate._CudaGraphCall.__new__(multirate._CudaGraphCall)
    graph.device, graph.last_use = device, event
    graph.inputs, graph.spec = multirate.tree_flatten((inputs[0], {'history': inputs[1]}))
    graph.outputs, graph.output_spec = multirate.tree_flatten(
        {'action': output_tensors[0], 'nested': (output_tensors[1], 'preserved')})
    graph.graph = SimpleNamespace(replay=replay)
    with monkeypatch.context() as mocked:
        mocked.setattr(torch.cuda, 'device', device_context)
        mocked.setattr(torch.cuda, 'current_stream', current_stream)
        mocked.setattr(torch.Tensor, 'copy_', copy_)
        mocked.setattr(torch.Tensor, 'clone', clone)
        first = graph(torch.tensor([2., 4.]), {'history': torch.tensor([1., 3.])})
        selected[0] = streams[1]
        second = graph(torch.tensor([7., 9.]), {'history': torch.tensor([2., 5.])})
    ordering = ['enter', 'current', 'wait', 'copy', 'copy', 'replay', 'clone', 'clone', 'record', 'exit']
    assert events == [(operation, name) for name in ('first', 'second') for operation in ordering]
    assert not entered
    torch.testing.assert_close(first['action'], torch.tensor([3., 7.]))
    torch.testing.assert_close(first['nested'][0], torch.tensor([1., 1.]))
    torch.testing.assert_close(second['action'], torch.tensor([9., 14.]))
    assert first['nested'][1] == second['nested'][1] == 'preserved'


@CUDA
def test_real_graph_replay_across_two_streams_preserves_each_output():
    # Hardware validation for the CPU submission-order regression above.
    device = torch.device('cuda', torch.cuda.current_device())
    first_input = torch.arange(4096, device=device, dtype=torch.float32) / 4096
    second_input = first_input + 2
    streams = [torch.cuda.Stream(device=device), torch.cuda.Stream(device=device)]
    function = lambda value: {'action': value.square() + 3, 'force': value * 2}
    graph = multirate._CudaGraphCall(function, (first_input,), device)
    ready = torch.cuda.Event()
    ready.record(torch.cuda.current_stream(device))
    for stream in streams:
        stream.wait_event(ready)
    results = []
    for _ in range(8):
        with torch.cuda.stream(streams[0]):
            results.append((first_input, graph(first_input)))
        with torch.cuda.stream(streams[1]):
            results.append((second_input, graph(second_input)))
    for stream in streams:
        stream.synchronize()
    for value, actual in results:
        for name, expected in function(value).items():
            torch.testing.assert_close(actual[name], expected, rtol=0, atol=0)


@CUDA
@pytest.mark.parametrize('mode', ['cuda_graph', 'compile_cuda_graph'])
def test_bfloat16_with_preserved_fp32_weights_graph_matches_eager(mode, monkeypatch):
    # Same BF16 conversion and native FP32 weight preservation as deployment.
    # Run after device recovery; this test has not been validated on lost GPU.
    import torch._inductor.config
    if not torch.cuda.is_bf16_supported():
        pytest.skip('BF16-capable CUDA hardware required')
    monkeypatch.setattr(torch._inductor.config, 'compile_threads', 2)
    m = model().to(device='cuda', dtype=torch.bfloat16)
    assert {value.dtype for value in m.parameters()} == {torch.float32, torch.bfloat16}
    s = sample(1)
    expected = MultirateRunner(m, mode='eager').infer(s)
    runner = MultirateRunner(m, mode=mode)
    try:
        actual = runner.infer(s)
        assert runner.diagnostics['graph_scope'] == 'sampler'
        assert not runner.diagnostics['fallback_reasons']
        for name in expected:
            assert torch.isfinite(actual[name]).all()
            torch.testing.assert_close(actual[name], expected[name], atol=.03, rtol=.03, msg=name)
    finally:
        runner.close()
