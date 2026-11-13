"""Census math, original-sampler fidelity and crash-safe shard integrity."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from fastwam.cosmos.jepa_contact_eval import AttentionCapture
from fastwam.cosmos.jepa_force_latent import ActionGuidedVisualReader
from fastwam.cosmos.jepa_query_census import (MeanAttentionCapture, observed_batch,
    infer_repeated_seed, attention_metrics, write_shard, verify_shard)
from fastwam.cosmos.noaction import NoActionJointModel


def test_streaming_mean_retains_every_head_query_and_matches_full_capture():
    torch.manual_seed(12)
    reader = ActionGuidedVisualReader(feature_dim=12, action_dim=8, state_dim=8,
        history_steps=32, text_dim=24, force_hidden=16, width=16, query_count=8,
        depth=2, heads=4).eval()
    observation = reader.prepare_observation(torch.randn(2,392,12), torch.ones(2,392,dtype=torch.bool),
        torch.randn(2,32,14), torch.ones(2,32,14,dtype=torch.bool), torch.randn(2,9), torch.randn(2,3,24))
    with torch.no_grad(), AttentionCapture(reader) as full, MeanAttentionCapture(reader, 3) as streaming:
        for step in range(3):
            reader(torch.randn(2,32,8), torch.full((2,), float(1000-step*300)), observation)
        actual = streaming.mean()
    expected = torch.stack([r['attention'] for r in full.records]).reshape(3,2,2,4,8,392).mean(0).transpose(0,1).numpy()
    np.testing.assert_allclose(actual, expected, atol=1e-7, rtol=1e-6)
    assert actual.shape == (2,2,4,8,392)
    with MeanAttentionCapture(reader, 3) as incomplete:
        with pytest.raises(ValueError, match='Incomplete'):
            incomplete.mean()


class Scheduler:
    def build_inference_schedule(self, steps, device, dtype):
        clocks = torch.linspace(1000, 0, steps+1, device=device, dtype=dtype)
        return clocks[:-1], torch.full((steps,), 1/steps, device=device, dtype=dtype)
    def step(self, prediction, dt, value):
        return value - prediction * dt


class SamplerHarness(nn.Module):
    """Original infer_joint calls against deterministic lightweight core operators."""
    infer_joint = NoActionJointModel.infer_joint
    device = torch.device('cpu')
    torch_dtype = torch.float32
    action_dim = 8
    num_inference_steps = 16
    def __init__(self):
        super().__init__()
        self.v_scheduler = self.f_scheduler = self.a_scheduler = Scheduler()
    def _conditions(self, sample):
        assert not any(k in sample for k in ('action','wrench_future','video_latents'))
        return sample['proprio']
    def prefill_static(self, first, conditions):
        return dict(first=first, conditions=conditions)
    def joint_denoise_core(self, video, force, action, tv, tf, ta, conditions, cache):
        value = conditions[:, :1, None]
        return video*.11, force*.22+value, action*.33+value


def source_sample(value):
    return dict(first_frame_latents=torch.full((16,1,4,4), value),
        history=torch.zeros(32,14), history_is_pad=torch.zeros(32,dtype=torch.bool),
        history_channel_valid=torch.ones(32,14,dtype=torch.bool), cosmos_text_index=torch.tensor(0),
        jepa_observation=torch.zeros(2,3,2,224,224), proprio=torch.full((32,9),value),
        action=torch.full((32,8),float('nan')), wrench_future=torch.full((32,6),float('nan')))


def test_repeated_seed_batch_matches_original_sampler_every_sample():
    model = SamplerHarness().eval()
    samples = [source_sample(.1), source_sample(.7), source_sample(-.2)]
    batched = infer_repeated_seed(model, observed_batch(samples), seed=113)
    for index, sample in enumerate(samples):
        original = model.infer_joint(observed_batch([sample]), seed=113,force_seed=114,video_seed=115)
        for key in original:
            torch.testing.assert_close(batched[key][index:index+1], original[key], atol=0,rtol=0)
    assert observed_batch(samples)['proprio'].shape == (3,9)


def test_uniform_attention_metrics_have_expected_concentration_and_redundancy():
    metrics = attention_metrics(np.full((2,4,8,392), 1/392, dtype=np.float32))
    assert metrics['entropy_normalized'] == pytest.approx(1., abs=1e-6)
    assert metrics['camera_main_mass'] == pytest.approx(.5)
    assert metrics['query_cosine_similarity'] == pytest.approx(1., abs=1e-6)
    assert metrics['peak_probability'] == pytest.approx(1/392)


def test_shard_atomic_identity_content_and_contiguous_index_verification(tmp_path):
    rows = [dict(sample_index=i,episode_index=0,frame_index=i,segment_start=0,segment_stop=40,
        observed_frame_indices=[0,i]) for i in range(2)]
    maps = np.full((2,2,4,8,392),1/392,dtype=np.float16)
    receipt = write_shard(tmp_path,0,rows,maps,[dict(sample_index=i) for i in range(2)],
        {'thumbnails/000000-main.jpg':b'placeholder'},np.zeros((2,32,8)),np.zeros((2,32,6)),'identity')
    result = verify_shard(tmp_path/receipt['path'],0,2,'identity')
    assert result['sha256'] == receipt['sha256']
    (tmp_path/'shard_000000_000002.json').unlink()  # crash after tar publication is recoverable
    assert verify_shard(tmp_path/receipt['path'],0,2,'identity')['count'] == 2
    with pytest.raises(ValueError, match='lineage'):
        verify_shard(tmp_path/receipt['path'],0,2,'different')
    with pytest.raises(FileExistsError):
        write_shard(tmp_path,0,rows,maps,[],{},[],[],'identity')


def test_event_allows_terminal_status(tmp_path):
    spec = importlib.util.spec_from_file_location('census_runner',Path(__file__).resolve().parents[2]/'scripts/cosmos/evaluate_jepa_query_census.py')
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    runner.event(tmp_path,'complete',status='complete',completed=84349)
    assert json.loads((tmp_path/'progress.json').read_text())['status'] == 'complete'
