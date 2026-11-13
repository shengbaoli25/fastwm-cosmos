"""CPU checks for passive reader attention and non-pseudoreplicated metrics."""
import math

import pytest
import torch
from torch.nn import functional as F

from fastwam.cosmos.jepa_force_latent import ActionGuidedVisualReader
from fastwam.cosmos.jepa_contact_eval import (
    AttentionCapture, attention_roi_metrics, cluster_bootstrap,
    paired_perturbation_metrics, token_layout,
)


def reader_inputs(dtype=torch.float32):
    torch.manual_seed(922)
    reader = ActionGuidedVisualReader(feature_dim=12, action_dim=8, state_dim=8,
        history_steps=4, text_dim=6, force_hidden=16, width=16, query_count=3,
        depth=2, heads=4, horizon=5).to(dtype=dtype).eval()
    valid = torch.ones(2, 8, dtype=torch.bool)
    valid[0, -2:] = False
    normal = lambda *s: torch.randn(*s).to(dtype=dtype)
    history_valid = torch.ones(2, 4, 14, dtype=torch.bool)
    observation = reader.prepare_observation(normal(2, 8, 12), valid,
        normal(2, 4, 14), history_valid, normal(2, 9), normal(2, 3, 6))
    return reader, normal(2, 5, 8), torch.tensor([250., 750.], dtype=dtype), observation


@pytest.mark.parametrize('dtype', [torch.float32, torch.bfloat16])
def test_capture_is_passive_and_all_weights_have_exact_key_provenance(dtype):
    reader, action, clock, observation = reader_inputs(dtype)
    with torch.no_grad():
        expected = reader(action, clock, observation)
        with AttentionCapture(reader) as capture:
            actual = reader(action, clock, observation)
            reader(action, clock / 2, observation)
        for name in expected:
            torch.testing.assert_close(actual[name], expected[name], atol=0, rtol=0)
    assert len(capture.records) == 4  # 2 denoising calls x 2 layers; no horizon reader
    for index, record in enumerate(capture.records):
        assert record['call_index'] == index // 2
        assert record['layer_index'] == index % 2
        attention = record['attention']
        assert attention.shape == (2, 4, 3, 8)
        assert attention.dtype == torch.float32 and not attention.requires_grad
        assert attention.device.type == 'cpu'
        assert attention[0, :, :, -2:].count_nonzero() == 0
        torch.testing.assert_close(attention.sum(-1), torch.ones(2, 4, 3))
        torch.testing.assert_close(record['valid'], observation['valid'])
        torch.testing.assert_close(record['action_clock'], clock.float() / (1 if index < 2 else 2))
    assert not reader._forward_pre_hooks
    assert all(not layer._forward_hooks and not layer.query._forward_hooks for layer in reader.readers)


def test_weights_reproduce_sdpa_value_read_and_corresponding_mha_weights():
    reader, action, clock, observation = reader_inputs()
    actual_args = []
    layer = reader.readers[0]
    handle = layer.register_forward_pre_hook(lambda module, args: actual_args.append(args))
    try:
        with torch.no_grad(), AttentionCapture(reader) as capture:
            reader(action, clock, observation)
    finally:
        handle.remove()
    query, (key, value), valid = actual_args[0]
    q = layer.query(layer.query_norm(query)).unflatten(-1, (4, 4)).transpose(1, 2)
    expected = F.scaled_dot_product_attention(q, key, value, attn_mask=valid[:, None, None, :])
    weights = capture.records[0]['attention']
    torch.testing.assert_close(weights @ value, expected, atol=2e-7, rtol=1e-5)
    # MHA configured with identity projections receives these exact projected Q/K/V.
    mha = torch.nn.MultiheadAttention(16, 4, dropout=0., batch_first=True)
    with torch.no_grad():
        mha.in_proj_weight.copy_(torch.eye(16).repeat(3, 1))
        mha.in_proj_bias.zero_()
        mha.out_proj.weight.copy_(torch.eye(16))
        mha.out_proj.bias.zero_()
    flatten = lambda x: x.transpose(1, 2).flatten(2)
    _, mha_weights = mha(flatten(q), flatten(key), flatten(value),
        key_padding_mask=~valid, need_weights=True, average_attn_weights=False)
    torch.testing.assert_close(weights, mha_weights, atol=1e-7, rtol=1e-5)


def test_capture_cleans_up_after_failure_and_rejects_training():
    reader, action, clock, observation = reader_inputs()
    reader.train()
    with pytest.raises(ValueError, match='evaluation-only'):
        with AttentionCapture(reader):
            pass
    reader.eval()
    with pytest.raises(RuntimeError, match='record limit'):
        with AttentionCapture(reader, max_records=1):
            reader(action, clock, observation)
    assert not reader._forward_pre_hooks
    assert all(not layer._forward_hooks and not layer.query._forward_hooks for layer in reader.readers)


def test_layout_maps_views_pixels_and_joint_observed_time():
    layout = token_layout(observed_frame_indices=[7, 11])
    assert layout['grid_shape'] == [2, 14, 14]
    assert not layout['frame_specific_attribution_available']
    assert layout['observed_frame_indices'] == [7, 11]
    assert layout['tokens'][0] == dict(token_index=0, camera_index=0, camera='cam_main',
        row=0, col=0, pixel_box_xyxy=[0, 0, 16, 16])
    assert layout['tokens'][195]['pixel_box_xyxy'] == [208, 208, 224, 224]
    assert layout['tokens'][196]['camera'] == 'cam_wrist'
    assert layout['tokens'][196]['pixel_box_xyxy'] == [0, 0, 16, 16]
    assert token_layout(observed_frame_indices=[7, 7])['observed_frame_indices'] == [7, 7]
    with pytest.raises(ValueError):
        token_layout(393)
    with pytest.raises(ValueError):
        token_layout(observed_frame_indices=[8, 7])


def test_uniform_attention_enrichment_is_one_and_masked_roi_area_is_correct():
    attention = torch.tensor([[[[.25, .25, .25, .25, 0., 0.]]]])
    valid = torch.tensor([[True, True, True, True, False, False]])
    roi = torch.tensor([True, False, False, False, True, False])
    metrics = attention_roi_metrics(attention, roi, valid)
    assert metrics['roi_mass'].item() == .25
    assert metrics['roi_area_fraction'].item() == .25
    assert metrics['roi_enrichment'].item() == 1.
    assert metrics['pointing_hit'].item() == 1.
    assert metrics['normalized_entropy'].item() == pytest.approx(1.)


def test_metrics_preserve_heads_queries_and_batch_masks():
    attention = torch.tensor([[[[.8, .2, 0., 0.], [.1, .9, 0., 0.]]],
                              [[[0., 0., .25, .75], [0., 0., .6, .4]]]])
    valid = torch.tensor([[True, True, False, False], [False, False, True, True]])
    roi = torch.tensor([[True, False, False, False], [False, False, False, True]])
    metrics = attention_roi_metrics(attention, roi, valid)
    torch.testing.assert_close(metrics['roi_mass'], torch.tensor([[[.8, .1]], [[.75, .4]]]))
    torch.testing.assert_close(metrics['roi_enrichment'], 2 * metrics['roi_mass'])
    torch.testing.assert_close(metrics['pointing_hit'], torch.tensor([[[1., 0.]], [[1., 0.]]]))


def test_empty_roi_is_missing_and_single_valid_key_has_zero_entropy():
    metrics = attention_roi_metrics(torch.tensor([0., 1., 0.]), torch.zeros(3, dtype=torch.bool),
        torch.tensor([False, True, False]))
    assert not metrics['roi_available'] and torch.isnan(metrics['roi_enrichment'])
    assert metrics['normalized_entropy'] == 0
    with pytest.raises(ValueError, match='sum to one'):
        attention_roi_metrics(torch.tensor([.3, .3]), torch.tensor([True, False]))
    with pytest.raises(ValueError, match='boolean'):
        attention_roi_metrics(torch.tensor([.5, .5]), torch.tensor([1., 0.]))


def test_paired_metrics_mask_padding_and_distinguish_change_from_error():
    clean = torch.tensor([[1., 2.], [float('nan'), float('nan')]])
    altered = torch.tensor([[2., 2.], [float('nan'), float('nan')]])
    target = torch.tensor([[2., 2.], [float('nan'), float('nan')]])
    result = paired_perturbation_metrics(clean, altered, target, torch.tensor([True, False]))
    assert result['prediction_change_mse'] == .5
    assert result['prediction_change_mae'] == .5
    assert result['delta_mse'] == -.5  # change can improve instead of worsening
    assert result['valid_component_count'] == 2


def test_bootstrap_equal_weights_episodes_not_correlated_frames():
    records = [dict(episode_index=0, effect=0.)] * 100 + [dict(episode_index=1, effect=2.)]
    result = cluster_bootstrap(records, 'effect', seed=47)
    assert result['mean'] == 1.  # not 2 / 101
    assert result['n_clusters'] == 2 and result['n_records'] == 101
    assert result == cluster_bootstrap(records, 'effect', seed=47)
    assert result['ci_low'] <= result['mean'] <= result['ci_high']


def test_bootstrap_reports_missing_values_and_does_not_fake_single_episode_ci():
    records = [dict(episode_index=2, effect=3.), dict(episode_index=3, effect=math.nan)]
    result = cluster_bootstrap(records, 'effect')
    assert result['mean'] == 3 and result['excluded_records'] == 1
    assert result['ci_low'] is None and result['ci_high'] is None
    assert cluster_bootstrap([], 'effect')['mean'] is None
    with pytest.raises(ValueError, match='cluster ID'):
        cluster_bootstrap([dict(effect=1.)], 'effect')
