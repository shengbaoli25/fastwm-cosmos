"""Evaluation-only checks: no target leakage, deterministic inventory, paired masks."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import torch

SPEC = importlib.util.spec_from_file_location('contact_runner', Path(__file__).resolve().parents[2] / 'scripts/cosmos/evaluate_jepa_contact.py')
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


def test_selector_preserves_valid_continuous_segments_and_never_reads_rgb():
    rows = [(ep, f, 10, 41) for ep in range(78) for f in range(10, 41)]
    selected = runner.select_anchors(rows)
    assert len(selected) == 36
    assert sorted({r['episode_index'] for r in selected}) == [0, 7, 14, 21, 28, 35, 42, 49, 56, 63, 70, 77]
    assert [r['frame_index'] for r in selected[:3]] == [16, 25, 34]
    assert all(r['segment_start'] <= r['observed_frame_indices'][0] <= r['frame_index'] for r in selected)


def test_observed_input_removes_future_targets_and_future_proprio():
    sample = {k: torch.ones(2) for k in runner.OBSERVED_KEYS}
    sample.update(proprio=torch.arange(32 * 9).reshape(32, 9),
                  action=torch.full((32, 8), float('nan')),
                  wrench_future=torch.full((32, 6), float('nan')),
                  video_latents=torch.full((16, 3, 4, 4), float('nan')))
    observed = runner.observed_input(sample)
    assert set(observed) == set(runner.OBSERVED_KEYS) | {'proprio'}
    assert observed['proprio'].shape == (1, 9)
    torch.testing.assert_close(observed['proprio'][0], sample['proprio'][0])


def test_intervention_selection_is_one_per_episode_and_predefined_phase():
    rows = runner.select_anchors([(ep, f, 0, 30) for ep in range(78) for f in range(30)])
    positions = runner.intervention_positions(rows, 12)
    selected = [rows[index] for index in positions]
    assert len({row['episode_index'] for row in selected}) == 12
    assert [row['phase'] for row in selected] == ['early', 'late'] * 6
    assert runner.intervention_positions(rows, 0) == []
    assert len(runner.intervention_positions(rows, 100)) == 12


def test_roi_union_fraction_area_and_rejects_malformed_rectangles():
    grid = runner.roi_grid(dict(status='visible', boxes=[[0, 0, .5, 1], [.25, 0, .75, 1]]))
    assert grid.mean() == pytest.approx(.75)
    with pytest.raises(ValueError):
        runner.roi_grid(dict(status='visible', boxes=[[.8, 0, .2, 1]]))
    assert not runner.roi_grid(dict(status='occluded', boxes=[])).any()


def test_matched_background_counts_and_two_observed_frame_occlusion():
    roi = np.zeros((2, 14, 14), dtype=bool)
    roi[0, 3:6, 4:8] = True
    roi[1, 2:4, 5:8] = True
    bg = runner.matching_random_mask(roi, np.random.default_rng(4), forbidden=roi)
    assert not (bg & roi).any()
    assert np.array_equal(bg.sum(axis=(1, 2)), roi.sum(axis=(1, 2)))
    rgb = torch.ones(1, 2, 3, 2, 224, 224)
    altered = runner.occlude_observation(rgb, roi)
    assert (altered == 0).sum() == int(roi.sum()) * 16 * 16 * 3 * 2
    torch.testing.assert_close(altered[:, :, :, 0], altered[:, :, :, 1])
    assert rgb.min() == 1


def test_translated_background_preserves_exact_topology_and_excludes_contact():
    roi = np.zeros((2, 14, 14), dtype=bool)
    roi[0, 4:6, 6:9] = True
    roi[0, 6, 6] = True
    roi[1, 1:3, 1:3] = True
    background = runner.matching_translated_mask(roi, np.random.default_rng(4))
    assert not (background & roi).any()
    for view in range(2):
        source_points = np.argwhere(roi[view])
        target_points = np.argwhere(background[view])
        assert np.array_equal(source_points - source_points.min(0), target_points - target_points.min(0))
    with pytest.raises(ValueError, match='No disjoint same-shape'):
        runner.matching_translated_mask(np.ones((2, 14, 14), dtype=bool), np.random.default_rng(4))


def test_target_errors_honor_padding_and_missing_wrench():
    sample = dict(action=torch.zeros(32, 8), wrench_future=torch.zeros(32, 6),
                  action_is_pad=torch.ones(32, dtype=torch.bool), force_is_pad=torch.zeros(32, dtype=torch.bool),
                  wrench_valid=torch.zeros(32, 6, dtype=torch.bool))
    sample['action_is_pad'][0] = False
    sample['wrench_valid'][0, 0] = True
    prediction = dict(action=torch.full((1, 32, 8), 999.), force=torch.full((1, 32, 6), 999.))
    prediction['action'][:, 0] = 2
    prediction['force'][:, 0, 0] = 3
    assert runner.target_errors(prediction, sample) == dict(action_mse_normalized=4., force_mse_normalized=9.)


def test_summarize_uniform_attention_is_unit_enrichment(tmp_path):
    (tmp_path / 'attention').mkdir()
    rows = [dict(sample_index=i, episode_index=i, frame_index=4, phase='middle') for i in range(2)]
    (tmp_path / 'samples.json').write_text(json.dumps(dict(samples=rows)))
    for row in rows:
        np.savez(tmp_path / 'attention' / f'{row["sample_index"]:06d}.npz', attention=np.full((3, 2, 4, 8, 392), 1/392))
    roi = dict(schema='jepa-contact-roi-v1', annotation_stage='raw_rgb_before_attention', samples={
        str(i): dict(cam_main=dict(status='visible', boxes=[[0, 0, .1, .1]])) for i in range(2)})
    path = tmp_path / 'roi.json'
    path.write_text(json.dumps(roi))
    report = runner.summarize(tmp_path, path)
    assert report['contact_cluster_ci']['mean'] == pytest.approx(1.)
    assert report['contact_cluster_ci']['n_clusters'] == 2
    assert len(report['contact_records']) == 4
