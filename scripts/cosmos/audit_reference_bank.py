"""Audit sampled reference augmentation on CPU; does not assess policy success.

Checks every episode, several fixed anchors, and multiple epoch realizations.
Write the report outside the immutable bank directory.
"""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'src'))

import numpy as np

from fastwam.cosmos.reference_bank import ReferenceBank, atomic_json, file_digest
from fastwam.cosmos.reference_trajectory import AugmentationConfig


def distribution(values):
    a = np.asarray(values, dtype=np.float64)
    if not len(a):
        return dict(count=0)
    return dict(count=len(a), min=float(a.min()), median=float(np.median(a)),
                mean=float(a.mean()), p95=float(np.quantile(a, .95)), max=float(a.max()))


def audit(bank_path, *, epochs=8, anchors_per_episode=3, seed=42, config=None):
    if epochs < 1 or anchors_per_episode < 1 or seed < 0:
        raise ValueError('epochs/anchors must be positive and seed nonnegative')
    config = AugmentationConfig() if config is None else config
    bank = ReferenceBank(bank_path)
    manifest = bank.manifest
    checks = {}

    def require(condition, name, context):
        if not condition:
            raise ValueError(f'{name} failed ({context})')
        checks[name] = checks.get(name, 0)+1

    input_keys = {'reference_pose', 'reference_valid', 'reference_channel_valid',
        'reference_order', 'reference_break_before', 'reference_present', 'current_tcp'}
    plan_lengths, base_lengths, duplicate_counts = [], [], []
    positive_gaps, masked_fractions, position_errors, rotation_errors = [], [], [], []
    dropped_plans = nonuniform_plans = enabled_samples = sample_count = 0
    reports = []
    for entry in manifest['episodes']:
        ep = entry['episode_index']
        with np.load(bank.root/entry['path'], allow_pickle=False) as z:
            all_anchors = z['valid_anchor_frame'].copy()
            observed = z['observation_tcp'].copy()
            base_frames = z['source_frame'].copy()
            base_ids = z['segment_id'].copy()
            bounds = z['segment_bounds'].copy()
        positions = np.unique(np.linspace(0, len(all_anchors)-1,
                                         min(anchors_per_episode, len(all_anchors)), dtype=np.int64))
        chosen_frames = all_anchors[positions]
        base_lengths.append(len(base_frames))
        lengths, drops = [], []
        for epoch in range(epochs):
            shared_plan = None
            for frame, observed_index in zip(chosen_frames, positions):
                context = f'episode={ep}, epoch={epoch}, frame={int(frame)}'
                sample = bank.sample(ep, int(frame), epoch=epoch, seed=seed, config=config)
                inputs, targets = sample['inputs'], sample['supervision']
                pose, valid = inputs['reference_pose'], inputs['reference_valid']
                frames, ids = targets['source_frame'], targets['segment_id']
                target = targets['alignment_target']
                n = len(frames)
                require(set(inputs) == input_keys, 'supervision_not_exposed_to_model', context)
                require(inputs['current_tcp'].shape == (8,) and np.array_equal(
                    inputs['current_tcp'], observed[observed_index]), 'current_tcp_is_single_causal_row', context)
                require(pose.shape == (n, 8) and valid.shape == (n,) and
                    np.isfinite(pose).all(), 'finite_variable_length_pose', context)
                require(np.allclose(np.linalg.norm(pose[:, 3:7], axis=1), 1, atol=1e-5),
                    'unit_reference_quaternions', context)
                require(np.array_equal(inputs['reference_channel_valid'],
                    np.repeat(valid[:, None], 3, axis=1)), 'channel_masks_match_geometry_mask', context)
                require(bool(inputs['reference_present']) == bool(valid.any()),
                    'presence_matches_valid_geometry', context)
                expected_placeholder = np.zeros((int((~valid).sum()), 8), dtype=np.float32)
                expected_placeholder[:, 6] = 1
                require(np.array_equal(pose[~valid], expected_placeholder),
                    'hidden_geometry_has_explicit_placeholder', context)
                require(np.array_equal(inputs['reference_order'], np.linspace(0, 1, n, dtype=np.float32)),
                    'reference_order_uses_augmented_slots_not_gt_time', context)
                require(np.array_equal(inputs['reference_break_before'], np.r_[True, ids[1:] != ids[:-1]]),
                    'segment_break_mask_preserved', context)
                require(np.array_equal(targets['segment_bounds'], bounds) and
                    np.array_equal(np.unique(ids), np.arange(len(bounds))),
                    'all_continuous_segments_preserved', context)
                for sid, (lo, hi) in enumerate(bounds):
                    local = frames[ids == sid]
                    require(len(local) >= 2 and local[0] == lo and local[-1] == hi-1
                        and (np.diff(local) >= 0).all()
                        and np.isin(local, base_frames[base_ids == sid]).all(),
                        'ordered_source_mapping_within_original_segment', context)
                anchor_sid = int(np.flatnonzero((bounds[:, 0] <= frame) & (frame < bounds[:, 1]))[0])
                enabled = bool(targets['alignment_enabled'])
                require(target.shape == (n,) and np.isfinite(target).all() and (target >= 0).all()
                    and np.isclose(target.sum(), float(enabled), atol=1e-6),
                    'alignment_mass_one_if_enabled_zero_otherwise', context)
                require(not (target[~valid] > 0).any(), 'no_alignment_mass_on_masked_geometry', context)
                require(not (target[ids != anchor_sid] > 0).any(),
                    'alignment_never_crosses_continuous_segment', context)
                require(np.isclose(targets['alignment_target_unmasked'].sum(), 1, atol=1e-6),
                    'diagnostic_unmasked_alignment_normalized', context)
                require(enabled == bool(valid[ids == anchor_sid].any()),
                    'alignment_enabled_only_with_local_geometry', context)
                sample_count += 1
                enabled_samples += int(enabled)
                current_plan = {k: value for k, value in inputs.items() if k != 'current_tcp'}
                if shared_plan is not None:
                    require(all(np.array_equal(current_plan[k], shared_plan[k]) for k in current_plan),
                        'same_episode_epoch_plan_stable_across_observed_anchors', context)
                    continue
                shared_plan = {k: value.copy() for k, value in current_plan.items()}
                # Count each sampled plan once, rather than once per anchor.
                lengths.append(n)
                plan_lengths.append(n)
                dropped = not bool(valid.any())
                drops.append(dropped)
                dropped_plans += int(dropped)
                gaps = np.diff(frames)[ids[1:] == ids[:-1]]
                duplicate_counts.append(int((gaps == 0).sum()))
                positive_gaps.extend(gaps[gaps > 0].tolist())
                nonuniform = any(len(np.unique(np.diff(frames[ids == sid]))) > 1
                                 for sid in range(len(bounds)))
                nonuniform_plans += int(nonuniform)
                masked_fractions.append(float((~valid).mean()))
                # Diagnostic errors are measured BEFORE masked placeholders.
                position_errors.extend(targets['reference_position_error'].tolist())
                rotation_errors.extend(targets['reference_rotation_error'].tolist())
        reports.append(dict(episode_index=ep, source_rows=entry['source_rows'],
            retained_anchors=entry['retained_anchors'], continuous_segments=len(bounds),
            base_reference_points=len(base_frames), checked_anchor_frames=chosen_frames.tolist(),
            augmented_lengths_by_epoch=lengths, dropped_by_epoch=drops,
            distinct_augmented_lengths=len(set(lengths))))
    total_plans = len(plan_lengths)
    return dict(
        schema='realman-reference-bank-cpu-audit-v1', status='passed',
        scope='Sampled data/interface invariants only; no model training, robot rollout, or success-rate evidence',
        bank_identity_sha256=manifest['identity_sha256'],
        source_inventory_sha256=manifest['identity']['source_inventory_sha256'],
        source_inventory_scope=manifest['identity']['source_inventory_scope'],
        current_augmentation_code_sha256=file_digest(ROOT/'src/fastwam/cosmos/reference_trajectory.py'),
        audit_code_sha256=file_digest(__file__),
        sampling=dict(seed=seed, epochs=epochs, anchors_per_episode=anchors_per_episode,
            anchor_selection='uniform retained-anchor positions, including first/middle/last for default 3',
            distinct_plans=total_plans, checked_samples=sample_count,
            augmentation_config=asdict(config)),
        dataset=dict(episodes=len(reports), source_rows=manifest['total_source_rows'],
            retained_anchors=manifest['total_retained_anchors'],
            base_reference_points=manifest['total_reference_points'],
            continuous_segments=sum(r['continuous_segments'] for r in reports)),
        measurements=dict(base_reference_length=distribution(base_lengths),
            augmented_reference_length=distribution(plan_lengths),
            episodes_with_variable_lengths=sum(r['distinct_augmented_lengths'] > 1 for r in reports),
            within_segment_positive_source_frame_gap=distribution(positive_gaps),
            plans_with_nonuniform_within_segment_gaps=nonuniform_plans,
            repeated_slots_per_plan=distribution(duplicate_counts),
            masked_slot_fraction_per_plan=distribution(masked_fractions),
            whole_reference_dropped_plans=dropped_plans,
            whole_reference_dropout_observed_fraction=dropped_plans/total_plans,
            enabled_alignment_samples=enabled_samples,
            disabled_alignment_samples=sample_count-enabled_samples,
            pre_mask_position_perturbation_m=distribution(position_errors),
            pre_mask_rotation_perturbation_rad=distribution(rotation_errors)),
        passed_invariant_check_counts=checks, episodes=reports)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bank', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--epochs', type=int, default=8)
    parser.add_argument('--anchors-per-episode', type=int, default=3)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--augmentation-config', type=Path,
                        help='Optional JSON object of AugmentationConfig keyword arguments')
    args = parser.parse_args()
    if args.output.resolve().is_relative_to(args.bank.resolve()):
        parser.error('Write audit reports outside the immutable reference bank')
    config = None if args.augmentation_config is None else AugmentationConfig(
        **json.loads(args.augmentation_config.read_text(encoding='utf-8')))
    result = audit(args.bank, epochs=args.epochs, anchors_per_episode=args.anchors_per_episode,
                   seed=args.seed, config=config)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(args.output, result)
    print(json.dumps({k: result[k] for k in ('status', 'dataset', 'measurements')}, indent=2))


if __name__ == '__main__':
    main()
