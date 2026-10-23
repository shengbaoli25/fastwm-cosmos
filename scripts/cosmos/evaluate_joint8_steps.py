"""Compare 16/8/4 sampling steps against recorded Joint8 GT; no robot execution.

Uses the original verified training bundle and raw demonstrations. Results are
training-window replay, not held-out performance or physical grasp success.
"""
import argparse
import csv
import gc
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'deploy'), str(Path(__file__).resolve().parent)]

import numpy as np
import torch

from fastwam.cache.latent import file_sha256
from fastwam.cosmos.noaction import SCHEMA
from fastwam.cosmos.offline_metrics import action_errors, force_errors, merge_stats
from fastwam.cosmos.realman_joint import RealManJointSource
from realman_deploy.model_adapter import CosmosJointPolicy
from validate_joint8_local import comparison, preflight

STEP1000_SHA = '38c402207973b9da74488ccf7475cd9a9f7b9280db1f700c9fda582ae4e2a425'


def model_input(first, history, state, text_index, device):
    """Accept only observations; future labels/masks cannot enter the model."""
    raw, pad, valid = history
    return dict(
        first_frame_latents=first,
        history=torch.tensor([raw], dtype=torch.bfloat16, device=device),
        history_is_pad=torch.tensor([pad], dtype=torch.bool, device=device),
        history_channel_valid=torch.tensor([valid], dtype=torch.bool, device=device),
        proprio=torch.tensor([[list(state) + [0.]]], dtype=torch.float32, device=device),
        action_is_pad=torch.zeros((1, 32), dtype=torch.bool, device=device),
        cosmos_text_index=torch.tensor([text_index], dtype=torch.long, device=device),
    )


def recorded_targets(source, cached, normalizer):
    index = int(cached['sample_index'])
    ep, frame, start, _ = source.rows[index]
    lo, hi = source.bounds[index]
    rows = start + (frame + torch.arange(32)).clamp(lo, hi - 1)
    amask = ~cached['action_is_pad'].numpy()
    fmask = (~cached['force_is_pad'].numpy())[:, None] & cached['wrench_valid'].numpy()
    return dict(sample=index, episode=ep, frame=frame,
        action=source.raw['action'][rows].numpy(),
        training_action=np.asarray(normalizer.decode('action', cached['action'].tolist())),
        force=cached['raw_wrench_future64'].numpy(),
        action_valid=amask, force_valid=fmask,
        measured_state=source.raw['state'][start + frame].numpy(),
        measured_force=source.raw['force'][start + frame].numpy())


def prediction_metrics(action, force, target):
    return dict(
        action_vs_recorded=action_errors(action, target['action'], target['action_valid']),
        action_vs_training_target=action_errors(action, target['training_action'], target['action_valid']),
        force_vs_recorded=force_errors(force, target['force'], target['force_valid']),
    )


def aggregate(records, steps):
    result = {}
    for step in steps:
        selected = [r for r in records if r['steps'] == step]
        if not selected:
            continue
        metrics = {section: {group: merge_stats([r['metrics'][section][group] for r in selected])
                            for group in selected[0]['metrics'][section]}
                   for section in selected[0]['metrics']}
        latencies = [r['joint_core_ms'] for r in selected]
        result[str(step)] = dict(records=len(selected), metrics=metrics,
            median_joint_core_ms=float(np.median(latencies)), p95_joint_core_ms=float(np.percentile(latencies, 95)))
    if '16' in result:
        for item in result.values():
            item['gt_mae_change_vs_16'] = {}
            for section in ('action_vs_recorded', 'force_vs_recorded'):
                for group, stats in item['metrics'][section].items():
                    ref = result['16']['metrics'][section][group]['pooled_mae']
                    value = stats['pooled_mae']
                    item['gt_mae_change_vs_16'][group] = dict(
                        absolute=None if ref is None or value is None else value - ref,
                        percent=None if not ref or value is None else (value / ref - 1) * 100)
    return result


def plot_overlay(path, predictions, target, valid, *, force=False):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    target = np.asarray(target, dtype=np.float64).copy()
    mask = np.asarray(valid, dtype=bool)
    if mask.ndim == 1:
        mask = np.broadcast_to(mask[:, None], target.shape)
    target[~mask] = np.nan
    predictions = {k: np.asarray(v, dtype=np.float64).copy() for k, v in predictions.items()}
    if force:
        labels = ['Fx (N)', 'Fy (N)', 'Fz (N)', 'Tx (Nm)', 'Ty (Nm)', 'Tz (Nm)']
    else:
        target[:, :7] = np.rad2deg(target[:, :7])
        for p in predictions.values():
            p[:, :7] = np.rad2deg(p[:, :7])
        labels = [f'Joint {i + 1} (deg)' for i in range(7)] + ['Gripper (openness)']
    x = (np.arange(len(target)) + int(force)) / 30
    fig, axes = plt.subplots(len(labels) // 2, 2, figsize=(12, len(labels) * 1.4), squeeze=False)
    for j, ax in enumerate(axes.flat):
        ax.plot(x, target[:, j], color='black', label='Recorded GT', linewidth=1.8)
        for step, p in predictions.items():
            p[~mask] = np.nan
            ax.plot(x, p[:, j], label=f'{step} steps', linewidth=1.1)
        ax.set(title=labels[j], xlabel='Horizon (s)'); ax.grid(alpha=.25)
    axes.flat[0].legend()
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def evaluate(args, normalizer, inputs, report, save):
    config = json.loads(args.runtime.read_text())
    # This only changes the in-memory backend. Keep the deployment step check.
    config['inference_backend'] = args.backend
    if config.get('joint_steps') != 16:
        raise ValueError('Expected the original 16-step runtime; do not modify its production step setting')
    norm_path = Path(config['normalization_file'])
    if not norm_path.is_absolute():
        norm_path = args.runtime.parent / norm_path
    if file_sha256(norm_path) != file_sha256(args.bundle / 'prepared/normalization.json'):
        raise ValueError('Runtime and evaluation normalization differ')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable')
    policy = infer = None
    prepared, outputs = [], {}
    try:
        policy = CosmosJointPolicy(args.checkpoint, config, config_path=args.runtime)
        report['model_warmup'] = policy.diagnostics['warmup']
        infer = policy.runner.infer_joint if policy.runner else policy.model.infer_joint
        for cached, observation, history in inputs:
            if int(cached['cosmos_text_index']) != policy.text_index:
                raise ValueError('Runtime task prompt differs from the recorded window')
            torch.cuda.synchronize(); started = time.perf_counter()
            with torch.inference_mode():
                first = policy.tokenizer.encode(policy._encode_image(observation)[:, :, :1])
            torch.cuda.synchronize()
            encoding_ms = (time.perf_counter() - started) * 1000
            sample = model_input(first, history, normalizer.encode('state', observation.state),
                                 policy.text_index, policy.device)
            target = recorded_targets(args.source, cached, normalizer)
            prepared.append((sample, target, encoding_ms))
        report['encoding_ms_per_window'] = [p[2] for p in prepared]
        report['inference'], report['parity'], report['step_warmup_ms'] = [], {}, {}
        torch.cuda.reset_peak_memory_stats()
        for step in args.steps:
            kwargs = dict(num_inference_steps=step, seed=42, video_seed=44, force_seed=43, use_cache=True)
            torch.cuda.synchronize(); started = time.perf_counter()
            with torch.inference_mode():
                warm = infer(prepared[0][0], **kwargs)
            torch.cuda.synchronize()
            report['step_warmup_ms'][str(step)] = (time.perf_counter() - started) * 1000
            # Check implementation drift separately from reducing sampler steps.
            if policy.runner is not None:
                with torch.inference_mode():
                    reference = policy.model.infer_joint(prepared[0][0], **kwargs)
                parity = {key: comparison(warm[key].float().cpu(), reference[key].float().cpu())
                          for key in warm}
                report['parity'][str(step)] = parity
                save()
                if any(v['max_abs'] > .05 or v['mean_abs'] > .01 for v in parity.values()):
                    raise ValueError('Optimized/reference numerical parity failed; rerun with joint_static_cache')
                del reference
            del warm
            for sample, target, encoding_ms in prepared:
                for seed_offset in args.seed_offsets:
                    kwargs.update(seed=42 + seed_offset, video_seed=44 + seed_offset, force_seed=43 + seed_offset)
                    torch.cuda.synchronize(); started = time.perf_counter()
                    with torch.inference_mode():
                        prediction = infer(sample, **kwargs)
                    torch.cuda.synchronize()
                    core_ms = (time.perf_counter() - started) * 1000
                    action_norm = prediction['action'][0].float().cpu().numpy()
                    force_norm = prediction['force'][0].float().cpu().numpy()
                    action = np.asarray(normalizer.decode('action', action_norm.tolist()))
                    force = np.asarray(normalizer.decode('force', force_norm.tolist()))
                    del prediction
                    metrics = prediction_metrics(action, force, target)
                    key = (target['sample'], seed_offset)
                    if step == 16:
                        outputs[key] = (action, force)
                    reference_action, reference_force = outputs[key]
                    metrics['action_vs_16'] = action_errors(action, reference_action, target['action_valid'])
                    metrics['force_vs_16'] = force_errors(force, reference_force, target['force_valid'])
                    hold_action = np.tile(target['measured_state'], (32, 1))
                    hold_force = np.tile(target['measured_force'], (32, 1))
                    metrics['hold_state_vs_recorded'] = action_errors(hold_action, target['action'], target['action_valid'])
                    metrics['hold_force_vs_recorded'] = force_errors(hold_force, target['force'], target['force_valid'])
                    record = dict(sample=target['sample'], episode=target['episode'], frame=target['frame'],
                        steps=step, seed_offset=seed_offset, seeds={k: kwargs[k] for k in ('seed', 'video_seed', 'force_seed')},
                        joint_core_ms=core_ms, metrics=metrics,
                        first_action_joint_delta_deg=np.rad2deg(action[0, :7] - target['measured_state'][:7]).tolist(),
                        recorded_first_joint_delta_deg=np.rad2deg(target['action'][0, :7] - target['measured_state'][:7]).tolist())
                    name = f'sample_{target["sample"]:06d}_seed{seed_offset}_steps{step}.npz'
                    np.savez_compressed(args.output / name, action=action, force=force,
                        recorded_action=target['action'], training_action=target['training_action'],
                        recorded_force=target['force'], action_valid=target['action_valid'], force_valid=target['force_valid'],
                        measured_state=target['measured_state'], measured_force=target['measured_force'],
                        normalized_action=action_norm, normalized_force=force_norm,
                        action_time_s=np.arange(32) / 30, force_time_s=np.arange(1, 33) / 30)
                    record['artifact'] = name
                    report['inference'].append(record); save()
                    print(json.dumps(dict(stage='gt_window_complete', sample=target['sample'], steps=step,
                        seed_offset=seed_offset, joint_mae_deg=metrics['action_vs_recorded']['joint_deg']['pooled_mae'])), flush=True)
        report['aggregate'] = aggregate(report['inference'], args.steps)
        report['aggregate_by_seed_offset'] = {str(offset): aggregate(
            [r for r in report['inference'] if r['seed_offset'] == offset], args.steps)
            for offset in args.seed_offsets}
        report['peak_gpu_allocated_gib'] = torch.cuda.max_memory_allocated() / 2**30
        report['peak_gpu_reserved_gib'] = torch.cuda.max_memory_reserved() / 2**30
        with (args.output / 'summary.csv').open('w', newline='') as stream:
            writer = csv.writer(stream)
            writer.writerow(['steps', 'target', 'group', 'valid_values', 'mae', 'rmse', 'max_abs'])
            for step, item in report['aggregate'].items():
                for section, groups in item['metrics'].items():
                    for group, stats in groups.items():
                        writer.writerow([step, section, group, sum(stats['valid_per_channel']),
                                         stats['pooled_mae'], stats['pooled_rmse'], stats['pooled_max_abs']])
        if not args.no_plots:
            for _, target, _ in prepared:
                for seed_offset in args.seed_offsets:
                    stem = f'sample_{target["sample"]:06d}_seed{seed_offset}'
                    prediction = {}
                    for step in args.steps:
                        with np.load(args.output / f'{stem}_steps{step}.npz') as saved:
                            prediction[step] = (saved['action'], saved['force'])
                    plot_overlay(args.output / f'{stem}_action.png', {s: p[0] for s, p in prediction.items()},
                                 target['action'], target['action_valid'])
                    plot_overlay(args.output / f'{stem}_force.png', {s: p[1] for s, p in prediction.items()},
                                 target['force'], target['force_valid'], force=True)
        save()
    finally:
        infer = None
        prepared.clear(); outputs.clear()
        if policy is not None:
            policy.close()
        policy = None
        gc.collect(); torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('bundle', 'dataset', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--runtime', type=Path, help='Default: BUNDLE/runtime.local.json')
    parser.add_argument('--checkpoint', type=Path, help='Default: BUNDLE/weights/step_001000.pt')
    parser.add_argument('--checkpoint-sha256', default=STEP1000_SHA)
    parser.add_argument('--steps', type=int, nargs='+', default=[16, 8, 4])
    parser.add_argument('--seed-offsets', type=int, nargs='+', default=[0, 1, 2])
    parser.add_argument('--sample-indices', type=int, nargs='+', help='Default: all sample_indices in receipt.json')
    parser.add_argument('--backend', choices=['joint_optimized', 'joint_static_cache'], default='joint_optimized')
    parser.add_argument('--preflight-only', action='store_true')
    parser.add_argument('--no-plots', action='store_true')
    args = parser.parse_args()
    if 16 not in args.steps or any(s < 1 for s in args.steps) or len(set(args.steps)) != len(args.steps):
        parser.error('--steps must include 16 and contain distinct positive values')
    if not args.seed_offsets or min(args.seed_offsets) < 0 or len(set(args.seed_offsets)) != len(args.seed_offsets):
        parser.error('--seed-offsets must contain distinct nonnegative values')
    args.steps = [16] + [s for s in args.steps if s != 16]
    args.bundle, args.dataset = args.bundle.resolve(), args.dataset.resolve()
    args.runtime = (args.runtime or args.bundle / 'runtime.local.json').resolve()
    args.checkpoint = (args.checkpoint or args.bundle / 'weights/step_001000.pt').resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    report = dict(schema='joint8-sampling-gt-v1', status='running', started=time.strftime('%Y-%m-%d %H:%M:%S %z'),
        scope='verified training-window replay; not held-out, closed-loop or grasp success',
        steps=args.steps, seed_offsets=args.seed_offsets, backend=args.backend,
        action_offsets=list(range(32)), force_offsets=list(range(1, 33)), fps=30,
        aggregation='pooled valid scalar counts across windows and seeds; overlapping windows/seeds are not independent trials',
        latency_scope='joint core only after warmup; RGB/VAE measured once per window separately; no robot I/O',
        parity_scope='first recorded window, A42/V44/F43, each step count; max_abs<=0.05 and mean_abs<=0.01 normalized numerical diagnostic only',
        command=sys.argv, paths=dict(bundle=str(args.bundle), dataset=str(args.dataset), runtime=str(args.runtime)))
    def save():
        path = args.output / 'report.json'
        tmp = path.with_suffix('.tmp')
        tmp.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + '\n'); tmp.replace(path)
    save()
    try:
        receipt = json.loads((args.bundle / 'receipt.json').read_text())
        if args.sample_indices is not None:
            if not args.sample_indices or len(set(args.sample_indices)) != len(args.sample_indices):
                raise ValueError('Sample indices must be nonempty and distinct')
            receipt = dict(receipt, sample_indices=args.sample_indices)
        if not receipt['sample_indices']:
            raise ValueError('No recorded windows selected')
        manifest = json.loads((args.bundle / 'prepared/features/manifest.json').read_text())
        args.source = RealManJointSource(args.dataset, manifest['identity']['realman_source']['source_inventory_sha256'])
        normalizer, report['inputs'], inputs = preflight(args.bundle, args.source, receipt)
        report['input_alignment'] = 'passed'; save()
        if args.preflight_only:
            report['status'] = 'preflight_passed_no_inference'
        else:
            actual_sha = file_sha256(args.checkpoint)
            if actual_sha != args.checkpoint_sha256:
                raise ValueError('Checkpoint SHA256 differs from the requested trained weight')
            payload = torch.load(args.checkpoint, map_location='cpu', mmap=True, weights_only=True)
            if payload.get('schema') != SCHEMA or int(payload.get('step', 0)) <= 0:
                raise ValueError('A trained joint checkpoint is required; official Video/init Action/Force is not a trained policy')
            if payload['manifest']['dataset_identity'] != manifest['identity_sha256']:
                raise ValueError('Checkpoint and evaluation training-cache identity differ')
            report['checkpoint'] = dict(path=str(args.checkpoint), sha256=actual_sha, step=payload['step'])
            report['provenance'] = dict(feature_identity=manifest['identity_sha256'],
                receipt_sha256=file_sha256(args.bundle / 'receipt.json'),
                runtime_sha256=file_sha256(args.runtime), evaluator_sha256=file_sha256(__file__),
                gpu=torch.cuda.get_device_name() if torch.cuda.is_available() else None, torch=torch.__version__)
            del payload
            evaluate(args, normalizer, inputs, report, save)
            report['status'] = 'gt_evaluation_complete'
    except Exception as exc:
        report.update(status='failed', error=repr(exc)); raise
    finally:
        report['finished'] = time.strftime('%Y-%m-%d %H:%M:%S %z'); save()
    print(args.output / 'report.json', flush=True)


if __name__ == '__main__':
    main()
