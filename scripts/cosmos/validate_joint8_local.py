"""Replay recorded Joint8 observations through the production policy, without hardware.

The asset bundle contains original, hashed training shards, not relabeled local
features. Preflight runs before weights arrive. Runtime replay reports pipeline
parity separately from training-data fit; it is not a held-out evaluation.
"""
import argparse
import gc
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'deploy')]

import numpy as np
import torch
from safetensors.torch import load_file
from torchvision.transforms import Resize

from fastwam.cache.latent import cache_identity_sha256, file_sha256
from fastwam.cosmos.realman_joint import RealManJointSource
from realman_deploy.contracts import Observation
from realman_deploy.model_adapter import CosmosJointPolicy
from realman_deploy.normalization import DatasetNormalizer, OnlineHistory


def comparison(a, b):
    a, b = torch.as_tensor(a).float(), torch.as_tensor(b).float()
    if a.shape != b.shape or not bool(torch.isfinite(a).all() & torch.isfinite(b).all()):
        raise ValueError('Comparison requires matching finite tensors')
    d = a - b
    return dict(max_abs=float(d.abs().max()), mean_abs=float(d.abs().mean()),
                rmse=float(d.square().mean().sqrt()), exact=torch.equal(a, b))


def require_equal(a, b, name):
    result = comparison(a, b)
    result['exact'] = torch.equal(torch.as_tensor(a), torch.as_tensor(b))
    if not result['exact']:
        raise ValueError(name + ' differs: ' + str(result))
    return result


def raw_frames(source, index):
    import av
    ep, frame, _, _ = source.rows[index]
    episode = source.episodes[ep]
    frames = {}
    for name in ('global', 'wrist'):
        camera = 'observation.images.' + name
        key = 'videos/' + camera + '/'
        path = source.root / source.info['video_path'].format(video_key=camera,
            chunk_index=episode[key + 'chunk_index'], file_index=episode[key + 'file_index'])
        target = round(episode[key + 'from_timestamp'] * 30) + frame
        with av.open(str(path)) as container:
            for j, image in enumerate(container.decode(video=0)):
                if j == target:
                    if (image.width, image.height) != (640, 480):
                        raise ValueError('Wrong recorded RGB dimensions')
                    frames[name] = image.to_ndarray(format='rgb24').tobytes()
                    break
        if name not in frames:
            raise ValueError('Missing recorded frame: ' + str(path))
    return frames


def replay_input(source, normalizer, index):
    ep, frame, start, _ = source.rows[index]
    lo, _ = source.bounds[index]
    history = OnlineHistory(normalizer)
    for f in range(max(lo, frame - 31), frame + 1):
        history.append(source.raw['state'][start + f].tolist(),
                       source.raw['force'][start + f].tolist())
    observation = Observation.now(index, source.raw['state'][start + frame].tolist(),
        source.raw['force'][start + frame].tolist(), raw_frames(source, index), action_space='joint8')
    return observation, history.tensors()


def training_sample(bundle, receipt, index):
    entry = next(e for e in receipt['shards'] if e['start'] <= index < e['stop'])
    path = bundle / 'prepared/features' / entry['path']
    if file_sha256(path) != entry['sha256']:
        raise ValueError('Training shard hash differs')
    payload = load_file(str(path))
    return {k: v[index - entry['start']].clone() for k, v in payload.items()}


def fit_metrics(predicted, target, valid):
    p, y = torch.as_tensor(predicted).float(), torch.as_tensor(target).float()
    valid = torch.as_tensor(valid).bool()
    if p.ndim != 2 or p.shape != y.shape or valid.shape not in (p.shape, p.shape[:1]):
        raise ValueError('Prediction, target and validity shapes differ')
    if valid.ndim == 1:
        valid = valid[:, None].expand_as(p)
    if not bool(torch.isfinite(p).all() & torch.isfinite(y).all()) or not bool(valid.any(0).all()):
        raise ValueError('Nonfinite predictions or missing valid target channels')
    d = p - y
    n = valid.sum(0)
    return dict(mae_per_channel=((d.abs() * valid).sum(0) / n).tolist(),
                rmse_per_channel=((d.square() * valid).sum(0) / n).sqrt().tolist(),
                valid_per_channel=n.tolist())


def plot_fit(path, prediction, target, mask, labels, units, start_offset=0):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(len(labels) // 2, 2, figsize=(12, len(labels) * 1.25), squeeze=False)
    x = (np.arange(32) + start_offset) / 30.
    for j, ax in enumerate(axes.flat):
        y = np.asarray(target)[:, j].copy()
        y[~np.asarray(mask).astype(bool)] = np.nan
        ax.plot(x, y, label='Recorded target', linewidth=1.5)
        ax.plot(x, np.asarray(prediction)[:, j], label='Production inference', linewidth=1.2)
        ax.set(title=labels[j], xlabel='Horizon (s)', ylabel=units[j])
        ax.grid(alpha=.25)
    axes.flat[0].legend()
    fig.tight_layout(); fig.savefig(path, dpi=160); plt.close(fig)


def preflight(bundle, source, receipt):
    for relative, expected in receipt['files'].items():
        if file_sha256(bundle / relative) != expected:
            raise ValueError('Downloaded asset hash differs: ' + relative)
    manifest = json.loads((bundle / 'prepared/features/manifest.json').read_text())
    if (cache_identity_sha256(manifest['identity']) != manifest['identity_sha256']
            or manifest['identity']['realman_source'] != source.identity):
        raise ValueError('Local source and training-cache identity differ')
    normalizer = DatasetNormalizer.from_json(bundle / 'prepared/normalization.json')
    norm = json.loads((bundle / 'prepared/normalization.json').read_text())
    if norm['source_identity_sha256'] != source.identity_sha256 or norm['statistics'] != source.stats:
        raise ValueError('Local statistics differ from training-bound normalization')
    from realman_force.contract import validate_contract, input_contract
    from realman_force.force_filter import make_manifest
    hardware = json.loads((bundle / 'hardware.local.json').read_text())
    calibration = json.loads(Path(hardware['force_calibration_file']).read_text())
    filter_config = json.loads(Path(hardware['force_filter_file']).read_text())
    validate_contract(norm['force_preprocessing'], calibration)
    if input_contract(calibration, make_manifest(filter_config, calibration)) != norm['force_preprocessing']:
        raise ValueError('Deployment force filter/calibration differs from training')
    from fastwam.cosmos.assets import read_text_bank
    bank, _, text_id = read_text_bank(bundle / 'prepared/text'); del bank
    if text_id != manifest['identity']['text_identity']:
        raise ValueError('Text-cache identity differs')
    lightweight = CosmosJointPolicy.__new__(CosmosJointPolicy)
    lightweight.config = dict(camera_order=['global', 'wrist'], camera_width=640, camera_height=480)
    lightweight.device = 'cpu'; lightweight._resize = Resize((224, 224), antialias=True)
    records = []; inputs = []
    for index in receipt['sample_indices']:
        cached = training_sample(bundle, receipt, index)
        numeric = source[index]
        checks = {k: require_equal(numeric[k], cached[k], k) for k in numeric}
        observation, history = replay_input(source, normalizer, index)
        for key, value in zip(('history', 'history_is_pad', 'history_channel_valid'), history):
            checks['online_' + key] = require_equal(torch.tensor(value), cached[key], key)
        checks['current_proprio'] = require_equal(normalizer.encode('state', observation.state) + [0.],
                                                  cached['proprio'][0], 'current proprio')
        pixels = lightweight._encode_image(observation)
        expected = source.window(index)[:, :1].unsqueeze(0).to(torch.bfloat16)
        checks['rgb_camera_order_resize'] = require_equal(pixels, expected, 'RGB preprocessing')
        records.append(dict(sample=index, episode=int(cached['episode_index']),
                            frame=int(cached['frame_index']), checks=checks))
        inputs.append((cached, observation, history))
        print(json.dumps(dict(stage='input_alignment_passed', sample=index)), flush=True)
    return normalizer, records, inputs


def run_inference(args, normalizer, inputs, report, save):
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable; restore the local GPU driver before inference. Downloaded weights and CPU input checks remain valid.')
    config = json.loads(args.runtime.read_text())
    if config.get('joint_steps') != 16 or config.get('inference_backend') != 'joint_static_cache':
        raise ValueError('Replay requires production joint_static_cache and 16 steps')
    policy = original = traced = None
    try:
        torch.cuda.reset_peak_memory_stats()
        policy = CosmosJointPolicy(args.checkpoint, config, config_path=args.runtime)
        report['warmup'] = policy.diagnostics['warmup']; save()
        original = policy.model.infer_joint
        records = []
        for cached, observation, history in inputs:
            capture = {}
            def traced(sample, **kwargs):
                capture['input'] = {k: v.detach().cpu().clone() for k, v in sample.items()}
                result = original(sample, **kwargs)
                capture['output'] = {k: v.detach().float().cpu() for k, v in result.items()}
                return result
            policy.model.infer_joint = traced
            try:
                policy.inference_index = 0
                actions = policy.infer(observation, None, history)
            finally:
                policy.model.infer_joint = original
            production = capture['output']; sample = {k: v.unsqueeze(0) for k, v in cached.items()}
            local_first = dict(sample, first_frame_latents=capture['input']['first_frame_latents'])
            with torch.inference_mode():
                numeric_output = original(local_first, num_inference_steps=16, seed=42, video_seed=44, force_seed=43, use_cache=True)
            numeric_output = {k: v.float().cpu() for k, v in numeric_output.items()}
            input_checks = {k: require_equal(capture['input'][k], local_first[k].to(capture['input'][k].dtype), k)
                            for k in ('history', 'history_is_pad', 'history_channel_valid', 'cosmos_text_index')}
            require_equal(capture['input']['proprio'][:, 0],
                          sample['proprio'][:, 0].to(capture['input']['proprio'].dtype), 'runtime current state')
            parity = {k: comparison(production[k], numeric_output[k]) for k in production}
            if not all(x['max_abs'] <= 1e-5 for x in parity.values()):
                raise ValueError('Production adapter and source-numeric replay differ: ' + str(parity))
            with torch.inference_mode():
                remote_output = original(sample, num_inference_steps=16, seed=42, video_seed=44, force_seed=43, use_cache=True)
            remote_output = {k: v.float().cpu() for k, v in remote_output.items()}
            cross_gpu = {k: comparison(production[k], remote_output[k]) for k in production}
            action = np.asarray(normalizer.decode('action', actions))
            force = np.asarray(normalizer.decode('force', production['force'][0].tolist()))
            target_action = np.asarray(normalizer.decode('action', cached['action'].tolist()))
            target_force = cached['raw_wrench_future64'].numpy()
            index = int(cached['sample_index']); ep, frame, start, _ = args.source.rows[index]
            lo, hi = args.source.bounds[index]
            target_rows = start + (frame + torch.arange(32)).clamp(lo, hi - 1)
            raw_action = args.source.raw['action'][target_rows].numpy()
            amask = ~cached['action_is_pad'].numpy(); fmask = ~cached['force_is_pad'].numpy()
            measured = args.source.raw['state'][start + frame].numpy()
            record = dict(sample=index, episode=ep, frame=frame, input_checks=input_checks,
                production_latency=policy.diagnostics['last_inference'], latency_includes_trace_copy=True, pipeline_parity=parity,
                tokenizer_h200_vs_3080=comparison(sample['first_frame_latents'], local_first['first_frame_latents']),
                remote_cached_latent_vs_local_rgb=cross_gpu,
                action_vs_recorded=fit_metrics(action, raw_action, amask),
                action_vs_training_target=fit_metrics(action, target_action, amask),
                force_vs_recorded=fit_metrics(force, target_force, fmask),
                hold_current_state_baseline=fit_metrics(np.tile(measured, (32, 1)), raw_action, amask),
                first_action_joint_delta_deg=((action[0, :7] - measured[:7]) * 180 / np.pi).tolist())
            record['action_joint_mae_deg'] = (np.array(record['action_vs_recorded']['mae_per_channel'][:7]) * 180 / np.pi).tolist()
            if args.check_uncached and not records:
                with torch.inference_mode():
                    uncached = original(local_first, num_inference_steps=16, seed=42, video_seed=44, force_seed=43, use_cache=False)
                record['static_cache_vs_uncached'] = {k: comparison(production[k], uncached[k].float().cpu()) for k in production}
                record['static_cache_action_tolerance'] = dict(max_abs=.05, mean_abs=.01, units='normalized')
                d = record['static_cache_vs_uncached']['action']
                record['static_cache_action_passed'] = d['max_abs'] <= .05 and d['mean_abs'] <= .01
                del uncached
            np.savez(args.output / f'sample_{index:06d}.npz', action=action, raw_action=raw_action,
                training_target_action=target_action, force=force, target_force=target_force,
                action_valid=amask, force_valid=fmask, measured_state=measured,
                **{'normalized_' + k: v.numpy() for k, v in production.items()})
            action_degrees, raw_degrees = action.copy(), raw_action.copy()
            action_degrees[:, :7] *= 180 / np.pi; raw_degrees[:, :7] *= 180 / np.pi
            plot_fit(args.output / f'sample_{index:06d}_action.png', action_degrees, raw_degrees, amask,
                     [f'Joint {j + 1}' for j in range(7)] + ['Gripper'], ['degree'] * 7 + ['openness'])
            plot_fit(args.output / f'sample_{index:06d}_force.png', force, target_force, fmask,
                     ['Fx', 'Fy', 'Fz', 'Tx', 'Ty', 'Tz'], ['N'] * 3 + ['Nm'] * 3, start_offset=1)
            records.append(record); report['inference'] = records; save()
            print(json.dumps(dict(stage='inference_replay_complete', sample=index,
                latency=record['production_latency'], joint_mae_deg=record['action_joint_mae_deg'])), flush=True)
        report['peak_gpu_allocated_gib'] = torch.cuda.max_memory_allocated() / 2**30
        report['peak_gpu_reserved_gib'] = torch.cuda.max_memory_reserved() / 2**30
        if any(not r.get('static_cache_action_passed', True) for r in records):
            raise ValueError('Static cache versus uncached action tolerance exceeded; inspect report')
    finally:
        if policy is not None:
            policy.close()
        original = traced = policy = None
        gc.collect(); torch.cuda.empty_cache()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('bundle', 'dataset', 'output'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--runtime', type=Path)
    p.add_argument('--checkpoint', type=Path)
    p.add_argument('--check-uncached', action='store_true')
    args = p.parse_args(); torch.set_num_threads(4)
    if bool(args.runtime) != bool(args.checkpoint):
        p.error('--runtime and --checkpoint must be provided together')
    args.output.mkdir(parents=True, exist_ok=False)
    report = dict(schema='joint8-local-validation-v1', started=time.strftime('%Y-%m-%d %H:%M:%S %z'),
                  scope='training-data input parity and fit; no hardware execution or held-out claim', status='running')
    def save():
        path = args.output / 'report.json'; tmp = path.with_suffix('.tmp')
        tmp.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n'); tmp.replace(path)
    save()
    try:
        receipt = json.loads((args.bundle / 'receipt.json').read_text())
        manifest = json.loads((args.bundle / 'prepared/features/manifest.json').read_text())
        args.source = RealManJointSource(args.dataset, manifest['identity']['realman_source']['source_inventory_sha256'])
        normalizer, report['inputs'], inputs = preflight(args.bundle, args.source, receipt); save()
        report['input_alignment'] = 'passed'
        if args.runtime:
            payload = torch.load(args.checkpoint, map_location='cpu', mmap=True, weights_only=True)
            if payload.get('step') != 1000:
                raise ValueError('This validation requires the requested step1000 checkpoint')
            report['checkpoint'] = dict(path=str(args.checkpoint.resolve()), sha256=file_sha256(args.checkpoint), step=payload['step'])
            del payload
            run_inference(args, normalizer, inputs, report, save)
        report['status'] = 'inference_passed' if args.runtime else 'preflight_passed_weights_pending'
    except Exception as exc:
        report.update(status='failed', error=repr(exc)); raise
    finally:
        report['finished'] = time.strftime('%Y-%m-%d %H:%M:%S %z'); save()
    print(args.output / 'report.json', flush=True)


if __name__ == '__main__':
    main()
