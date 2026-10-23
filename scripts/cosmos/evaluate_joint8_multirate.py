"""Local trained-weight V4/F8/A8 experiment with fresh V8/F8/A8 and V4/F4/A4 controls."""
import argparse
import gc
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'deploy'), str(Path(__file__).resolve().parent)]
import numpy as np
import torch

from evaluate_joint8_steps import (STEP1000_SHA, aggregate, model_input,
                                  prediction_metrics, recorded_targets)
from validate_joint8_local import comparison, preflight
from fastwam.cache.latent import file_sha256
from fastwam.cosmos.noaction import SCHEMA
from fastwam.cosmos.noaction_inference import NoActionInferenceRunner
from fastwam.cosmos.multirate_inference import MultirateInferenceRunner, infer_multirate_reference
from fastwam.cosmos.offline_metrics import action_errors, force_errors
from fastwam.cosmos.realman_joint import RealManJointSource
from realman_deploy.model_adapter import CosmosJointPolicy

MODES = {'v8_f8_a8': (8, 8), 'v4_f4_a4': (4, 4), 'v4_f8_a8': (4, 8)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('bundle', 'dataset', 'runtime', 'historical', 'output'):
        parser.add_argument('--' + key, required=True, type=Path)
    parser.add_argument('--sample-indices', nargs='+', type=int)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    checkpoint = args.bundle / 'weights/step_001000.pt'
    report = dict(schema='joint8-multirate-gt-v1', status='running', command=sys.argv,
        started=time.strftime('%Y-%m-%d %H:%M:%S %z'), modes=MODES, seed_offsets=[0, 1, 2],
        scope='50 selected training windows; not held-out or robot grasp success',
        schedule='Video state/clock held for two control substeps, then one Video Euler update; Force/Action freshly evaluated each substep',
        cache='Exact per-layer Video K/V reuse within fixed-state subcycle; no velocity reuse',
        latency_scope='Warmed compiled Force/Action plus whole CUDA Graph; joint core wall time incl request prefill, excludes RGB/VAE and robot/network I/O',
        ordering='Three modes interleaved per window/seed; order rotated each seed/window',
        inference=[], parity={}, warmup_ms={}, video_equal_to_4step=[], historical_baseline_parity=[],
        paths={k: str(getattr(args, k).resolve()) for k in ('bundle', 'dataset', 'runtime', 'historical')})
    def save():
        path = args.output / 'report.json'
        temp = path.with_suffix('.tmp')
        temp.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
        temp.replace(path)
    def progress(stage, **kwargs):
        print(json.dumps(dict(stage=stage, **kwargs)), flush=True)
    policy = None
    runners = {}
    prepared = []
    save()
    try:
        receipt = json.loads((args.bundle / 'receipt.json').read_text())
        if args.sample_indices:
            receipt = dict(receipt, sample_indices=args.sample_indices)
        manifest = json.loads((args.bundle / 'prepared/features/manifest.json').read_text())
        source = RealManJointSource(args.dataset, manifest['identity']['realman_source']['source_inventory_sha256'])
        normalizer, report['inputs'], inputs = preflight(args.bundle, source, receipt)
        report['input_alignment'] = 'passed'
        sha = file_sha256(checkpoint)
        if sha != STEP1000_SHA:
            raise ValueError('Trained checkpoint SHA256 mismatch')
        payload = torch.load(checkpoint, map_location='cpu', mmap=True, weights_only=True)
        if payload['schema'] != SCHEMA or payload['step'] != 1000 or payload['manifest']['dataset_identity'] != manifest['identity_sha256']:
            raise ValueError('Wrong trained checkpoint/dataset')
        report['checkpoint'] = dict(path=str(checkpoint), sha256=sha, step=payload['step'])
        del payload
        report['provenance'] = dict(gpu=torch.cuda.get_device_name(), torch=torch.__version__,
            runtime_sha256=file_sha256(args.runtime), receipt_sha256=file_sha256(args.bundle / 'receipt.json'),
            source_hashes={str(p.relative_to(ROOT)): file_sha256(p) for p in
                [Path(__file__), ROOT / 'src/fastwam/cosmos/multirate_inference.py',
                 ROOT / 'src/fastwam/cosmos/noaction_inference.py']})
        config = json.loads(args.runtime.read_text())
        if config['joint_steps'] != 16:
            raise ValueError('Expected unchanged deployment config')
        config['inference_backend'] = 'joint_static_cache'
        progress('loading_trained_policy')
        policy = CosmosJointPolicy(checkpoint, config, config_path=args.runtime)
        for cached, observation, history in inputs:
            if int(cached['cosmos_text_index']) != policy.text_index:
                raise ValueError('Text prompt mismatch')
            torch.cuda.synchronize(); started = time.perf_counter()
            with torch.inference_mode():
                first = policy.tokenizer.encode(policy._encode_image(observation)[:, :, :1])
            torch.cuda.synchronize()
            encoding_ms = (time.perf_counter() - started) * 1000
            sample = model_input(first, history, normalizer.encode('state', observation.state),
                                 policy.text_index, policy.device)
            target = recorded_targets(source, cached, normalizer)
            prepared.append((sample, target, encoding_ms))
        report['encoding_ms_per_window'] = [item[2] for item in prepared]
        inputs.clear()
        for mode, (video_steps, control_steps) in MODES.items():
            runner = (MultirateInferenceRunner(policy.model, video_steps=video_steps)
                      if video_steps != control_steps else NoActionInferenceRunner(policy.model))
            runners[mode] = runner
            progress('compile_and_capture', mode=mode)
            torch.cuda.synchronize(); started = time.perf_counter()
            with torch.inference_mode():
                warm = runner.infer_joint(prepared[0][0], num_inference_steps=control_steps)
            torch.cuda.synchronize()
            report['warmup_ms'][mode] = (time.perf_counter() - started) * 1000
            with torch.inference_mode():
                reference = (infer_multirate_reference(policy.model, prepared[0][0], video_steps=video_steps,
                                                       control_steps=control_steps)
                    if video_steps != control_steps else policy.model.infer_joint(
                        prepared[0][0], num_inference_steps=control_steps))
            parity = {key: comparison(warm[key].float().cpu(), reference[key].float().cpu()) for key in warm}
            report['parity'][mode] = parity
            if any(value['max_abs'] > .05 or value['mean_abs'] > .01 for value in parity.values()):
                raise ValueError('Full-size optimized/reference parity failed: ' + mode)
            progress('parity_passed', mode=mode, parity=parity)
            del reference, warm
            save()
        torch.cuda.reset_peak_memory_stats()
        modes = list(MODES)
        for window_index, (sample, target, _) in enumerate(prepared):
            for offset in range(3):
                rotate = (window_index * 3 + offset) % 3
                videos = {}
                old_path = args.historical / f'sample_{target["sample"]:06d}_seed{offset}_steps16.npz'
                with np.load(old_path) as old:
                    if not np.array_equal(old['recorded_action'], target['action']) or not np.array_equal(old['recorded_force'], target['force']):
                        raise ValueError('Historical GT does not match this experiment')
                    old_action, old_force = old['action'].copy(), old['force'].copy()
                for mode in modes[rotate:] + modes[:rotate]:
                    video_steps, control_steps = MODES[mode]
                    seeds = dict(seed=42 + offset, video_seed=44 + offset, force_seed=43 + offset)
                    torch.cuda.synchronize(); started = time.perf_counter()
                    with torch.inference_mode():
                        prediction = runners[mode].infer_joint(sample, num_inference_steps=control_steps, **seeds)
                    torch.cuda.synchronize()
                    core_ms = (time.perf_counter() - started) * 1000
                    action_norm = prediction['action'][0].float().cpu().numpy()
                    force_norm = prediction['force'][0].float().cpu().numpy()
                    if video_steps == 4:
                        videos[mode] = prediction['video_latents'].float().cpu()
                    action = np.asarray(normalizer.decode('action', action_norm.tolist()))
                    force = np.asarray(normalizer.decode('force', force_norm.tolist()))
                    del prediction
                    metrics = prediction_metrics(action, force, target)
                    metrics['action_vs_historical16'] = action_errors(action, old_action, target['action_valid'])
                    metrics['force_vs_historical16'] = force_errors(force, old_force, target['force_valid'])
                    if video_steps == control_steps:
                        path = args.historical / f'sample_{target["sample"]:06d}_seed{offset}_steps{control_steps}.npz'
                        with np.load(path) as prior:
                            report['historical_baseline_parity'].append(dict(mode=mode, sample=target['sample'],
                                seed_offset=offset, action=comparison(action_norm, prior['normalized_action']),
                                force=comparison(force_norm, prior['normalized_force'])))
                    name = f'sample_{target["sample"]:06d}_seed{offset}_{mode}.npz'
                    np.savez_compressed(args.output / name, action=action, force=force,
                        recorded_action=target['action'], recorded_force=target['force'],
                        training_action=target['training_action'], action_valid=target['action_valid'],
                        force_valid=target['force_valid'], measured_state=target['measured_state'],
                        measured_force=target['measured_force'], normalized_action=action_norm, normalized_force=force_norm,
                        action_time_s=np.arange(32) / 30, force_time_s=np.arange(1, 33) / 30)
                    report['inference'].append(dict(mode=mode, steps=mode, video_steps=video_steps,
                        force_steps=control_steps, action_steps=control_steps, sample=target['sample'],
                        episode=target['episode'], frame=target['frame'], seed_offset=offset, seeds=seeds,
                        joint_core_ms=core_ms, metrics=metrics, artifact=name))
                video_parity = comparison(videos['v4_f8_a8'], videos['v4_f4_a4'])
                report['video_equal_to_4step'].append(dict(sample=target['sample'], seed_offset=offset, **video_parity))
                if not video_parity['exact']:
                    raise ValueError('Video independence or 4-step schedule violated')
                del videos
            save()
            progress('window_complete', windows=window_index + 1, total=len(prepared),
                     predictions=len(report['inference']))
        report['aggregate'] = aggregate(report['inference'], list(MODES))
        report['runner_stats'] = {mode: runner.stats for mode, runner in runners.items()}
        report['peak_allocated_gib'] = torch.cuda.max_memory_allocated() / 2**30
        report['peak_reserved_gib'] = torch.cuda.max_memory_reserved() / 2**30
        report['status'] = 'gt_evaluation_complete'
    except Exception as exc:
        report.update(status='failed', error=repr(exc))
        raise
    finally:
        report['finished'] = time.strftime('%Y-%m-%d %H:%M:%S %z')
        save()
        for runner in runners.values():
            runner.close()
        runners.clear(); prepared.clear()
        if policy is not None:
            policy.close()
        gc.collect(); torch.cuda.empty_cache()
    progress('complete', output=str(args.output))


if __name__ == '__main__':
    main()
