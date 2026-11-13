"""Exhaustive, resumable JEPA query census over every verified training anchor.

No target-fed query shortcut: use all 16 original joint denoising updates. A
bounded benchmark verifies original B1 parity and batched numerical parity before
selecting throughput. Every head/query map is kept, averaged only across steps.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT)]


def atomic_json(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def event(output, phase, **values):
    value = dict(phase=phase, status='running', updated_at=datetime.now(timezone.utc).isoformat())
    value.update(values)
    atomic_json(output / 'progress.json', value)
    print(json.dumps(value, allow_nan=False), flush=True)


def arguments(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--checkpoint', type=Path)
    p.add_argument('--mode', choices=['benchmark', 'full'], default='benchmark')
    p.add_argument('--expected-samples', type=int, default=84349)
    p.add_argument('--expected-episodes', type=int, default=78)
    p.add_argument('--seed', type=int, default=113)
    p.add_argument('--steps', type=int, default=16, choices=[16])
    p.add_argument('--batch-size', default='auto', choices=['auto', '1', '4', '8'])
    p.add_argument('--benchmark-samples', type=int, default=8)
    p.add_argument('--shard-size', type=int, default=500)
    p.add_argument('--start-index', type=int, default=0)
    p.add_argument('--stop-index', type=int)
    p.add_argument('--max-hours', type=float, default=22.)
    p.add_argument('--batch-action-rms-tolerance', type=float, default=.0005)
    p.add_argument('--batch-attention-rms-tolerance', type=float, default=.0001)
    return p.parse_args(argv)


def jpeg_bytes(sample, view):
    import numpy as np
    from PIL import Image
    pixels = ((sample['jepa_observation'][view, :, -1].permute(1, 2, 0).numpy() + 1) * 127.5).round().clip(0, 255).astype(np.uint8)
    stream = io.BytesIO()
    Image.fromarray(pixels).save(stream, format='JPEG', quality=85, optimize=False)
    return stream.getvalue()


def run_forward(model, samples, args, *, original=False, pad_to=None):
    import numpy as np
    import torch
    from fastwam.cosmos.jepa_query_census import MeanAttentionCapture, observed_batch, infer_repeated_seed
    active_count = len(samples)
    if pad_to is not None and active_count < pad_to:
        samples = samples + [samples[-1]] * (pad_to - active_count)
    observations = observed_batch(samples)
    with torch.inference_mode(), MeanAttentionCapture(model.dit['jepa_reader'], args.steps) as capture:
        if original:
            if len(samples) != 1:
                raise ValueError('Original sampler reference must be B1')
            prediction = model.infer_joint(observations, num_inference_steps=args.steps,
                seed=args.seed, force_seed=args.seed + 1, video_seed=args.seed + 2, use_cache=True)
        else:
            prediction = infer_repeated_seed(model, observations, steps=args.steps, seed=args.seed)
        mean = capture.mean()
    result = {key: value[:active_count].detach().float().cpu().numpy() for key, value in prediction.items()}
    if any(not np.isfinite(value).all() for value in result.values()):
        raise ValueError('Nonfinite census prediction')
    return result, mean[:active_count]


def benchmark(model, dataset, args, identity):
    import numpy as np
    import torch
    count = max(8, args.benchmark_samples)
    ids = np.linspace(0, len(dataset) - 1, min(count, len(dataset)), dtype=int).tolist()
    samples = [dataset[index] for index in ids]
    event(args.output, 'benchmark_warmup', sample_indices=ids, original_sampler_steps=args.steps)
    run_forward(model, samples[:1], args, original=True)
    original_predictions, original_attentions = [], []
    torch.cuda.synchronize()
    start = time.monotonic()
    for sample in samples:
        prediction, attention = run_forward(model, [sample], args, original=True)
        original_predictions.append(prediction)
        original_attentions.append(attention)
    torch.cuda.synchronize()
    original_seconds = time.monotonic() - start
    reference = {key: np.concatenate([p[key] for p in original_predictions]) for key in original_predictions[0]}
    reference_attention = np.concatenate(original_attentions)
    candidates = []
    for batch_size in (1, 4, 8):
        event(args.output, 'benchmark_batch', batch_size=batch_size, original_seconds=original_seconds)
        try:
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            predictions, attentions = [], []
            torch.cuda.synchronize()
            start = time.monotonic()
            for begin in range(0, len(samples), batch_size):
                prediction, attention = run_forward(model, samples[begin:begin + batch_size], args, pad_to=batch_size)
                predictions.append(prediction)
                attentions.append(attention)
            torch.cuda.synchronize()
            elapsed = time.monotonic() - start
            result = {key: np.concatenate([p[key] for p in predictions]) for key in reference}
            attention = np.concatenate(attentions)
            differences = {}
            for key in reference:
                delta = result[key].astype(np.float64) - reference[key].astype(np.float64)
                per_sample = np.sqrt(np.mean(delta.reshape(len(samples), -1) ** 2, axis=1))
                differences[key] = dict(max_abs=float(np.abs(delta).max()), rms=float(np.sqrt(np.mean(delta ** 2))),
                    per_sample_rms=per_sample.tolist(), max_sample_rms=float(per_sample.max()))
            delta = attention.astype(np.float64) - reference_attention.astype(np.float64)
            per_sample = np.sqrt(np.mean(delta.reshape(len(samples), -1) ** 2, axis=1))
            differences['attention_mean'] = dict(max_abs=float(np.abs(delta).max()), rms=float(np.sqrt(np.mean(delta ** 2))),
                per_sample_rms=per_sample.tolist(), max_sample_rms=float(per_sample.max()))
            b1_exact = all(v['max_abs'] <= 2e-6 for v in differences.values())
            finite = all(np.isfinite(v).all() for v in result.values()) and np.isfinite(attention).all()
            accepted = bool(finite and (b1_exact if batch_size == 1 else
                differences['action']['max_sample_rms'] <= args.batch_action_rms_tolerance and
                differences['attention_mean']['max_sample_rms'] <= args.batch_attention_rms_tolerance))
            peak = torch.cuda.max_memory_allocated()
            candidates.append(dict(batch_size=batch_size, passed=accepted, original_b1_exact=b1_exact,
                samples=len(samples), seconds=elapsed, seconds_per_sample=elapsed / len(samples),
                samples_per_second=len(samples) / elapsed, peak_allocated_bytes=peak,
                peak_reserved_bytes=torch.cuda.max_memory_reserved(), differences=differences))
        except torch.cuda.OutOfMemoryError as error:
            candidates.append(dict(batch_size=batch_size, passed=False, error='cuda_out_of_memory', message=str(error)))
            gc.collect()
            torch.cuda.empty_cache()
        print(json.dumps(dict(status='benchmark_candidate', **candidates[-1])), flush=True)
    if not candidates[0]['passed']:
        raise ValueError('B1 adapter does not reproduce the original joint sampler; census blocked')
    passing = [candidate for candidate in candidates if candidate['passed']]
    if args.batch_size != 'auto':
        passing = [c for c in passing if c['batch_size'] == int(args.batch_size)]
        if not passing:
            raise ValueError('Requested batch size failed numerical parity; choose auto or B1')
    selected = min(passing, key=lambda value: value['seconds_per_sample'])
    jpeg_size = sum(len(jpeg_bytes(sample, view)) for sample in samples for view in range(2)) / len(samples)
    total = args.stop_index - args.start_index
    summary = dict(schema='jepa-query-census-benchmark-v1', run_identity=identity,
        sample_indices=ids, original_single_seconds_per_sample=original_seconds/len(samples),
        candidates=candidates, selected_batch_size=selected['batch_size'],
        action_rms_tolerance=args.batch_action_rms_tolerance,
        attention_rms_tolerance=args.batch_attention_rms_tolerance,
        scope='Batched BF16 results may differ numerically; identical per-example noise and sampler semantics.',
        estimated_model_hours=total * selected['seconds_per_sample'] / 3600,
        estimated_hours_with_20percent_io_margin=total * selected['seconds_per_sample'] * 1.2 / 3600,
        uncompressed_attention_bytes=total * 2 * 4 * 8 * 392 * 2,
        estimated_thumbnail_bytes=int(total * jpeg_size),
        prediction_float32_bytes=total * 32 * (8 + 6) * 4,
        note='Storage estimates exclude JSON and compression savings; time margin is an estimate, not completion promise.')
    atomic_json(args.output / 'benchmark.json', summary)
    print(json.dumps(dict(status='benchmark_passed', **summary)), flush=True)
    return summary


def execute(args):
    import numpy as np
    import torch
    from omegaconf import OmegaConf
    from fastwam.cosmos.jepa_training_data import JepaForceTrainingDataset
    from fastwam.cosmos.jepa_force_latent import create_jepa_force_latent_cosmos
    from fastwam.cosmos.jepa_query_census import attention_metrics, digest_file, write_shard, verify_shard
    torch.set_num_threads(4)
    started = time.monotonic()
    if args.max_hours <= 0 or args.shard_size < 1 or args.benchmark_samples < 1:
        raise ValueError('Positive hours, shard size and benchmark sample count required')
    if torch.cuda.device_count() != 1:
        raise ValueError('Exactly one visible GPU required')
    args.output.mkdir(parents=True, exist_ok=True)
    event(args.output, 'verify_inputs', expected_samples=args.expected_samples, expected_episodes=args.expected_episodes)
    config_path = args.run_root / 'resolved.yaml'
    checkpoint = args.checkpoint or args.run_root / 'training/checkpoints/weights/step_030000.pt'
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    cfg = OmegaConf.to_container(OmegaConf.load(config_path), resolve=True, throw_on_missing=True)
    data_cfg = dict(cfg['data']['train'])
    if data_cfg.pop('_target_') != 'fastwam.cosmos.jepa_training_data.JepaForceTrainingDataset':
        raise ValueError('Exact trained JEPA cache required')
    data_cfg.pop('_recursive_', None)
    event(args.output, 'verify_cache_hashes')
    dataset = JepaForceTrainingDataset(**data_cfg)
    episodes = sorted({row[0] for row in dataset.rows})
    if len(dataset) != args.expected_samples or episodes != list(range(args.expected_episodes)):
        raise ValueError(f'Census cardinality differs: {len(dataset)} anchors, episodes {episodes}')
    if args.stop_index is None:
        args.stop_index = len(dataset)
    if not 0 <= args.start_index < args.stop_index <= len(dataset):
        raise ValueError('Invalid contiguous census range')
    event(args.output, 'cardinality_verified', samples=len(dataset), episodes=len(episodes),
        start=args.start_index, stop=args.stop_index, checkpoint=str(checkpoint))
    lineage = dict(schema='jepa-query-census-lineage-v1', checkpoint=str(checkpoint), checkpoint_step=30000,
        checkpoint_sha256=digest_file(checkpoint), config_sha256=digest_file(config_path),
        base_cache_identity=dataset.identity_sha256, observation_cache_identity=dataset.observation_identity_sha256,
        source_helper_sha256=digest_file(ROOT / 'src/fastwam/cosmos/jepa_query_census.py'),
        script_sha256=digest_file(__file__), sample_count=len(dataset), episode_count=len(episodes),
        start=args.start_index, stop=args.stop_index, shard_size=args.shard_size, steps=args.steps, seed=args.seed,
        seed_policy='per-example original B1 draws replicated across batch positions',
        aggregation='mean_over_16_denoising_steps', attention_dtype='float16',
        action_rms_tolerance=args.batch_action_rms_tolerance,
        attention_rms_tolerance=args.batch_attention_rms_tolerance,
        requested_batch_size=args.batch_size)
    identity = hashlib.sha256(json.dumps(lineage, sort_keys=True).encode()).hexdigest()
    run_path = args.output / 'run.json'
    if run_path.exists():
        previous = json.loads(run_path.read_text())
        if previous['run_identity'] != identity or previous['lineage'] != lineage:
            raise ValueError('Resume lineage differs; do not mix checkpoints/configs/seeds/code')
        if previous['gpu'] != torch.cuda.get_device_name() or previous['torch_version'] != torch.__version__:
            raise ValueError('Resume requires the same GPU model and Torch version as its benchmark')
    else:
        atomic_json(run_path, dict(schema='jepa-query-census-run-v1', run_identity=identity, lineage=lineage,
            gpu=torch.cuda.get_device_name(), torch_version=torch.__version__,
            training_distribution_only=True, labels='concentration/redundancy descriptors, not force-contact correctness',
            observed_offsets=[-4, 0], temporal_scope='single fused tubelet',
            no_future_targets_in_query=True, created_at=datetime.now(timezone.utc).isoformat()))
    model_cfg = dict(cfg['model'])
    if model_cfg.pop('_target_') != 'fastwam.cosmos.jepa_force_latent.create_jepa_force_latent_cosmos':
        raise ValueError('Exact trained JEPA factory required')
    model_cfg.pop('_recursive_', None)
    model_cfg['device'] = 'cuda'
    event(args.output, 'construct_model_and_strict_restore')
    model = create_jepa_force_latent_cosmos(**model_cfg)
    if model.load_checkpoint(checkpoint) != 30000:
        raise ValueError('Expected step30000 trained checkpoint')
    model.eval().requires_grad_(False)
    model.gradient_checkpointing = False
    benchmark_path = args.output / 'benchmark.json'
    if benchmark_path.exists():
        measured = json.loads(benchmark_path.read_text())
        if measured['run_identity'] != identity:
            raise ValueError('Benchmark lineage mismatch')
        event(args.output, 'resume_reusing_verified_benchmark', batch_size=measured['selected_batch_size'])
    else:
        measured = benchmark(model, dataset, args, identity)
    if args.mode == 'benchmark':
        event(args.output, 'benchmark_complete', status='benchmark_complete', planned_samples=args.stop_index - args.start_index,
            selected_batch_size=measured['selected_batch_size'], estimated_hours=measured['estimated_hours_with_20percent_io_margin'])
        return
    batch_size = measured['selected_batch_size']
    shard_dir = args.output / 'shards'
    shard_dir.mkdir(exist_ok=True)
    completed, archived_bytes = 0, 0
    shard_index = []
    for start in range(args.start_index, args.stop_index, args.shard_size):
        stop = min(start + args.shard_size, args.stop_index)
        archive = shard_dir / f'shard_{start:06d}_{stop:06d}.tar.gz'
        if archive.exists():
            receipt = verify_shard(archive, start, stop, identity)
            atomic_json(shard_dir / f'shard_{start:06d}_{stop:06d}.json', receipt)
            completed += stop - start
            archived_bytes += receipt['bytes']
            shard_index.append(receipt)
            atomic_json(args.output / 'shards.json', dict(schema='jepa-query-census-shards-v1',
                run_identity=identity, planned_samples=args.stop_index-args.start_index,
                completed_samples=completed, shards=shard_index))
            event(args.output, 'verified_existing_shard', completed=completed, total=args.stop_index - args.start_index, shard=archive.name)
            continue
        if (time.monotonic() - started) / 3600 >= args.max_hours:
            event(args.output, 'time_budget_checkpoint', status='continuation_required', completed=completed,
                total=args.stop_index - args.start_index, next_index=start, archived_bytes=archived_bytes)
            return
        rows, maps, summaries, thumbnails, actions, forces = [], [], [], {}, [], []
        for begin in range(start, stop, batch_size):
            end = min(begin + batch_size, stop)
            samples = [dataset[index] for index in range(begin, end)]
            prediction, attention = run_forward(model, samples, args, pad_to=batch_size)
            for offset, index in enumerate(range(begin, end)):
                ep, frame, lo, hi = dataset.rows[index]
                row = dict(sample_index=index, episode_index=ep, frame_index=frame, segment_start=lo,
                    segment_stop=hi, observed_frame_indices=[max(lo, frame - 4), frame])
                rows.append(row)
                maps.append(attention[offset].astype(np.float16))
                summary = dict(sample_index=index, episode_index=ep, frame_index=frame,
                    **attention_metrics(attention[offset]))
                summary['fp16_max_absolute_quantization_error'] = float(np.max(np.abs(attention[offset] - maps[-1].astype(np.float32))))
                summaries.append(summary)
                actions.append(prediction['action'][offset])
                forces.append(prediction['force'][offset])
                for view, name in enumerate(('main', 'wrist')):
                    thumbnails[f'thumbnails/{index:06d}-{name}.jpg'] = jpeg_bytes(samples[offset], view)
            if end == stop or end - start < batch_size + 1 or (end - start) % (batch_size * 10) == 0:
                event(args.output, 'inference', completed=completed, uncommitted_in_shard=end-start,
                    total=args.stop_index - args.start_index, next_index=end, batch_size=batch_size,
                    elapsed_hours=(time.monotonic() - started) / 3600)
        event(args.output, 'publish_shard', start=start, stop=stop)
        receipt = write_shard(shard_dir, start, rows, maps, summaries, thumbnails, actions, forces, identity)
        completed += stop - start
        archived_bytes += receipt['bytes']
        shard_index.append(receipt)
        atomic_json(args.output / 'shards.json', dict(schema='jepa-query-census-shards-v1',
            run_identity=identity, planned_samples=args.stop_index-args.start_index,
            completed_samples=completed, shards=shard_index))
        event(args.output, 'shard_committed', completed=completed, total=args.stop_index - args.start_index,
            next_index=stop, archived_bytes=archived_bytes, shard=receipt)
    event(args.output, 'complete', status='complete', completed=completed,
        total=args.stop_index - args.start_index, all_training_anchors=completed == len(dataset), archived_bytes=archived_bytes)


def main(argv=None):
    args = arguments(argv)
    for key in ('HF_HUB_OFFLINE', 'TRANSFORMERS_OFFLINE', 'HF_DATASETS_OFFLINE'):
        os.environ[key] = '1'
    args.output.mkdir(parents=True, exist_ok=True)
    import fcntl
    lock = (args.output / '.census.lock').open('a+')
    try:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        lock.close()
        raise RuntimeError('Another process is already writing this census output')
    try:
        execute(args)
    except BaseException as error:
        failure = dict(status='failed', error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc(),
            updated_at=datetime.now(timezone.utc).isoformat())
        print(json.dumps(failure), flush=True)
        args.output.mkdir(parents=True, exist_ok=True)
        atomic_json(args.output / 'failure.json', failure)
        atomic_json(args.output / 'progress.json', failure)
        raise
    finally:
        lock.close()


if __name__ == '__main__':
    main()
