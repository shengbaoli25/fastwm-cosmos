"""Paired frozen-backbone JEPA reader adaptation at the plug 25k checkpoint.

Prepare copies exact training-cache slices outside the 12 diagnostic episodes.
Train precomputes the frozen ViT-L patches and clean online Force targets once.
Arms differ only in the slot-to-condition RMS ratio. This auxiliary-loss study
does not optimize action flow matching. Its endpoint policy runs are full 16-call
joint inference on an episode-disjoint adaptation evaluation split.
"""
from __future__ import annotations

import argparse
from bisect import bisect_right
from collections import defaultdict
from copy import deepcopy
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
import torch
from omegaconf import OmegaConf
from safetensors import safe_open
from safetensors.torch import save_file

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'deploy'), str(ROOT)]
from fastwam.cache.latent import cache_identity_sha256, file_sha256
from fastwam.cosmos.assets import TOKENIZER_SHA
from fastwam.cosmos.cache import SCHEMA as BASE_SCHEMA, _path
from fastwam.cosmos.jepa_force_latent import SCHEMA as CHECKPOINT_SCHEMA
from fastwam.cosmos.jepa_training_data import OBSERVATION_SCHEMA, _read_manifest, anchor_rows, observation_indices
from fastwam.cosmos.noaction import NoActionJointModel


SOURCE_SHA = 'b24b701239f54ad38d16c819b9ea9f6fad5f9374a683a84a68d2cd7ef27067a8'
CHECKPOINT_STEP = 25000
CHECKPOINT_SHA = 'efb9d666b1593753974af8967065f701c2953e017ce42817116759a09ee1c958'
VALIDATION_BUNDLE_SHA = 'edb28fc1d865be0572fe11f8b6eace7ff213cb536e667512d6544730f546d844'
VALIDATION_EPISODES = tuple(range(0, 78, 7))
PACKET_SCHEMA = 'jepa-plug-reader-ablation-samples-v1'
LOCAL_PACKET_SCHEMA = 'jepa-plug-reader-ablation-local-reconstructed-samples-v1'
PRODUCTION_BASE_MANIFEST_SHA = '7f16008720ae08f7438a966681bb1630936c191de48877d64fdf5534b848946b'
NUMERIC = ('first_frame_latents', 'history', 'history_is_pad', 'history_channel_valid', 'cosmos_text_index',
           'proprio', 'action', 'action_is_pad', 'wrench_future', 'wrench_valid', 'force_is_pad',
           'sample_index', 'episode_index', 'frame_index')


def atomic_json(path, data):
    partial = path.with_suffix(path.suffix + '.tmp')
    partial.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    partial.replace(path)


def event(**data):
    print(json.dumps(data, allow_nan=False), flush=True)


def read_config(path):
    config = OmegaConf.to_container(OmegaConf.load(path), resolve=True, throw_on_missing=True)
    if config.get('jepa_experiment', {}).get('raw_source_sha256') != SOURCE_SHA:
        raise ValueError('This experiment requires the original plug78 training configuration')
    return config


def open_selection(path):
    if file_sha256(path) != VALIDATION_BUNDLE_SHA:
        raise ValueError('Unexpected canonical 36-window evaluation bundle')
    with safe_open(str(path), framework='pt', device='cpu') as handle:
        manifest = json.loads(handle.metadata()['manifest'])
    rows = manifest['samples']
    if len(rows) != 36 or tuple(sorted({row['episode_index'] for row in rows})) != VALIDATION_EPISODES:
        raise ValueError('Unexpected evaluation window/episode split')
    return manifest


def choose_anchors(rows, count, seed, excluded_episodes):
    eligible = defaultdict(list)
    for index, (episode, frame, start, stop) in enumerate(rows):
        if episode not in excluded_episodes and frame >= start + 31 and frame + 32 < stop:
            eligible[episode].append(index)
    if len(eligible) < 24 or count < len(eligible) or count > sum(map(len, eligible.values())):
        raise ValueError('Need at least one full 32-step window per adaptation episode')
    rng = np.random.default_rng(seed)
    shuffled = {episode: rng.permutation(eligible[episode]).tolist() for episode in sorted(eligible)}
    selected = []
    while len(selected) < count:
        for episode in rng.permutation(list(shuffled)):
            if shuffled[episode]:
                selected.append(shuffled[episode].pop())
            if len(selected) == count:
                break
    return sorted(selected)


def manifest_sources(config, validation):
    jepa = config['data']['train']
    base = jepa['base_config']
    root, manifest = _read_manifest(base['cache_root'], BASE_SCHEMA, base['expected_identity_sha256'])
    rgb_root, rgb = _read_manifest(jepa['observation_cache_root'], OBSERVATION_SCHEMA, jepa['expected_observation_identity'])
    identity = manifest['identity']
    source = identity['realman_source']
    obs = rgb['identity']
    if (manifest['identity_sha256'] != validation['base_cache_identity']
            or rgb['identity_sha256'] != validation['observation_cache_identity']
            or obs['base_identity_sha256'] != manifest['identity_sha256']
            or identity['text_identity'] != validation['text_identity']
            or source['source_inventory_sha256'] != SOURCE_SHA
            or source['source_inventory_sha256'] != validation['source_inventory_sha256']
            or obs['camera_order'] != ['cam_main', 'cam_wrist'] or obs['offsets'] != [-4, 0]
            or obs['storage'] != 'uint8_round_after_baseline_resize'):
        raise ValueError('Training-cache and validation-cache lineage differ')
    rows = anchor_rows(identity)
    if len(rows) != 84349 or len(identity['episode_lengths']) != 78:
        raise ValueError('Expected 84,349 retained anchors from 78 plug demonstrations')
    shards = manifest['shards']
    stop = 0
    for entry in shards:
        if entry['start'] != stop or entry['stop'] <= stop:
            raise ValueError('Latent shard inventory is not contiguous')
        stop = entry['stop']
    if stop != len(rows):
        raise ValueError('Latent shard inventory does not cover the anchor index')
    return root, manifest, rgb_root, rgb, rows


def prepare(args):
    started = time.monotonic()
    config = read_config(args.config)
    validation = open_selection(args.validation)
    root, manifest, rgb_root, rgb, rows = manifest_sources(config, validation)
    selected = choose_anchors(rows, args.windows, args.seed, VALIDATION_EPISODES)
    entries = manifest['shards']
    shard_starts = [entry['start'] for entry in entries]
    by_shard = defaultdict(list)
    for index in selected:
        by_shard[bisect_right(shard_starts, index) - 1].append(index)
    observations, records, payload = {}, [], []
    for shard_idx, indices in by_shard.items():
        entry = entries[shard_idx]
        path = _path(root, entry['path'])
        if file_sha256(path) != entry['sha256']:
            raise ValueError(f'Feature payload SHA256 differs: {path}')
        with safe_open(str(path), framework='pt', device='cpu') as shard:
            for index in indices:
                episode, frame, start, stop = rows[index]
                if episode not in observations:
                    obs = rgb['episodes'][episode]
                    pixels = _path(rgb_root, obs['path'])
                    if obs['episode_index'] != episode or file_sha256(pixels) != obs['sha256']:
                        raise ValueError(f'Observed RGB payload identity differs: {pixels}')
                    tensor = np.load(pixels, mmap_mode='r', allow_pickle=False)
                    if tensor.dtype != np.uint8 or list(tensor.shape) != obs['shape'] or tensor.shape[1:] != (2, 3, 224, 224):
                        raise ValueError(f'Invalid observed RGB layout: {pixels}')
                    observations[episode] = tensor
                item = {key: shard.get_slice(key)[index - entry['start']].contiguous().clone() for key in NUMERIC}
                if tuple(int(item[key]) for key in ('sample_index', 'episode_index', 'frame_index')) != (index, episode, frame):
                    raise ValueError(f'Cached latent anchors were substituted at {index}')
                frames = observation_indices(frame, start, stop)
                item['observation_uint8'] = torch.from_numpy(np.array(observations[episode][frames], copy=True))
                records.append(dict(sample_index=index, episode_index=episode, frame_index=frame,
                                    segment_start=start, segment_stop=stop, observed_frame_indices=frames))
                payload.append(item)
        if shard_idx == len(entries) - 1 or len(payload) == len(selected) or len(payload) % 64 < len(indices):
            event(status='packet_progress', completed=len(payload), total=len(selected))
    metadata = dict(schema=PACKET_SCHEMA, samples=records, train_episodes=sorted(observations),
                    validation_episodes=list(VALIDATION_EPISODES), selection_seed=args.seed,
                    selection='episode-balanced random draw from full-context 32-step windows, without replacement',
                    source_inventory_sha256=SOURCE_SHA, base_cache_identity=manifest['identity_sha256'],
                    observation_cache_identity=rgb['identity_sha256'], config_sha256=file_sha256(args.config),
                    text_identity=validation['text_identity'])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    partial = args.output.with_suffix('.safetensors.tmp')
    if args.output.exists() or partial.exists():
        raise FileExistsError(args.output)
    save_file({key: torch.stack([item[key] for item in payload]) for key in payload[0]}, str(partial),
              metadata={'manifest': json.dumps(metadata, separators=(',', ':'))})
    partial.rename(args.output)
    receipt = dict(status='packet_ready', samples=len(records), train_episodes=len(observations),
                   bytes=args.output.stat().st_size, sha256=file_sha256(args.output),
                   config_sha256=metadata['config_sha256'], seconds=time.monotonic() - started)
    atomic_json(args.output.with_suffix('.receipt.json'), receipt)
    event(**receipt)


def load_evaluator(path):
    spec = importlib.util.spec_from_file_location('query_scale_contact_support', path)
    if spec is None or spec.loader is None:
        raise ValueError(f'Cannot import the established evaluation support at {path}')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def check_checkpoint(args, config):
    if file_sha256(args.checkpoint) != CHECKPOINT_SHA:
        raise ValueError('Expected the verified weak de-duplication plug 25k checkpoint')
    payload = torch.load(args.checkpoint, map_location='cpu', weights_only=True, mmap=True)
    if payload.get('schema') != CHECKPOINT_SCHEMA or payload.get('step') != CHECKPOINT_STEP or 'optimizer' in payload:
        raise ValueError('Invalid starting checkpoint schema/step')
    manifest = payload['manifest']
    config_model = config['model']
    if (manifest['dataset_identity'] != config_model['expected_data_identity']
            or manifest['normalization_sha256'] != config_model['expected_normalization_sha256']
            or manifest['text_identity'] != config_model['expected_text_identity']):
        raise ValueError('Starting weights do not belong to the supplied configuration')
    objective = manifest['jepa_force_latent']['query_redundancy']
    if (objective['margin'], objective['weight'], objective['warmup_steps']) != (.95, .01, 2500):
        raise ValueError('Invalid starting weak de-duplication objective')
    return payload


@torch.no_grad()
def precompute(model, evaluator, path, rows, chunk, *, training):
    """Store only terms independent of the trainable reader and action noise."""
    parts = defaultdict(list)
    raw_batches = []
    dtype, device = model.torch_dtype, model.device
    with safe_open(str(path), framework='pt', device='cpu') as packet:
        for start in range(0, len(rows), chunk):
            batch_rows = rows[start:start + chunk]
            examples = [evaluator.read_sample(packet, start + offset, row) for offset, row in enumerate(batch_rows)]
            batch = {key: torch.stack([example[key] for example in examples]) for key in examples[0]}
            if training and (batch['action_is_pad'].any() or batch['history_is_pad'].any()
                             or batch['force_is_pad'].any() or not batch['wrench_valid'].all()):
                raise ValueError('Adaptation packet contains an incomplete window')
            if not training:
                raw_batches.extend(examples)
            conditions = NoActionJointModel._conditions(model, batch)
            target = model.clean_force_target(batch, conditions)
            patches = model.dit['jepa_observation'](batch['jepa_observation'].to(device)).to(dtype)
            history_valid = batch['history_channel_valid'] & ~batch['history_is_pad'][..., None]
            proprio = batch['proprio'][:, 0] if batch['proprio'].ndim == 3 else batch['proprio']
            values = dict(patches=patches, target=target, task=conditions[0].mean(1, keepdim=True),
                          history=batch['history'].to(device, dtype), history_valid=history_valid.to(device),
                          proprio=proprio.to(device, dtype), action=batch['action'].to(device, dtype),
                          wrench_future=batch['wrench_future'].to(device), wrench_valid=batch['wrench_valid'].to(device),
                          force_is_pad=batch['force_is_pad'].to(device))
            for key, value in values.items():
                parts[key].append(value.detach())
            event(status='frozen_cache_progress', split='adaptation' if training else 'evaluation',
                  completed=start + len(batch_rows), total=len(rows))
    return {key: torch.cat(values) for key, values in parts.items()}, raw_batches


class CachedForceObjective:
    """Call the existing auxiliary objective with the exact frozen label pass result."""

    def __init__(self, model):
        self.model = model
        self.training = True
        self._auxiliary_step = CHECKPOINT_STEP

    def __getattr__(self, name):
        return getattr(self.model, name)

    def clean_force_target(self, sample, conditions):
        return conditions['target']

    def loss(self, cache, context):
        return self.model.__class__._auxiliary_training_loss(self, cache,
            dict(action_force_context=context, conditions={'target': cache['target']}))


def reader_loss(reader, cache, noisy_action, clock, objective):
    with torch.autocast(device_type=cache['patches'].device.type, dtype=torch.bfloat16):
        patches = cache['patches']
        valid = torch.ones(patches.shape[:2], dtype=torch.bool, device=patches.device)
        observation = reader.prepare_observation(patches, valid, cache['history'],
            cache['history_valid'], cache['proprio'], cache['task'])
        result = reader(noisy_action, clock, observation, query_redundancy_margin=objective.query_redundancy_margin)
        return objective.loss(cache, result)


def noisy_candidates(cache, scheduler, generator):
    action = cache['action']
    unit = torch.rand((len(action),), device=action.device, dtype=torch.float32, generator=generator)
    clock = (scheduler._phi(unit, scheduler.shift) * scheduler.num_train_timesteps).to(action.dtype)
    noise = torch.randn(action.shape, device=action.device, dtype=action.dtype, generator=generator)
    return scheduler.add_noise(action, noise, clock), clock


def batch_indices(size, batch_size, updates, seed):
    generator = torch.Generator(device='cpu').manual_seed(seed)
    result = []
    while len(result) < updates:
        permutation = torch.randperm(size, generator=generator)
        result.extend(permutation.split(batch_size))
    return torch.stack(result[:updates])


def index_cache(cache, index):
    return {key: value[index] for key, value in cache.items()}


@torch.no_grad()
def validate_aux(reader, cache, objective, seed):
    # All 36 held-fixed windows share the same eight draws across checkpoints and arms.
    generator = torch.Generator(device=cache['action'].device).manual_seed(seed)
    totals = defaultdict(float)
    reader.eval()
    for _ in range(8):
        noisy_action, clock = noisy_candidates(cache, objective.a_scheduler, generator)
        loss, metrics = reader_loss(reader, cache, noisy_action, clock, objective)
        for key, value in dict(loss=float(loss), **metrics).items():
            totals[key] += value / 8
    reader.train()
    return dict(totals)


def save_reader(reader, path, arm, step, packet_sha, args):
    state = {key: value.detach().cpu().contiguous() for key, value in reader.state_dict().items()}
    config = dict(schema='jepa-plug-reader-ablation-weights-v1', starting_checkpoint_sha256=CHECKPOINT_SHA,
                  train_packet_sha256=packet_sha, arm=arm, adaptation_updates=step,
                  query_slot_condition_ratio=reader.query_slot_condition_ratio,
                  validation_bundle_sha256=VALIDATION_BUNDLE_SHA,
                  learning_rate=args.learning_rate, batch_size=args.batch_size, seed=args.seed)
    partial = path.with_suffix('.safetensors.tmp')
    save_file(state, str(partial), metadata={'ablation': json.dumps(config, separators=(',', ':'))})
    partial.rename(path)


@torch.no_grad()
def policy_eval(model, reader, evaluator, rows, samples, path, *, seed, roi, arm, updates, reader_sha, validation):
    """Evaluate the unmodified 16-call sampler with just this trained reader swapped in."""
    prior = model.dit['jepa_reader']
    path.mkdir()
    for sub in ('attention', 'raw', 'heatmaps', 'predictions'):
        (path / sub).mkdir()
    model.dit['jepa_reader'] = deepcopy(reader).to(device=model.device, dtype=model.torch_dtype).requires_grad_(False).eval()
    metrics = []
    try:
        atomic_json(path / 'samples.json', dict(schema='jepa-contact-eval-samples-v1', samples=rows,
                    selection='fixed 36 plug windows from 12 episodes excluded from reader adaptation',
                    base_cache_identity=validation['base_cache_identity'],
                    observation_cache_identity=validation['observation_cache_identity'],
                    train_distribution_only=True, inference_target_keys_excluded=True))
        atomic_json(path / 'run.json', dict(schema='jepa-contact-eval-run-v1',
                    starting_checkpoint_sha256=CHECKPOINT_SHA, checkpoint_step=CHECKPOINT_STEP,
                    checkpoint_sha256=reader_sha, checkpoint=f'paired_reader_{arm}_{updates:04d}',
                    adaptation_updates=updates, reader_arm=arm, variant='vitl',
                    query_slot_condition_ratio=reader.query_slot_condition_ratio,
                    token_layout=evaluator.token_layout(), seed=seed, inference_steps=16,
                    sample_bundle_sha256=VALIDATION_BUNDLE_SHA, contact_roi_sha256=file_sha256(roi)))
        duplicate_by_episode = defaultdict(list)
        for row, sample in zip(rows, samples, strict=True):
            with evaluator.SlotCapture(model.dit['jepa_reader']) as capture:
                predicted = model.infer_joint(evaluator.observed_input(sample), num_inference_steps=16,
                                              seed=seed, force_seed=seed + 1, video_seed=seed + 2, use_cache=True)
            view = evaluator.save_attention(path, row, capture, 16, len(prior.readers))
            spatial = view.reshape(16, 2, 4, 8, 2, 196).transpose(0, 1, 2, 4, 3, 5).astype(np.float64)
            norm = np.linalg.norm(spatial, axis=-1, keepdims=True)
            unit = spatial / np.maximum(norm, 1e-20)
            i, j = np.triu_indices(8, 1)
            cosine = (unit @ unit.swapaxes(-1, -2))[..., i, j]
            duplicate_by_episode[row['episode_index']].append(np.stack([
                cosine.mean(axis=(0, 2, 4)), (cosine > .95).mean(axis=(0, 2, 4)),
                spatial.sum(-1).mean(axis=(0, 2, 4)),
            ]))
            key = f'{row["sample_index"]:06d}'
            evaluator.save_raw_sheet(path / 'raw' / f'{key}.png', sample, row)
            evaluator.heat_sheet(path / 'heatmaps' / f'{key}.png', sample, row, view)
            np.savez_compressed(path / 'predictions' / f'{key}-baseline.npz',
                                **{name: predicted[name].detach().float().cpu().numpy()
                                   for name in ('action', 'force', 'video_latents')})
            metric = dict(**row, condition='baseline', **evaluator.target_errors(predicted, sample),
                          action_change_rms_normalized=0., force_change_max_normalized=0., video_latent_change_max=0.)
            metrics.append(metric)
            with (path / 'metrics.jsonl').open('a', encoding='utf-8') as stream:
                stream.write(json.dumps(metric, allow_nan=False) + '\n')
            event(status='policy_validation_progress', arm=arm, updates=updates, completed=len(metrics), total=len(rows))
        scores = evaluator.summarize(path, roi)
        duplicate = np.stack([np.stack(values).mean(axis=0) for values in duplicate_by_episode.values()]).mean(axis=0)
        result = dict(contact=scores['contact_cluster_ci_by_view_and_layer'],
                      query_cosine_by_layer_view=duplicate[0].tolist(),
                      queries_above_095_by_layer_view=duplicate[1].tolist(),
                      camera_mass_by_layer_view=duplicate[2].tolist())
        for name in ('action_mse_normalized', 'force_mse_normalized'):
            values = defaultdict(list)
            for metric in metrics:
                if metric[name] is not None:
                    values[metric['episode_index']].append(metric[name])
            result[name] = dict(mean=float(np.mean([np.mean(v) for v in values.values()])) if values else None,
                                episodes=len(values), samples=sum(map(len, values.values())))
        atomic_json(path / 'progress.json', dict(status='complete', completed=len(metrics), total=len(rows)))
        return result
    finally:
        model.dit['jepa_reader'] = prior


def compare_predictions(output, rows, updates):
    """Measure policy change and guard the no-writeback Force/Video contract."""
    records = []
    for row in rows:
        name = f'{row["sample_index"]:06d}-baseline.npz'
        left = output / 'A' / f'rollout_step_{updates:04d}' / 'predictions' / name
        right = output / 'B' / f'rollout_step_{updates:04d}' / 'predictions' / name
        with np.load(left, allow_pickle=False) as a, np.load(right, allow_pickle=False) as b:
            action = a['action'] - b['action']
            records.append(dict(sample=row['sample_index'], episode=row['episode_index'],
                                action_change_rms_normalized=float(np.sqrt(np.mean(action**2))),
                                force_change_max_normalized=float(np.max(np.abs(a['force'] - b['force']))),
                                video_latent_change_max=float(np.max(np.abs(a['video_latents'] - b['video_latents'])))))
    return dict(anchors=records,
                max_force_change=max(row['force_change_max_normalized'] for row in records),
                max_video_change=max(row['video_latent_change_max'] for row in records))


def policy_differences(result, step):
    """Positive values mean B exceeds A; calculate change in the gap separately."""
    a, b = (result['arms'][arm]['checkpoints'][str(step)]['full_policy_eval'] for arm in ('A', 'B'))
    diff = {}
    for name in ('action_mse_normalized', 'force_mse_normalized'):
        diff[name] = b[name]['mean'] - a[name]['mean'] if None not in (a[name]['mean'], b[name]['mean']) else None
    for name in ('query_cosine_by_layer_view', 'queries_above_095_by_layer_view', 'camera_mass_by_layer_view'):
        diff[name] = (np.asarray(b[name]) - a[name]).tolist()
    diff['roi_enrichment'] = {name: b['contact'][name]['mean'] - a['contact'][name]['mean']
                              for name in a['contact'].keys() & b['contact'].keys()}
    return diff


def verify_local_reconstruction(args, packet, packet_sha, validation, config):
    """Verify this raw reconstruction against its own receipt and the pinned training inventory."""
    with args.local_reconstruction_receipt.open(encoding='utf-8-sig') as source:
        receipt = json.load(source)
    common = dict(schema=LOCAL_PACKET_SCHEMA, producer='local_reconstructed', selection_seed=812,
                  production_base_manifest_sha256=PRODUCTION_BASE_MANIFEST_SHA,
                  source_inventory_sha256=SOURCE_SHA, config_sha256=file_sha256(args.config),
                  reference_base_cache_identity=validation['base_cache_identity'],
                  reference_observation_cache_identity=validation['observation_cache_identity'],
                  text_identity=validation['text_identity'], tokenizer_sha256=TOKENIZER_SHA,
                  normalization_sha256=config['model']['expected_normalization_sha256'])
    if (any(packet.get(key) != value for key, value in common.items())
            or receipt.get('status') != 'packet_ready' or receipt.get('samples') != 512
            or receipt.get('config_sha256') != common['config_sha256'] or receipt.get('train_episodes') != 66
            or receipt.get('sha256') != packet_sha or receipt.get('bytes') != args.train_packet.stat().st_size
            or any(key in packet for key in ('base_cache_identity', 'observation_cache_identity'))):
        raise ValueError('Local reconstruction receipt/hash/assets differ from the recorded contract')
    if file_sha256(args.normalization) != common['normalization_sha256']:
        raise ValueError('Local reconstruction and evaluation normalization differ')
    parity = packet['first_frame_latent_parity']
    proof = packet['raw_inventory_proof']
    rgb_parity = packet['observation_rgb_parity']
    preprocessing = {key: packet[key] for key in ('observation_rgb_producer', 'first_frame_latent_producer',
                                                 'device', 'torch_version', 'script_sha256', 'selection_runner_sha256')}
    if (not all(isinstance(value, str) and value for value in preprocessing.values())
            or not all(len(preprocessing[key]) == 64 and all(c in '0123456789abcdef' for c in preprocessing[key])
                       for key in ('script_sha256', 'selection_runner_sha256'))
            or packet.get('reference_validation_bundle_sha256') != VALIDATION_BUNDLE_SHA
            or parity.get('exact_reference_sha256') != VALIDATION_BUNDLE_SHA or parity.get('windows') != 36
            or not 0 <= parity['relative_rmse'] <= .01 or not math.isfinite(parity['max_abs'])
            or not 0 <= parity['exact_windows'] <= 36
            or rgb_parity.get('float_equal') is not True or rgb_parity.get('max_abs') != 0
            or rgb_parity.get('checked_observation_frames', 0) < 1
            or (proof.get('matching_files'), proof.get('missing_files'), proof.get('changed_files')) != (554, 0, 0)):
        raise ValueError('Local packet is missing valid raw inventory, preprocessing or VAE parity evidence')
    if file_sha256(args.production_cache_manifest) != PRODUCTION_BASE_MANIFEST_SHA:
        raise ValueError('Reference production cache manifest SHA256 differs')
    production = json.loads(args.production_cache_manifest.read_text(encoding='utf-8'))
    identity = production['identity']
    if (production['status'] != 'complete' or production['completed_samples'] != 84349
            or cache_identity_sha256(identity) != production['identity_sha256']
            or production['identity_sha256'] != common['reference_base_cache_identity']
            or identity['realman_source']['source_inventory_sha256'] != SOURCE_SHA
            or cache_identity_sha256(identity['realman_source']['files']) != SOURCE_SHA
            or cache_identity_sha256(identity['realman_source']) != packet['source_identity_sha256']):
        raise ValueError('Reference production cache source/size differs')
    rows = anchor_rows(identity)
    selection = choose_anchors(rows, 512, 812, VALIDATION_EPISODES)
    records = packet['samples']
    if [row['sample_index'] for row in records] != selection or len(packet['train_episodes']) != 66:
        raise ValueError('Local adaptation windows differ from the deterministic production index selection')
    for item, index in zip(records, selection, strict=True):
        episode, frame, start, stop = rows[index]
        if (tuple(item[k] for k in ('episode_index', 'frame_index', 'segment_start', 'segment_stop')) != (episode, frame, start, stop)
                or item['observed_frame_indices'] != observation_indices(frame, start, stop)):
            raise ValueError(f'Local adaptation anchor {index} does not match the pinned source index')
    if (args.batch_size, args.updates, args.seed, args.eval_seed, args.ratio, args.learning_rate) != (64, 1000, 42, 113, .5, 1e-4):
        raise ValueError('Local reconstruction is pinned to the planned paired adaptation recipe')
    return dict(source='local_raw_reconstruction', receipt=str(args.local_reconstruction_receipt),
                receipt_sha256=file_sha256(args.local_reconstruction_receipt),
                reference_production_cache_manifest=str(args.production_cache_manifest),
                **common, preprocessing=preprocessing, first_frame_latent_parity=parity,
                observation_rgb_parity=rgb_parity, raw_inventory_proof=proof,
                production_cached_tensors=False, validation_uses_production_cache=True)


def train(args):
    started = time.monotonic()
    config = read_config(args.config)
    validation = open_selection(args.validation)
    packet_sha = file_sha256(args.train_packet)
    if packet_sha != args.train_packet_sha256.lower():
        raise ValueError('Adaptation packet SHA256 differs from the verified receipt')
    with safe_open(str(args.train_packet), framework='pt', device='cpu') as data:
        packet = json.loads(data.metadata()['manifest'])
    rows = packet['samples']
    episodes = sorted({row['episode_index'] for row in rows})
    if (episodes != packet['train_episodes'] or len(episodes) < 24
            or len(rows) % args.batch_size or len(rows) < args.batch_size
            or len({row['sample_index'] for row in rows}) != len(rows)
            or packet['validation_episodes'] != list(VALIDATION_EPISODES)
            or set(episodes) & set(VALIDATION_EPISODES) or packet['source_inventory_sha256'] != SOURCE_SHA
            or packet['config_sha256'] != file_sha256(args.config)):
        raise ValueError('Adaptation packet differs from the fixed source/split/config contract')
    if args.local_reconstruction_receipt is None:
        if (packet['schema'] != PACKET_SCHEMA
                or any(packet[key] != validation[key] for key in ('base_cache_identity', 'observation_cache_identity', 'text_identity'))):
            raise ValueError('Adaptation packet differs from the frozen production cache')
        data_origin = dict(source='exact_production_cache', production_cached_tensors=True)
    else:
        data_origin = verify_local_reconstruction(args, packet, packet_sha, validation, config)
    if not math.isfinite(args.ratio) or args.ratio <= 0:
        raise ValueError('Arm B ratio must be positive and finite')
    evaluator = load_evaluator(args.evaluator)
    if args.upstream is None:
        args.upstream = Path(config['model']['upstream'])
    if args.text_cache is None:
        args.text_cache = Path(config['model']['text_cache'])
    payload = check_checkpoint(args, config)
    model = evaluator.load_model(config, args, payload)
    del payload
    original = model.dit['jepa_reader']
    if not hasattr(original, 'query_slot_condition_ratio') or original.query_slot_condition_ratio is not None:
        raise ValueError('Load the opt-in query-balance implementation with default disabled')
    model.requires_grad_(False).eval()
    train_cache, _ = precompute(model, evaluator, args.train_packet, rows, args.precompute_batch, training=True)
    validation_cache, eval_samples = precompute(model, evaluator, args.validation, validation['samples'],
                                                args.precompute_batch, training=False)
    args.output.mkdir(parents=True, exist_ok=False)
    schedule = batch_indices(len(rows), args.batch_size, args.updates, args.seed)
    schedule_sha = hashlib.sha256(schedule.numpy().tobytes()).hexdigest()
    objective = CachedForceObjective(model)
    milestones = sorted({s for s in (0, 250, 500, args.updates) if s <= args.updates})
    result = dict(schema='jepa-plug-query-scale-ablation-v1', status='running', train_packet_sha256=packet_sha,
                  starting_checkpoint_sha256=CHECKPOINT_SHA, starting_checkpoint_step=CHECKPOINT_STEP,
                  training_data_provenance=data_origin, training_runner_sha256=file_sha256(Path(__file__)),
                  torch_version=torch.__version__, validation_bundle_sha256=VALIDATION_BUNDLE_SHA,
                  training_variant='reader_only_frozen_backbone_auxiliary_objective',
                  objective='existing Force latent alignment + wrench SmoothL1 + weak query duplication; action FM not optimized',
                  ratio=args.ratio, updates_each_arm=args.updates, windows=len(rows), train_episodes=episodes,
                  evaluation_episodes=list(VALIDATION_EPISODES), evaluation_windows=len(eval_samples),
                  evaluation_pretraining_overlap=True, seed=args.seed, optimizer='AdamW', learning_rate=args.learning_rate,
                  weight_decay=.01, clip_grad_norm=1., batch_size=args.batch_size, schedule_sha256=schedule_sha,
                  action_clock='original shifted continuous FM training distribution', gpu=torch.cuda.get_device_name(model.device),
                  auxiliary_warmup_reference_step=CHECKPOINT_STEP, arms={})
    atomic_json(args.output / 'ablation.json', result)
    for arm, ratio in (('A', None), ('B', args.ratio)):
        reader = deepcopy(original).requires_grad_(True).float().train()
        reader.query_slot_condition_ratio = ratio
        optimizer = torch.optim.AdamW(reader.parameters(), lr=args.learning_rate, weight_decay=.01)
        generator = torch.Generator(device=model.device).manual_seed(args.seed + 1)
        directory = args.output / arm
        directory.mkdir()
        arm_record = result['arms'][arm] = dict(query_slot_condition_ratio=ratio, checkpoints={})
        arm_started = time.monotonic()
        for step in range(args.updates + 1):
            objective._auxiliary_step = CHECKPOINT_STEP + step
            if step in milestones:
                file = directory / f'reader_step_{step:04d}.safetensors'
                save_reader(reader, file, arm, step, packet_sha, args)
                entry = arm_record['checkpoints'][str(step)] = dict(
                    weights=file.name, weights_sha256=file_sha256(file),
                    aux_eval=validate_aux(reader, validation_cache, objective, args.seed + 2),
                )
                if step in (0, args.updates):
                    entry['full_policy_eval'] = policy_eval(model, reader, evaluator, validation['samples'], eval_samples,
                        directory / f'rollout_step_{step:04d}', seed=args.eval_seed, roi=args.roi, arm=arm, updates=step,
                        reader_sha=entry['weights_sha256'], validation=validation)
                atomic_json(args.output / 'ablation.json', result)
                event(status='checkpoint_complete', arm=arm, adaptation_updates=step, **entry)
            if step == args.updates:
                break
            batch = index_cache(train_cache, schedule[step].to(model.device))
            noisy_action, clock = noisy_candidates(batch, objective.a_scheduler, generator)
            optimizer.zero_grad(set_to_none=True)
            loss, metrics = reader_loss(reader, batch, noisy_action, clock, objective)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(reader.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
            if step == 0 or (step + 1) % 10 == 0:
                metric = dict(arm=arm, adaptation_updates=step + 1, total_loss=float(loss.detach()),
                              gradient_norm=float(grad_norm), seconds=time.monotonic() - arm_started, **metrics)
                with (directory / 'training.jsonl').open('a', encoding='utf-8') as stream:
                    stream.write(json.dumps(metric, allow_nan=False) + '\n')
                event(status='train_update', **metric)
        del reader, optimizer
        arm_record['seconds'] = time.monotonic() - arm_started
        atomic_json(args.output / 'ablation.json', result)
    result['paired_policy_change'] = {str(step): compare_predictions(args.output, validation['samples'], step)
                                     for step in (0, args.updates)}
    initial = policy_differences(result, 0)
    final = policy_differences(result, args.updates)
    gap_change = {}
    for key, value in final.items():
        if isinstance(value, dict):
            gap_change[key] = {name: delta - initial[key][name] for name, delta in value.items()}
        elif value is None or initial[key] is None:
            gap_change[key] = None
        else:
            gap_change[key] = (np.asarray(value) - initial[key]).tolist()
    result['policy_b_minus_a'] = {'initial': initial, 'final': final, 'change_in_gap': gap_change}
    result['frozen_stream_isolation_exact'] = all(
        change['max_force_change'] == 0 and change['max_video_change'] == 0
        for change in result['paired_policy_change'].values()
    )
    result.update(status='complete', seconds=time.monotonic() - started)
    atomic_json(args.output / 'ablation.json', result)
    event(status='complete', seconds=result['seconds'], output=str(args.output))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='phase', required=True)
    build = commands.add_parser('prepare', help='Export a deterministic episode-disjoint adaptation packet')
    run = commands.add_parser('train', help='Run both arms from the exact 25k checkpoint')
    for phase in (build, run):
        phase.add_argument('--config', type=Path, required=True)
        phase.add_argument('--validation', type=Path, required=True)
        phase.add_argument('--output', type=Path, required=True)
    build.add_argument('--windows', type=int, default=512)
    build.add_argument('--seed', type=int, default=812)
    for name in ('checkpoint', 'evaluator', 'normalization', 'train-packet', 'roi'):
        run.add_argument('--' + name, type=Path, required=True)
    run.add_argument('--train-packet-sha256', required=True)
    run.add_argument('--local-reconstruction-receipt', type=Path)
    run.add_argument('--production-cache-manifest', type=Path)
    run.add_argument('--upstream', type=Path)
    run.add_argument('--text-cache', type=Path)
    run.add_argument('--device', default='cuda:0')
    run.add_argument('--updates', type=int, default=1000)
    run.add_argument('--batch-size', type=int, default=64)
    run.add_argument('--precompute-batch', type=int, default=8)
    run.add_argument('--learning-rate', type=float, default=1e-4)
    run.add_argument('--ratio', type=float, default=.5)
    run.add_argument('--seed', type=int, default=42)
    run.add_argument('--eval-seed', type=int, default=113)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.phase == 'prepare':
        prepare(args)
        return
    if (min(args.updates, args.batch_size, args.precompute_batch) < 1 or not math.isfinite(args.learning_rate)
            or args.learning_rate <= 0 or not torch.cuda.is_available() or not torch.device(args.device).type == 'cuda'):
        raise ValueError('Training requires a CUDA device and positive update/batch/learning-rate settings')
    if bool(args.production_cache_manifest) != bool(args.local_reconstruction_receipt):
        raise ValueError('Local reconstruction requires both its receipt and pinned production-cache manifest')
    train(args)


if __name__ == '__main__':
    main()
