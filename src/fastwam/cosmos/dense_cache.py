"""Immutable two-view patch-feature cache for the 71-episode Joint8 source."""
from bisect import bisect_right
from collections import OrderedDict
import json
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

from fastwam.cache.latent import cache_identity_sha256, file_sha256
from .dense_encoder import load_dense_teacher
from .realman_joint import open_realman_source


SCHEMA = 'fastwam-dense-feature-cache-v1'
FEATURE_DIM = 768
VIEW_GRID = 14
TOKENS = 2 * VIEW_GRID * VIEW_GRID
SHARD_FRAMES = 128


def _atomic(path, payload):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=True,
                                    allow_nan=False) + '\n')
    temporary.replace(path)


def episode_partition(lengths, world_size=4):
    if world_size < 1 or not lengths:
        raise ValueError('Positive world size and episode inventory required')
    bins = [[] for _ in range(world_size)]
    sizes = [0] * world_size
    for episode in sorted(range(len(lengths)), key=lambda i: (-lengths[i], i)):
        rank = min(range(world_size), key=lambda i: (sizes[i], i))
        bins[rank].append(episode)
        sizes[rank] += lengths[episode]
    return [sorted(items) for items in bins]


def _identity(source, teacher_manifest):
    return dict(schema=SCHEMA, dataset_source_sha256=source.source_sha,
        source_identity_sha256=source.identity_sha256,
        episode_lengths=source.lengths, episodes=len(source.lengths),
        teacher=teacher_manifest, feature_dtype='bfloat16', feature_shape=[TOKENS, FEATURE_DIM],
        output_view_grid=VIEW_GRID, camera_order=['cam_main', 'cam_wrist'],
        target_offsets=[0, 16, 32], sample_stride=4,
        future_target_slots='one endpoint after each 4 stride-4 frame group',
        exporter_schema=SCHEMA)


def build_teacher_manifest(teacher_name, weight_sha256):
    if teacher_name == 'vjepa21':
        input_size, source_grid = 384, 24
        upstream_commit = '204698b45b3712590f06245fbfba32d3be539812'
    elif teacher_name == 'dinov3':
        input_size, source_grid = 224, 14
        upstream_commit = '6876159a11b4df116f30f667f8c9888617df0751'
    else:
        raise ValueError('Unsupported dense feature teacher')
    return dict(name=teacher_name, upstream_commit=upstream_commit,
        weight_sha256=weight_sha256,
        checkpoint_format='official_vjepa2_1_ema_encoder' if teacher_name == 'vjepa21'
        else 'modelscope_huggingface_safetensors_mapped_to_official_dinov3',
        preprocessing='two-view square resize bicubic antialias; ImageNet mean/std',
        input_size=input_size, source_grid=source_grid, output_grid=VIEW_GRID,
        feature_dim=FEATURE_DIM, feature_tokens=TOKENS,
        token_order='cam_main then cam_wrist, row-major')


def write_rank_cache(*, dataset, source_sha, teacher_name, teacher_repo, weights,
                     weight_sha256, root, rank, world_size=4, frame_batch=32):
    if rank not in range(world_size) or frame_batch < 1:
        raise ValueError('Invalid dense feature rank or batch size')
    source = open_realman_source(dataset, source_sha)
    if source.identity.get('action_space') != 'joint8' or len(source.episodes) != 71:
        raise ValueError('The PlugInsertion71 Joint8 dataset is required')
    teacher = load_dense_teacher(teacher_name, teacher_repo, weights, weight_sha256)
    identity = _identity(source, teacher.manifest())
    digest = cache_identity_sha256(identity)
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    shard_root = root / 'shards'
    shard_root.mkdir(exist_ok=True)
    rank_receipt = root / f'rank_{rank:02d}.json'
    if rank_receipt.exists():
        raise FileExistsError('Feature rank receipt already exists; use a fresh cache root')
    assignment = episode_partition(source.lengths, world_size)[rank]
    receipt = dict(schema=SCHEMA, rank=rank, world_size=world_size,
        episodes=assignment, identity_sha256=digest, status='running', shards=[])
    _atomic(rank_receipt, receipt)
    for episode in assignment:
        frames = source.decode_episode(episode)
        episode_shards = []
        feature_parts = []
        episode_root = shard_root / f'episode_{episode:06d}'
        episode_root.mkdir(exist_ok=True)
        for first in range(0, len(frames), frame_batch):
            last = min(len(frames), first + frame_batch)
            encoded = teacher.encode(frames[first:last].to('cuda'))
            feature_parts.append(encoded.to('cpu', torch.bfloat16))
            while sum(len(part) for part in feature_parts) >= SHARD_FRAMES:
                payload = torch.cat(feature_parts, 0)
                count = len(payload)
                take, remainder = payload[:SHARD_FRAMES], payload[SHARD_FRAMES:]
                if len(take) < SHARD_FRAMES:
                    break
                start = sum(item['stop'] - item['start'] for item in episode_shards)
                stop = start + SHARD_FRAMES
                relative = f'shards/episode_{episode:06d}/{start:06d}-{stop:06d}.safetensors'
                path = root / relative
                if path.exists():
                    raise FileExistsError('Refusing to replace feature shard: ' + str(path))
                save_file({'features': take.contiguous()}, str(path))
                episode_shards.append(dict(episode=episode, start=start, stop=stop,
                    path=relative, sha256=file_sha256(path)))
                feature_parts = [remainder] if len(remainder) else []
        if feature_parts:
            features = torch.cat(feature_parts, 0)
            start = sum(item['stop'] - item['start'] for item in episode_shards)
            stop = start + len(features)
            relative = f'shards/episode_{episode:06d}/{start:06d}-{stop:06d}.safetensors'
            path = root / relative
            if path.exists():
                raise FileExistsError('Refusing to replace feature shard: ' + str(path))
            save_file({'features': features.contiguous()}, str(path))
            episode_shards.append(dict(episode=episode, start=start, stop=stop,
                path=relative, sha256=file_sha256(path)))
        if sum(item['stop'] - item['start'] for item in episode_shards) != len(frames):
            raise ValueError('Encoded frame count differs from source episode')
        receipt['shards'].extend(episode_shards)
        receipt['completed_episodes'] = len([x for x in assignment
            if any(e['episode'] == x for e in receipt['shards'])])
        _atomic(rank_receipt, receipt)
        print(json.dumps(dict(rank=rank, episode=episode, frames=len(frames)), ensure_ascii=True), flush=True)
        source._frames = None
        source._video_episode = None
        del frames
    receipt['status'] = 'complete'
    _atomic(rank_receipt, receipt)
    return receipt


def assemble_cache(*, dataset, source_sha, teacher_name, teacher_repo, weights,
                   weight_sha256, root, world_size=4):
    source = open_realman_source(dataset, source_sha)
    identity = _identity(source, build_teacher_manifest(teacher_name, weight_sha256))
    digest = cache_identity_sha256(identity)
    root = Path(root).resolve()
    expected_episodes = episode_partition(source.lengths, world_size)
    entries = []
    for rank in range(world_size):
        receipt_path = root / f'rank_{rank:02d}.json'
        receipt = json.loads(receipt_path.read_text())
        if (receipt.get('status') != 'complete' or receipt.get('rank') != rank
                or receipt.get('world_size') != world_size or receipt.get('identity_sha256') != digest
                or receipt.get('episodes') != expected_episodes[rank]):
            raise ValueError('Incomplete/mismatched rank feature-cache receipt')
        entries.extend(receipt['shards'])
    entries.sort(key=lambda item: (item['episode'], item['start']))
    by_episode = {episode: [] for episode in range(len(source.lengths))}
    for item in entries:
        path = (root / item['path']).resolve()
        if (not path.is_relative_to(root) or not path.is_file()
                or file_sha256(path) != item['sha256']):
            raise ValueError('Missing/mutated feature shard: ' + item['path'])
        features = load_file(str(path))['features']
        if (features.shape != (item['stop'] - item['start'], TOKENS, FEATURE_DIM)
                or features.dtype != torch.bfloat16 or not bool(torch.isfinite(features).all())):
            raise ValueError('Feature shard tensor contract differs')
        by_episode[item['episode']].append(item)
    for episode, shards in by_episode.items():
        cursor = 0
        for item in shards:
            if item['start'] != cursor:
                raise ValueError('Episode feature shards have gaps/overlap')
            cursor = item['stop']
        if cursor != source.lengths[episode]:
            raise ValueError('Episode feature shard count differs from source')
    manifest = dict(identity=identity, identity_sha256=digest, status='complete',
        completed_frames=sum(source.lengths), shards=entries)
    path = root / 'manifest.json'
    if path.exists():
        raise FileExistsError('Refusing to replace a dense feature manifest')
    _atomic(path, manifest)
    return manifest


class DenseFeatureCachedDataset(torch.utils.data.Dataset):
    def __init__(self, *, dataset_root, source_sha256, feature_root,
                 expected_cache_identity, expected_teacher, expected_weight_sha256,
                 cache_capacity=4):
        self.source = open_realman_source(dataset_root, source_sha256)
        if self.source.identity.get('action_space') != 'joint8' or len(self.source.episodes) != 71:
            raise ValueError('Exactly 71 PlugInsertion Joint8 episodes required')
        self.root = Path(feature_root).resolve()
        self.manifest = json.loads((self.root / 'manifest.json').read_text())
        identity = self.manifest.get('identity', {})
        if (self.manifest.get('status') != 'complete'
                or self.manifest.get('identity_sha256') != expected_cache_identity
                or cache_identity_sha256(identity) != expected_cache_identity
                or identity.get('dataset_source_sha256') != source_sha256
                or identity.get('teacher', {}).get('name') != expected_teacher
                or identity.get('teacher', {}).get('weight_sha256') != expected_weight_sha256
                or identity.get('feature_shape') != [TOKENS, FEATURE_DIM]
                or identity.get('episode_lengths') != self.source.lengths):
            raise ValueError('Dense feature cache identity/dataset/teacher differs')
        self.shards = {episode: [] for episode in range(71)}
        for item in self.manifest['shards']:
            self.shards[item['episode']].append(item)
        for shards in self.shards.values():
            shards.sort(key=lambda item: item['start'])
        self.capacity = int(cache_capacity)
        if self.capacity < 1:
            raise ValueError('Positive feature shard cache capacity required')
        self.loaded = OrderedDict()

    def __len__(self):
        return len(self.source)

    def _frame_features(self, episode, frame):
        shards = self.shards[episode]
        starts = [item['start'] for item in shards]
        index = bisect_right(starts, frame) - 1
        if index < 0 or frame >= shards[index]['stop']:
            raise IndexError('Source frame outside feature inventory')
        item = shards[index]
        key = item['path']
        if key not in self.loaded:
            path = (self.root / key).resolve()
            if not path.is_relative_to(self.root) or file_sha256(path) != item['sha256']:
                raise ValueError('Dense feature shard SHA256 differs')
            features = load_file(str(path))['features']
            if (features.shape != (item['stop'] - item['start'], TOKENS, FEATURE_DIM)
                    or features.dtype != torch.bfloat16 or not bool(torch.isfinite(features).all())):
                raise ValueError('Dense feature shard tensor contract differs')
            self.loaded[key] = features
            while len(self.loaded) > self.capacity:
                self.loaded.popitem(last=False)
        self.loaded.move_to_end(key)
        return self.loaded[key][frame - item['start']]

    def __getitem__(self, index):
        sample = self.source[index]
        episode, frame, _, _ = self.source.rows[index]
        lo, hi = self.source.bounds[index]
        offsets = torch.tensor([0, 16, 32])
        selected = (frame + offsets).clamp(lo, hi - 1).tolist()
        sample['visual_features'] = torch.stack([
            self._frame_features(episode, selected_frame) for selected_frame in selected])
        return sample
