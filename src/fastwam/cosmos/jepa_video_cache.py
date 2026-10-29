"""Two-camera DINOv3/V-JEPA temporal features for PlugInsertion71."""
from bisect import bisect_right
from collections import OrderedDict
import json
from pathlib import Path
import copy

import torch
from safetensors.torch import load_file, save_file
from safetensors import safe_open

from fastwam.cache.latent import cache_identity_sha256, file_sha256
from .dense_encoder import load_dense_teacher
from .realman_joint import open_realman_source
from fastwam.datasets.lerobot.utils.normalizer import SingleFieldLinearNormalizer


SCHEMA = 'fastwam-temporal-feature-window-v2'
OFFSETS = (-4, 0, 4, 8, 12, 16, 20, 24, 28, 32)
GRID = 14
TOKENS = 2 * GRID * GRID
SHARD_WINDOWS = 64


def episode_splits():
    order = torch.randperm(71, generator=torch.Generator().manual_seed(42)).tolist()
    return dict(train=sorted(order[:55]), val=sorted(order[55:63]),
                test=sorted(order[63:]))


def valid_windows(source):
    result = []
    for index, (episode, frame, _, _) in enumerate(source.rows):
        lo, hi = source.bounds[index]
        if frame + OFFSETS[0] >= lo and frame + OFFSETS[-1] < hi:
            result.append((index, episode, frame))
    return result


def identity(source, teacher_manifest):
    return dict(schema=SCHEMA, source_sha256=source.source_sha,
        source_identity_sha256=source.identity_sha256,
        episode_lengths=source.lengths, teacher=teacher_manifest,
        observation_offsets=list(OFFSETS[:2]), target_offsets=list(OFFSETS[2:]),
        temporal_slots=[list(OFFSETS[index:index+2]) for index in range(0, len(OFFSETS), 2)],
        temporal_pooling=('adjacent_frame_mean_v1' if teacher_manifest['name'] == 'dinov3_vith16plus'
                          else 'official_video_tubelet2'),
        video_tubelet=2, cameras=['cam_main', 'cam_wrist'],
        feature_dtype='bfloat16', feature_shape=[5, TOKENS, teacher_manifest['feature_dim']],
        split_seed=42, splits=episode_splits(), window_rule='strict_contiguous_-4_to_+32')


def write_json(path, payload):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=True, allow_nan=False) + '\n')
    tmp.replace(path)


def episode_partition(lengths, world_size=4):
    bins, totals = [[] for _ in range(world_size)], [0] * world_size
    for episode in sorted(range(len(lengths)), key=lambda i: (-lengths[i], i)):
        rank = min(range(world_size), key=lambda i: (totals[i], i))
        bins[rank].append(episode)
        totals[rank] += lengths[episode]
    return [sorted(items) for items in bins]


def write_rank_cache(*, dataset, source_sha, teacher_name, teacher_repo, weights,
                     weight_sha256, root, rank, world_size=4, frame_batch=4):
    if (teacher_name not in ('vjepa21', 'vjepa21_vitg', 'dinov3_vith16plus')
            or rank not in range(world_size) or frame_batch < 1):
        raise ValueError('Valid temporal teacher, rank and window batch required')
    source = open_realman_source(dataset, source_sha)
    if source.identity.get('action_space') != 'joint8' or len(source.episodes) != 71:
        raise ValueError('PlugInsertion71 Joint8 source required')
    teacher = load_dense_teacher(teacher_name, teacher_repo, weights, weight_sha256)
    device = next(teacher.encoder.parameters()).device
    teacher_manifest = dict(teacher.manifest(), video_input_size=224,
                            video_tubelet=2, video_grid=GRID)
    spec = identity(source, teacher_manifest)
    digest = cache_identity_sha256(spec)
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    receipt_path = root / f'rank_{rank:02d}.json'
    if receipt_path.exists():
        raise FileExistsError('Video cache rank receipt already exists')
    assigned = episode_partition(source.lengths, world_size)[rank]
    receipt = dict(schema=SCHEMA, rank=rank, world_size=world_size,
                   episodes=assigned, identity_sha256=digest, status='running', shards=[])
    write_json(receipt_path, receipt)
    windows_by_episode = {ep: [] for ep in assigned}
    for item in valid_windows(source):
        if item[1] in windows_by_episode:
            windows_by_episode[item[1]].append(item)
    for episode in assigned:
        frames = source.decode_episode(episode)
        windows = windows_by_episode[episode]
        for first in range(0, len(windows), SHARD_WINDOWS):
            chunk = windows[first:first + SHARD_WINDOWS]
            pieces = []
            for start in range(0, len(chunk), frame_batch):
                batch = chunk[start:start + frame_batch]
                indexes = torch.tensor([[frame + offset for offset in OFFSETS]
                                        for _, _, frame in batch], dtype=torch.long)
                clips = frames[indexes].permute(0, 2, 1, 3, 4).contiguous().to(device)
                if teacher_name == 'dinov3_vith16plus':
                    features = teacher.encode_video(clips)
                else:
                    observed = teacher.encode_video(clips[:, :, :2], context_only=True)
                    future = teacher.encode_video(clips, context_only=False)[:, 1:]
                    features = torch.cat((observed, future), 1).to(torch.bfloat16)
                pieces.append(features.cpu())
                del clips, features
                if teacher_name != 'dinov3_vith16plus':
                    del observed, future
            tensor = torch.cat(pieces, 0).contiguous()
            relative = f'shards/episode_{episode:06d}/{first:06d}-{first+len(chunk):06d}.safetensors'
            path = root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            if path.exists():
                raise FileExistsError('Refusing to overwrite video cache shard')
            save_file({'features': tensor}, str(path))
            receipt['shards'].append(dict(episode=episode, start=first,
                stop=first+len(chunk), path=relative, sha256=file_sha256(path),
                size_bytes=path.stat().st_size, mtime_ns=path.stat().st_mtime_ns))
            write_json(receipt_path, receipt)
        print(json.dumps(dict(rank=rank, episode=episode, windows=len(windows))), flush=True)
        source._frames = None
        source._video_episode = None
        del frames
    receipt['status'] = 'complete'
    write_json(receipt_path, receipt)
    return receipt


def assemble_cache(*, dataset, source_sha, teacher_name, teacher_repo, weights,
                   weight_sha256, root, world_size=4):
    source = open_realman_source(dataset, source_sha)
    teacher = load_dense_teacher(teacher_name, teacher_repo, weights, weight_sha256, device='cpu')
    spec = identity(source, dict(teacher.manifest(), video_input_size=224,
                                 video_tubelet=2, video_grid=GRID))
    digest = cache_identity_sha256(spec)
    root = Path(root).resolve()
    entries = []
    partition = episode_partition(source.lengths, world_size)
    for rank in range(world_size):
        receipt = json.loads((root / f'rank_{rank:02d}.json').read_text())
        if (receipt.get('status') != 'complete' or receipt.get('identity_sha256') != digest
                or receipt.get('episodes') != partition[rank]):
            raise ValueError('Incomplete or mismatched video cache receipt')
        entries.extend(receipt['shards'])
    counts = {episode: 0 for episode in range(71)}
    width = spec['teacher']['feature_dim']
    for item in entries:
        path = (root / item['path']).resolve()
        if (not path.is_relative_to(root) or file_sha256(path) != item['sha256']):
            raise ValueError('Video cache shard SHA256 differs')
        feature = load_file(str(path))['features']
        if (feature.shape != (item['stop'] - item['start'], 5, TOKENS, width)
                or feature.dtype != torch.bfloat16 or not bool(torch.isfinite(feature).all())
                or item['start'] != counts[item['episode']]):
            raise ValueError('Video cache shard tensor/sequence differs')
        counts[item['episode']] = item['stop']
    expected = {episode: 0 for episode in range(71)}
    for _, episode, _ in valid_windows(source):
        expected[episode] += 1
    if counts != expected:
        raise ValueError('Video cache window inventory incomplete')
    manifest = dict(identity=spec, identity_sha256=digest, status='complete',
                    completed_windows=sum(counts.values()), shards=entries)
    path = root / 'manifest.json'
    if path.exists():
        raise FileExistsError('Refusing to overwrite video cache manifest')
    write_json(path, manifest)
    return manifest


class JepaVideoCachedDataset(torch.utils.data.Dataset):
    def __init__(self, *, dataset_root, source_sha256, feature_root,
                 expected_cache_identity, expected_teacher, expected_weight_sha256,
                 split='train', cache_capacity=2):
        self.source = open_realman_source(dataset_root, source_sha256)
        self.root = Path(feature_root).resolve()
        manifest = json.loads((self.root / 'manifest.json').read_text())
        identity_ = manifest.get('identity', {})
        if (manifest.get('status') != 'complete'
                or identity_.get('schema') != SCHEMA
                or manifest.get('identity_sha256') != expected_cache_identity
                or cache_identity_sha256(identity_) != expected_cache_identity
                or identity_.get('source_sha256') != source_sha256
                or identity_.get('source_identity_sha256') != self.source.identity_sha256
                or identity_.get('observation_offsets') != list(OFFSETS[:2])
                or identity_.get('target_offsets') != list(OFFSETS[2:])
                or identity_.get('video_tubelet') != 2
                or identity_.get('temporal_slots') != [list(OFFSETS[index:index+2])
                                                        for index in range(0, len(OFFSETS), 2)]
                or identity_.get('temporal_pooling') != (
                    'adjacent_frame_mean_v1' if expected_teacher == 'dinov3_vith16plus'
                    else 'official_video_tubelet2')
                or identity_.get('teacher', {}).get('video_input_size') != 224
                or identity_.get('teacher', {}).get('video_grid') != GRID
                or identity_.get('teacher', {}).get('name') != expected_teacher
                or identity_.get('teacher', {}).get('weight_sha256') != expected_weight_sha256
                or identity_.get('feature_shape') != [5, TOKENS, identity_.get('teacher', {}).get('feature_dim')]
                or identity_.get('splits') != episode_splits() or split not in episode_splits()):
            raise ValueError('Video cache identity, teacher or split differs')
        self.windows = [item for item in valid_windows(self.source)
                        if item[1] in identity_['splits'][split]]
        # All splits use statistics fitted exclusively on training episodes.
        train_episodes = set(episode_splits()['train'])
        selected = torch.tensor([start + frame for episode, frame, start, _ in self.source.rows
                                 if episode in train_episodes])
        self.normalization = {}
        for name, values in self.source.raw.items():
            mean = values[selected].double().mean(0).float()
            std = values[selected].double().std(0, correction=0).float()
            if not bool(torch.isfinite(std).all() & (std > 0).all()):
                raise ValueError('Training-episode normalization is degenerate')
            self.normalization[name] = dict(mean=mean.tolist(), std=std.tolist())
            self.source.normalized[name] = SingleFieldLinearNormalizer(
                dict(mean=mean, std=std), mode='z-score').forward(values)
        self.by_episode = {episode: [] for episode in range(71)}
        for item in manifest['shards']:
            self.by_episode[item['episode']].append(item)
        for shards in self.by_episode.values():
            shards.sort(key=lambda item: item['start'])
        self.local_index = {}
        counts = {episode: 0 for episode in range(71)}
        for _, episode, frame in valid_windows(self.source):
            self.local_index[(episode, frame)] = counts[episode]
            counts[episode] += 1
        self.capacity = int(cache_capacity)
        self.loaded = OrderedDict()
        if self.capacity < 1 or not self.windows:
            raise ValueError('Nonempty split and positive cache capacity required')

    def __len__(self):
        return len(self.windows)

    def view(self, split):
        if split not in episode_splits():
            raise ValueError('Unknown episode split')
        result = copy.copy(self)
        result.windows = [item for item in valid_windows(self.source)
                          if item[1] in episode_splits()[split]]
        return result

    def __getitem__(self, index):
        source_index, episode, frame = self.windows[index]
        local = self.local_index[(episode, frame)]
        shards = self.by_episode[episode]
        starts = [item['start'] for item in shards]
        pos = bisect_right(starts, local) - 1
        if pos < 0 or local >= shards[pos]['stop']:
            raise IndexError('Video window absent from cache')
        item = shards[pos]
        path = (self.root / item['path']).resolve()
        if (not path.is_relative_to(self.root) or path.stat().st_size != item['size_bytes']
                or path.stat().st_mtime_ns != item['mtime_ns']):
            raise ValueError('Immutable video cache shard metadata differs')
        sample = self.source[source_index]
        # mmap only this window; shards contain many training windows.
        with safe_open(str(path), framework='pt', device='cpu') as handle:
            offset = local-item['start']
            sample['visual_features'] = handle.get_slice('features')[offset:offset+1][0]
        if self.source.raw['action'].shape[1:] == (8,):
            _, _, start, _ = self.source.rows[source_index]
            sample['raw_action_future64'] = self.source.raw['action'][start+frame:start+frame+32].double()
        return sample
