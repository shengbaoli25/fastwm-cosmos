"""Immutable, leak-free two-camera video targets for 71/78-episode sources."""
from bisect import bisect_right
from collections import OrderedDict
import copy
import json
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file  # load_file kept for legacy callers

from fastwam.cache.latent import cache_identity_sha256, file_sha256
from fastwam.datasets.lerobot.utils.normalizer import SingleFieldLinearNormalizer
from .dense_encoder import (build_teacher_manifest, load_dense_teacher, teacher_spec,
                            validate_teacher_assets)
from .realman_joint import open_realman_source


SCHEMA = 'fastwam-jepa-video-window-v2'
LEGACY_SCHEMA = 'fastwam-jepa-video-window-v1'
OFFSETS = (-4, 0, 4, 8, 12, 16, 20, 24, 28, 32)
GRID = 14
TOKENS = 2 * GRID * GRID
SHARD_WINDOWS = 64


def episode_splits(episode_count=71):
    """Keep historical PlugInsertion71 and fixed RM71D78 seed-42 splits."""
    if episode_count not in (71, 78):
        raise ValueError('Only the pinned 71- or 78-episode dataset is supported')
    train_stop, val_stop = (55, 63) if episode_count == 71 else (60, 69)
    order = torch.randperm(episode_count, generator=torch.Generator().manual_seed(42)).tolist()
    return dict(train=sorted(order[:train_stop]), val=sorted(order[train_stop:val_stop]),
                test=sorted(order[val_stop:]))


def valid_windows(source):
    result = []
    for index, (episode, frame, _, _) in enumerate(source.rows):
        lo, hi = source.bounds[index]
        if frame + OFFSETS[0] >= lo and frame + OFFSETS[-1] < hi:
            result.append((index, episode, frame))
    return result


def cache_teacher_manifest(name, weight_sha256):
    spec = teacher_spec(name)
    return dict(build_teacher_manifest(name, weight_sha256), video_input_size=224,
                video_tubelet=2, video_grid=GRID, feature_dim=spec['feature_dim'],
                temporal_aggregation=spec['temporal_aggregation'])


def identity(source, teacher_manifest):
    name = teacher_manifest['name']
    expected = cache_teacher_manifest(name, teacher_manifest['weight_sha256'])
    # Permit legacy manifest callers, but never permit a false teacher width.
    if any(key in teacher_manifest and teacher_manifest[key] != value
           for key, value in expected.items()):
        raise ValueError('Teacher manifest differs from the pinned backbone specification')
    return dict(schema=SCHEMA, source_sha256=source.source_sha,
        source_identity_sha256=source.identity_sha256,
        episode_lengths=source.lengths, teacher=expected,
        observation_offsets=list(OFFSETS[:2]), target_offsets=list(OFFSETS[2:]),
        video_tubelet=2, cameras=['cam_main', 'cam_wrist'],
        feature_dtype='bfloat16', feature_shape=[5, TOKENS, expected['feature_dim']],
        split_seed=42, splits=episode_splits(len(source.lengths)),
        window_rule='strict_contiguous_-4_to_+32',
        observation_encoding='separate_two_frame_forward_before_full_future_forward')


def write_json(path, payload):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=True, allow_nan=False) + '\n')
    tmp.replace(path)


def episode_partition(lengths, world_size=4):
    if world_size < 1:
        raise ValueError('Positive world size required')
    bins, totals = [[] for _ in range(world_size)], [0] * world_size
    for episode in sorted(range(len(lengths)), key=lambda i: (-lengths[i], i)):
        rank = min(range(world_size), key=lambda i: (totals[i], i))
        bins[rank].append(episode)
        totals[rank] += lengths[episode]
    return [sorted(items) for items in bins]


def _expected_counts(source):
    counts = dict.fromkeys(range(len(source.lengths)), 0)
    for _, episode, _ in valid_windows(source):
        counts[episode] += 1
    return counts


def _inventory(entries, source, *, complete):
    expected = _expected_counts(source)
    counts = dict.fromkeys(expected, 0)
    paths = set()
    for item in sorted(entries, key=lambda x: (x['episode'], x['start'])):
        ep, start, stop = item['episode'], item['start'], item['stop']
        if (ep not in counts or start != counts[ep] or stop <= start
                or stop > expected[ep] or item['path'] in paths):
            raise ValueError('Video cache shard tensor/sequence differs')
        paths.add(item['path'])
        counts[ep] = stop
    if complete and counts != expected:
        raise ValueError('Video cache window inventory incomplete')
    return counts


def _shard_path(root, item, *, check_hash=False):
    path = (root / item['path']).resolve()
    if (not path.is_relative_to(root) or not path.is_file()
            or path.stat().st_size != item['size_bytes']
            or path.stat().st_mtime_ns != item['mtime_ns']):
        raise ValueError('Immutable video cache shard metadata differs')
    if check_hash and file_sha256(path) != item['sha256']:
        raise ValueError('Video cache shard SHA256 differs')
    return path


def _validate_shard(path, item, width):
    # Read at most one window at a time: a full ViT-g shard exceeds 350 MB.
    with safe_open(str(path), framework='pt', device='cpu') as handle:
        if list(handle.keys()) != ['features']:
            raise ValueError('Video cache shard keys differ')
        feature = handle.get_slice('features')
        expected = [item['stop'] - item['start'], 5, TOKENS, width]
        if feature.get_shape() != expected:
            raise ValueError('Video cache shard tensor/sequence differs')
        for index in range(expected[0]):
            window = feature[index:index + 1]
            if window.dtype != torch.bfloat16 or not bool(torch.isfinite(window).all()):
                raise ValueError('Video cache shard dtype or finite values differ')


def write_rank_cache(*, dataset, source_sha, teacher_name, teacher_repo, weights,
                     weight_sha256, root, rank, world_size=4, frame_batch=4,
                     device='cuda', resume=False):
    teacher_spec(teacher_name)
    if world_size < 1 or rank not in range(world_size) or frame_batch < 1:
        raise ValueError('Valid teacher, rank, world size and window batch required')
    source = open_realman_source(dataset, source_sha)
    if source.identity.get('action_space') != 'joint8' or len(source.lengths) not in (71, 78):
        raise ValueError('Pinned 71/78-episode Joint8 source required')
    spec = identity(source, cache_teacher_manifest(teacher_name, weight_sha256))
    digest = cache_identity_sha256(spec)
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    receipt_path = root / f'rank_{rank:02d}.json'
    assigned = episode_partition(source.lengths, world_size)[rank]
    if receipt_path.exists():
        if not resume:
            raise FileExistsError('Video cache rank receipt already exists')
        receipt = json.loads(receipt_path.read_text())
        if (receipt.get('schema') != SCHEMA or receipt.get('identity_sha256') != digest
                or receipt.get('episodes') != assigned or receipt.get('rank') != rank
                or receipt.get('world_size') != world_size
                or receipt.get('status') not in ('running', 'complete')
                or any(item['episode'] not in assigned for item in receipt['shards'])):
            raise ValueError('Partial cache receipt belongs to another preparation')
        _inventory(receipt['shards'], source, complete=False)
        expected = _expected_counts(source)
        for item in receipt['shards']:
            if (item['start'] % SHARD_WINDOWS
                    or item['stop'] != min(item['start'] + SHARD_WINDOWS, expected[item['episode']])):
                raise ValueError('Partial cache window range differs')
            _shard_path(root, item, check_hash=True)
        if receipt['status'] == 'complete':
            expected = _expected_counts(source)
            counts = _inventory(receipt['shards'], source, complete=False)
            if any(counts[ep] != expected[ep] for ep in assigned):
                raise ValueError('Complete rank receipt is missing windows')
            validate_teacher_assets(teacher_name, teacher_repo, weights, weight_sha256)
            return receipt
    else:
        receipt = dict(schema=SCHEMA, rank=rank, world_size=world_size,
                       episodes=assigned, identity_sha256=digest, status='running', shards=[])
    teacher = load_dense_teacher(teacher_name, teacher_repo, weights, weight_sha256, device=device)
    if identity(source, teacher.manifest()) != spec:
        raise ValueError('Loaded teacher differs from requested cache backbone')
    write_json(receipt_path, receipt)
    completed = {(item['episode'], item['start']): item for item in receipt['shards']}
    windows_by_episode = {ep: [] for ep in assigned}
    for item in valid_windows(source):
        if item[1] in windows_by_episode:
            windows_by_episode[item[1]].append(item)
    width = spec['teacher']['feature_dim']
    for episode in assigned:
        windows = windows_by_episode[episode]
        if all((episode, first) in completed for first in range(0, len(windows), SHARD_WINDOWS)):
            continue
        frames = source.decode_episode(episode)
        for first in range(0, len(windows), SHARD_WINDOWS):
            chunk = windows[first:first + SHARD_WINDOWS]
            if (episode, first) in completed:
                if completed[episode, first]['stop'] != first + len(chunk):
                    raise ValueError('Partial cache window range differs')
                continue
            pieces = []
            for start in range(0, len(chunk), frame_batch):
                batch = chunk[start:start + frame_batch]
                indexes = torch.tensor([[frame + offset for offset in OFFSETS]
                                        for _, _, frame in batch], dtype=torch.long)
                clips = frames[indexes].permute(0, 2, 1, 3, 4).contiguous().to(device)
                # Context must be a separate forward: slicing a full video teacher
                # output would allow future frames to alter the observation token.
                observed = teacher.encode_video(clips[:, :, :2], context_only=True)
                future = teacher.encode_video(clips, context_only=False)[:, 1:]
                features = torch.cat((observed, future), 1).cpu().to(torch.bfloat16)
                if features.shape != (len(batch), 5, TOKENS, width) or not bool(torch.isfinite(features).all()):
                    raise ValueError('Teacher cache output geometry or finite values differ')
                pieces.append(features)
                del clips, observed, future
            tensor = torch.cat(pieces, 0).contiguous()
            relative = f'shards/episode_{episode:06d}/{first:06d}-{first + len(chunk):06d}.safetensors'
            path = root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            if path.exists():
                raise FileExistsError('Unrecorded video cache shard exists; inspect before resume')
            temporary = path.with_suffix('.safetensors.tmp')
            save_file({'features': tensor}, str(temporary))
            temporary.replace(path)
            receipt['shards'].append(dict(episode=episode, start=first,
                stop=first + len(chunk), path=relative, sha256=file_sha256(path),
                size_bytes=path.stat().st_size, mtime_ns=path.stat().st_mtime_ns))
            write_json(receipt_path, receipt)
        print(json.dumps(dict(rank=rank, episode=episode, windows=len(windows))), flush=True)
        source._frames = None
        source._video_episode = None
        del frames
    receipt['shards'].sort(key=lambda item: (item['episode'], item['start']))
    receipt['status'] = 'complete'
    write_json(receipt_path, receipt)
    return receipt


def assemble_cache(*, dataset, source_sha, teacher_name, teacher_repo, weights,
                   weight_sha256, root, world_size=4):
    # No encoder construction or checkpoint tensor loading is needed for assembly.
    validate_teacher_assets(teacher_name, teacher_repo, weights, weight_sha256)
    source = open_realman_source(dataset, source_sha)
    spec = identity(source, cache_teacher_manifest(teacher_name, weight_sha256))
    digest = cache_identity_sha256(spec)
    root = Path(root).resolve()
    entries = []
    partition = episode_partition(source.lengths, world_size)
    for rank in range(world_size):
        receipt = json.loads((root / f'rank_{rank:02d}.json').read_text())
        if (receipt.get('schema') != SCHEMA or receipt.get('status') != 'complete'
                or receipt.get('rank') != rank or receipt.get('world_size') != world_size
                or receipt.get('identity_sha256') != digest
                or receipt.get('episodes') != partition[rank]
                or any(item['episode'] not in partition[rank] for item in receipt['shards'])):
            raise ValueError('Incomplete or mismatched video cache receipt')
        entries.extend(receipt['shards'])
    entries.sort(key=lambda item: (item['episode'], item['start']))
    counts = _inventory(entries, source, complete=True)
    for item in entries:
        _validate_shard(_shard_path(root, item, check_hash=True), item,
                        spec['teacher']['feature_dim'])
    manifest = dict(identity=spec, identity_sha256=digest, status='complete',
                    completed_windows=sum(counts.values()), shards=entries)
    path = root / 'manifest.json'
    if path.exists():
        if json.loads(path.read_text()) == manifest:
            return manifest
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
        teacher = identity_.get('teacher', {})
        pinned = cache_teacher_manifest(expected_teacher, expected_weight_sha256)
        self.splits = episode_splits(len(self.source.lengths))
        self.feature_dim = teacher_spec(expected_teacher)['feature_dim']
        legacy = identity_.get('schema') == LEGACY_SCHEMA and expected_teacher.startswith('vjepa21')
        if (manifest.get('status') != 'complete'
                or identity_.get('schema') not in (SCHEMA, LEGACY_SCHEMA)
                or manifest.get('identity_sha256') != expected_cache_identity
                or cache_identity_sha256(identity_) != expected_cache_identity
                or identity_.get('source_sha256') != source_sha256
                or identity_.get('source_identity_sha256') != self.source.identity_sha256
                or identity_.get('episode_lengths') != self.source.lengths
                or identity_.get('observation_offsets') != list(OFFSETS[:2])
                or identity_.get('target_offsets') != list(OFFSETS[2:])
                or identity_.get('video_tubelet') != 2
                or identity_.get('cameras') != ['cam_main', 'cam_wrist']
                or identity_.get('feature_dtype') != 'bfloat16'
                or identity_.get('split_seed') != 42
                or identity_.get('window_rule') != 'strict_contiguous_-4_to_+32'
                or teacher.get('video_input_size') != 224
                or teacher.get('video_grid') != GRID
                or teacher.get('name') != expected_teacher
                or teacher.get('weight_sha256') != expected_weight_sha256
                or teacher.get('feature_dim') != self.feature_dim
                or identity_.get('feature_shape') != [5, TOKENS, self.feature_dim]
                or identity_.get('splits') != self.splits or split not in self.splits
                or (not legacy and identity_ != identity(self.source, pinned))
                or (legacy and teacher.get('upstream_commit', pinned['upstream_commit']) != pinned['upstream_commit'])):
            raise ValueError('Video cache identity, teacher or split differs')
        _inventory(manifest['shards'], self.source, complete=True)
        self.windows = [item for item in valid_windows(self.source)
                        if item[1] in self.splits[split]]
        # Fit every split exclusively on retained source rows from train episodes.
        train_episodes = set(self.splits['train'])
        selected = torch.tensor([start + frame for episode, frame, start, _ in self.source.rows
                                 if episode in train_episodes], dtype=torch.long)
        self.normalization = {}
        for name, values in self.source.raw.items():
            mean = values[selected].double().mean(0).float()
            std = values[selected].double().std(0, correction=0).float()
            if not bool(torch.isfinite(std).all() & (std > 0).all()):
                raise ValueError('Training-episode normalization is degenerate')
            self.normalization[name] = dict(mean=mean.tolist(), std=std.tolist())
            self.source.normalized[name] = SingleFieldLinearNormalizer(
                dict(mean=mean, std=std), mode='z-score').forward(values)
        self.by_episode = {episode: [] for episode in range(len(self.source.lengths))}
        for item in manifest['shards']:
            self.by_episode[item['episode']].append(item)
        for shards in self.by_episode.values():
            shards.sort(key=lambda item: item['start'])
        self.local_index = {}
        counts = dict.fromkeys(self.by_episode, 0)
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
        if split not in self.splits:
            raise ValueError('Unknown episode split')
        result = copy.copy(self)
        result.windows = [item for item in valid_windows(self.source)
                          if item[1] in self.splits[split]]
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
        path = _shard_path(self.root, item)
        sample = self.source[source_index]
        with safe_open(str(path), framework='pt', device='cpu') as handle:
            offset = local - item['start']
            features = handle.get_slice('features')[offset:offset + 1][0]
        if features.shape != (5, TOKENS, self.feature_dim) or features.dtype != torch.bfloat16:
            raise ValueError('Video cache shard tensor/sequence differs')
        sample['visual_features'] = features
        if self.source.raw['action'].shape[1:] == (8,):
            _, _, start, _ = self.source.rows[source_index]
            sample['raw_action_future64'] = self.source.raw['action'][start + frame:start + frame + 32].double()
        return sample
