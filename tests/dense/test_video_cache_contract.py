"""Episode split, normalization, and shard-access contracts for video cache."""

import json
import os
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from fastwam.cache.latent import cache_identity_sha256, file_sha256
from fastwam.cosmos import jepa_video_cache as cache


class FakeJointSource:
    def __init__(self):
        splits = cache.episode_splits()
        episodes = [splits['train'][0], splits['train'][1],
                    splits['val'][0], splits['test'][0]]
        self.source_sha = 'source-test'
        self.identity_sha256 = 'numeric-adapter-test'
        self.lengths = [40] * 71
        self.rows = [(episode, 4, episode * 40, 40) for episode in episodes]
        self.bounds = [(0, 40)] * len(self.rows)
        self.raw = {name: torch.zeros(71 * 40, 1) for name in ('state', 'action', 'force')}
        for index, episode in enumerate(episodes):
            value = (1., 3., 100., -100.)[index]
            for field in self.raw.values():
                field[episode * 40 + 4] = value
        self.normalized = {}

    def __getitem__(self, index):
        episode, frame, start, _ = self.rows[index]
        return {'episode_index': episode,
                'normalized_state': self.normalized['state'][start + frame].clone()}


def make_cache(tmp_path, monkeypatch):
    source = FakeJointSource()
    original_spec = cache.teacher_spec
    monkeypatch.setattr(cache, 'teacher_spec',
                        lambda name: dict(original_spec(name), feature_dim=8))
    teacher = {'name': 'vjepa21_vitg', 'weight_sha256': 'teacher-test',
               'feature_dim': 8, 'video_input_size': 224, 'video_grid': 14}
    spec = cache.identity(source, teacher)
    digest = cache_identity_sha256(spec)
    shards = []
    for _, episode, _ in cache.valid_windows(source):
        relative = f'episode_{episode:06d}.safetensors'
        path = tmp_path / relative
        features = torch.full((1, 5, cache.TOKENS, 8), episode,
                              dtype=torch.bfloat16)
        save_file({'features': features}, str(path))
        stat = path.stat()
        shards.append(dict(episode=episode, start=0, stop=1, path=relative,
                           sha256=file_sha256(path), size_bytes=stat.st_size,
                           mtime_ns=stat.st_mtime_ns))
    manifest = dict(identity=spec, identity_sha256=digest,
                    status='complete', shards=shards)
    (tmp_path / 'manifest.json').write_text(json.dumps(manifest))
    monkeypatch.setattr(cache, 'open_realman_source', lambda *_: source)
    kwargs = dict(dataset_root='unused', source_sha256=source.source_sha,
                  feature_root=tmp_path, expected_cache_identity=digest,
                  expected_teacher=teacher['name'],
                  expected_weight_sha256=teacher['weight_sha256'])
    return source, kwargs, shards


def test_train_only_normalization_and_split_views(tmp_path, monkeypatch):
    source, kwargs, _ = make_cache(tmp_path, monkeypatch)
    dataset = cache.JepaVideoCachedDataset(**kwargs, split='train')
    assert len(dataset) == 2
    assert dataset.normalization['state']['mean'] == [2.]
    assert dataset.normalization['state']['std'] == [1.]
    val, test = dataset.view('val'), dataset.view('test')
    assert len(val) == len(test) == 1
    assert val.normalization is dataset.normalization
    assert test.normalization is dataset.normalization
    assert val[0]['normalized_state'].item() == 5.  # Existing normalizer clips to 5.
    assert test[0]['normalized_state'].item() == -5.
    assert {dataset[0]['episode_index'], dataset[1]['episode_index']}.isdisjoint(
        {val[0]['episode_index'], test[0]['episode_index']})
    assert source.normalized['state'].shape == source.raw['state'].shape


def test_single_window_slice_and_immutable_metadata(tmp_path, monkeypatch):
    _, kwargs, shards = make_cache(tmp_path, monkeypatch)
    dataset = cache.JepaVideoCachedDataset(**kwargs)
    monkeypatch.setattr(cache, 'load_file',
                        lambda *_args, **_kwargs: pytest.fail('whole-shard load'))
    real_open = cache.safe_open
    accesses = []

    class SliceOnlyOpen:
        def __init__(self, *args, **kwargs):
            self.handle = real_open(*args, **kwargs)

        def __enter__(self):
            self.handle.__enter__()
            return self

        def __exit__(self, *args):
            return self.handle.__exit__(*args)

        def get_slice(self, key):
            actual_slice = self.handle.get_slice(key)

            class TrackedSlice:
                def __getitem__(self, selection):
                    accesses.append((key, selection))
                    return actual_slice[selection]

            return TrackedSlice()

        def get_tensor(self, _key):
            pytest.fail('whole-shard tensor requested')

    monkeypatch.setattr(cache, 'safe_open', SliceOnlyOpen)
    sample = dataset[0]
    assert accesses == [('features', slice(0, 1))]
    assert sample['visual_features'].shape == (5, cache.TOKENS, 8)
    assert sample['visual_features'].dtype == torch.bfloat16
    assert torch.equal(sample['visual_features'],
                       torch.full_like(sample['visual_features'],
                                       sample['episode_index']))
    path = Path(kwargs['feature_root']) / next(item['path'] for item in shards
                                               if item['episode'] == sample['episode_index'])
    stat = path.stat()
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000_000))
    with pytest.raises(ValueError, match='metadata differs'):
        dataset[0]


def test_numeric_source_identity_change_is_rejected(tmp_path, monkeypatch):
    source, kwargs, _ = make_cache(tmp_path, monkeypatch)
    source.identity_sha256 = 'different-adapter'
    with pytest.raises(ValueError, match='Video cache identity'):
        cache.JepaVideoCachedDataset(**kwargs)
