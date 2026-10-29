"""Episode split, normalization, and shard-access contracts for video cache."""

import json
import os
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from fastwam.cache.latent import cache_identity_sha256, file_sha256
from fastwam.cosmos import jepa_video_cache as cache
from fastwam.cosmos.dense_encoder import DenseTeacher, pool_adjacent_frame_features


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


def make_cache(tmp_path, monkeypatch, teacher_name='vjepa21_vitg'):
    source = FakeJointSource()
    width = 1280 if teacher_name == 'dinov3_vith16plus' else 8
    teacher = {'name': teacher_name, 'weight_sha256': 'teacher-test',
               'feature_dim': width, 'video_input_size': 224, 'video_grid': 14}
    spec = cache.identity(source, teacher)
    digest = cache_identity_sha256(spec)
    shards = []
    for _, episode, _ in cache.valid_windows(source):
        relative = f'episode_{episode:06d}.safetensors'
        path = tmp_path / relative
        features = torch.full((1, 5, cache.TOKENS, width), episode,
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


def test_dinov3_adjacent_frame_pooling_preserves_time_and_patch_order():
    features = torch.empty(2, 10, cache.TOKENS, 3, dtype=torch.bfloat16)
    for frame in range(10):
        features[:, frame] = frame
    pooled = pool_adjacent_frame_features(features)
    assert pooled.shape == (2, 5, cache.TOKENS, 3)
    assert pooled.dtype == torch.bfloat16
    torch.testing.assert_close(pooled[0, :, 0, 0],
                               torch.tensor([0.5, 2.5, 4.5, 6.5, 8.5], dtype=torch.bfloat16))
    torch.testing.assert_close(pooled[:, :, -1, -1], pooled[:, :, 0, 0])


def test_dinov3_pooling_rejects_wrong_frame_or_token_geometry():
    with pytest.raises(ValueError, match='finite'):
        pool_adjacent_frame_features(torch.zeros(1, 9, cache.TOKENS, 8))


@pytest.mark.parametrize('field,value', [
    ('schema', 'fastwam-jepa-video-window-v1'),
    ('temporal_pooling', 'official_video_tubelet2'),
    ('temporal_slots', [[0, -4], [4, 8], [12, 16], [20, 24], [28, 32]]),
    ('feature_shape', [5, 392, 768]),
])
def test_hplus_cache_rejects_incompatible_representation(tmp_path, monkeypatch, field, value):
    _, kwargs, _ = make_cache(tmp_path, monkeypatch, 'dinov3_vith16plus')
    assert cache.JepaVideoCachedDataset(**kwargs)[0]['visual_features'].shape == (5, 392, 1280)
    path = tmp_path / 'manifest.json'
    manifest = json.loads(path.read_text())
    manifest['identity'][field] = value
    digest = cache_identity_sha256(manifest['identity'])
    manifest['identity_sha256'] = kwargs['expected_cache_identity'] = digest
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='Video cache identity'):
        cache.JepaVideoCachedDataset(**kwargs)


def test_hplus_cache_rejects_other_teacher_or_weights(tmp_path, monkeypatch):
    _, kwargs, _ = make_cache(tmp_path, monkeypatch, 'dinov3_vith16plus')
    for change in ({'expected_teacher': 'vjepa21_vitg'}, {'expected_weight_sha256': 'other'}):
        with pytest.raises(ValueError, match='Video cache identity'):
            cache.JepaVideoCachedDataset(**dict(kwargs, **change))


def test_hplus_writer_multiple_batches_and_shards_assemble(tmp_path, monkeypatch):
    from types import SimpleNamespace

    class ImageEncoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.ones(()))
            self.batch_sizes = []

        def forward_features(self, images):
            self.batch_sizes.append(len(images))
            values = images[:, 0, 0, 0]
            return {'x_norm_patchtokens': values[:, None, None].expand(-1, 196, 1280)}

    encoder = ImageEncoder().eval().requires_grad_(False)
    teacher = DenseTeacher('dinov3_vith16plus', encoder, 224, 14, 14, 'teacher-test')
    rgb = torch.linspace(-0.8, 0.8, 40)[:, None, None, None].expand(-1, 3, 224, 448)
    source = SimpleNamespace(source_sha='test-source', identity_sha256='test-numeric',
        identity={'action_space': 'joint8'}, episodes=list(range(71)), lengths=[40] * 71,
        rows=[(0, frame, 0, 40) for frame in range(4, 8)], bounds=[(0, 40)] * 4,
        decode_episode=lambda episode: rgb)
    monkeypatch.setattr(cache, 'open_realman_source', lambda *_: source)
    monkeypatch.setattr(cache, 'load_dense_teacher', lambda *_args, **_kwargs: teacher)
    monkeypatch.setattr(cache, 'SHARD_WINDOWS', 3)
    kwargs = dict(dataset='unused', source_sha=source.source_sha,
        teacher_name=teacher.name, teacher_repo='unused', weights='unused',
        weight_sha256=teacher.weight_sha256, root=tmp_path, world_size=1)
    receipt = cache.write_rank_cache(**kwargs, rank=0, frame_batch=2)
    assert encoder.batch_sizes == [40, 20, 20]
    assert [(item['start'], item['stop']) for item in receipt['shards']] == [(0, 3), (3, 4)]
    manifest = cache.assemble_cache(**kwargs)
    assert manifest['completed_windows'] == 4
    assert manifest['identity']['feature_shape'] == [5, 392, 1280]
    assert manifest['identity']['temporal_pooling'] == 'adjacent_frame_mean_v1'
    tensors = torch.cat([cache.load_file(str(tmp_path / item['path']))['features']
                         for item in manifest['shards']])
    for index, (_, frame, _, _) in enumerate(source.rows):
        offsets = torch.tensor(cache.OFFSETS) + frame
        normalized = (((rgb[offsets, 0, 0, 0] + 1) * 0.5 - 0.485) / 0.229).bfloat16()
        expected = normalized.float().reshape(5, 2).mean(1).bfloat16()
        torch.testing.assert_close(tensors[index, :, 0, 0], expected, rtol=0, atol=0)
        torch.testing.assert_close(tensors[index, :, -1, -1], expected, rtol=0, atol=0)
    with pytest.raises(FileExistsError, match='receipt already exists'):
        cache.write_rank_cache(**kwargs, rank=0, frame_batch=2)
