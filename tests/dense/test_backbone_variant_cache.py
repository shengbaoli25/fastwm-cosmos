"""Matched backbone caches pin episode splits, teacher geometry and temporal use."""
import json

import pytest
import torch
from safetensors.torch import load_file, save_file

from fastwam.cache.latent import cache_identity_sha256, file_sha256
from fastwam.cosmos import jepa_video_cache as cache


class FakeSource78:
    def __init__(self):
        self.source_sha = 'source-test'
        self.identity_sha256 = 'joint8-adapter-test'
        self.identity = {'action_space': 'joint8'}
        self.lengths = [40] * 78
        split = cache.episode_splits(78)
        episodes = [*split['train'][:2], split['val'][0], split['test'][0]]
        self.rows = [(ep, 4, ep * 40, 40) for ep in episodes]
        self.bounds = [(0, 40)] * len(self.rows)
        self.raw = {name: torch.zeros(78 * 40, 1) for name in ('state', 'action', 'force')}
        self.normalized = {}
        for ep, value in zip(episodes, (1., 3., 1000., -1000.)):
            for values in self.raw.values():
                values[ep * 40 + 4] = value

    def __getitem__(self, index):
        ep, frame, start, _ = self.rows[index]
        return {'episode_index': ep, 'normalized_state': self.normalized['state'][start + frame]}


def make_cache(tmp_path, monkeypatch, *, teacher_name='vjepa21'):
    source = FakeSource78()
    original_spec = cache.teacher_spec
    monkeypatch.setattr(cache, 'teacher_spec', lambda name: dict(original_spec(name), feature_dim=8))
    spec = cache.identity(source, cache.cache_teacher_manifest(teacher_name, 'teacher-sha'))
    shards = []
    for _, ep, _ in cache.valid_windows(source):
        path = tmp_path / f'{ep}.safetensors'
        save_file({'features': torch.zeros(1, 5, cache.TOKENS, 8, dtype=torch.bfloat16)}, str(path))
        shards.append(dict(episode=ep, start=0, stop=1, path=path.name,
                           sha256=file_sha256(path), size_bytes=path.stat().st_size,
                           mtime_ns=path.stat().st_mtime_ns))
    manifest = dict(status='complete', identity=spec,
                    identity_sha256=cache_identity_sha256(spec), shards=shards,
                    completed_windows=len(shards))
    (tmp_path / 'manifest.json').write_text(json.dumps(manifest))
    monkeypatch.setattr(cache, 'open_realman_source', lambda *_: source)
    args = dict(dataset_root='unused', source_sha256=source.source_sha,
                feature_root=tmp_path, expected_teacher=teacher_name,
                expected_weight_sha256='teacher-sha',
                expected_cache_identity=manifest['identity_sha256'])
    return source, manifest, args


def test_seed42_episode_partitions_are_disjoint_complete_and_stable():
    for total, lengths in [(71, [55, 8, 8]), (78, [60, 9, 9])]:
        split = cache.episode_splits(total)
        assert [len(v) for v in split.values()] == lengths
        assert sorted(ep for group in split.values() for ep in group) == list(range(total))
        assert split == cache.episode_splits(total)
    with pytest.raises(ValueError, match='71- or 78-episode'):
        cache.episode_splits(77)
    ranks = cache.episode_partition([40] * 78, 4)
    assert sorted(ep for group in ranks for ep in group) == list(range(78))
    assert max(map(len, ranks)) - min(map(len, ranks)) <= 1


def test_real_teacher_width_is_enforced_before_any_shard_loading():
    source = FakeSource78()
    assert cache.teacher_spec('vjepa21')['feature_dim'] == 768
    assert cache.teacher_spec('dinov3_vithplus')['feature_dim'] == 1280
    for name in ('vjepa21', 'dinov3_vithplus'):
        teacher = cache.cache_teacher_manifest(name, 'sha')
        teacher['feature_dim'] = 8
        with pytest.raises(ValueError, match='pinned backbone'):
            cache.identity(source, teacher)


def test_rm71d78_statistics_use_only_train_rows(tmp_path, monkeypatch):
    source, _, args = make_cache(tmp_path, monkeypatch)
    train = cache.JepaVideoCachedDataset(**args)
    assert train.normalization['state'] == {'mean': [2.], 'std': [1.]}
    assert len(train) == 2 and len(train.view('val')) == len(train.view('test')) == 1
    assert train.view('val').normalization is train.normalization
    assert train.view('val')[0]['normalized_state'].item() == 5.
    assert train.view('test')[0]['normalized_state'].item() == -5.
    assert train[0]['visual_features'].shape == (5, 392, 8)


@pytest.mark.parametrize('tamper', ['teacher', 'width', 'aggregation', 'split', 'source'])
def test_dataset_rejects_self_consistent_but_wrong_cache_identity(tmp_path, monkeypatch, tamper):
    _, manifest, args = make_cache(tmp_path, monkeypatch)
    spec = manifest['identity']
    if tamper == 'teacher':
        args['expected_teacher'] = 'dinov3_vithplus'
    elif tamper == 'width':
        spec['teacher']['feature_dim'] = 9
        spec['feature_shape'][-1] = 9
    elif tamper == 'aggregation':
        spec['teacher']['temporal_aggregation'] = 'full_future_sliced_as_observation'
    elif tamper == 'split':
        spec['splits']['train'][0], spec['splits']['test'][0] = spec['splits']['test'][0], spec['splits']['train'][0]
    else:
        spec['source_identity_sha256'] = 'wrong-adapter'
    # Attacker/caller recomputing a self-consistent digest still cannot change contract.
    args['expected_cache_identity'] = manifest['identity_sha256'] = cache_identity_sha256(spec)
    (tmp_path / 'manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='Video cache identity'):
        cache.JepaVideoCachedDataset(**args)


def test_assembly_uses_metadata_only_and_single_window_reads(tmp_path, monkeypatch):
    source, manifest, _ = make_cache(tmp_path, monkeypatch)
    (tmp_path / 'manifest.json').unlink()
    receipt = dict(schema=cache.SCHEMA, rank=0, world_size=1, episodes=list(range(78)),
                   status='complete', identity_sha256=manifest['identity_sha256'], shards=manifest['shards'])
    (tmp_path / 'rank_00.json').write_text(json.dumps(receipt))
    monkeypatch.setattr(cache, 'validate_teacher_assets', lambda *_: None)
    monkeypatch.setattr(cache, 'load_dense_teacher', lambda *_a, **_k: pytest.fail('assembly constructed a teacher'))
    monkeypatch.setattr(cache, 'load_file', lambda *_a, **_k: pytest.fail('assembly loaded a whole shard'))
    real_open = cache.safe_open
    selections = []

    class WindowOnlyOpen:
        def __init__(self, *args, **kwargs):
            self.handle = real_open(*args, **kwargs)
        def __enter__(self):
            self.handle.__enter__()
            return self
        def __exit__(self, *args):
            return self.handle.__exit__(*args)
        def keys(self):
            return self.handle.keys()
        def get_slice(self, key):
            actual = self.handle.get_slice(key)
            class WindowSlice:
                def get_shape(self):
                    return actual.get_shape()
                def __getitem__(self, selection):
                    assert selection.stop - selection.start == 1
                    selections.append(selection)
                    return actual[selection]
            return WindowSlice()
        def get_tensor(self, _key):
            pytest.fail('assembly requested whole-shard tensor')
    monkeypatch.setattr(cache, 'safe_open', WindowOnlyOpen)
    args = dict(dataset='unused', source_sha=source.source_sha, teacher_name='vjepa21',
                teacher_repo='unused', weights='unused', weight_sha256='teacher-sha', root=tmp_path, world_size=1)
    result = cache.assemble_cache(**args)
    assert result['completed_windows'] == 4 and len(selections) == 4
    assert cache.assemble_cache(**args) == result
    receipt['shards'].pop()
    (tmp_path / 'rank_00.json').write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match='inventory incomplete'):
        cache.assemble_cache(**args)


def test_writer_observation_is_separate_forward_and_resume_checks_hash(tmp_path, monkeypatch):
    source = FakeSource78()
    source.rows, source.bounds = [(0, 4, 0, 40)], [(0, 40)]
    original_spec = cache.teacher_spec
    monkeypatch.setattr(cache, 'teacher_spec', lambda name: dict(original_spec(name), feature_dim=8))
    monkeypatch.setattr(cache, 'open_realman_source', lambda *_: source)
    monkeypatch.setattr(cache, 'validate_teacher_assets', lambda *_: None)
    values = torch.zeros(40, 1, 1, 1)
    values[8:] = 0.25
    source.decode_episode = lambda _ep: values.expand(-1, 3, 224, 448)
    calls = []

    class Teacher:
        def manifest(self):
            return cache.cache_teacher_manifest('vjepa21', 'teacher-sha')
        def encode_video(self, video, *, context_only):
            calls.append((video.shape[2], context_only))
            slots = 1 if context_only else 5
            return video.mean((1, 2, 3, 4))[:, None, None, None].expand(-1, slots, 392, 8)
    monkeypatch.setattr(cache, 'load_dense_teacher', lambda *_a, **_k: Teacher())
    args = dict(dataset='unused', source_sha=source.source_sha, teacher_name='vjepa21',
                teacher_repo='unused', weights='unused', weight_sha256='teacher-sha',
                rank=0, world_size=1, frame_batch=1, device='cpu')
    first_root = tmp_path / 'first'
    first = cache.write_rank_cache(**args, root=first_root)
    first_features = load_file(str(first_root / first['shards'][0]['path']))['features']
    values[8:] = 0.75
    second_root = tmp_path / 'second'
    second = cache.write_rank_cache(**args, root=second_root)
    second_features = load_file(str(second_root / second['shards'][0]['path']))['features']
    assert calls == [(2, True), (10, False), (2, True), (10, False)]
    assert torch.equal(first_features[:, 0], second_features[:, 0])
    assert not torch.equal(first_features[:, 1:], second_features[:, 1:])
    monkeypatch.setattr(cache, 'load_dense_teacher', lambda *_a, **_k: pytest.fail('completed resume loaded teacher'))
    assert cache.write_rank_cache(**args, root=first_root, resume=True) == first
    path = first_root / first['shards'][0]['path']
    with path.open('r+b') as handle:
        handle.seek(-1, 2)
        handle.write(b'X')
    with pytest.raises(ValueError, match='metadata differs|SHA256 differs'):
        cache.write_rank_cache(**args, root=first_root, resume=True)


def test_legacy_vjepa_manifest_remains_readable_with_pinned_width(tmp_path, monkeypatch):
    _, manifest, args = make_cache(tmp_path, monkeypatch)
    spec = manifest['identity']
    spec['schema'] = cache.LEGACY_SCHEMA
    spec.pop('observation_encoding')
    spec['teacher'].pop('temporal_aggregation')
    args['expected_cache_identity'] = manifest['identity_sha256'] = cache_identity_sha256(spec)
    (tmp_path / 'manifest.json').write_text(json.dumps(manifest))
    assert len(cache.JepaVideoCachedDataset(**args)) == 2
