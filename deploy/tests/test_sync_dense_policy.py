"""CPU contract tests with small stand-ins; no official weights or GPU required."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
from types import ModuleType

import pytest
import torch

from realman_deploy.normalization import DatasetNormalizer
from realman_sync.dense_policy import DensePolicy


SCHEMA = 'fastwam-jepa-video-context-cosmos-joint8-v1'


def force_contract():
    value = {'schema': 'realman_model_force_v1', 'field': 'observation.force_filtered',
             'units': ['N'] * 3 + ['Nm'] * 3, 'sampling': {'fps': 30}}
    value['id'] = hashlib.sha256(json.dumps(value, sort_keys=True,
        separators=(',', ':')).encode()).hexdigest()
    return value


def stats():
    return {name: {'mean': [0.25] * width, 'std': [2.] * width}
            for name, width in (('state', 8), ('action', 8), ('force', 6))}


def make_manifest(teacher='vjepa21_vitg'):
    width = {'vjepa21_vitg': 1408, 'dinov3_vith16plus': 1280}[teacher]
    architecture = dict(feature_dim=width, hidden_dim=768, depth=28, heads=12,
        view_grid=14, action_hidden=768, action_ffn=3072, force_hidden=768,
        force_ffn=2048, joint_heads=16, joint_head_dim=128, context_dim=1024)
    return dict(schema=SCHEMA, architecture=architecture, action_space='joint8',
        teacher={'name': teacher, 'feature_dim': width, 'weight_sha256': 'a' * 64},
        observation_offsets=[-4, 0], future_offsets=[4, 8, 12, 16, 20, 24, 28, 32],
        stride=4, temporal_slots=[[-4, 0], [4, 8], [12, 16], [20, 24], [28, 32]],
        temporal_pooling='adjacent_frame_mean_v1' if teacher.startswith('dino') else 'official_video_tubelet2',
        normalization='training episodes only', dataset_source_sha256='b' * 64,
        cache_identity_sha256='c' * 64, splits={'train': [1, 2], 'val': [3], 'test': [4]},
        lambda_future=1., num_inference_steps=16)


def model_config(manifest):
    return dict(_target_='fastwam.cosmos.dense_features.create_dense_feature_joint',
        manifest=deepcopy(manifest), **manifest['architecture'], lambda_future=1.,
        force_shift=1.25, action_seed=42, force_seed=43, proprio_seed=44,
        gradient_checkpointing=True, num_inference_steps=16)


class FakeTeacher:
    def __init__(self, name):
        self.name = name
        self.inputs = []
        self.width = 1280 if name.startswith('dino') else 1408

    def manifest(self):
        return {'name': self.name, 'feature_dim': self.width, 'weight_sha256': 'a' * 64}

    def encode_video(self, pixels, *, context_only):
        assert context_only is True
        self.inputs.append(pixels.clone())
        return torch.full((1, 1, 392, self.width), 0.125)


class FakeModel(torch.nn.Module):
    constructed = []

    def __init__(self, **kwargs):
        super().__init__()
        self.manifest = deepcopy(kwargs['manifest'])
        self.kwargs = kwargs
        self.weight = torch.nn.Parameter(torch.zeros(2))
        self.calls = []
        self.output = torch.zeros(1, 32, 8, device='cpu')
        self.constructed.append(self)

    def infer_joint(self, sample, **kwargs):
        self.calls.append((sample, kwargs))
        return {'action': self.output}


@pytest.fixture
def deployment(tmp_path, monkeypatch):
    model_module = ModuleType('fastwam.cosmos.dense_features')
    model_module.DenseFeatureJointModel, model_module.SCHEMA = FakeModel, SCHEMA
    teacher_module = ModuleType('fastwam.cosmos.dense_encoder')
    calls = []
    def load_teacher(name, source, weights, expected_sha256, device):
        calls.append((name, source, weights, expected_sha256, device))
        return FakeTeacher(name)
    teacher_module.load_dense_teacher = load_teacher
    monkeypatch.setitem(sys.modules, 'fastwam.cosmos.dense_features', model_module)
    monkeypatch.setitem(sys.modules, 'fastwam.cosmos.dense_encoder', teacher_module)
    root = tmp_path / 'bundle'
    root.mkdir()
    (root / 'official').mkdir()
    (root / 'teacher.pt').write_bytes(b'placeholder; mocked official weight loader')
    manifest = make_manifest()
    trained = {'teacher': 'vjepa21_vitg', 'model': model_config(manifest)}
    (root / 'config.yaml').write_text(json.dumps(trained))
    normalization = dict(statistics=stats(), cache_identity='c' * 64,
                         normalization='training episodes only', train_episodes=[1, 2])
    (root / 'normalization.json').write_text(json.dumps(normalization))
    (root / 'force.json').write_text(json.dumps({'model_input': force_contract()}))
    payload = dict(schema=SCHEMA, manifest=manifest, model={'weight': torch.tensor([3., 4.])})
    torch.save(payload, root / 'step_020000.pt')
    config = dict(training_config='config.yaml', normalization_file='normalization.json',
        teacher_source='official', teacher_weights='teacher.pt',
        force_preprocessing_file='force.json', device='cpu', camera_width=8, camera_height=6)
    return root, config, calls, payload


def observation(sequence, *, value=0, force_id=None):
    return dict(sequence=sequence, stamp=100. + sequence / 30,
        state=[0.5] * 8, force=[1.] * 6, action_space='joint8',
        force_meta={'preprocessing_id': force_id or force_contract()['id']}, frame_meta={},
        frames={'global': {'width': 8, 'height': 6, 'data': bytes([value]) * (8 * 6 * 3)},
                'wrist': {'width': 8, 'height': 6, 'data': bytes([255 - value]) * (8 * 6 * 3)}})


def load_policy(deployment, **changes):
    root, config, _, _ = deployment
    return DensePolicy(dict(config, **changes), root / 'serve.json', 'step_020000.pt')


def test_paths_checkpoint_state_and_saved_scheduler_are_used(deployment):
    root, _, calls, _ = deployment
    policy = load_policy(deployment)
    assert calls == [('vjepa21_vitg', root / 'official', root / 'teacher.pt', 'a' * 64, 'cpu')]
    assert torch.equal(policy.model.weight, torch.tensor([3., 4.]))
    assert policy.model.kwargs['force_shift'] == 1.25
    assert not policy.model.kwargs['gradient_checkpointing']
    assert policy.metadata['video_context_frames'] == 2
    assert policy.metadata['video_context_stride'] == 4
    assert policy.metadata['force_preprocessing'] == force_contract()
    policy.close()
    assert policy.model is None and policy.teacher is None


@pytest.mark.parametrize('teacher_name', ['vjepa21_vitg', 'dinov3_vith16plus'])
def test_online_images_history_and_denormalized_actions_match_contract(deployment, teacher_name):
    root, _, _, payload = deployment
    payload['manifest'] = make_manifest(teacher_name)
    torch.save(payload, root / 'step_020000.pt')
    (root / 'config.yaml').write_text(json.dumps({'teacher': teacher_name,
        'model': model_config(payload['manifest'])}))
    policy = load_policy(deployment, action_horizon=5)
    previous, current = observation(6, value=32), observation(10, value=128)
    history = [observation(i) for i in range(7, 10)] + [current]
    action = policy.predict(current, history, [previous, current])
    assert action == [[0.25] * 8] * 5
    pixels = policy.teacher.inputs[0]
    assert pixels.shape == (1, 3, 2, 224, 448)
    torch.testing.assert_close(pixels[0, :, 0, :, :224], torch.full((3, 224, 224), 32 / 127.5 - 1))
    torch.testing.assert_close(pixels[0, :, 1, :, 224:], torch.full((3, 224, 224), 127 / 127.5 - 1), atol=1e-7, rtol=1e-4)
    sample, kwargs = policy.model.calls[0]
    assert set(sample) == {'history', 'history_is_pad', 'history_channel_valid', 'proprio', 'visual_features'}
    assert sample['visual_features'].shape == (1, 1, 392, policy.teacher.width)
    assert sample['history'].shape == (1, 32, 14)
    assert sample['proprio'].shape == (1, 9)
    assert sample['proprio'][0, -1] == 0
    assert sample['history_is_pad'].sum() == 28
    assert not sample['history_channel_valid'][0, :28].any()
    assert sample['history_channel_valid'][0, 28:].all()
    assert not sample['history'][0, :28].any()
    expected = torch.tensor(policy.normalizer.encode('state', current['state']) + policy.normalizer.encode('force', current['force']))
    torch.testing.assert_close(sample['history'][0, -1], expected)
    assert kwargs == {'num_inference_steps': 16, 'seed': 42, 'force_seed': 43}
    policy.model.output[0, :, 7] = 10.
    assert all(row[7] == 1. for row in policy.predict(current, history, [previous, current]))
    assert policy.model.calls[-1][1]['seed'] == 43


@pytest.mark.parametrize('change,match', [
    ('wrong_schema', 'weights checkpoint'), ('wrong_shape', 'size mismatch'),
    ('wrong_architecture', 'architecture differ'), ('missing_force_shift', 'incomplete'),
    ('unresolved_manifest', 'unresolved experiment'), ('wrong_normalization', 'training-only'),
])
def test_checkpoint_and_run_artifact_mismatches_fail_before_teacher(deployment, change, match):
    root, _, calls, payload = deployment
    trained = json.loads((root / 'config.yaml').read_text())
    if change == 'wrong_schema':
        payload['schema'] = 'legacy'
    elif change == 'wrong_shape':
        payload['model']['weight'] = torch.ones(3)
    elif change == 'wrong_architecture':
        trained['model']['hidden_dim'] += 12
    elif change == 'missing_force_shift':
        del trained['model']['force_shift']
    elif change == 'unresolved_manifest':
        trained['model']['manifest'] = {}
    elif change == 'wrong_normalization':
        data = json.loads((root / 'normalization.json').read_text())
        data['cache_identity'] = 'other'
        (root / 'normalization.json').write_text(json.dumps(data))
    (root / 'config.yaml').write_text(json.dumps(trained))
    torch.save(payload, root / 'step_020000.pt')
    with pytest.raises((ValueError, RuntimeError), match=match):
        load_policy(deployment)
    assert calls == []


@pytest.mark.parametrize('change,match', [
    ('consecutive_video', 'source offsets'), ('old_video', 'source offsets'),
    ('short_video', 'Two measured'), ('wrong_current_image', 'images differ'),
    ('gap_history', 'consecutive'), ('wrong_history_anchor', 'end at'),
    ('force_identity', 'preprocessing differs'), ('bad_rgb', 'RGB24'),
    ('bad_action', 'normalized action'), ('nan_action', 'normalized action'),
])
def test_invalid_temporal_sensor_or_model_payload_is_rejected(deployment, change, match):
    policy = load_policy(deployment)
    current, older = observation(10, value=128), observation(6, value=32)
    history = [observation(9), current]
    video = [older, current]
    if change == 'consecutive_video':
        video[0] = observation(9)
    elif change == 'old_video':
        video[0]['stamp'] -= 3
    elif change == 'short_video':
        video = [current]
    elif change == 'wrong_current_image':
        video[1] = observation(10, value=129)
    elif change == 'gap_history':
        history[0] = observation(7)
    elif change == 'wrong_history_anchor':
        history = [observation(9)]
    elif change == 'force_identity':
        history[0]['force_meta']['preprocessing_id'] = 'wrong'
    elif change == 'bad_rgb':
        older['frames']['global']['data'] = b'bad'
    elif change == 'bad_action':
        policy.model.output = torch.zeros(1, 16, 8)
    elif change == 'nan_action':
        policy.model.output[0, 0, 0] = float('nan')
    with pytest.raises(ValueError, match=match):
        policy.predict(current, history, video)
    assert policy.inference_index == 0


def test_normalization_matches_training_fp32_scale_offset(deployment):
    policy = load_policy(deployment)
    values = [0., 1., -10., 1e7, -1e7, 0.1, 0.5, 0.9]
    means, stds = torch.tensor(stats()['state']['mean']), torch.tensor(stats()['state']['std'])
    expected = (torch.tensor(values) / (stds + 1e-8) - means / (stds + 1e-8)).clamp(-5, 5)
    torch.testing.assert_close(torch.tensor(policy.normalizer.encode('state', values)), expected, rtol=0, atol=0)

def test_unchanged_dense_model_loads_and_runs_one_online_flow_step(tmp_path, monkeypatch):
    """Exercise actual model construction/scheduler/shape with a tiny CPU model."""
    from fastwam.cosmos.dense_features import DenseFeatureJointModel, SCHEMA as real_schema
    from fastwam.cosmos import dense_encoder
    monkeypatch.setattr(dense_encoder, 'load_dense_teacher',
        lambda name, source, weights, expected_sha256, device: FakeTeacher(name))
    root = tmp_path
    (root / 'official').mkdir()
    (root / 'teacher.pt').write_bytes(b'mocked frozen teacher')
    base = make_manifest()
    architecture = dict(feature_dim=1408, hidden_dim=24, depth=1, heads=2, view_grid=14,
        action_hidden=24, action_ffn=48, force_hidden=24, force_ffn=48,
        joint_heads=2, joint_head_dim=12, context_dim=32)
    base.pop('architecture')
    base['num_inference_steps'] = 1
    kwargs = dict(manifest=base, **architecture, lambda_future=1., force_shift=1.25,
        action_seed=42, force_seed=43, proprio_seed=44, gradient_checkpointing=False,
        num_inference_steps=1)
    original = DenseFeatureJointModel(**kwargs)
    original.save_checkpoint(root / 'step_000001.pt', step=1)
    del original
    trained = {'teacher': 'vjepa21_vitg', 'model': dict(kwargs, _target_='fastwam.cosmos.dense_features.create_dense_feature_joint')}
    (root / 'config.yaml').write_text(json.dumps(trained))
    (root / 'normalization.json').write_text(json.dumps(dict(statistics=stats(), cache_identity='c' * 64,
        normalization='training episodes only', train_episodes=[1, 2])))
    policy = DensePolicy(dict(training_config='config.yaml', normalization_file='normalization.json',
        teacher_source='official', teacher_weights='teacher.pt', force_preprocessing=force_contract(),
        device='cpu', camera_width=8, camera_height=6), root / 'serve.json', 'step_000001.pt')
    current, older = observation(10), observation(6)
    actions = policy.predict(current, [observation(9), current], [older, current])
    assert torch.tensor(actions).shape == (32, 8)
    assert torch.isfinite(torch.tensor(actions)).all()
    assert all(0 <= row[7] <= 1 for row in actions)
