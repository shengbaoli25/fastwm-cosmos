from unittest.mock import patch

import pytest
import torch

from fastwam.cache.latent import cache_identity_sha256
from fastwam.cosmos.jepa_training_data import (JepaForceTrainingDataset, build_observation_cache,
                                             observation_indices)


def test_observed_indices_respect_current_anchor_and_discontinuity():
    assert observation_indices(12, 10, 20) == [10, 12]
    assert observation_indices(16, 10, 20) == [12, 16]
    with pytest.raises(ValueError):
        observation_indices(20, 10, 20)


def fixture_data(tmp_path):
    class Source:
        identity = {'segments': [[0, 2, 6]]}
        identity_sha256 = cache_identity_sha256(identity)
        lengths = [6]
        bounds = [(2, 6)] * 4
        rows = [(0, i, 0, 6) for i in range(2, 6)]

        def decode_episode(self, ep):
            images = torch.zeros(6, 3, 224, 448)
            for i in range(6):
                images[i, ..., :224] = -1 + i / 10
                images[i, ..., 224:] = .3 + i / 10
            return images

    class Base:
        root = tmp_path / 'base'
        identity_sha256 = 'a' * 64
        manifest = dict(identity_sha256=identity_sha256,
            identity=dict(realman_source=Source.identity, episode_lengths=Source.lengths))
        external_labels = False

        def __len__(self):
            return 4

        def __getitem__(self, j):
            result = dict(sample_index=torch.tensor(j), episode_index=torch.tensor(0),
                          frame_index=torch.tensor(j + 2), action=torch.full((32, 8), float(j)),
                          history=torch.full((32, 14), float(j)), action_is_pad=torch.zeros(32, dtype=torch.bool))
            if self.external_labels:
                result['force_teacher_features'] = torch.zeros(32, 768)
            return result

    manifest = build_observation_cache(Source(), Base.manifest, tmp_path / 'observations')
    return Base(), manifest, Source()


def test_dataset_preserves_exact_numeric_rows_and_camera_time_order(tmp_path):
    base, obs, _ = fixture_data(tmp_path)
    with patch('fastwam.cosmos.jepa_training_data.CosmosCachedDataset', return_value=base):
        dataset = JepaForceTrainingDataset({}, tmp_path / 'observations', obs['identity_sha256'])
    sample = dataset[3]
    for key, value in base[3].items():
        torch.testing.assert_close(sample[key], value, rtol=0, atol=0)
    assert sample['jepa_observation'].shape == (2, 3, 2, 224, 224)
    # [-4,0] clamps to this segment's frame2, then current frame5; never frame1.
    torch.testing.assert_close(sample['jepa_observation'][0, 0, :, 0, 0], torch.tensor([-.8, -.5]), atol=1/255, rtol=0)
    torch.testing.assert_close(sample['jepa_observation'][1, 0, :, 0, 0], torch.tensor([.5, .8]), atol=1/255, rtol=0)
    assert set(sample) - set(base[3]) == {'jepa_observation'}
    assert dataset.root == base.root and dataset.identity_sha256 == base.identity_sha256
    base.external_labels = True
    with pytest.raises(ValueError, match='External Force-teacher'):
        dataset[3]


def test_observation_cache_rejects_wrong_identity_and_modified_payload(tmp_path):
    base, obs, _ = fixture_data(tmp_path)
    with patch('fastwam.cosmos.jepa_training_data.CosmosCachedDataset', return_value=base):
        with pytest.raises(ValueError, match='identity'):
            JepaForceTrainingDataset({}, tmp_path / 'observations', 'b' * 64)
        path = tmp_path / 'observations' / obs['episodes'][0]['path']
        with path.open('ab') as stream:
            stream.write(b'changed')
        with pytest.raises(ValueError, match='payload changed'):
            JepaForceTrainingDataset({}, tmp_path / 'observations', obs['identity_sha256'])


def test_complete_observation_cache_reuse_verifies_without_decoding(tmp_path):
    base, original, source = fixture_data(tmp_path)
    with patch.object(source, 'decode_episode', side_effect=AssertionError('Complete RGB cache must be reused')):
        actual = build_observation_cache(source, base.manifest, tmp_path / 'observations', resume=True)
    assert actual == original


def test_hydra_instantiates_nested_baseline_once_and_batches_observations(tmp_path):
    from hydra.utils import instantiate
    from omegaconf import OmegaConf

    base, observed, _ = fixture_data(tmp_path)
    baseline_config = dict(_target_='fastwam.cosmos.cache.CosmosCachedDataset',
                           cache_root=str(base.root), expected_identity_sha256=base.identity_sha256,
                           expected_action_representation='delta_current_joint7')
    config = OmegaConf.create(dict(_target_='fastwam.cosmos.jepa_training_data.JepaForceTrainingDataset',
        _recursive_=False, base_config=baseline_config, observation_cache_root=str(tmp_path / 'observations'),
        expected_observation_identity=observed['identity_sha256']))
    with patch('fastwam.cosmos.jepa_training_data.CosmosCachedDataset', return_value=base) as factory:
        dataset = instantiate(config)
    expected = {key: value for key, value in baseline_config.items() if key != '_target_'}
    factory.assert_called_once_with(**expected)
    batch = next(iter(torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=False)))
    assert batch['jepa_observation'].shape == (2, 2, 3, 2, 224, 224)
    assert batch['jepa_observation'].dtype == torch.float32
    torch.testing.assert_close(batch['action'], torch.stack([base[j]['action'] for j in range(2)]),
                               rtol=0, atol=0)


def test_reference_jepa_keeps_epoch_and_pads_only_the_reference(tmp_path):
    from fastwam.cosmos.reference_dataset import reference_collate
    from test_reference_dataset_cpu import reference_item

    base, observed, _ = fixture_data(tmp_path)
    seen = []

    class ReferenceBase(type(base)):
        accepts_epoch_indices = True
        collate_fn = staticmethod(reference_collate)
        reference_binding = {'status': 'bound'}
        reference_augmentation_config = {'dropout_probability': .1}

        def __getitem__(self, index):
            anchor, epoch = index
            seen.append((anchor, epoch))
            sample = super().__getitem__(anchor)
            reference = reference_item(2 + anchor)
            sample.update(reference_inputs=reference['reference_inputs'],
                          reference_supervision=reference['reference_supervision'])
            return sample

    with patch('fastwam.cosmos.reference_dataset.ReferenceCachedDataset', return_value=ReferenceBase()):
        dataset = JepaForceTrainingDataset({'_target_': 'fastwam.cosmos.reference_dataset.ReferenceCachedDataset'},
            tmp_path / 'observations', observed['identity_sha256'])
    loader = torch.utils.data.DataLoader(dataset, batch_size=2, sampler=[(0, 7), (1, 7)],
                                        collate_fn=dataset.collate_fn)
    batch = next(iter(loader))
    assert dataset.accepts_epoch_indices and seen == [(0, 7), (1, 7)]
    assert dataset.reference_binding['status'] == 'bound'
    assert batch['reference_inputs']['reference_pose'].shape == (2, 3, 8)
    assert batch['reference_inputs']['reference_valid'][0].tolist() == [True, True, False]
    assert batch['jepa_observation'].shape == (2, 2, 3, 2, 224, 224)
