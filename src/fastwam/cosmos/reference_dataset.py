"""Full-episode references joined to immutable Cosmos windows by source frame.

The reference has variable M, independent of the 32 future action targets.
Only batching pads M; the auditor learns which 32 local tokens to produce.
Augmentation epoch travels with each sampler index, so worker lifetime, rank,
prefetching and checkpoint resume cannot silently freeze/change the reference.
"""
from dataclasses import asdict
from numbers import Integral

import numpy as np
import torch
from torch.utils.data import default_collate

from .cache import CosmosCachedDataset
from .reference_bank import ReferenceBank
from .reference_binding import verify_binding
from .reference_trajectory import AugmentationConfig


INPUT_KEYS = frozenset(('reference_pose', 'reference_valid', 'reference_channel_valid',
    'reference_order', 'reference_break_before', 'reference_present', 'current_tcp'))
SUPERVISION_KEYS = frozenset(('alignment_target', 'alignment_enabled'))
PADDED_KEYS = frozenset(('reference_pose', 'reference_valid', 'reference_channel_valid',
    'reference_order', 'reference_break_before'))


def reference_collate(samples):
    """Pad only references; provenance and alignment labels never become inputs."""
    if not samples:
        raise ValueError('Cannot collate an empty reference batch')
    for item in samples:
        if set(item['reference_inputs']) != INPUT_KEYS:
            raise ValueError('Unexpected reference input fields; refuse possible label leakage')
        if set(item['reference_supervision']) != SUPERVISION_KEYS:
            raise ValueError('Unexpected reference supervision fields')
    sizes = [len(x['reference_inputs']['reference_pose']) for x in samples]
    if min(sizes) < 1:
        raise ValueError('Full reference must contain at least one slot')
    maximum = max(sizes)
    inputs, supervision = {}, {}
    for key in INPUT_KEYS:
        values = [torch.as_tensor(x['reference_inputs'][key]) for x in samples]
        if key not in PADDED_KEYS:
            inputs[key] = torch.stack(values)
            continue
        if any(len(value) != size for value, size in zip(values, sizes)):
            raise ValueError('Reference fields have inconsistent lengths')
        padded = torch.zeros((len(samples), maximum, *values[0].shape[1:]), dtype=values[0].dtype)
        if key == 'reference_pose':
            padded[..., 6] = 1  # finite identity quaternion; validity remains false
        for row, value in enumerate(values):
            padded[row, :len(value)] = value
        inputs[key] = padded
    target = torch.zeros((len(samples), maximum), dtype=torch.float32)
    for row, (item, size) in enumerate(zip(samples, sizes)):
        value = torch.as_tensor(item['reference_supervision']['alignment_target'])
        if tuple(value.shape) != (size,):
            raise ValueError('Alignment supervision length differs from reference')
        target[row, :size] = value
    supervision['alignment_target'] = target
    supervision['alignment_enabled'] = torch.stack([
        torch.as_tensor(x['reference_supervision']['alignment_enabled']) for x in samples])
    batch = default_collate([{k:v for k,v in item.items()
                              if k not in ('reference_inputs', 'reference_supervision')}
                             for item in samples])
    batch.update(reference_inputs=inputs, reference_supervision=supervision)
    return batch


class ReferenceCachedDataset(CosmosCachedDataset):
    """Use verified cached observations/GT and independently augmented GT intent."""
    accepts_epoch_indices = True
    collate_fn = staticmethod(reference_collate)

    def __init__(self, cache_root, expected_identity_sha256, expected_text_identity,
                 reference_bank, expected_reference_identity, require_production=True,
                 max_cached_shards=2, expected_action_representation='delta_current_joint7',
                 reference_seed=42, reference_augmentation=None):
        if not require_production or expected_action_representation != 'delta_current_joint7':
            raise ValueError('Reference training requires the verified production Joint8 delta cache')
        super().__init__(cache_root, expected_identity_sha256, expected_text_identity,
                         require_production, max_cached_shards, expected_action_representation)
        self.reference_bank = ReferenceBank(reference_bank)
        if self.reference_bank.manifest['identity_sha256'] != expected_reference_identity:
            raise ValueError('Reference bank identity differs from resolved training config')
        self.reference_identity_sha256 = expected_reference_identity
        self.reference_binding = verify_binding(reference_bank, self.root/'manifest.json')
        if (self.reference_binding['status'] != 'bound'
                or self.reference_binding['cache_identity_sha256'] != self.identity_sha256
                or self.reference_binding['reference_identity_sha256'] != expected_reference_identity):
            raise ValueError('Reference/cache provenance rejected: '+str(self.reference_binding['errors']))
        if not isinstance(reference_seed, Integral) or reference_seed < 0:
            raise ValueError('Nonnegative deterministic reference seed required')
        self.reference_seed = int(reference_seed)
        self.reference_augmentation = AugmentationConfig(**dict(reference_augmentation or {}))
        self.reference_augmentation_config = asdict(self.reference_augmentation)

    def __getitem__(self, index):
        # Integer access is useful for diagnostics and always means epoch zero.
        epoch = 0
        if isinstance(index, tuple) and len(index) == 2:
            index, epoch = index
        if (not isinstance(index, Integral) or isinstance(index, bool)
                or not isinstance(epoch, Integral) or isinstance(epoch, bool) or epoch < 0):
            raise IndexError((index, epoch))
        sample = super().__getitem__(int(index))
        episode = int(sample['episode_index'].item())
        frame = int(sample['frame_index'].item())
        reference = self.reference_bank.sample(episode, frame, epoch=int(epoch),
            seed=self.reference_seed, config=self.reference_augmentation)
        sample['reference_inputs'] = {key:torch.from_numpy(np.asarray(value).copy())
                                      for key,value in reference['inputs'].items()}
        sample['reference_supervision'] = {
            key:torch.from_numpy(np.asarray(reference['supervision'][key]).copy())
            for key in SUPERVISION_KEYS}
        return sample
