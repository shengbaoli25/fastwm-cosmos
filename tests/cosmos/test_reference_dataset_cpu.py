"""No model weights/GPU: variable reference batching and exact resume epochs."""
from pathlib import Path

import numpy as np
import unittest
from unittest.mock import patch
try:
    import torch
except ImportError as exc:
    raise unittest.SkipTest('Dataset integration requires CPU torch') from exc
from torch.utils.data import DataLoader, Dataset

from fastwam.cosmos.cache import CosmosCachedDataset
from fastwam.cosmos.reference_dataset import ReferenceCachedDataset, reference_collate, INPUT_KEYS
from fastwam.cosmos.reference_trajectory import AugmentationConfig
from fastwam.utils.samplers import ResumableEpochSampler


def reference_item(length, *, present=True):
    pose = torch.zeros(length, 8)
    pose[:, 6] = 1
    valid = torch.full((length,), present, dtype=torch.bool)
    target = torch.zeros(length)
    if present:
        target[length//2] = 1
    return dict(action=torch.ones(32,8), episode_index=torch.tensor(5), frame_index=torch.tensor(172),
        reference_inputs=dict(reference_pose=pose, reference_valid=valid,
            reference_channel_valid=valid[:,None].expand(-1,3).clone(),
            reference_order=torch.linspace(0,1,length), reference_break_before=torch.zeros(length,dtype=torch.bool),
            reference_present=torch.tensor(present), current_tcp=torch.tensor([0.,0.,0.,0.,0.,0.,1.,0.])),
        reference_supervision=dict(alignment_target=target,alignment_enabled=torch.tensor(present)))


def test_full_plan_padding_is_independent_of_action_horizon():
    batch = reference_collate([reference_item(89), reference_item(218), reference_item(137,present=False)])
    assert batch['action'].shape == (3,32,8)
    inputs, labels = batch['reference_inputs'], batch['reference_supervision']
    assert set(inputs) == INPUT_KEYS
    assert inputs['reference_pose'].shape == (3,218,8)
    assert inputs['current_tcp'].shape == (3,8)
    assert inputs['reference_present'].tolist() == [True,True,False]
    assert not inputs['reference_valid'][0,89:].any()
    assert not inputs['reference_channel_valid'][0,89:].any()
    assert not inputs['reference_valid'][2].any()
    assert torch.all(inputs['reference_pose'][0,89:,6] == 1)
    assert labels['alignment_target'].sum(-1).tolist() == [1.,1.,0.]
    assert labels['alignment_enabled'].tolist() == [True,True,False]
    assert not labels['alignment_target'][0,89:].any()


def test_collate_rejects_provenance_as_input():
    item = reference_item(103)
    item['reference_inputs']['source_frame'] = torch.arange(103)
    with unittest.TestCase().assertRaisesRegex(ValueError,'label leakage'):
        reference_collate([item])


def test_wrapper_joins_source_episode_frame_not_cache_sample_index():
    calls = []
    source = dict(episode_index=torch.tensor(5),frame_index=torch.tensor(172),sample_index=torch.tensor(9))
    def read(self, index):
        assert index == 9
        return dict(source)
    class Bank:
        def sample(self, ep, frame, *, epoch, seed, config):
            calls.append((ep,frame,epoch,seed))
            item = reference_item(101)
            labels = item['reference_supervision']
            labels['source_frame'] = np.arange(101)  # retained only inside the bank
            return dict(inputs=item['reference_inputs'],supervision=labels)
    wrapped = ReferenceCachedDataset.__new__(ReferenceCachedDataset)
    wrapped.reference_bank, wrapped.reference_seed = Bank(),42
    wrapped.reference_augmentation = AugmentationConfig()
    with patch.object(CosmosCachedDataset,'__getitem__',read):
        sample = wrapped[(9,7)]
    assert calls == [(5,172,7,42)]
    assert set(sample['reference_inputs']) == INPUT_KEYS
    assert set(sample['reference_supervision']) == {'alignment_target','alignment_enabled'}
    assert sample['reference_inputs']['reference_pose'].shape == (101,8)
    with unittest.TestCase().assertRaises(IndexError):
        wrapped[(9,-1)]


class EpochDataset(Dataset):
    accepts_epoch_indices = True
    def __len__(self):
        return 23
    def __getitem__(self,index):
        sample, epoch = index
        return dict(index=sample,epoch=epoch)


def sampler(dataset):
    return ResumableEpochSampler(dataset,seed=42,batch_size=2,num_processes=2)


def test_resume_keeps_absolute_epoch_and_exact_remaining_indices():
    data = EpochDataset()
    uninterrupted = sampler(data)
    uninterrupted.set_epoch(7)
    full = list(uninterrupted)
    resumed = sampler(data)
    resumed.set_epoch_offset(7)
    resumed.set_resume_batch_offset(2)
    assert list(resumed) == full[8:]
    resumed.clear_resume_batch_offset()
    resumed.set_epoch(1)
    uninterrupted.set_epoch(8)
    assert list(resumed) == list(uninterrupted)
    assert all(epoch == 8 for _,epoch in resumed)


def check_epoch_reaches_prefetched_and_persistent_workers(workers):
    data = EpochDataset()
    indices = sampler(data)
    loader = DataLoader(data,batch_size=4,sampler=indices,num_workers=workers,
                        persistent_workers=bool(workers))
    for epoch in (0,1,11):
        indices.set_epoch(epoch)
        batches = list(loader)
        assert torch.cat([x['epoch'] for x in batches]).tolist() == [epoch]*len(data)
        assert len(set(torch.cat([x['index'] for x in batches]).tolist())) == len(data)


def test_epoch_reaches_single_process_loader():
    check_epoch_reaches_prefetched_and_persistent_workers(0)


def test_epoch_reaches_prefetched_and_persistent_workers():
    check_epoch_reaches_prefetched_and_persistent_workers(2)


def test_unchanged_dataset_still_receives_integer_indices():
    class Existing:
        def __len__(self):
            return 11
    assert all(type(i) is int for i in sampler(Existing()))


def test_plan_does_not_import_torch_or_touch_remote_paths():
    import importlib.util
    path = Path(__file__).resolve().parents[2]/'scripts/cosmos/train_reference_audit.py'
    spec = importlib.util.spec_from_file_location('reference_train_entry',path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    args = module.parser().parse_args(['--plan'])
    plan = module.describe(args)
    assert (plan['ranks'],plan['batch_per_rank'],plan['global_batch']) == (4,64,256)
    assert (plan['max_steps'],plan['scheduler_total_steps'],plan['warmup_steps']) == (30000,50000,2500)
    assert plan['vae_reencode'] is False
    assert plan['model_output'] == 'unchanged 32xJoint8 delta action chunk'


def load_tests(loader, tests, pattern):
    suite = unittest.TestSuite()
    for name, function in globals().items():
        if name.startswith('test_'):
            suite.addTest(unittest.FunctionTestCase(function))
    return suite


if __name__ == '__main__':
    unittest.main()
