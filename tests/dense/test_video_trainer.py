"""Offline validation reports physical errors without touching held-out test data."""
import json
from contextlib import nullcontext
from types import SimpleNamespace

import torch
from torch.nn import functional as F

from fastwam.cosmos.jepa_validation import JepaValidationMixin as JepaVideoTrainer


class FakeDataset:
    windows = [(i, i // 4, i) for i in range(32)]
    normalization = {'action': {'std': [2.] * 8, 'mean': [0.] * 8},
                     'force': {'std': [3.] * 6, 'mean': [0.] * 6}}

    def __getitem__(self, index):
        return dict(action=torch.zeros(32, 8), wrench_future=torch.zeros(32, 6),
                    raw_action_future64=torch.zeros(32, 8).double(),
                    raw_wrench_future64=torch.zeros(32, 6).double(),
                    visual_features=torch.arange(5*2*8).reshape(5, 2, 8).float())


class FakePolicy(torch.nn.Module):
    def infer_joint(self, data, **kwargs):
        return dict(action=data['action']+1, force=data['wrench_future']+2,
                    future_features=F.layer_norm(data['visual_features'][:, 1:], (8,)))


def test_validation_physically_scales_errors_and_writes_once(tmp_path):
    trainer = JepaVideoTrainer.__new__(JepaVideoTrainer)
    model = FakePolicy().train()
    trainer.model, trainer.jepa_validation = model, FakeDataset()
    trainer.global_step, trainer.eval_num_inference_steps = 1000, 16
    trainer.eval_dir, trainer._last_jepa_validation = str(tmp_path), -1
    trainer.accelerator = SimpleNamespace(unwrap_model=lambda x: x, process_index=0,
        num_processes=1, autocast=nullcontext, reduce=lambda x, **kwargs: x,
        is_main_process=True, wait_for_everyone=lambda: None)
    trainer.validate_video_policy()
    report = json.loads((tmp_path/'step_001000.json').read_text())
    assert report['samples'] == 32 and model.training
    metrics = report['metrics']
    assert metrics['future_l1'] == 0
    assert metrics['action_mse_normalized'] == 1
    assert metrics['force_mse_normalized'] == 4
    assert metrics['joint_mae_rad'] == metrics['gripper_mae'] == 2
    assert metrics['force_mae_N'] == metrics['moment_mae_Nm'] == 6
    trainer.jepa_validation = None  # A second final save must not rerun validation.
    trainer.validate_video_policy()
