"""Numeric video-policy validation independent of the distributed trainer."""
import json
from pathlib import Path

import torch
from torch.nn import functional as F
from torch.utils.data._utils.collate import default_collate


class JepaValidationMixin:
    @torch.no_grad()
    def validate_video_policy(self):
        if self._last_jepa_validation == self.global_step:
            return
        model = self.accelerator.unwrap_model(self.model)
        was_training = model.training
        model.eval()
        dataset = self.jepa_validation
        by_episode = {}
        for index, (_, episode, _) in enumerate(dataset.windows):
            by_episode.setdefault(episode, []).append(index)
        selected = []
        for episode in sorted(by_episode):
            windows = by_episode[episode]
            positions = torch.linspace(0, len(windows)-1, 4).round().long().tolist()
            selected.extend(windows[p] for p in positions)
        local = selected[self.accelerator.process_index::self.accelerator.num_processes]
        names = ('future_l1', 'persistence_l1', 'action_mse_normalized', 'force_mse_normalized',
                 'joint_mae_rad', 'gripper_mae', 'force_mae_N', 'moment_mae_Nm')
        rows = []
        try:
            for first in range(0, len(local), 4):
                data = default_collate([dataset[index] for index in local[first:first+4]])
                with self.accelerator.autocast():
                    prediction = model.infer_joint(data, num_inference_steps=self.eval_num_inference_steps)
                device = prediction['action'].device
                future = prediction['future_features'].float()
                target = F.layer_norm(data['visual_features'][:, 1:].to(device).float(), (future.shape[-1],))
                persistent = F.layer_norm(data['visual_features'][:, :1].to(device).float(), (future.shape[-1],))
                action = prediction['action'].float() - data['action'].to(device).float()
                force = prediction['force'].float() - data['wrench_future'].to(device).float()
                action_scale = torch.tensor(dataset.normalization['action']['std'], device=device)
                force_scale = torch.tensor(dataset.normalization['force']['std'], device=device)
                action_mean = torch.tensor(dataset.normalization['action']['mean'], device=device)
                force_mean = torch.tensor(dataset.normalization['force']['mean'], device=device)
                physical_action = (prediction['action'].float() * (action_scale+1e-8)
                                   + action_mean - data['raw_action_future64'].to(device).float())
                physical_force = (prediction['force'].float() * (force_scale+1e-8)
                                  + force_mean - data['raw_wrench_future64'].to(device).float())
                rows.append(torch.stack(((future-target).abs().flatten(1).mean(1),
                    (persistent-target).abs().flatten(1).mean(1),
                    action.square().flatten(1).mean(1), force.square().flatten(1).mean(1),
                    physical_action[:, :, :7].abs().mean((1, 2)),
                    physical_action[:, :, 7].abs().mean(1),
                    physical_force[:, :, :3].abs().mean((1, 2)),
                    physical_force[:, :, 3:].abs().mean((1, 2))), 1))
            values = torch.cat(rows, 0)
            totals = torch.cat((values.sum(0), values.new_tensor([len(values)])))
            totals = self.accelerator.reduce(totals, reduction='sum')
            report = dict(step=self.global_step, split='val', episodes=sorted(by_episode),
                          samples=int(totals[-1]), metrics=dict(zip(names, (totals[:-1]/totals[-1]).cpu().tolist())),
                          inference_steps=self.eval_num_inference_steps,
                          scope='fixed 4 windows per validation episode; not a closed-loop success estimate')
            if self.accelerator.is_main_process:
                path = Path(self.eval_dir) / f'step_{self.global_step:06d}.json'
                path.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
                print('[jepa-validation] ' + json.dumps(report, allow_nan=False), flush=True)
            self._last_jepa_validation = self.global_step
        finally:
            model.train(was_training)
        self.accelerator.wait_for_everyone()
