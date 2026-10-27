"""Numeric held-out validation for video latents; no RGB reconstruction metrics."""
from pathlib import Path
import shutil


from fastwam.trainer import Wan22Trainer
from .jepa_validation import JepaValidationMixin


class JepaVideoTrainer(JepaValidationMixin, Wan22Trainer):
    def __init__(self, model, train_dataset, val_dataset=None, *, cfg):
        super().__init__(model, train_dataset, None, cfg=cfg)
        self.jepa_validation = train_dataset.view('val')
        self._last_jepa_validation = -1

    def save_checkpoint(self):
        result = super().save_checkpoint()
        interval = int(getattr(self.cfg, 'validation_every', 1000))
        if self.global_step > 0 and self.global_step % interval == 0:
            self.validate_video_policy()
        # Keep 5k milestones and the two most recent complete states.
        self.accelerator.wait_for_everyone()
        if self.accelerator.is_main_process:
            states = sorted(Path(self.state_dir).glob('step_*'))
            recent = {p.name for p in states[-2:]}
            for path in states:
                try:
                    step = int(path.name.removeprefix('step_'))
                except ValueError:
                    continue
                if path.name not in recent and step % 5000:
                    shutil.rmtree(path)
                    (Path(self.weights_dir) / f'{path.name}.pt').unlink(missing_ok=True)
        self.accelerator.wait_for_everyone()
        return result
