"""Offline multirate Euler sampler; fixed Video during each control subcycle.

Video reads only observed/video tokens. Its per-layer K/V can therefore be
reused exactly while both its state and noise clock stay fixed. Force/Action
are freshly evaluated at every control step. This changes the joint sampling
schedule, not weights. It is separate from approximate velocity caching.
"""
import torch

from .noaction import BRANCH_READS
from .noaction_inference import NoActionInferenceRunner


def multirate_schedules(model, video_steps, control_steps):
    if video_steps < 1 or control_steps < video_steps or control_steps % video_steps:
        raise ValueError('Positive Video steps must divide Force/Action steps')
    ratio = control_steps // video_steps
    vt, vd = model.v_scheduler.build_inference_schedule(video_steps, model.device, model.torch_dtype)
    expanded_delta = torch.zeros(control_steps, device=model.device, dtype=model.torch_dtype)
    expanded_delta[ratio - 1::ratio] = vd
    return [(vt.repeat_interleave(ratio), expanded_delta)] + [
        s.build_inference_schedule(control_steps, model.device, model.torch_dtype)
        for s in (model.f_scheduler, model.a_scheduler)]


class MultirateInferenceRunner(NoActionInferenceRunner):
    def __init__(self, model, *, video_steps=4, **kwargs):
        if model.manifest.get('branch_reads') != BRANCH_READS:
            raise ValueError('Multirate cache requires the verified one-way Video dependency')
        self.video_steps = int(video_steps)
        self._multirate_key = None
        super().__init__(model, **kwargs)
        self.stats.update(sampler='multirate_block_hold', video_steps=self.video_steps,
                          velocity_cache=False)

    def _build_schedules(self, steps):
        return multirate_schedules(self.model, self.video_steps, steps)

    def _prepare_schedule(self, first, text, steps):
        if self.video_steps < 1 or steps < self.video_steps or steps % self.video_steps:
            raise ValueError('Positive Video steps must divide Force/Action steps')
        key = (self.video_steps, steps)
        if key != self._multirate_key:
            self._schedule_key = None
            self._multirate_key = key
        self.ratio = steps // self.video_steps
        self.stats.update(video_evaluations=self.video_steps, force_evaluations=steps,
                          action_evaluations=steps)
        super()._prepare_schedule(first, text, steps)

    def _rollout(self, video, force, action, first, static, texts, ftext, masks):
        m, fe, ae = self.model, self.model.dit['force'], self.model.dit['action']
        # These caches belong to this invocation, including each graph replay.
        video_kv = None
        pv = None
        for index, (prepared, mods, deltas) in enumerate(self.schedule):
            refresh = index % self.ratio == 0
            if refresh:
                vt = m.video.net.x_embedder(torch.cat(
                    (video[:, :, 1:].to(first.dtype), self.pad_channels), 1))
                video_kv = []
            ft = fe.force_encoder(force) + fe.position[fe.history_steps:] + ftext
            at = ae.action_encoder(action)
            for layer_index, (layer, mod, kv, text) in enumerate(zip(self.layers, mods, static, texts)):
                if refresh:
                    vt, vk, vv = layer.video_forward(vt, mod[0], kv, text[0], self.positions)
                    video_kv.append((vk, vv))
                else:
                    vk, vv = video_kv[layer_index]
                ft, at = layer.control(ft, at, mod[1:], kv[2:], text[1],
                                       self.positions[2:], masks, vk, vv)
            if refresh:
                pv = m.video.finish(vt.flatten(1, 3), prepared)
            force = force + fe.head(ft) * deltas[1]
            action = action + ae.head(at) * deltas[2]
            if (index + 1) % self.ratio == 0:
                video = torch.cat((first, video[:, :, 1:] + pv * deltas[0]), 2)
        return video, force, action


@torch.no_grad()
def infer_multirate_reference(model, sample, *, video_steps=4, control_steps=8,
                              seed=42, video_seed=44, force_seed=43):
    """Independent slow reference: recompute full joint core each control step.

    Do not reuse dynamic Video K/V. Repeated video predictions are identical
    because video state and clock do not advance until the subcycle ends.
    """
    if model.training:
        raise ValueError('Evaluation mode required')
    schedules = multirate_schedules(model, video_steps, control_steps)
    ratio = control_steps // video_steps
    first = sample['first_frame_latents'].to(model.device, model.torch_dtype)
    conditions = model._conditions(sample)
    static = model.prefill_static(first, conditions)
    def noise(shape, value):
        return torch.randn(shape, generator=torch.Generator(device='cpu').manual_seed(value)).to(
            model.device, model.torch_dtype)
    video = noise((*first.shape[:2], 3, *first.shape[3:]), video_seed)
    video[:, :, :1] = first
    force = noise((len(first), 32, 6), force_seed)
    action = noise((len(first), 32, model.action_dim), seed)
    for index in range(control_steps):
        tv, tf, ta = (pair[0][index].expand(len(first)) for pair in schedules)
        pv, pf, pa = model.joint_denoise_core(video, force, action, tv, tf, ta, conditions, static)
        force = model.f_scheduler.step(pf, schedules[1][1][index], force)
        action = model.a_scheduler.step(pa, schedules[2][1][index], action)
        if (index + 1) % ratio == 0:
            video = model.v_scheduler.step(pv, schedules[0][1][index], video)
            video[:, :, :1] = first
    return dict(video_latents=video, force=force, action=action)
