"""Read-only Joint8 SDPA probes. The production SDPA output is never replaced."""
from contextlib import contextmanager
import numpy as np
import torch


def mean_key_probabilities(q, k, valid=None, query_stride=1, bias=None):
    attention_dtype = q.dtype
    q = q[..., ::query_stride, :].float()
    logits = q @ k.float().transpose(-1, -2) / q.shape[-1] ** .5
    if bias is not None:
        # The production attend() casts its additive mask to q's dtype first.
        bias = bias.to(dtype=attention_dtype).float()
        if bias.shape[-2] != q.shape[-2] and query_stride > 1:
            bias = bias[..., ::query_stride, :]
        logits = logits + bias
    if valid is not None:
        logits = logits.masked_fill(~valid[:, None, None, :], -torch.inf)
    return logits.softmax(-1).mean(-2).detach().cpu().numpy()  # batch, head, key


def per_query_key_probabilities(q, k, valid=None, bias=None):
    """Average heads while retaining each temporal query's key distribution."""
    attention_dtype = q.dtype
    logits = q.float() @ k.float().transpose(-1, -2) / q.shape[-1] ** .5
    if bias is not None:
        logits = logits + bias.to(dtype=attention_dtype).float()
    if valid is not None:
        logits = logits.masked_fill(~valid[:, None, None, :], -torch.inf)
    return logits.softmax(-1).mean(1).detach().cpu().numpy()  # batch, query, key


class AttentionProbe:
    def __init__(self, model, layers=(0, 6, 13, 20, 27), steps=(0, 7, 15)):
        self.model, self.layers, self.steps = model, layers, steps
        self.records = {}; self.query_records = {}; self.step = -1; self.active = None; self.ordinal = 0

    def _record_queries(self, module, args, output):
        if self.active not in self.layers or self.step not in self.steps:
            return
        if not isinstance(output, (tuple, list)) or len(output) < 2:
            raise ValueError('Visual query selector must return (maps, bias, ...)')
        maps = output[0]
        bias = output[1]
        if not isinstance(maps, torch.Tensor) or maps.ndim != 4 or maps.shape[-1] != 392:
            raise ValueError('Expected visual query maps [B, action_time, queries, 392]')
        if maps.shape[1] != 32:
            raise ValueError('Expected query maps for 32 action tokens')
        if not isinstance(bias, torch.Tensor) or bias.shape != (maps.shape[0], maps.shape[1], 392):
            raise ValueError('Expected the actual applied query routing bias [B, action_time, 392]')
        prefix = f's{self.step:02d}_l{self.active:02d}'
        self.query_records[prefix + '_query_maps'] = maps.detach().float().cpu().numpy()
        self.query_records[prefix + '_query_bias'] = bias.detach().float().cpu().numpy()

    @contextmanager
    def recording(self):
        import fastwam.cosmos.noaction as module
        core, layer, sdpa = self.model.joint_denoise_core, self.model._dynamic_layer, module.attend
        self.records = {}; self.query_records = {}; self.step = -1
        selector = self.model.dit['visual_queries'] if 'visual_queries' in self.model.dit else None
        selector_hook = None if selector is None else selector.register_forward_hook(self._record_queries)
        def traced_core(*args, **kwargs):
            self.step += 1
            return core(*args, **kwargs)
        def traced_layer(*args, **kwargs):
            self.active = args[0]; self.ordinal = 0
            try:
                return layer(*args, **kwargs)
            finally:
                self.active = None
        def traced_sdpa(q, k, v, valid=None, *sdpa_args, **sdpa_kwargs):
            bias = sdpa_kwargs.get('bias')
            if bias is None and sdpa_args:
                bias = sdpa_args[0]
            result = sdpa(q, k, v, valid, *sdpa_args, **sdpa_kwargs)
            if self.active is not None:
                branch = ('video', 'force', 'action')[self.ordinal]
                self.ordinal += 1
                if self.active in self.layers and self.step in self.steps:
                    # All force/action queries; bounded, evenly sampled video queries.
                    p = mean_key_probabilities(q, k, valid, 8 if branch == 'video' else 1,
                                               bias=bias)
                    self.records[f's{self.step:02d}_l{self.active:02d}_{branch}'] = p
                    if branch == 'force':
                        # Preserve temporal alignment for diagnosing which
                        # action/force horizon routes to observed visual keys.
                        self.records[f's{self.step:02d}_l{self.active:02d}_force_query_probs'] = (
                            per_query_key_probabilities(q, k, valid, bias=bias))
            return result
        self.model.joint_denoise_core, self.model._dynamic_layer = traced_core, traced_layer
        module.attend = traced_sdpa
        try:
            yield self
        finally:
            if selector_hook is not None:
                selector_hook.remove()
            self.model.joint_denoise_core, self.model._dynamic_layer, module.attend = core, layer, sdpa

    def save(self, path):
        np.savez_compressed(path, **self.records, **self.query_records)
