"""Inference-only, exact dependency caching for multirate V/F/A Euler sampling.

Each expert evaluates its integration velocity at the beginning of its interval,
holds its latent and clock throughout that interval, then updates at the end.
An upstream change refreshes downstream *features*, not a held Euler velocity.
For example, 8/4/16 refreshes Force K/V after every Video update, even during a
Force interval. This module does not change checkpoint or training semantics.

The reference implementation deliberately calls the original complete model at
every fine-grid tick. The optimized implementation reuses per-layer Video and
Force K/V only while their complete dependencies remain unchanged. No learned
features are reused between requests.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import operator
import threading
import warnings

import torch
from torch.nn import functional as F
from torch.utils._pytree import tree_flatten, tree_unflatten

from fastwam.models.wan22.wan_video_dit import modulate, sinusoidal_embedding_1d
from .noaction import attend, frame_slice


@dataclass(frozen=True)
class MultirateSteps:
    video: int = 4
    force: int = 4
    action: int = 16

    def __post_init__(self):
        for field in ("video", "force", "action"):
            value = getattr(self, field)
            try:
                parsed = operator.index(value)
            except TypeError as exc:
                raise ValueError("Sampling steps must be positive integers") from exc
            if isinstance(value, bool) or parsed < 1:
                raise ValueError("Sampling steps must be positive integers")
            object.__setattr__(self, field, parsed)
        if any(self.total % value for value in self.counts):
            raise ValueError("The maximum sampling step count must be divisible by every expert count")

    @property
    def counts(self):
        return self.video, self.force, self.action

    @property
    def total(self):
        return max(self.counts)

    @property
    def strides(self):
        return tuple(self.total // value for value in self.counts)


@contextmanager
def _evaluating(model):
    modes = {module: module.training for module in model.modules()}
    model.eval()
    try:
        with torch.no_grad():
            yield
    finally:
        for module, mode in modes.items():
            module.training = mode


def _initialize(model, sample, steps, seed, video_seed, force_seed):
    first = sample["first_frame_latents"].to(model.device, model.torch_dtype)
    if first.ndim != 5 or first.shape[2] != 1 or min(first.shape) < 1:
        raise ValueError("first_frame_latents must have shape [B,C,1,H,W]")
    if not torch.isfinite(first).all():
        raise ValueError("first_frame_latents must be finite")
    conditions = model._conditions(sample)

    def noise(shape, value):
        # CPU noise preserves the existing sampler's reproducible seed contract.
        return torch.randn(shape, generator=torch.Generator(device="cpu").manual_seed(value)).to(
            model.device, model.torch_dtype)

    video = noise((*first.shape[:2], 3, *first.shape[3:]), video_seed)
    video[:, :, :1] = first
    force = noise((len(first), 32, 6), force_seed)
    action = noise((len(first), 32, model.action_dim), seed)
    schedulers = (model.v_scheduler, model.f_scheduler, model.a_scheduler)
    schedules = tuple(scheduler.build_inference_schedule(count, model.device, model.torch_dtype)
                      for scheduler, count in zip(schedulers, steps.counts))
    return first, conditions, (video, force, action), schedulers, schedules


def _video_pass(model, video, tv, raw, static):
    prepared = model._video_prepare(video, tv, raw)
    future = frame_slice(prepared, 1, prepared["grid"][0])
    tokens = future["tokens"]
    layers = []
    for i in range(model.depth):
        ok, ov, _, _ = static["layers"][i]
        q, k, v, residual = model.video.block_qkv(i, tokens, future)
        layers.append((k, v))
        tokens = model.video.block_finish(i, attend(q, torch.cat((ok, k), 2),
            torch.cat((ov, v), 2)), residual, future, static["text"][i][0])
    velocity = torch.cat((torch.zeros_like(video[:, :, :1]), model.video.finish(tokens, future)), 2)
    return velocity, tuple(layers)


def _force_pass(model, force, tf, tv, context, hvalid, static, video_layers):
    tokens, modulation = model.dit["force"].prepare_future(force, tf, tv, context)
    layers = []
    for i, block in enumerate(model.dit["force"].blocks):
        ok, ov, hk, hv = static["layers"][i]
        vk, vv = video_layers[i]
        q, k, v, residual = block.qkv(tokens, modulation)
        layers.append((k, v))
        valid = torch.cat((torch.ones(len(force), ok.shape[2] + vk.shape[2],
            device=force.device, dtype=torch.bool), hvalid,
            torch.ones(len(force), k.shape[2], device=force.device, dtype=torch.bool)), 1)
        tokens = block(attend(q, torch.cat((ok, vk, hk, k), 2),
            torch.cat((ov, vv, hv, v), 2), valid), residual)
    return model.dit["force"].head(tokens), tuple(layers)


def _action_pass(model, action, ta, tv, tf, context, cmask, hvalid, static,
                 video_layers, force_layers):
    expert = model.dit["action"]
    tokens, _, modulation, _, text_mask, freq = expert.prepare(action, ta, context, cmask)
    source_time = torch.cat((sinusoidal_embedding_1d(256, tv),
                             sinusoidal_embedding_1d(256, tf)), -1)
    modulation = modulation + model.dit["action_source_time"](source_time).unflatten(
        -1, (6, expert.hidden_dim))
    for i, block in enumerate(expert.blocks):
        ok, ov, hk, hv = static["layers"][i]
        vk, vv = video_layers[i]
        fk, fv = force_layers[i]
        q, k, v, residual, gate, shift, scale, mlp_gate, _ = model.ops._build_expert_attention_io(
            expert, block, tokens, freq, modulation)
        q, k, v = (model._split(value) for value in (q, k, v))
        valid = torch.cat((torch.ones(len(action), ok.shape[2] + vk.shape[2],
            device=action.device, dtype=torch.bool), hvalid,
            torch.ones(len(action), fk.shape[2] + k.shape[2], device=action.device, dtype=torch.bool)), 1)
        mixed = attend(q, torch.cat((ok, vk, hk, fk, k), 2),
            torch.cat((ov, vv, hv, fv, v), 2), valid).transpose(1, 2).flatten(2)
        tokens = block.gate(residual, gate, block.self_attn.o(mixed))
        cq = model._split(block.cross_attn.norm_q(block.cross_attn.q(block.norm3(tokens))))
        ck, cv = static["text"][i][1]
        cross = F.scaled_dot_product_attention(cq, ck, cv,
            attn_mask=text_mask[:, None], dropout_p=0.)
        tokens = tokens + block.cross_attn.o(cross.transpose(1, 2).flatten(2))
        tokens = block.gate(tokens, mlp_gate, block.ffn(modulate(block.norm2(tokens), shift, scale)))
    return expert.post(tokens)


_BRANCHES = {"video": _video_pass, "force": _force_pass, "action": _action_pass}


class _CudaGraphCall:
    """One shape-specialized branch graph; all request inputs are copied anew."""

    def __init__(self, function, args, device):
        self.device = torch.device(device)
        leaves, self.spec = tree_flatten(args)
        if any(isinstance(value, torch.Tensor) and value.device != self.device for value in leaves):
            raise ValueError("CUDA graph inputs must all be on the model CUDA device")
        self.inputs = [value.detach().clone() if isinstance(value, torch.Tensor) else value for value in leaves]
        with torch.cuda.device(self.device):
            stream = torch.cuda.Stream(device=self.device)
            stream.wait_stream(torch.cuda.current_stream(self.device))
            with torch.cuda.stream(stream):
                for _ in range(3):
                    function(*tree_unflatten(self.inputs, self.spec))
            torch.cuda.current_stream(self.device).wait_stream(stream)
            stream.synchronize()
            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph, stream=stream):
                try:
                    output = function(*tree_unflatten(self.inputs, self.spec))
                except Exception:
                    # capture_end can replace the original error when the stream
                    # was invalidated. Log it before leaving the graph context.
                    import logging
                    logging.getLogger(__name__).exception("CUDA graph capture body failed")
                    raise
            self.outputs, self.output_spec = tree_flatten(output)
            torch.cuda.current_stream(self.device).wait_stream(stream)

    def __call__(self, *args):
        leaves, spec = tree_flatten(args)
        if spec != self.spec:
            raise ValueError("CUDA graph input structure changed")
        with torch.cuda.device(self.device):
            for target, source in zip(self.inputs, leaves):
                if isinstance(target, torch.Tensor):
                    target.copy_(source)
            self.graph.replay()
            # Callers hold some integration velocities until a later interval.
            # Clones prevent subsequent replay from mutating those values.
            outputs = [value.clone() if isinstance(value, torch.Tensor) else value for value in self.outputs]
        return tree_unflatten(outputs, self.output_spec)


def _input_signature(args):
    leaves, spec = tree_flatten(args)
    return str(spec), tuple((tuple(value.shape), tuple(value.stride()), value.dtype, value.device)
        if isinstance(value, torch.Tensor) else (type(value), value) for value in leaves)


def _owned_output(output):
    """Keep outputs alive independently of compiler-managed CUDA Graph pools."""
    leaves, spec = tree_flatten(output)
    return tree_unflatten([value.clone() if isinstance(value, torch.Tensor) else value
                           for value in leaves], spec)


class MultirateRunner:
    """Reusable optional executor; default eager is portable and needs no GPU.

    ``compile`` uses torch.compile; ``cuda_graph`` captures each complete expert
    pass on the model's actual CUDA index. ``auto`` selects compile on CUDA and
    eager elsewhere. Unsupported compilation/capture falls back to eager with a
    warning and ``fallback_reason``. First-call setup is not steady-state latency.
    A runner serializes its own requests. Do not train or mutate the model during
    inference. Device/dtype/parameter updates invalidate its execution caches.
    """

    def __init__(self, model, mode="eager"):
        if mode not in {"eager", "compile", "cuda_graph", "auto"}:
            raise ValueError("mode must be eager, compile, cuda_graph, or auto")
        self.model, self.mode = model, mode
        self.fallback_reason = None
        self._functions, self._graphs = {}, {}
        self._signature = None
        self._lock = threading.RLock()
        self._disabled = set()
        self._fallback_reasons = {}

    @property
    def active_mode(self):
        if self.mode == "auto":
            return "compile" if self.model.device.type == "cuda" else "eager"
        if self.mode == "cuda_graph" and self.model.device.type != "cuda":
            return "eager"
        return self.mode

    @property
    def diagnostics(self):
        """JSON-safe execution status; requested acceleration is not a guarantee.

        A branch is only reported as compiled/captured after its first successful
        call. This exposes partial fallback without declaring the whole sampler
        accelerated when one or more branches still use eager execution.
        """
        with self._lock:
            branches = {name: ("eager" if name in self._disabled or self.active_mode == "eager"
                               else "compile" if name in self._functions
                               else "cuda_graph" if name in self._graphs else "pending")
                        for name in _BRANCHES}
            modes = set(branches.values())
            return {"requested_mode": self.mode,
                    "active_mode": self.active_mode,
                    "effective_mode": next(iter(modes)) if len(modes) == 1 else "mixed",
                    "branch_modes": branches,
                    "fallback_reason": self.fallback_reason,
                    "fallback_reasons": dict(self._fallback_reasons)}

    def clear(self):
        with self._lock:
            self._functions.clear()
            self._graphs.clear()
            self._disabled.clear()
            self._fallback_reasons.clear()
            self._signature = None
            self.fallback_reason = None

    def close(self):
        """Release this runner's compiled callables and captured GPU buffers."""
        self.clear()

    def _begin(self):
        signature = (self.model.device, self.model.torch_dtype, self.model.fps,
            tuple((id(value), value.data_ptr(), value._version, tuple(value.shape), value.dtype, value.device)
                  for value in (*self.model.parameters(), *self.model.buffers())))
        if signature != self._signature:
            self.clear()
            self._signature = signature
        if self.mode == "cuda_graph" and self.model.device.type != "cuda" and not self.fallback_reason:
            self.fallback_reason = "CUDA Graph requires a CUDA device; using eager"
            self._fallback_reasons["device"] = self.fallback_reason
            warnings.warn(self.fallback_reason, RuntimeWarning, stacklevel=3)
        if self.active_mode == "compile" and hasattr(torch, "compiler"):
            marker = getattr(torch.compiler, "cudagraph_mark_step_begin", None)
            if marker is not None and self.model.device.type == "cuda":
                marker()

    def _call(self, branch, *args):
        function = lambda *values: _BRANCHES[branch](self.model, *values)
        mode = self.active_mode
        if mode == "eager" or branch in self._disabled:
            return function(*args)
        try:
            if mode == "compile":
                if branch not in self._functions:
                    self._functions[branch] = torch.compile(function, mode="reduce-overhead", fullgraph=True)
                # reduce-overhead may capture a CUDA Graph whose next invocation
                # reuses output storage. Euler velocities and K/V must survive
                # later calls (e.g. Force feature refresh midway through a held
                # Force interval), including calls to another compiled branch.
                return _owned_output(self._functions[branch](*args))
            signature = _input_signature(args)
            if branch not in self._graphs or self._graphs[branch][0] != signature:
                # Keep at most one shape per branch, avoiding unbounded VRAM use.
                self._graphs[branch] = (signature, _CudaGraphCall(function, args, self.model.device))
            return self._graphs[branch][1](*args)
        except (RuntimeError, NotImplementedError, AttributeError) as exc:
            # A broken device or exhausted GPU is not an optimization fallback.
            # Let the deployment layer report failure instead of declaring ready.
            fatal = ("out of memory", "device-side assert", "illegal memory access",
                     "unspecified launch failure", "driver shutting down", "device lost",
                     "no kernel image", "driver version is insufficient")
            if isinstance(exc, torch.OutOfMemoryError) or any(value in str(exc).lower() for value in fatal):
                raise
            self._disabled.add(branch)
            self._functions.pop(branch, None)
            self._graphs.pop(branch, None)
            self.fallback_reason = f"{mode} {branch} failed ({type(exc).__name__}: {exc}); using eager"
            self._fallback_reasons[branch] = self.fallback_reason
            warnings.warn(self.fallback_reason, RuntimeWarning, stacklevel=3)
            return function(*args)

    def infer(self, sample, **kwargs):
        return infer_multirate(self.model, sample, runner=self, **kwargs)


def infer_multirate(model, sample, *, video_steps=4, force_steps=4, action_steps=16,
                    seed=42, video_seed=44, force_seed=43, use_cache=True, runner=None):
    """Sample V/F/A with independently configured integer-divisor Euler grids.

    Returns ``action``, ``force`` and ``video_latents`` like ``infer_joint``.
    ``use_cache=False`` selects the independent complete-model reference.
    """
    steps = MultirateSteps(video_steps, force_steps, action_steps)
    if runner is not None and runner.model is not model:
        raise ValueError("The executor belongs to a different model")
    if not use_cache:
        return infer_multirate_reference(model, sample, video_steps=steps.video,
            force_steps=steps.force, action_steps=steps.action,
            seed=seed, video_seed=video_seed, force_seed=force_seed)
    runner = runner if runner is not None else MultirateRunner(model)
    with runner._lock, _evaluating(model):
        runner._begin()
        first, conditions, values, schedulers, schedules = _initialize(
            model, sample, steps, seed, video_seed, force_seed)
        raw, context, cmask, _, hvalid = conditions
        static = model.prefill_static(first, conditions)
        video, force, action = values
        sv, sf, sa = steps.strides
        held = [None, None, None]
        video_layers = force_layers = None
        for tick in range(steps.total):
            indices = (tick // sv, tick // sf, tick // sa)
            tv, tf, ta = (schedule[0][index].expand(len(first))
                          for schedule, index in zip(schedules, indices))
            video_dirty, force_dirty, action_start = tick % sv == 0, tick % sf == 0, tick % sa == 0
            if video_dirty:
                held[0], video_layers = runner._call("video", video, tv, raw, static)
            if video_dirty or force_dirty:
                velocity, force_layers = runner._call("force", force, tf, tv,
                    context, hvalid, static, video_layers)
                if force_dirty:
                    held[1] = velocity
            if action_start:
                held[2] = runner._call("action", action, ta, tv, tf, context,
                    cmask, hvalid, static, video_layers, force_layers)
            if (tick + 1) % sv == 0:
                video = schedulers[0].step(held[0], schedules[0][1][indices[0]], video)
                video[:, :, :1] = first
            if (tick + 1) % sf == 0:
                force = schedulers[1].step(held[1], schedules[1][1][indices[1]], force)
            if (tick + 1) % sa == 0:
                action = schedulers[2].step(held[2], schedules[2][1][indices[2]], action)
        return {"action": action, "force": force, "video_latents": video}


def infer_multirate_reference(model, sample, *, video_steps=4, force_steps=4,
                              action_steps=16, seed=42, video_seed=44, force_seed=43):
    """Slow full-model reference with no static or dynamic K/V cache.

    This intentionally recomputes all experts every tick. Only an expert's
    interval-start velocity is integrated at its interval end, making the
    numerical protocol separately checkable from branch/cache optimizations.
    """
    steps = MultirateSteps(video_steps, force_steps, action_steps)
    with _evaluating(model):
        first, conditions, values, schedulers, schedules = _initialize(
            model, sample, steps, seed, video_seed, force_seed)
        values = list(values)
        velocities = [None, None, None]
        for tick in range(steps.total):
            indices = tuple(tick // stride for stride in steps.strides)
            times = [schedule[0][index].expand(len(first))
                     for schedule, index in zip(schedules, indices)]
            predictions = model.joint_denoise_core(*values, *times, conditions, static_cache=None)
            for expert, stride in enumerate(steps.strides):
                if tick % stride == 0:
                    velocities[expert] = predictions[expert]
                if (tick + 1) % stride == 0:
                    values[expert] = schedulers[expert].step(velocities[expert],
                        schedules[expert][1][indices[expert]], values[expert])
                    if expert == 0:
                        values[expert][:, :, :1] = first
        return {"action": values[2], "force": values[1], "video_latents": values[0]}
