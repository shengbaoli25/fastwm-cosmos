"""Inference-only reuse of fixed conditions for the Video-isolated checkpoint.

Training, checkpoint tensors, schedules and the reference implementation remain
unchanged. Cached conditions are refreshed on every inference request; CUDA
graph buffers persist only to reuse allocations and recorded kernels. The Video rollout can
omit Force because the checkpoint's mask excludes every Force-owned key from
every Video query. The final clean WM pass still computes Force and all AE K/V.
"""
from __future__ import annotations

import torch
from torch.nn import functional as F
from torch.utils._pytree import tree_flatten, tree_unflatten

from .joint import JOINT_PROTOCOL, action_prefix_mask, grouped_attention_mask, joint_token_layout
from fastwam.models.wan22.wan_video_dit import flash_attention, sinusoidal_embedding_1d


class _CapturedCall:
    """Fixed-shape CUDA replay with explicit copies for every request input."""
    def __init__(self, function, args):
        flat, self.spec = tree_flatten(args)
        self.inputs = [value.clone() for value in flat]
        self.signature = [(value.shape, value.dtype, value.device) for value in flat]
        static_args = tree_unflatten(self.inputs, self.spec)
        stream = torch.cuda.Stream(device=flat[0].device)
        stream.wait_stream(torch.cuda.current_stream(flat[0].device))
        with torch.cuda.stream(stream):
            for _ in range(2):
                function(*static_args)
        torch.cuda.current_stream(flat[0].device).wait_stream(stream)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph, stream=stream):
            self.outputs = function(*static_args)

    def __call__(self, args):
        flat, spec = tree_flatten(args)
        if spec != self.spec or [(v.shape, v.dtype, v.device) for v in flat] != self.signature:
            raise ValueError("CUDA graph input shape/dtype/device changed")
        for buffer, value in zip(self.inputs, flat):
            buffer.copy_(value)
        self.graph.replay()
        return self.outputs


class JointInferenceRunner:
    def __init__(self, model, *, use_cuda_graph=False, compile_future=False,
                 compile_mode="max-autotune-no-cudagraphs"):
        if model.manifest.get("joint_protocol") != JOINT_PROTOCOL:
            raise ValueError("Inference reuse requires the Video-isolated O/R/V/F/A protocol")
        self.model = model
        self.use_cuda_graph = use_cuda_graph
        self._graphs = {}
        self._compiled_future = None
        if compile_future and model.device.type == "cuda":
            self._compiled_future = torch.compile(self._future_velocity, fullgraph=True, mode=compile_mode)

    def _execute(self, stage, function, *args):
        if not self.use_cuda_graph or self.model.device.type != "cuda":
            return function(*args)
        flat, spec = tree_flatten(args)
        key = (stage, str(spec), tuple((tuple(x.shape), x.dtype, x.device) for x in flat))
        if key not in self._graphs:
            # Bound shape variants rather than retaining arbitrary GPU pools.
            if len(self._graphs) >= 2:
                self._graphs.clear()
            self._graphs[key] = _CapturedCall(function, args)
        return self._graphs[key](args)

    def close(self):
        self._graphs.clear()
        self._compiled_future = None

    def _prepare_video(self, x, clock, raw, controls, *, validate=False):
        return self.model.video.prepare(x, clock.float() / 1000., raw,
            candidate_actions=controls, fps=self.model.fps, context_is_projected=True,
            observed_time="clean_zero", validate_values=validate)

    def _video_text_kv(self, raw):
        result = []
        for block in self.model.video.blocks:
            attention = block.cross_attn
            split = lambda x: x.unflatten(-1, (attention.n_heads, attention.head_dim))
            result.append((attention.k_norm(split(attention.k_proj(raw))),
                           attention.v_norm(split(attention.v_proj(raw)))))
        return tuple(result)

    def _video_finish(self, i, mixed, state, prepared, text_kv):
        block = self.model.video.blocks[i]
        x = state["residual"]
        flat = mixed.transpose(1, 2).flatten(2)
        self_output = block.self_attn.output_dropout(block.self_attn.output_proj(flat)).reshape_as(x)
        x = x + state["modulations"]["self_attn"][2] * self_output
        shift, scale, gate = state["modulations"]["cross_attn"]
        normalized = block.layer_norm_cross_attn(x) * (1 + scale) + shift
        attention = block.cross_attn
        q = attention.q_norm(attention.q_proj(normalized.flatten(1, 3)).unflatten(
            -1, (attention.n_heads, attention.head_dim)))
        x = x + gate * attention.compute_attention(q, *text_kv).reshape_as(x)
        shift, scale, gate = state["modulations"]["mlp"]
        normalized = block.layer_norm_mlp(x) * (1 + scale) + shift
        return (x + gate * block.mlp(normalized)).flatten(1, 3)

    def _video_velocity(self, x, clock, raw, controls, mask, text_kv):
        prepared = self._prepare_video(x, clock, raw, controls)
        tokens = prepared["tokens"]
        observed_count = prepared["tokens_per_frame"]
        observed = []
        for i in range(self.model.depth):
            q, k, v, state = self.model.video.block_qkv(i, tokens, prepared)
            observed.append(tuple(value[:, :, :observed_count].clone() for value in (q, k, v)))
            # The exact mask has two rectangular Video blocks: O reads O and
            # future Video reads all Video. Dense subcalls enable fused SDPA.
            mixed = torch.cat((F.scaled_dot_product_attention(q[:, :, :observed_count],
                k[:, :, :observed_count], v[:, :, :observed_count], dropout_p=0.),
                F.scaled_dot_product_attention(q[:, :, observed_count:], k, v, dropout_p=0.)), 2)
            tokens = self._video_finish(i, mixed, state, prepared, text_kv[i])
        return self.model.video.finish(tokens, prepared), tuple(observed)

    @staticmethod
    def _future_prepared(prepared):
        """Slice frame-local modulation and absolute RoPE without reindexing time."""
        count = prepared["tokens_per_frame"]
        state = dict(prepared["state"])
        state["time"] = state["time"][:, 1:]
        state["adaln"] = state["adaln"][:, 1:] if state["adaln"] is not None else None
        state["rope"] = state["rope"][count:]
        state["extra"] = state["extra"][:, 1:] if state["extra"] is not None else None
        return {**prepared, "tokens": prepared["tokens"][:, count:],
                "grid": (prepared["grid"][0] - 1, *prepared["grid"][1:]), "state": state}

    def _future_velocity(self, x, clock, raw, controls, mask, text_kv, observed):
        full = self._prepare_video(x, clock, raw, controls)
        prepared = self._future_prepared(full)
        tokens = prepared["tokens"]
        for i, (_, ok, ov) in enumerate(observed):
            q, k, v, state = self.model.video.block_qkv(i, tokens, prepared)
            mixed = F.scaled_dot_product_attention(q, torch.cat((ok, k), 2), torch.cat((ov, v), 2), dropout_p=0.)
            tokens = self._video_finish(i, mixed, state, prepared, text_kv[i])
        future_velocity = self.model.video.finish(tokens, prepared)
        # The scheduler always restores the observed frame; its velocity is unused.
        return torch.cat((torch.zeros_like(future_velocity[:, :, :1]), future_velocity), 2)

    def _rollout(self, x, first, raw, controls, mask, video_text_kv, times, deltas):
        observed = None
        for index, (t, dt) in enumerate(zip(times, deltas)):
            if index == 0:
                velocity, observed = self._video_velocity(x, t.expand(len(x)), raw, controls, mask, video_text_kv)
            else:
                future = self._compiled_future or self._future_velocity
                velocity = future(x, t.expand(len(x)), raw, controls, mask, video_text_kv, observed)
            x = self.model.v_scheduler.step(velocity, dt, x)
            x[:, :, :1] = first
        return x, observed

    def _prefix(self, x, raw, controls, force_input, valid, mask, text_kv, observed):
        zero = torch.zeros(len(x), device=x.device, dtype=x.dtype)
        prepared = self._future_prepared(self._prepare_video(x, zero, raw, controls))
        tokens, force_tokens = prepared["tokens"], force_input
        caches = []
        force = self.model.dit["force"]
        for i, block in enumerate(force.blocks):
            q, k, v, state = self.model.video.block_qkv(i, tokens, prepared)
            oq, ok, ov = observed[i]
            k, v = torch.cat((ok, k), 2), torch.cat((ov, v), 2)
            attention = block.self_attention
            normalized = block.norm1(force_tokens)
            fk, fv = attention.kv(normalized)
            full_k, full_v = torch.cat((k, fk), 2), torch.cat((v, fv), 2)
            force_mixed = F.scaled_dot_product_attention(attention.split(attention.q(normalized)),
                full_k, full_v, attn_mask=mask[:, :, k.shape[2]:], dropout_p=0.)
            next_force = force_tokens + attention.out(force_mixed.transpose(1, 2).flatten(2))
            next_force = next_force + block.ff(block.norm2(next_force))
            if i + 1 < self.model.depth:
                mixed = F.scaled_dot_product_attention(q, k, v, dropout_p=0.)
                tokens = self._video_finish(i, mixed, state, prepared, text_kv[i])
            force_tokens = torch.where(valid[..., None], next_force, 0)
            caches.append((full_k, full_v))
        # Video's final velocity head is unused in the clean cache refresh.
        wrench = force.head(force_tokens[:, force.history_steps + force.horizon:])
        return tuple(caches), wrench

    def _action_text_kv(self, context):
        expert = self.model.dit["action"]
        text = expert.text_embedding(context)
        return tuple((block.cross_attn.norm_k(block.cross_attn.k(text)), block.cross_attn.v(text))
                     for block in expert.blocks)

    def _action_velocity(self, action, clock, memory, allowed, text_kv, text_mask, apad):
        expert = self.model.dit["action"]
        t = expert.time_embedding(sinusoidal_embedding_1d(expert.freq_dim, clock))
        mod = expert.time_projection(t).unflatten(1, (6, expert.hidden_dim))
        x, freq = expert.action_encoder(action), expert.get_freqs(action.shape[1])
        split = lambda tensor: tensor.unflatten(-1, (expert.num_heads, self.model.video.attn_head_dim)).transpose(1, 2)
        for i, (pk, pv) in enumerate(memory):
            block = expert.blocks[i]
            q, k, v, residual, gate, shift, scale, mlp_gate, _ = self.model.ops._build_expert_attention_io(
                expert, block, x, freq, mod)
            mixed = F.scaled_dot_product_attention(split(q), torch.cat((pk, split(k)), 2),
                torch.cat((pv, split(v)), 2), attn_mask=allowed, dropout_p=0.).transpose(1, 2).flatten(2)
            x = block.gate(residual, gate, block.self_attn.o(mixed))
            attention = block.cross_attn
            query = attention.norm_q(attention.q(block.norm3(x)))
            cross = flash_attention(query, *text_kv[i], num_heads=attention.num_heads, ctx_mask=text_mask)
            x = x + attention.o(cross)
            x = block.gate(x, mlp_gate, block.ffn(block.norm2(x) * (1 + scale) + shift))
            x = torch.where((~apad)[..., None], x, 0)
        return expert.post(x)

    def _actions(self, noise, memory, allowed, text_kv, text_mask, apad, times, deltas):
        action = noise
        for t, dt in zip(times, deltas):
            velocity = self._action_velocity(action, t.expand(len(action)), memory, allowed, text_kv, text_mask, apad)
            action = self.model.a_scheduler.step(velocity, dt, action)
        return action

    def _policy_core(self, x, raw, controls, force_input, valid, mask, video_text_kv,
                     observed, noise, allowed, text_kv, text_mask, apad, times, deltas):
        memory, wrench = self._prefix(x, raw, controls, force_input, valid, mask, video_text_kv, observed)
        return wrench, self._actions(noise, memory, allowed, text_kv, text_mask, apad, times, deltas)

    @torch.no_grad()
    def infer_forcewm(self, sample, *, action_condition, num_inference_steps=16,
                      seed=0, video_seed=44, generate_action=True):
        model = self.model
        prior_modes = {module: module.training for module in model.modules()}
        model.eval()
        try:
            if num_inference_steps < 1:
                raise ValueError("Positive Action generation steps required")
            first = sample["first_frame_latents"].to(model.device, model.torch_dtype)
            if first.ndim != 5 or first.shape[1:3] != (16, 1):
                raise ValueError("Only observed Cosmos latent accepted")
            action = action_condition.to(model.device, model.torch_dtype)
            if action.shape != (len(first), 32, 7):
                raise ValueError("Explicit clean Action32x7 required")
            raw, context, cmask = model._contexts(sample)
            apad = sample["action_is_pad"].to(model.device, torch.bool)
            controls = model.dit["controls"](action, apad)
            force = model.dit["force"]
            force_input, _, valid, _, _ = force.prepare(
                history=sample["history"].to(model.device, model.torch_dtype),
                action_condition=action, context=context, context_mask=cmask,
                history_is_pad=sample["history_is_pad"].to(model.device, torch.bool), candidate_is_pad=apad,
                history_channel_valid=sample["history_channel_valid"].to(model.device, torch.bool),
                observation_latents=first)
            # Local generators reproduce the reference's seeds without changing
            # its persistent training RNG, even during warmup or graph capture.
            shape = (*first.shape[:2], 3, *first.shape[3:])
            x = torch.randn(shape, generator=torch.Generator().manual_seed(video_seed)).to(model.device, model.torch_dtype)
            x[:, :, :1] = first
            times, deltas = model.v_scheduler.build_inference_schedule(model.generated_video_steps, model.device, x.dtype)
            prepared = self._prepare_video(x, times[0].expand(len(x)), raw, controls, validate=True)
            groups, prefix_valid, branches = joint_token_layout(prepared["tokens"].shape[1],
                prepared["tokens_per_frame"], valid, force.history_steps, force.horizon)
            mask = grouped_attention_mask(groups, prefix_valid, branches)
            video_text_kv = self._video_text_kv(raw)
            x, observed = self._execute("video", self._rollout,
                x, first, raw, controls, mask, video_text_kv, times, deltas)
            result = {"video_latents": x}
            if generate_action:
                noise = torch.randn((len(first), 32, 7), generator=torch.Generator().manual_seed(seed)).to(model.device, model.torch_dtype)
                atimes, adeltas = model.a_scheduler.build_inference_schedule(num_inference_steps, model.device, model.torch_dtype)
                allowed = action_prefix_mask(prefix_valid, apad)
                text_mask = cmask[:, None, None, :].expand(-1, 1, 32, -1)
                result["force"], result["action"] = self._execute("policy", self._policy_core,
                    x, raw, controls, force_input, valid, mask, video_text_kv,
                    observed, noise, allowed, self._action_text_kv(context), text_mask, apad, atimes, adeltas)
            else:
                _, result["force"] = self._prefix(x, raw, controls, force_input, valid, mask, video_text_kv, observed)
            if any(not bool(torch.isfinite(value).all()) for value in result.values()):
                raise ValueError("Nonfinite inference output")
            # Graph outputs are reused on the next request. Return owned values.
            return {key: value.clone() for key, value in result.items()}
        finally:
            for module, training in prior_modes.items():
                module.training = training
