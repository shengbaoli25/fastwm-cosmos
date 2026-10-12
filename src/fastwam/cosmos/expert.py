"""Expose the official Cosmos Base DiT to the FastWAM cached training contract.

The adapter retains native patching, QKV norms/RoPE, three AdaLN branches,
text cross-attention and output head. The native video path remains
bidirectional. Policy observation K/V comes from a separate clean single-frame
pass that cannot see teacher future video or action.
"""
from __future__ import annotations

from contextlib import nullcontext

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class CosmosExpert(nn.Module):
    """Native Base adapter with explicit global-time or observed-only mode.

    Video uses native bidirectional attention and one global diffusion time.
    The policy observation stream contains one clean latent and sigma0, and
    never receives teacher future video. Action conditioning is rejected.
    """

    def __init__(self, net: nn.Module):
        super().__init__()
        required = ("prepare_embedded_sequence", "t_embedder", "t_embedding_norm", "blocks",
                    "final_layer", "unpatchify")
        if any(not hasattr(net, name) for name in required):
            raise ValueError("CosmosExpert requires the native base Cosmos DiT")
        if getattr(net, "patch_temporal", None) != 1:
            raise ValueError("CosmosExpert requires patch_temporal=1 ")
        if not len(net.blocks):
            raise ValueError("CosmosExpert requires at least one transformer block")
        self.net = net
        self.hidden_dim = int(net.model_channels)
        self.num_heads = int(net.num_heads)
        self.attn_head_dim = int(net.blocks[0].self_attn.head_dim)
        self.num_layers = len(net.blocks)
        self.patch_size = (1, int(net.patch_spatial), int(net.patch_spatial))
        if self.num_heads * self.attn_head_dim != self.hidden_dim:
            raise ValueError("Cosmos attention width must match native model channels")
        for block in net.blocks:
            if (block.self_attn.n_heads != self.num_heads or block.self_attn.head_dim != self.attn_head_dim
                    or getattr(block, "cp_size", None) not in (None, 1)):
                raise ValueError("CosmosExpert supports matching single-device native attention blocks")

    @property
    def blocks(self):
        return self.net.blocks

    @staticmethod
    def _fp32_region(module, value):
        # The loader preserves native modulation/final projection weights in
        # FP32. Disabling caller autocast is sufficient and also works on CPU.
        return (torch.autocast(value.device.type, enabled=False)
                if getattr(module, "use_wan_fp32_strategy", False) else nullcontext())

    def prepare(self, latents: Tensor, sigma_video: Tensor, reason1_context: Tensor,
                candidate_actions: Tensor | None = None, observed_only: bool = False,
                fps: float | Tensor = 30, *, observed_time: str = "native_global", context_is_projected: bool = False) -> dict:
        if (latents.ndim != 5 or min(latents.shape) < 1 or not latents.is_floating_point()
                or not torch.isfinite(latents).all()):
            raise ValueError("Cosmos latents must be finite floating [B,C,T,H,W]")
        batch, channels, frames, height, width = latents.shape
        if channels != self.net.out_channels or height % self.patch_size[1] or width % self.patch_size[2]:
            raise ValueError("Cosmos latent channels/spatial patch dimensions do not match the network")
        if not isinstance(observed_only, bool) or (observed_only and (frames != 1 or candidate_actions is not None)):
            raise ValueError("observed_only requires exactly one observed latent frame and no candidate actions")
        if not isinstance(sigma_video, Tensor) or sigma_video.shape != (batch,):
            raise ValueError("sigma_video must be [B] in the normalized [0,1] RF convention")
        if not sigma_video.is_floating_point() or not torch.isfinite(sigma_video).all() or torch.any((sigma_video < 0) | (sigma_video > 1)):
            raise ValueError("sigma_video must be finite floating values in [0,1]")
        if observed_only and torch.count_nonzero(sigma_video):
            raise ValueError("Observed-only K/V prefill requires sigma_video=0")
        if observed_time not in {"clean_zero", "native_global"}:
            raise ValueError("observed_time must be clean_zero or native_global")
        if observed_only and observed_time != "clean_zero":
            raise ValueError("Observed-only K/V prefill requires clean_zero observed time")
        if (reason1_context.ndim != 3 or reason1_context.shape[0] not in (1, batch)
                or reason1_context.shape[1] < 1 or not reason1_context.is_floating_point()
                or not torch.isfinite(reason1_context).all()):
            raise ValueError("Reason1 context must be finite floating [B,L,D], with B or one batch")
        expected_context = (self.net.crossattn_proj_in_channels if self.net.use_crossattn_projection and not context_is_projected
                            else self.net.blocks[0].cross_attn.k_proj.in_features)
        if reason1_context.shape[-1] != expected_context:
            raise ValueError("Reason1 context dimension differs from the native text projection")
        if isinstance(fps, Tensor):
            if fps.shape != (batch,):
                raise ValueError("fps tensor must have shape [B]")
            frame_rate = fps.to(device=latents.device, dtype=torch.float32)
        else:
            frame_rate = torch.full((batch,), float(fps), device=latents.device)
        if not torch.isfinite(frame_rate).all() or torch.any(frame_rate <= 0):
            raise ValueError("fps must be finite and positive")

        net_dtype = next(self.net.x_embedder.parameters()).dtype
        input_latents = latents.to(dtype=net_dtype)
        observed_mask = torch.zeros((batch, 1, frames, height, width), device=latents.device, dtype=net_dtype)
        observed_mask[:, :, 0] = 1
        padding = torch.zeros((batch, 1, height, width), device=latents.device, dtype=net_dtype)
        x, rope, extra = self.net.prepare_embedded_sequence(
            torch.cat((input_latents, observed_mask), dim=1), fps=frame_rate, padding_mask=padding)
        grid = tuple(x.shape[1:4])
        if grid[0] != frames:
            raise ValueError("Native patching changed the temporal layout")
        context = reason1_context.to(device=latents.device, dtype=net_dtype).expand(batch, -1, -1)
        if self.net.use_crossattn_projection and not context_is_projected:
            context = self.net.crossattn_proj(context)

        # The official model accepts 0..1000 timesteps, then applies its
        # timestep_scale (0.001 in the pinned checkpoint). Native video keeps
        # global time even for clean observed pixels; isolated K/V uses zero.
        timestep = sigma_video.to(device=latents.device, dtype=torch.float32)[:, None].expand(batch, frames) * 1000
        if observed_time == "clean_zero":
            timestep = torch.cat((torch.zeros_like(timestep[:, :1]), timestep[:, 1:]), dim=1)
        timestep = timestep * self.net.timestep_scale
        with self._fp32_region(self.net, latents):
            time, adaln = self.net.t_embedder(timestep)
        if candidate_actions is not None:
            raise ValueError("This Base video branch never accepts action conditioning")
        with self._fp32_region(self.net, time):
            time = self.net.t_embedding_norm(time)
        if extra is not None and extra.shape != x.shape:
            raise ValueError("Native extra positional embedding has an unsupported layout")
        return {"tokens": x.flatten(1, 3), "tokens_per_frame": grid[1] * grid[2], "grid": grid,
                "state": {"time": time, "adaln": adaln, "context": context, "rope": rope,
                          "extra": extra, "observed_only": observed_only, "batch_size": batch,
                          "observed_time": observed_time}}

    def _grid_tokens(self, tokens: Tensor, prepared: dict) -> Tensor:
        frames, height, width = prepared["grid"]
        if tokens.shape != (prepared["state"]["batch_size"], frames * height * width, self.hidden_dim):
            raise ValueError("Video tokens do not match the prepared grid")
        return tokens.reshape(tokens.shape[0], frames, height, width, self.hidden_dim)

    def block_qkv(self, layer: int, tokens: Tensor, prepared: dict):
        """Return native normalized/rotated Q,K,V as [B,H,S,D], and residual state."""
        block = self.net.blocks[layer]
        x = self._grid_tokens(tokens, prepared)
        state = prepared["state"]
        if state["extra"] is not None:
            x = x + state["extra"]
        modulations = {}
        with self._fp32_region(block, x):
            for branch in ("self_attn", "cross_attn", "mlp"):
                modulation = getattr(block, f"adaln_modulation_{branch}")(state["time"])
                if block.use_adaln_lora:
                    modulation = modulation + state["adaln"]
                modulations[branch] = tuple(value[:, :, None, None].to(dtype=x.dtype)
                                            for value in modulation.chunk(3, dim=-1))
        shift, scale, _ = modulations["self_attn"]
        normalized = block.layer_norm_self_attn(x) * (1 + scale) + shift
        q, k, v = block.self_attn.compute_qkv(normalized.flatten(1, 3), rope_emb=state["rope"])
        return q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), {"residual": x, "modulations": modulations}

    def block_finish(self, layer: int, mixed: Tensor, post_state: dict, prepared: dict) -> Tensor:
        """Apply Cosmos output projection, gated cross-attention and gated MLP."""
        block = self.net.blocks[layer]
        x = post_state["residual"]
        batch, frames, height, width, _ = x.shape
        if mixed.shape != (batch, self.num_heads, frames * height * width, self.attn_head_dim):
            raise ValueError("Mixed attention output must be [B,H,S,D] matching the Cosmos token grid")
        flat = mixed.transpose(1, 2).flatten(2)
        self_output = block.self_attn.output_dropout(block.self_attn.output_proj(flat)).reshape_as(x)
        x = x + post_state["modulations"]["self_attn"][2] * self_output
        shift, scale, gate = post_state["modulations"]["cross_attn"]
        normalized = block.layer_norm_cross_attn(x) * (1 + scale) + shift
        cross_output = block.cross_attn(normalized.flatten(1, 3), prepared["state"]["context"],
                                       rope_emb=prepared["state"]["rope"]).reshape_as(x)
        x = x + gate * cross_output
        shift, scale, gate = post_state["modulations"]["mlp"]
        normalized = block.layer_norm_mlp(x) * (1 + scale) + shift
        return (x + gate * block.mlp(normalized)).flatten(1, 3)

    def finish(self, tokens: Tensor, prepared: dict) -> Tensor:
        """Decode expert tokens to native RF velocity (not reconstructed RGB)."""
        x = self._grid_tokens(tokens, prepared)
        state = prepared["state"]
        with self._fp32_region(self.net.final_layer, x):
            output = self.net.final_layer(x, state["time"], adaln_lora_B_T_3D=state["adaln"])
        return self.net.unpatchify(output)

    def forward(self, latents: Tensor, sigma_video: Tensor, reason1_context: Tensor,
                candidate_actions: Tensor | None = None, observed_only: bool = False,
                fps: float | Tensor = 30, attention_mask: Tensor | None = None,
                *, observed_time: str = "native_global") -> Tensor:
        """Standalone expert for native parity checks; optional explicit diagnostic mask.

        With mask=None, native_global reproduces the native global-time video
        path. Policy context always uses a separate observed-only pass, never
        the future-containing training video.
        """
        prepared = self.prepare(latents, sigma_video, reason1_context, candidate_actions, observed_only, fps,
                                observed_time=observed_time)
        tokens = prepared["tokens"]
        if attention_mask is not None and (attention_mask.dtype != torch.bool or attention_mask.shape != (tokens.shape[1], tokens.shape[1])):
            raise ValueError("Standalone Cosmos attention_mask must be a boolean [S,S] allowed-attention matrix")
        for layer in range(self.num_layers):
            q, k, v, state = self.block_qkv(layer, tokens, prepared)
            mixed = F.scaled_dot_product_attention(q, k, v, attn_mask=attention_mask)
            tokens = self.block_finish(layer, mixed, state, prepared)
        return self.finish(tokens, prepared)
