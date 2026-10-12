"""Pinned Cosmos kernels with differentiable PyTorch RMSNorm/RoPE and SDPA.

The AST adaptation changes only imports of unavailable TE/megatron and local
checkpoint I/O. DiT, Wan VAE, RF scheduler and UniPC implementations remain
upstream. Each rank uses these native Torch kernels; Accelerate/ZeRO owns data parallelism.
No context parallelism or alternate model architecture is introduced.
"""
from __future__ import annotations

import ast
import hashlib
import importlib.util
import os
from pathlib import Path
import sys
import types

NET_MODULE = "cosmos_predict2._src.predict2.networks.minimal_v4_dit"
VAE_MODULE = "cosmos_predict2._src.predict2.tokenizers.wan2pt1"

PINNED_CORE_SHA256 = {
    'cosmos_predict2/_src/predict2/action/networks/action_conditioned_minimal_v1_lvg_dit.py': 'a30a664c990790c42e5c8cf06801adac4432ba7a0eefaf8628433577c40f0845',
    "cosmos_predict2/_src/predict2/networks/minimal_v4_dit.py": "f672511a9c7944d87835cd257923149b0511ce798203c8a73fe488e59b2073b6",
    "cosmos_predict2/_src/predict2/networks/minimal_v1_lvg_dit.py": "2d3ddcf94158c61223bc610f31ede475f7e4e27cc3b604ff3557b73d891e665f",
    "cosmos_predict2/_src/predict2/tokenizers/wan2pt1.py": "885aaa3250f60f026b279f099dc08b2548d904604b839c57000c6c65a8d165bb",
}


def verify_upstream(repo: Path) -> dict[str, str]:
    """Fail before allocation if critical native code differs from the pin."""
    actual = {}
    for relative, expected in PINNED_CORE_SHA256.items():
        path = repo / relative
        actual[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual[relative] != expected:
            raise ValueError(f"Cosmos source does not match pinned a2c298b: {relative}")
    return actual

def load_adapted_module(name: str, path: Path, injected: dict, remove_prefixes: tuple[str, ...]):
    class RemoveImports(ast.NodeTransformer):
        def visit_Import(self, node):
            remaining = [a for a in node.names if not a.name.startswith(remove_prefixes)]
            return ast.copy_location(ast.Import(names=remaining), node) if remaining else ast.copy_location(ast.Pass(), node)

        def visit_ImportFrom(self, node):
            if (node.module or "").startswith(remove_prefixes):
                return ast.copy_location(ast.Pass(), node)
            return node

    tree = ast.fix_missing_locations(RemoveImports().visit(ast.parse(path.read_text(encoding="utf-8"))))
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    module.__dict__.update(injected)
    sys.modules[name] = module
    try:
        exec(compile(tree, str(path), "exec"), module.__dict__)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module


def native_imports(repo: Path):
    import torch
    verify_upstream(repo)
    # This flag excludes distributed/optional imports; torch autograd remains enabled.
    os.environ["COSMOS_TRAINING"] = "false"
    existing = sys.modules.get(NET_MODULE)
    if existing is not None and not getattr(existing, "_realman_native_torch", False):
        raise RuntimeError("Load native_torch in a fresh process, before another Cosmos backend")

    class NativeRMSNorm(torch.nn.Module):
        def __init__(self, hidden_size: int, eps: float = 1e-6):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(hidden_size))
            self.eps = eps

        def reset_parameters(self):
            torch.nn.init.ones_(self.weight)

        def get_extra_state(self):
            return None

        def set_extra_state(self, state):
            # TE stores kernel/FP8 metadata here; this BF16-only module has no such state.
            self.pretrained_te_extra_state = state

        def forward(self, x):
            value = x.float()
            value = value * torch.rsqrt(value.square().mean(dim=-1, keepdim=True) + self.eps)
            return (value * self.weight.float()).to(x.dtype)

    def native_rope(x, freqs, tensor_format="bshd", fused=True, **kwargs):
        if tensor_format != "bshd" or kwargs:
            raise ValueError("This runner supports only the target model's default noninterleaved bshd RoPE")
        if freqs.shape[0] != x.shape[1] or freqs.shape[-1] != x.shape[-1]:
            raise ValueError(f"RoPE shape mismatch: {x.shape}, {freqs.shape}")
        angles = freqs.transpose(0, 1)
        half = x.shape[-1] // 2
        rotated = torch.cat((-x[..., half:], x[..., :half]), dim=-1)
        return (x.float() * angles.cos() + rotated.float() * angles.sin()).to(x.dtype)

    # The native backend only reads local .pt files. Avoid importing unused S3/MSC clients.
    local_io = types.ModuleType("cosmos_predict2._src.imaginaire.utils.easy_io.easy_io")
    def local_load(file, **kwargs):
        if not isinstance(file, (str, Path)) or not Path(file).is_file():
            raise ValueError(f"Only existing local weight files are supported: {file}")
        return torch.load(file, map_location="cpu", weights_only=True, mmap=True)
    local_io.load = local_load
    local_package = types.ModuleType("cosmos_predict2._src.imaginaire.utils.easy_io")
    local_package.__path__ = []
    local_package.easy_io = local_io
    sys.modules[local_package.__name__] = local_package
    sys.modules[local_io.__name__] = local_io

    sys.path.insert(0, str(repo))
    # Upstream's package initializer requires a CUDA-extension sentinel even
    # when its extension kernels are unused. Import the unchanged model modules directly.
    cosmos_package = types.ModuleType("cosmos_predict2")
    cosmos_package.__path__ = [str(repo / "cosmos_predict2")]
    cosmos_package.__file__ = str(repo / "cosmos_predict2/__init__.py")
    sys.modules["cosmos_predict2"] = cosmos_package
    net_module = load_adapted_module(
        NET_MODULE, repo / Path(*NET_MODULE.split(".")).with_suffix(".py"),
        {"te": types.SimpleNamespace(pytorch=types.SimpleNamespace(RMSNorm=NativeRMSNorm)),
         "apply_rotary_pos_emb": native_rope}, ("transformer_engine",),
    )
    net_module._realman_native_torch = True
    no_parallel = types.SimpleNamespace(is_initialized=lambda: False)
    vae_module = load_adapted_module(
        VAE_MODULE, repo / Path(*VAE_MODULE.split(".")).with_suffix(".py"),
        {"parallel_state": no_parallel}, ("megatron",),
    )

    def load_local_vae(pretrained_path=None, z_dim=None, device="cpu", **kwargs):
        if kwargs.get("load_mean_std", False):
            raise ValueError("The target pretrained tokenizer does not use additional mean/std files")
        config = dict(dim=96, z_dim=z_dim, dim_mult=[1, 2, 4, 4], num_res_blocks=2,
                      attn_scales=[], temperal_downsample=[False, True, True], dropout=0.0)
        config["temporal_window"] = kwargs.get("temporal_window", 16)
        with torch.device("meta"):
            model = vae_module.WanVAE_(**config)
        weights = local_load(pretrained_path)
        model.load_state_dict(weights, strict=True, assign=True)
        model = model.to(device=device)
        return (model, torch.zeros(1, 1, 1, 1, 1, device=device),
                torch.ones(1, 1, 1, 1, 1, device=device),
                torch.zeros(1, 1, 50, 1, 1, device=device),
                torch.ones(1, 1, 50, 1, 1, device=device))

    vae_module._video_vae = load_local_vae
    from cosmos_predict2._src.predict2.action.networks.action_conditioned_minimal_v1_lvg_dit import ActionChunkConditionedMinimalV1LVGDiT
    from cosmos_predict2._src.predict2.conditioner import DataType
    return torch, net_module, vae_module, ActionChunkConditionedMinimalV1LVGDiT, DataType


def target_net_config() -> dict:
    return dict(
        max_img_h=240, max_img_w=240, max_frames=128, in_channels=16, out_channels=16,
        patch_spatial=2, patch_temporal=1, model_channels=2048, num_blocks=28, num_heads=16,
        concat_padding_mask=True, pos_emb_cls="rope3d", pos_emb_learnable=True,
        pos_emb_interpolation="crop", use_adaln_lora=True, adaln_lora_dim=256,
        atten_backend="torch", extra_per_block_abs_pos_emb=False,
        rope_enable_fps_modulation=False, rope_h_extrapolation_ratio=3.0,
        rope_w_extrapolation_ratio=3.0, rope_t_extrapolation_ratio=1.0,
        timestep_scale=0.001, use_crossattn_projection=True,
        crossattn_proj_in_channels=100352, crossattn_emb_channels=1024,
        use_wan_fp32_strategy=True, action_dim=7, temporal_compression_ratio=4,
    )


def preserve_fp32_strategy(net):
    net.t_embedder.float()
    for block in net.blocks:
        block.adaln_modulation_self_attn.float()
        block.adaln_modulation_cross_attn.float()
        block.adaln_modulation_mlp.float()
    net.final_layer.adaln_modulation.float()
    net.final_layer.linear.float()
