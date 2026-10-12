#!/usr/bin/env python3
"""Verify every data/release payload and original model tensor shape before GPU billing."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shared-repo", required=True)
    parser.add_argument("--pretrained-checkpoint", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    import torch
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf
    from fastwam.baseline.analysis import file_sha256
    from fastwam.models.wan22.action_dit import ActionDiT
    from fastwam.models.wan22.mot import MoT
    from fastwam.models.wan22.wan_video_dit import WanVideoDiT
    from evaluate import data_identity

    root = Path(__file__).resolve().parents[2]
    shared = Path(args.shared_repo).resolve()
    checkpoint = Path(args.pretrained_checkpoint).resolve()
    release = json.loads((root / "scripts/baseline/released_weights.json").read_text())
    if checkpoint.stat().st_size != release["size_bytes"] or file_sha256(checkpoint) != release["sha256"]:
        raise ValueError("Official full trained FastWAM checkpoint size/SHA256 mismatch")
    with initialize_config_dir(config_dir=str(root / "configs"), version_base="1.3"):
        cfg = compose(config_name="train", overrides=["task=forcevla_baseline"])
    with torch.device("meta"):
        mot = MoT({"video": WanVideoDiT(**OmegaConf.to_container(cfg.model.video_dit_config, resolve=True)),
                   "action": ActionDiT(**OmegaConf.to_container(cfg.model.action_dit_config, resolve=True))})
        proprio = torch.nn.Linear(8, 4096)
    # mmap avoids resident materialization of the giant shared checkpoint storage.
    payload = torch.load(checkpoint, map_location="cpu", mmap=True, weights_only=True)
    for key, expected in (("mot", mot.state_dict()), ("proprio_encoder", proprio.state_dict())):
        if set(payload[key]) != set(expected):
            raise ValueError(f"Released {key} tensor keys do not match the original baseline")
        for name, tensor in expected.items():
            if tuple(payload[key][name].shape) != tuple(tensor.shape):
                raise ValueError(f"Released tensor shape mismatch: {key}.{name}")
    stats = shared / "data/forcevla_fastwam/shared_stats/dataset_stats.json"
    stats_sha = file_sha256(stats)
    if stats_sha != "6d0ba7d47ce5bf5f290e3b5392ec8c13c07c2139e9a4f18d65a46cd5cb3a0dd4":
        raise ValueError("Baseline must reuse the exact frozen wipe normalization statistics")
    vae = shared / "checkpoints/DiffSynth-Studio/Wan-Series-Converted-Safetensors/Wan2.2_VAE.safetensors"
    if file_sha256(vae) != "0e913a2ca571c75fcb63385a8edadcca73454af5842596cb1ad11e4142590996":
        raise ValueError("Original Wan VAE checksum mismatch")
    cache = shared / "data/text_embeds_cache/forcevla"
    cache_files = {p.name: file_sha256(p) for p in sorted(cache.glob("*.pt"))}
    if not cache_files:
        raise ValueError("Official T5 task cache is missing")
    result = {"status": "complete", "release": release, "released_step": payload["step"],
              "strict_model_tensor_keys_and_shapes": True, "mot_tensors": len(payload["mot"]),
              "trainable_parameters": sum(p.numel() for p in mot.parameters()) + sum(p.numel() for p in proprio.parameters()),
              "stats_sha256": stats_sha, "vae_sha256": file_sha256(vae), "text_cache": cache_files,
              "data": data_identity(shared / "data/forcevla_fastwam/wipe"),
              "note": "CPU-only asset check; no GPU training, no optimizer construction"}
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: result[key] for key in ("status", "released_step", "mot_tensors", "trainable_parameters")}))


if __name__ == "__main__":
    main()
