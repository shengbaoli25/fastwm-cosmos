# RM71D78 ForceVLA ablation

Current branch: **`codex/ablation-rm71d78-forcevla`**.

This independent branch adapts the official ForceVLA pi0 + force-aware LIMoE
model to the 78-demonstration `rm71d_plug_transfer_20261029` dataset and the
existing RealMan joint8 HTTP client. The new ablation entrypoints support only
ForceVLA. They are based on `057f78767f52f358b229f4a05b097c0f1455fa3d`; existing
model training entrypoints are retained unchanged.

Start with the [Chinese run guide](docs/ablation/README.zh.md), then the
[official model/environment and configuration guide](docs/ablation/forcevla.md).
See [pinned sources](configs/ablation/upstream-lock.json) and the
[example configuration](configs/ablation/forcevla.example.json).

```bash
# Plan only. Does not download weights or start training.
python scripts/ablation/forcevla.py --config configs/ablation/forcevla.example.json
```

The branch includes verified-data export, official statistics/train/resume
wrappers, training/checkpoint identity checks, deployment export, and a server
supporting `/health`, `/infer`, and `/session/release`. Predictions are 32 rows
of absolute joint8: seven joint angles in radians and gripper openness.

**No training was launched and no large weights were downloaded.** Validation
covers CPU data/contracts, reviewed upstream source and loopback HTTP with
explicit test predictors. Real dataset validation, real pretrained/checkpoint
loading, GPU updates/resume and measured model latency/memory remain pending.
[Validation receipt](docs/ablation/validation.json) reports this branch's checks.
