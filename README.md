# RM71D78 · V-JEPA ViT-B · 28-layer Action/Force

**Branch: `codex/rm71d78-vjepa-vitb-28l-50k`.** Frozen V-JEPA2.1 ViT-B/16
features supervise a fresh 28-layer, width768 predictor with matched Action/Force experts.
The new DINOv3 ViT-H+ branch shares the downstream geometry, data split and 50k recipe.

| Parameter scope | V-JEPA ViT-B (this branch) | DINOv3 ViT-H+ |
|---|---:|---:|
| Trainable model | **1,119,958,542** | **1,120,746,510** |
| Frozen visual encoder | **86,833,152** | **840,633,600** |
| Logical RGB-to-policy total | **1,206,791,694** | **1,961,380,110** |

See the [comparison, exact counts and runnable preparation/training commands](docs/cosmos/rm71d78-backbone-comparison.md)
and [reproducible parameter audit](docs/cosmos/backbone-parameter-counts.json).
The active default is in `configs/backbone_variant.json`; use `scripts/backbones/run.py`.
Both recipes use RM71D78, 4 H200, batch64/rank, BF16 and50,000 updates.
Code and CPU/meta checks are prepared; caches, new training and measured inference results are pending.

The sections below retain historical baselines and deployment evidence.

# FastWAM · Cosmos28 · ForceWM28

**Historical branch: `codex/realman-cup-joint-30k` — no-reference joint diffusion.**

**Joint8 step10000 inference evaluation (2026-10-21):** independent Video/Force/Action
sampling, compiled inference and CUDA Graph runners, 27-configuration offline replay,
and matched GB10 / RTX 4090 measurements are now included. Start with the
[experiment guide](docs/cosmos/experiments/joint8-step10000-20261021/README.md),
[Chinese report](docs/cosmos/experiments/joint8-step10000-20261021/评测报告.md), or
[PDF](docs/cosmos/experiments/joint8-step10000-20261021/joint8-step10000_三专家采样与GB10-4090评测报告.pdf).
V/F/A=4/4/16 is the first robot-validation candidate: RGB-to-output P50/P95 is
763/779 ms on GB10 and 344/351 ms on RTX 4090. Quality measurements replay training
episodes and do not establish closed-loop grasp success. Independent expert counts
are currently exposed by the offline runners; the deployment adapter's
`joint_optimized` backend retains its existing equal-step checkpoint contract.

The RealMan training entry points now select `task=cosmos_noaction_joint_4gpu_30k`:
official non-action-conditioned Cosmos Video2World, AE 768/3072, Force diffusion
768/2048, 28 layers each, and a shared layer core for training and 16-step joint
inference. Video reads Video; Force reads Video/Force; AE reads all three.
There is no reference trajectory or future action conditioning in either WM.
The next RealMan run uses seven absolute joint angles in radians plus gripper
openness (8D); measured joints form the state, and recorded Q×48 filtered force is
read without filtering twice. Training, normalization and deployment share this contract.
See the [current training and deployment guide](docs/cosmos/noaction-joint-diffusion.md).
The next run waits for the newly collected force dataset; this change launches no training.

The material below documents **historical action-conditioned baselines**.
Their configs, checkpoint loaders and experiment records remain available for reproduction.

**Joint-attention comparison branch:** `codex/cosmos-force-joint-self-attention`
adds bidirectional per-layer Cosmos/Force self-attention while keeping the original
RealAction recipe, clean GT actions in both WMs, and noisy actions in AE. Select
`task=cosmos_forcewm_joint_4gpu_30k`; see the [architecture, matched configuration,
validation and run guide](docs/cosmos/joint-self-attention.md). The following
description documents the retained RealAction baseline.

This experiment replaces the Wan video branch with the **official Cosmos-Predict2.5-2B robot/action-cond**, retains an Action Expert and a supervised direct-chunk ForceWM, and aligns all three branches to **28 layers**. The implementation inherits the LIBERO two-camera FastWAM data processing and Accelerator/ZeRO trainer. Legacy Wan entry points remain in the tree for provenance; this experiment uses only `scripts/cosmos/` and `configs/task/cosmos_forcewm_4gpu_30k.yaml`.

**Status:** implementation and synthetic CPU contract tests exist. Actual official-weight loading, complete Cosmos feature export, four-GPU training, distributed checkpoint restore and evaluation must be established by runtime receipts. CPU tests are not GPU/quality evidence. See [design and operations](docs/cosmos/design.md).

| Setting | This experiment |
|---|---|
| Video | Official Cosmos-Predict2.5-2B robot/action-cond, 28 blocks, width2048, 16×128 attention |
| Action Expert | Random initialization, 28 blocks, width1024, FFN4096, 16×128 attention |
| ForceWM | Random initialization, 28 blocks, width1280, FFN4608, 16 heads |
| Force objective | 32 learned queries predict the whole future wrench chunk; normalized masked six-dimensional MSE |
| Images | Main left / wrist right, each224², combined224×448; nine frames at offsets0:33:4 |
| Offline video features | Independent nine-frame `[16,3,28,56]` plus independently encoded current frame `[16,1,28,56]` |
| Text | Official Cosmos-Reason1-7B, full concatenation of28 normalized hidden layers,512×100352; cached offline |
| Batch | 4 ranks ×16 per rank ×accumulation1 =64; eight-card allocation does not change this |
| Updates / save | 30000 / every5000 |
| Optimizer | AdamW1e-4, betas(.9,.95), weight_decay.01; cosine,1500 warmup |
| Precision / distribution | BF16 with native Cosmos FP32 modulation/head regions; original ZeRO1 |
| Seeds | global/AE42, Force43, generated-video memory44 |
| Losses | VideoFM + ActionFM + ForceMSE, coefficients1/1/1 |
| Free video | Explicit clean-action-conditioned, 16 Euler steps + video clock0 feature refresh once; detached/no-grad |
| Control mapping | Random shared Linear32→7, grouping all 32 commands into 8 disjoint intervals with validity bits; seed45 |

Only Cosmos pretrained modules are loaded. AE, Force, proprio projection, Action's two memory connections and the dataset-to-control projection are random. Both official action embedders load from the robot checkpoint. **No previous Wan/Action/Force checkpoint is inherited.** This is an initialization difference as well as a video-backbone replacement; report it when comparing training curves.

```bash
git clone --recurse-submodules https://github.com/shengbaoli25/fastwm-cosmos.git
cd fastwm-cosmos
# Use the existing pinned FastWAM Torch2.7.1+cu128 environment on the GPU host.
python -m pip install -r requirements-cosmos.txt
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=src python -m pytest -q tests/cosmos
```

The source pin and exact official asset hashes are in `src/fastwam/cosmos/assets.py`, `native.py` and `reason1_weights.json`. The action-conditioned checkpoint is required; Base, Wan VAE/T5 features and random text features are not substitutes.

Both video supervision and generated memory use the same clean ground-truth action chunk. Force consumes `H[-31..0]`, clean `U[0..31]` and 32 learned queries, producing `W[1..32]` in one pass. Future wrench labels appear only in the loss. There is no previous-wrench feedback or Action diffusion clock in Force. Its W tokens retain a causal attention mask, which does not introduce autoregressive execution.

Action FM retains its noisy input and original loss. It reads fixed, detached generated Video/Force memory throughout denoising. With GT actions conditioning those memories, its metrics are **conditional reconstruction**, not observation-only policy performance. WM inference requires an explicit normalized `action_condition`; no noise or inferred fallback is supplied. Video latent diffusion still uses 16 steps. Inference now has one video generation (17 video passes including feature refresh), while joint training retains 19 video passes per batch.

This branch uses the new v3 checkpoint/manifest schema and rejects old AR checkpoints. See [clean-action chunk contract and launch](docs/cosmos/clean-action-chunk.md).

The unresolved `???` fields are deliberate: training requires the identity of an actual complete Cosmos feature/text cache. Follow the ordered preparation, probe, restore and foreground launch commands in [the operation guide](docs/cosmos/operations.md). Existing RGB and numeric caches are read only. VAE/text encoding happens offline; the training model has no VAE or RGB decoder.
