# DINOv3 H+/16 with Cosmos Joint8 Action/Force

This experiment uses the frozen official [DINOv3 ViT-H+/16](https://modelscope.cn/models/facebook/dinov3-vith16plus-pretrain-lvd1689m) image encoder (about 840M parameters). The vendored official implementation has exactly 840,633,600 parameters; the HF-format checkpoint stores 840,592,640 tensor elements because it omits the masked key biases. It produces 1,280-wide normalized patch features from 224×224 RGB frames, with 32 transformer blocks, 20 attention heads, four register tokens, and a 16×16 patch size.

The staged ModelScope artifact is `model.safetensors`, 3,362,432,800 bytes, SHA256 `3e1d4d18b9bfa9f28fad8e9de6a783f1313532d3460efa4cd0b12521d81d1a4d`, at repository revision `98b7096b2938406ade58801a0bf75eca3198b5bd`. The launcher checks both size and digest before creating a run. The vendored official DINOv3 source is pinned to `6876159a11b4df116f30f667f8c9888617df0751`. Accept Meta's DINOv3 terms before setting `DINOV3_LICENSE_ACCEPTED=1` in the training environment.

## Temporal Features

DINOv3 processes a single image at a time. The cache builder independently encodes each of the ten sampled frames and averages features at matching camera and patch positions within adjacent pairs:

| Slot | Source frame offsets |
|---|---|
| Observation | `[-4, 0]` |
| Future 1 | `[4, 8]` |
| Future 2 | `[12, 16]` |
| Future 3 | `[20, 24]` |
| Future 4 | `[28, 32]` |

Each image yields 196 patches per camera. Main-camera patches precede wrist-camera patches, giving 392 tokens per slot. The immutable BF16 cache shape is `[5,392,1280]`. Pooling averages each pair in FP32 before storing BF16. It does not preserve order or motion within a pair; temporal prediction is learned by the trainable video predictor over the five slots. The observed slot is built only from offsets `[-4,0]`; future slots are read only as targets during training and do not enter inference context.

The main experiment uses `loss = action_FM + force_FM + L1(predicted_future, layer_norm(DINO_future))`. The control has the same initialization, architecture, data split, and optimizer but multiplies that future L1 by zero. The DINO encoder stays frozen and is not part of the trainable checkpoint. Cosmos Action and Force remain 28-layer experts.

## Launch

Stage the ModelScope weights and dataset on the H200 job filesystem. The experiment runner does not download weights during distributed work.

```bash
export DINOV3_LICENSE_ACCEPTED=1
python scripts/dense/run_experiment.py \
  --teacher dinov3_vith16plus --experiment main \
  --weights /absolute/shared/model.safetensors \
  --teacher-source /absolute/repo/third_party/dinov3 \
  --dataset /absolute/shared/plug-insertion-dataset \
  --run-root /absolute/shared/new-dinov3-hplus-main
```

The run keeps the existing four-H200, batch-64-per-rank, BF16, 20,000-update recipe. It builds a new immutable feature cache, runs a two-update/save preflight and full-state restore check, then starts fresh training. The DINO H+ main and no-future control configs are under `configs/experiments/`.

The default data remains the audited 71-episode PlugInsertion71 Joint8 source, split into 55 train / 8 validation / 8 test episodes using seed 42. The historical branch name contains `cup` and `30k`; it does not change the existing dataset or 20k recipe. Normalization is fitted only on training episodes. Action/Force losses and expert geometry retain the previous experiment's definitions.

For the control, use `--experiment control`, a fresh `--run-root`, and `--cache-root /absolute/shared/new-dinov3-hplus-main/features`. The cache schema is now `fastwam-temporal-feature-window-v2`; old V-JEPA and ViT-B caches cannot be reused for H+. Reuse verifies the teacher/weight identity, temporal slots, pooling, width, split, and shard digests. H+ starts with fresh predictor/Action/Force weights and optimizer state; no previous policy checkpoint is loaded.

The policy model's `infer_joint` consumes precomputed `visual_features`. For online integration, `DenseTeacher.encode_video(observed_rgb, context_only=True)` accepts the two observed paired-camera frames `[B,3,2,224,448]` in `[-1,1]` and returns `[B,1,392,1280]` with the same pooling as the cache. The existing RealMan serving adapter is for historical Cosmos models and does not yet route to this DINO policy. CPU contract tests do not establish actual-weight inference, four-H200 throughput, or closed-loop robot performance.

## Validation

On 2026-10-30, the CPU regression suite completed with **232 passed and 6 skipped** using Python 3.10 and PyTorch 2.9.1 after initializing the pinned Cosmos submodule:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=src:deploy OMP_NUM_THREADS=2 \
  python -m pytest -q tests/cosmos tests/dense \
  --ignore=tests/cosmos/test_reason1_native.py --tb=short
```

`test_reason1_native.py` was excluded because the local environment lacks `transformers`; the six skips are reported by the remaining suite. The checks cover the H+ architecture on the meta device, strict checkpoint key/shape mapping with small synthetic models, paired-camera preprocessing, observed/future isolation, cache shard ordering, and startup identity guards. The full ModelScope weights were not downloaded or executed, and no H200 training or robot deployment was started.
