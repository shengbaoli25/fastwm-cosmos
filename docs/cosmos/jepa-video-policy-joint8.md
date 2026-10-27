# JEPA video context with Cosmos Joint8 Action/Force

This branch replaces the former single-frame feature-flow experiment. The main
arm uses `lambda_future: 1`; the control uses `lambda_future: 0`. Both arms have
identical architecture, initialization, data split, schedules and inference.
Old feature caches and old checkpoints are rejected by the new schema.

## Structure and scope

The frozen official V-JEPA 2.1 ViT-g target encoder has 1,013,267,968 parameters.
The local artifact is `vjepa2_1_vitg_384.pt` (16,878,318,788 bytes), SHA256
`b417628f1618c8bd52c0f419800b802f65794288c6ef2ff85c1341a1ae587cba`.
The hash provenance is the pinned ModelScope repository metadata; local official
bytes must match it before any encoding or training starts. Pinned upstream
source is `204698b45b3712590f06245fbfba32d3be539812`.

Each camera uses 224×224 input, native 14×14 patches, tubelet size 2, no spatial
pooling. A sample has raw 30 Hz frame offsets `[-4,0,4,8,12,16,20,24,28,32]`.
The context path encodes only `[-4,0]` through the video patch stem and every
Transformer block. The target path separately encodes the complete 10-frame
clip and retains the four future tubelets. Consequently teacher future content
cannot enter observed tokens. Cameras are encoded independently and ordered
main then wrist within each tubelet. Cached tensor shape is `[5,392,1408]`.

The independently initialized predictor has 28 layers, width 768, 12×64 self
attention, 3072-wide GELU FFNs, and condition cross attention. Observed tokens
and 1,568 learned future-query positions interact bidirectionally. A shared
learned mask token, separate view embeddings and 3D rotary coordinates define
the queries; row/column coordinates reset for each camera. Predictor conditions
are a learned single-task identifier, current proprio9 and measured 32×14
joint/force history with channel validity.

Action and Force are the original Cosmos `ActionDiT` and
`ForceDiffusionExpert`: 28 layers, residual width 768, 16×128 attention,
Action FFN3072 / Force FFN2048, condition width1024. Their entire state_dict key
and tensor shape sets match the Cosmos Joint8 reference (770 Action keys,
381 Force keys). Source-clock modules are preserved; deterministic visual
clock equals clean0, while Action and Force clocks remain independent.

To preserve native Cosmos attention geometry, each predictor layer exports
only its updated observed-context positions through a learned K/V projection
768→4096, split into K and V of width2048. K receives 3D RoPE128 and V does not.
This **post-layer context adapter** is an explicit adaptation of V-JEPA Policy,
not a claim of exact paper reproduction: the paper reads its native pre-layer
context self-attention K/V. The adapter allows all 28 Cosmos expert layers to
have a one-to-one predictive context interface and preserves their tensor
shapes. Future-query K/V never directly enter either expert.

Force reads context/history/Force; Action reads context/history/Force/Action.
Force K/V seen by Action come from the same layer's input, as in Cosmos.
The predictor and measured history run once at inference, then cached context
feeds 16 Action/Force flow steps. No visual flow diffusion remains. Teacher
weights are excluded from the trainable model and need not reside on the GPUs
after feature precomputation.

| Trainable component | Parameters |
| --- | ---: |
| Predictor including 28 context bridges | 355,893,376 |
| Action | 491,060,488 |
| Force | 270,040,838 |
| Current proprio projection | 10,240 |
| Action source-clock module | 3,937,536 |
| Fixed task embedding | 1,024 |
| Total | 1,120,943,502 |

Combined with the frozen encoder: 2,134,211,470 parameters. The architecture is
larger than the earlier proposed12-layer predictor because matching Cosmos
experts requires28layers; it must pass the real four-H200 B64/rank preflight.

## Data and losses

Episode partition uses `torch.randperm(71, generator=manual_seed(42))`: first55
train, next8 validation, last8 test. Strict windows require all -4…+32 samples
inside one continuous valid source segment. Incomplete tails and discontinuity
crossings are removed. Historical32-step values can be padded and their validity
is respected. No future padding controls predictor query construction.

Normalization is fitted only on retained source rows from training episodes;
validation/test reuse those exact statistics. Visual cache assembly does not
learn across samples. Numeric training normalization is persisted in the run
metadata. All experiments use the same windows and split.

`loss = action_FM + force_FM + lambda_future * L1(prediction, layer_norm(target))`.
The target normalization is over each token's1408feature dimensions. Even in
the control arm the future head and L1 are computed and multiplied by0, so all
parameters remain part of the graph while action/force loss can train queries.
The reported raw futureL1 still diagnoses the control representation.

4×H200, physical batch64/rank, global256, accumulation1, BF16, ZeRO1,
activation checkpointing, AdamW1e-4/0.01/(0.9,0.95), warmup1000, cosine to1e-6,
20,000fresh steps. No old optimizer or model warm start. Save every1000; retain
5k milestones and two latest complete checkpoints.

Validation runs at saves every1000steps, using a fixed set of4windows from each
of8validation episodes and16flow steps. JSON logs contain futureL1, persistence
baselineL1, normalized action/force errors, jointMAErad, gripperMAE, forceMAEN,
and momentMAENm. Physical errors use original unclipped action/wrench targets;
normalized errors use the training normalization contract. These offline errors do not establish closed-loop insertion
success; held-out test episodes remain untouched by training validation.

## Offline launch

Stage the pinned source, dependency environment and verified official g weights
on the shared filesystem before launching any distributed job. No launch stage
has a network-download fallback.

```bash
python scripts/dense/run_experiment.py \
  --teacher vjepa21_vitg --experiment main \
  --weights /absolute/shared/vjepa2_1_vitg_384.pt \
  --teacher-source /absolute/repo/third_party/vjepa21 \
  --dataset /absolute/shared/dataset --run-root /absolute/shared/new-main-run
```

The main run builds the immutable video cache, checks two four-rank updates and
full-state save/restore, then starts fresh20k training. To reuse the completed
cache for the control:

```bash
python scripts/dense/run_experiment.py \
  --teacher vjepa21_vitg --experiment control \
  --weights /absolute/shared/vjepa2_1_vitg_384.pt \
  --teacher-source /absolute/repo/third_party/vjepa21 \
  --dataset /absolute/shared/dataset --run-root /absolute/shared/new-control-run \
  --cache-root /absolute/shared/new-main-run/features
```

The cache builder hashes all output shards. Reuse rechecks all hashes once before
workers start; per-window dataset access checks immutable metadata and uses
safetensors mmap slices rather than hashing/loading entire shards repeatedly.
The CLI requires `TRAIN_JOB_ID` and exactly four H200s. Cache building and formal
training failures are recorded per stage and stop the pipeline. Physical batch64
is never silently replaced by gradient accumulation.
