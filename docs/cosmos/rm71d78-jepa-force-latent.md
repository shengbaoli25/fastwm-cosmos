# All78 delta + JEPA observation / online Force latent

This experiment starts directly from official Cosmos and JEPA assets. Action and
Force experts start fresh. No robot-policy checkpoint, teacher pretraining run,
offline Force-feature cache, or optimizer state is required.

The baseline remains the no-reference triangular V/F/A model descended from
`0ff5871`, using the existing all78 `delta_current_joint7` data contract. Each
joint chunk is relative to the current measured joint7 anchor; gripper openness
remains absolute. The existing baseline run and prepared artifacts are read-only.

## Actual model and information flow

The reader receives noisy Action and its diffusion clock, measured force/state
history, current proprioception, task context, and **observed JEPA patches only**.
It has eight queries, width 256, two attention layers, and four heads. Patch values
must pass through visual attention; there is no direct query residual bypass.

At layers 7, 14, 21 and 28, a gated residual modifies the Force K/V **copy read by
Action**. It does not modify the original Force tokens or write to Video/history.
Video continues reading O/V, Force reads O/H/V/F, and Action reads O/H/V/F/A plus
the new visual correction. This retains the no-action-conditioned world models.

During training, a separate `no_grad` pass reuses the current Force parameters
with clean measured future wrench at clock zero. That label pass attends only to
observed O/H and clean F; it contains no future video. Its final pre-head Force
features are stopped-gradient targets for the JEPA reader. They are not model
inputs, and the label pass is absent at inference. Main Force FM trains the shared
Force parameters; this is an evolving online target, not a pretrained teacher.

The three original Video/Force/Action FM losses retain weights 1:1:1. Auxiliary
Force-feature loss has weight 1 after a linear 2500-update ramp. The auxiliary
wrench regression weight is 0.1 from the start. The first matched trial has
appearance-consistency weight **0**; adding that objective later is a distinct
ablation, not an unrecorded recipe change.

## Matched ViT-B / ViT-L runs

The literal shared recipe is `configs/experiments/jepa_force_variants.json`.
Both runs use 78 training episodes, no validation/test split, the same original
anchors, B64 per rank on four H200 ranks (global256), AdamW lr1e-4, weight decay
0.01, BF16, accumulation1, and seed42. Training stops at 30000 updates while the
cosine schedule retains its 50000-update horizon and 2500 warmup updates. Save
every 5000 updates. The new reader/adapter initialization seed is 47.

| Encoder | Width / depth / heads | Official checkpoint |
|---|---|---|
| ViT-B | 768 / 12 / 12 | `vjepa2_1_vitb_dist_vitG_384.pt` |
| ViT-L | 1024 / 24 / 16 | `vjepa2_1_vitl_dist_vitG_384.pt` |

Both use patch16/tubelet2, 224x224 per camera, camera-major 392 output patches,
and rank8/alpha8 Q/V LoRA in the final four encoder blocks. Other official JEPA
parameters remain frozen; the encoder executes online so LoRA receives gradients.
The checkpoint filename describes its official asset; the recorded experiment
uses the same 224 input and positional handling for both variants. The caller
must provide each checkpoint's actual SHA256; no digest is fabricated by a plan.

## Data reuse and observation cache

The default baseline prepared directory is:

```text
/inspire/ssd/project/long-range-embodied-collaborative-intelligence/ky26340/rm71d-cosmos-videoisolated-delta30k-20261103/online/job-83b69f78-eedb-40d3-9f6d-1fad8a22cba1/prepared
```

The orchestrator checks the raw-source inventory, all78 source identity, complete
anchor inventory, exact delta statistics/normalization, every feature shard,
text pairing, official Cosmos weights, and pinned upstream. It calls the existing
`resolve_config.py` for the baseline, then changes only the JEPA additions and
dataset wrapper.

For each anchor, `jepa_observation` has shape `[2,3,2,224,224]`, RGB in `[-1,1]`,
main camera then wrist, and observed frame offsets `[-4,0]`. Both indices clamp
to the same continuous source segment; there are no future frames. All original
numeric, mask and Cosmos-latent tensors remain unchanged.

Two runs share a single observation cache:

```text
/inspire/ssd/project/long-range-embodied-collaborative-intelligence/ky26340/rm71d-jepa-force-latent-20261104/prepared/observations
```

Each episode is decoded with the baseline decoder and bilinear 224 resize, then
rounded into a uint8 `.npy` mmap. This adds at most 1/255 quantization error in
normalized RGB. Storage is 301056 bytes per recorded source frame, about 25.4 GB
for 84349 frames; exact size depends on the source-frame count, not retained
anchor count. No JEPA features are cached. A shared kernel file lock prevents
two variants writing concurrently; interrupted preparation can append after
checking every already committed episode hash. A complete cache is reused only
when its source, baseline, geometry and decoder identities match.

## Planning and execution

Printing a plan does not access assets or start training:

```bash
python scripts/cosmos/train_rm71d78_jepa.py \
  --variant vitb --jepa-checkpoint /absolute/path/vjepa2_1_vitb_dist_vitG_384.pt \
  --jepa-sha256 "$VERIFIED_VITB_SHA256" --plan
```

In a newly allocated four-H200 platform job, use the original native environment:

```bash
bash scripts/cosmos/launch_rm71d78_jepa.sh \
  --variant vitb --jepa-checkpoint /absolute/path/vjepa2_1_vitb_dist_vitG_384.pt \
  --jepa-sha256 "$VERIFIED_VITB_SHA256"
```

The second job passes `--variant vitl` and its matching checkpoint/hash. The
variant is mandatory even when the branch recipe lists a default. Output paths
are separate `.../rm71d-jepa-force-latent-20261104/<variant>/online/$TRAIN_JOB_ID`.
Optional `--output-root`, `--prepared`, `--dataset`, `--observation-cache`,
`--checkpoint` and `--upstream` relocate explicit paths without weakening identity
checks. `--source-sha` must retain the pinned original all78 inventory.

After observation preparation, the existing `distributed_chunk.py` runs the
actual four-rank two-Adam/full-save gate and an independent four-rank exact-restore
gate before formal training. The JEPA preflight additionally audits real LoRA
gradients and frozen encoder parameters. A failed gate stops this attempt; the
launcher does not reduce batch size, change resources, or fall back to another
model. `pipeline-status.json`, `experiment.json`, resolved configs, gate reports
and the normal full training checkpoints record the actual outcome.

Native model inference uses `create_jepa_force_latent_cosmos`, `load_checkpoint`
and `infer_joint`, with causal `jepa_observation` added to the observed input. It
does not require future labels. The existing `export_noaction_deployment.py`,
RealMan deployment adapter and optimized runner do not support this new schema;
deploying these checkpoints needs a separate integration change.

The all78 recipe intentionally has no held-out set. These runs establish training
behavior; claims about appearance robustness require separate
held-out environments and closed-loop evaluation. The launch scripts do not
claim that those evaluations or GPU gates have already passed.
