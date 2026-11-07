# Clean30 JEPA ViT-L

This independent run inherits the exact ViT-L model/training configuration and
the startup dependency fix from `52c01ac`. The original all78 entrypoint and
model implementation are unchanged. It starts from official Cosmos and JEPA
weights with fresh Action/Force modules, optimizer and scheduler, as the current
experiment does; it does not resume a learned robot checkpoint or use a teacher.

## Dataset and task

The immutable source is `C:/Users/ASUS/Desktop/dataset/clean`, with inventory SHA
`c6854415da322c8051395525891b17f89e5469b762cca1322c6952b19299e4be`.
The specification is `configs/experiments/jepa_clean30_dataset.json`:
30 episodes, 46,870 recorded rows, 45,329 retained anchors, 285 continuous segments.
Every recorded task label remains exactly `七轴相对遥操作演示`. All 30 episodes
are training data; no new holdout split or task relabeling is introduced.

The existing adapter keeps only causal, fresh, executed continuous data and
clamps windows within each segment. The action representation remains
`delta_current_joint7` (seven target-joint deltas from the anchor's measured
joints, followed by absolute gripper openness). State and force statistics are
recomputed over retained anchors; action statistics are recomputed over valid
unpadded target positions. The existing z-score/clipping behavior is unchanged.

The new run creates its own `normalization.json`, Cosmos feature cache and JEPA
observed-RGB cache. It may reuse an official Reason1 text bank only after both
provenance validation and exact recorded-prompt equality; otherwise it encodes
the actual Chinese prompt with official Reason1. Old all78 numeric statistics or
video caches are rejected by full source identity equality.

## Recipe and commands

The recipe remains 4 H200 ranks, per-rank B64/global B256, accumulation 1,
30,000 optimizer updates, 50,000-step cosine horizon, 2,500 warmup steps,
lr 1e-4, weight decay .01, BF16/ZeRO1, seed 42 and save every 5,000 updates.
ViT-L is 24x1024; only Q/V rank-8 LoRA in its last four blocks train. Reader
width 256, depth 2, eight queries, four heads; residual layers 7/14/21/28;
latent weight 1 with 2,500-step ramp, wrench weight .1; no external teacher.

Local plan (no CUDA, dataset loading, downloads or output creation):

```powershell
python scripts/cosmos/train_jepa_dataset.py --dataset C:/Users/ASUS/Desktop/dataset/clean --plan
```

After uploading the unchanged dataset and code, a new four-H200 job uses:

```bash
bash /absolute/new-code/scripts/cosmos/launch_jepa_clean30.sh \
  --dataset /absolute/uploaded/clean \
  --text-cache /absolute/verified/matching-prompt/text
```

If no exact text bank exists, replace `--text-cache` with
`--reason1-checkpoint /absolute/official/Cosmos-Reason1-7B`. Do not substitute a
different task embedding. Both modes verify official assets. If the host needs
the installed timm overlay, keep the current `JEPA_DEPENDENCY_DIR` environment
variable; the launcher runs the inherited runtime import check before execution.

Default output is the fresh path
`/inspire/ssd/project/long-range-embodied-collaborative-intelligence/ky26340/rm71d-jepa-clean30-20261107/vitl/online/$TRAIN_JOB_ID`.
Override it with `--output-root` only to another fresh absolute directory.
The prepared subdirectory contains fresh normalization, text and four-rank VAE
features. The run's `observations/` contains main/wrist RGB for JEPA `[-4,0]`.
`resolve/resolved.yaml` binds their SHA identities to the unchanged model recipe.

`--prepared /absolute/already-verified-clean30/prepared` may reuse a completed
clean30 feature cache; it must match this source and its freshly recomputed
normalization exactly. Reused prepared data is not modified. Observed RGB is
created in the new run. Interrupted feature encoding is not resumed implicitly;
use a fresh attempt or provide a complete verified cache.

The pipeline runs the existing two-update gradient/save probe and independent
four-rank restore before formal training. Local CPU tests do not claim these
remote GPU gates passed. No platform job is submitted by preparing this code.

Adapter and exporter identities include source file hashes. Generate production
caches from the deployed LF Git archive; do not transplant a normalization
identity generated from a CRLF Windows checkout into the Linux run.
