# Joint8 query post-training evaluation: 20k vs 25k

This protocol compares the original 20,000-step checkpoint with the 25,000-step query post-training checkpoint on the same merged50 training windows. It is a training-fit comparison, not a held-out generalization test or real-robot evaluation.

## Fixed evaluation data and sampling

- Source: `/inspire/hdd/project/long-range-embodied-collaborative-intelligence/ky26340/realman-noaction-joint8/data/1020_cup_merged50`; source SHA256 `e4feb413686fdcc44e1bc046e4510250d187ef0bd6ae1dc6c1c32e1aad02d60d`.
- The dataset contains 50 episodes, 28,992 recorded frames, and 27,783 valid training rows. Evaluation takes one anchor every 32 valid rows without crossing capture gaps (974 windows in the earlier protocol).
- Both checkpoints use 16-step joint sampling with identical seeds: Action 42, Force 43, Video 44. Inference receives current visual observation, first-row proprioception, history, and text; target future action/force/video is excluded.
- Action targets are the next 32 absolute joint-angle/gripper values. Force targets are the following 32 world-frame, causally filtered six-axis wrench values. Metrics are reported in physical units: joint MAE/RMSE in degrees, gripper openness, force in N, and torque in Nm, with valid-channel masks, per-horizon errors, and a hold-current baseline.
- Video prediction is scored at recorded future offsets for the deterministically selected median-coverage episode; prior runs select episode 15 and score the global and wrist views separately with PSNR. This is chunked reconstruction against recorded current observations, not free-running rollout.

## Checkpoint-specific configuration

Evaluate each model in a separate process and output directory. The two configurations may differ, so pass each model's own config and checkpoint explicitly:

```bash
python -u scripts/cosmos/joint8_eval/evaluate.py \
  --config "$BASELINE_CONFIG" --dataset "$MERGED50_DATASET" \
  --source-sha e4feb413686fdcc44e1bc046e4510250d187ef0bd6ae1dc6c1c32e1aad02d60d \
  --checkpoint "$CHECKPOINT_20K" --expected-step 20000 --tokenizer "$TOKENIZER" \
  --output "$RESULTS/step020000" --skip-plots

python -u scripts/cosmos/joint8_eval/evaluate.py \
  --config "$POSTTRAIN_CONFIG" --dataset "$MERGED50_DATASET" \
  --source-sha e4feb413686fdcc44e1bc046e4510250d187ef0bd6ae1dc6c1c32e1aad02d60d \
  --checkpoint "$CHECKPOINT_25K" --expected-step 25000 --tokenizer "$TOKENIZER" \
  --output "$RESULTS/step025000" --skip-plots
```

Each process must be assigned exactly one GPU. The evaluator accepts only the known baseline and query-post-training schemas, verifies that schema against the manifest, and for 25k requires a warm-start receipt bound to the verified 20k checkpoint SHA, step, and fresh optimizer/scheduler. The instantiated model still performs strict full-manifest and tensor-shape loading; use the baseline config for 20k and query-post-training config for 25k.

After both complete receipts exist, render action/force curves and videos:

```bash
python scripts/cosmos/joint8_eval/compare.py \
  --runs "$RESULTS/step020000" "$RESULTS/step025000" \
  --output "$RESULTS/comparison"

python scripts/cosmos/joint8_eval/render_attention_video.py \
  --runs "$RESULTS/step020000" "$RESULTS/step025000" \
  --output "$RESULTS/query-attention"
```

The latter renders the existing V/F/A-to-observed-video attention probe for both checkpoints, reconstructs Force probabilities with the actual observation-attention bias, and saves the 25k per-query maps separately with ROI, entropy/effective-support, and pairwise-cosine diagnostics. The 20k baseline has no learned-query module, so the query-map panel is post-training-only; compare its ROI statistics with baseline Force-to-observation attention as descriptive evidence. Probe instrumentation must leave all production sampler outputs bitwise unchanged.

## Contact-region interpretation

`scripts/cosmos/joint8_eval/contact-roi.json` contains GT-only, view-local coarse boxes around visible cup/end-effector regions in episode 15 at frames 0, 96, 192, 288, 384, 480, and 517. These boxes are not contact-state labels. The 14x14 token grid per camera limits spatial precision. Report view mass, ROI mass, within-view fraction, and area-corrected enrichment (1 means approximately uniform), plus per-query map cosine similarity and entropy/effective support once learned query maps are available. ROI enrichment alone cannot establish true contact localization or causal force use; describe it as alignment with a coarse visible region.

The prior 5k/15k results in `joint8-eval-5k15k-20261021.md` are methodological context only. They are not a substitute for reevaluating both 20k and 25k with this paired protocol.
