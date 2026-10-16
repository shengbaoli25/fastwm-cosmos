# AE trajectory denoising

Base: clean-action-force-chunk, commit 1f1756e. This branch adds a fresh four-GPU training comparison and an explicit trajectory
denoising sampler. Three loss coefficients, model parameters, original
normalization/masks, Video16 and Force32-query prediction remain unchanged.
Training already samples GT/noise interpolations for Action FM.

All 32 steps and seven normalized action dimensions receive independent noise.
The default starting state is `x = 0.9 * reference + 0.1 * epsilon`, with epsilon
sampled from the original seeded standard Gaussian. The strength is a mixture
coefficient applied to every value, not a fraction of points selected for noise.
Use the original training normalizer before this operation. Do not interpret
0.1 as metres, radians, or an additive physical-unit standard deviation.

The Action clock starts at sigma*1000 and decreases to zero in 16 evaluations.
Using a weakly perturbed trajectory with clock1000 would give the model a state
inconsistent with its training. Sigma1 reproduces the original Gaussian sampler;
sigma0 returns the reference unchanged and is only a diagnostic control.

```python
result = model.infer_trajectory_denoising(
    sample,
    action_condition=clean_normalized_gt_action,  # both WMs, unchanged
    action_reference=clean_normalized_gt_action,  # AE starting reference only
    noise_strength=0.1, num_inference_steps=16, seed=0, video_seed=44,
)
```

Both action inputs must be supplied explicitly. Future Force/RGB labels never
enter generation. AE states never condition the WMs. Padding/metric masks stay
the original ones; noisy padding tokens are masked as in Gaussian inference.
`configs/inference/cosmos_ae_noisy_trajectory.yaml` is an inference config, not a
training override. Callers using `infer_forcewm` with no sampling options retain
the original Gaussian behavior for comparisons.

Compare fixed seeds/windows/checkpoints at sigma0.1,0.25,0.5 and Gaussian1. Report
the noisy starting trajectory's error as well as the denoised error, against the
same FP64 targets and validity masks. This distinguishes actual correction from
the advantage of beginning near the answer. GT-reference reconstruction cannot
establish autonomous action prediction. Record `ActionSampling.metadata()` and
the explicit reference source/hash with each evaluation. Keep the 129-window
five-episode full timeline protocol and Video/Force outputs unchanged.

The original RealAction step30000 evaluation keeps Gaussian AE initialization.
It runs independently on one RTX4090, without the noisy-reference group.
Future checkpoint comparisons must use an identified run and the same strength,
seeds, conditioning and metric masks; never mix Gaussian and noisy-reference
results into a single checkpoint improvement claim.

## Four-GPU training comparison

`ae_training_noise_max` is optional. Omitted or 1.0 preserves the original training
clock sampling and RNG consumption exactly. The new task config sets 0.1:
each sample draws sigma uniformly in [0, 0.1), then perturbs every normalized
32x7 action entry using `(1-sigma)*GT + sigma*Gaussian`. AE receives this noisy
trajectory and the matching clock1000*sigma. Both WMs continue to receive the
same clean GT actions, never the AE state. The target remains Gaussian-GT, and
the original scheduler loss weighting is retained without renormalization.
The explicit three loss coefficients remain 1/1/1; restricting sampled clocks
changes the sampled scheduler weights, so action training-loss magnitudes are
not directly comparable to the old full-range distribution.

Use task `cosmos_ae_noisy_trajectory_4gpu_30k`: 4H200/4rank/B16/accum1/global64,
30000steps/save5000/BF16/ZeRO1/lr1e-4/warmup1500/cosine/seed42. Start from the
same official release with fresh AE/Force initialization, not from trained
RealAction weights. New manifests bind the AE training distribution, so a full
checkpoint cannot silently resume with a different distribution. Fresh GPU
two-Adam/save/four-rank restore gates are still required before formal training.
Compare matched checkpoints using identical noisy-start strengths/seeds/windows
and record both starting and denoised errors for both models.
