# FastWAM with the LIBERO two-camera input layout

This repository contains the standard FastWAM adaptation for the ForceVLA
whiteboard-wiping data, using the LIBERO main-camera / wrist-camera layout.
The baseline keeps the complete official Video30 and Action30 models, MoT,
attention masks, Gaussian flow-matching objective, and official LIBERO Base
initialization.

Each camera is resized independently to 224 × 224. The main-camera image is
on the left and the wrist-camera image is on the right, forming a
224 × 448 video frame. A causal training window uses nine frames selected as
`0:33:4`. Each complete window is encoded independently with the original VAE;
episode-latent slicing and concatenating independently encoded single frames
are not equivalent to this representation.

## Training recipe

| Setting | Value |
| --- | --- |
| Dataset | 50 wiping episodes, 34,808 frame anchors, no train/validation split |
| GPUs / processes | 8; rank 0 through rank 7 |
| Batch per GPU | 16 |
| Gradient accumulation | 1 |
| Effective global batch | 128 |
| Maximum optimizer updates | 30,000 |
| Checkpoint interval | 5,000 updates |
| Precision / optimizer partitioning | BF16 / original DeepSpeed ZeRO stage 1 |
| Learning rate / warmup | 1e-4 / 1,500 updates |
| Seed | 42 |
| Training input | Completed window-level VAE feature cache |

The dedicated task is
`configs/task/forcevla_libero_2cam_cached_8gpu_30k.yaml`; its launcher is
`scripts/accelerate_configs/accelerate_zero1_ds_8gpu.yaml`.
The original four-process recipe is also retained for reproducibility.

`scripts/forcevla/prepare_cached_training.py --num-processes 8 --check-cache`
verifies completed cache payloads and exports the resolved training plan.
`scripts/forcevla/preflight_cached_gpu.py` performs two disposable optimizer
updates with the real trainer and the matching eight-process launcher.
Its results must be checked for every rank, including finite parameters and
Adam state, two actual updates without skipped steps, GPU memory, throughput,
and storage estimates. Formal training starts fresh from the same complete
official Base checkpoint; the disposable probe is not continued as training.

Preparation alone does not establish that 30,000 updates and the required
saves/evaluation fit a resource budget. The original five-hour branch cap
must be assessed using actual measurements before formal launch. The original
DeepSpeed JSON has no `gradient_clipping` field, so a configured
`max_grad_norm=1` does not establish that the engine clips gradients at 1.

## Relevant code

- `scripts/forcevla/prepare_forcevla.py`: data conversion and main/wrist mapping.
- `src/fastwam/cache/`: audited pixel inputs, independent complete-window
  encoding, cache identities, and feature-only datasets.
- `scripts/forcevla/prepare_pixel_cache.py` and
  `scripts/forcevla/precompute_pixel_cache_distributed.py`: cache preparation.
- `scripts/forcevla/prepare_cached_training.py` and
  `scripts/forcevla/preflight_cached_gpu.py`: resolved configuration and real
  distributed training checks.
- `scripts/baseline/` and `src/fastwam/baseline/`: model initialization,
  evaluation, and analysis.

Evaluation uses the 119 fixed windows from episodes 0, 12, 24, 36, and 49.
Keep raw numeric ground truth, reset Hold independently for each window, and
report main-camera and wrist-camera PSNR separately. These are training-set
conditional-fitting measurements; they do not establish generalization or
closed-loop control. The Video branch does not read Action tokens.

This code tree is based on the reviewed `c9b0dc1e376f7999ee6a891e36eac6f10feb7478`
baseline snapshot. Source availability and CPU tests do not establish an
eight-GPU training result. Model weights, private datasets, and production
checkpoints are stored outside Git. Force30 is maintained separately from
this standard two-camera FastWAM baseline.
