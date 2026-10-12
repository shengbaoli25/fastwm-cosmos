# Original FastWAM baseline on the complete wipe dataset

This independent checkout starts at official FastWAM `7faa71108368fbb3b6885649f112af607427a2d4`. Its origin is the user-provided `https://github.com/shengbaoli25/fastwam_codebase`; that repository currently advertises no refs. Nothing has been pushed. The active Wan/Cosmos comparison checkout is untouched.

The model is the original `FastWAM` class: Wan video30, ActionDiT30, and the original MoT. The official video/action classes, MoT, flow-matching schedulers, losses, inference, dataset reader, processor and transforms are unchanged. This baseline has no Force WM, AE14, future-memory pass, candidate reference trajectory or teacher-future inputs to action inference.

## Complete trained official weights

Use the published **FastWAM Base**, not the Optional IDM variant: [official model repository](https://huggingface.co/yuanty/fastwam), [pinned model README](https://huggingface.co/yuanty/fastwam/blob/8eaceeb24c3cc92ff2a9c9a9d266a4941b836705/README.md), and [checkpoint](https://huggingface.co/yuanty/fastwam/resolve/8eaceeb24c3cc92ff2a9c9a9d266a4941b836705/libero_uncond_2cam224.pt). Its revision, size, SHA256 and metadata audit are in `scripts/baseline/released_weights.json`. Public metadata confirms step21700, all1649 MoT tensors, action input7 and proprio projection8. The checkpoint is12,041,735,140 bytes; no large weight was downloaded during local code preparation.

The factory uses official `skip_dit_load_from_pretrain=true` and then strictly loads **every** video, action and proprio tensor from this full trained checkpoint. It avoids reading the Wan5B and exported ActionDiT initialization weights again. The official Wan VAE is still needed. Missing/mismatching tensors or a changed checksum cause an error; no silent partial fallback is allowed. The final evaluation similarly loads the complete fine-tuned checkpoint strictly.

The released policy was trained on LIBERO with two cameras. This ForceVLA adaptation uses its actual single camera at224×224, no fabricated second view. The camera/task/control/coordinate/normalization domains differ; this is fine-tuning, not a claim that the released policy works on this robot before adaptation. The source raw action7 and state7 order remain unchanged. To preserve all official proprio weights, stock `ConcatLeftAlign(state_target_dim=8)` appends one zero **after** normalizing the source7 fields; the eighth dimension is explicitly padding. Curve inverse transforms operate on the original action7 only.

Current official code defaults to action shift1, which remains the fine-tuning/inference setting. The README notes that the original base release used shift5; this transfer difference is disclosed. The original LIBERO statistics are not used to normalize ForceVLA: the same frozen all50-episode wipe statistics used by the ongoing comparison are reused.

## Architecture, attention and training

Original read mask (`True` means allowed): video cannot read action; video first-frame tokens cannot read future video; action can read all action tokens and only the first video frame. A real three-layer MoT CPU perturbation test confirms that changing future video cannot change action output through an indirect path, while changing the observed frame can. Inference receives only the first image, cached task text and current proprio. `action=None`; future GT is used only to calculate metrics.

Data: all50 episodes /34,808 frames at30Hz, no split; source absolute pose+gripper actions, one224 camera, official action32/video9/ratio4, global stride1, z-score and the official `[-5,5]` normalization clamp. Official RoboTwin supports this normalization. Gaussian flow matching, official AdamW(lr1e-4, betas0.9/0.95, weight decay0.01), official5% warmup+cosine, BF16 and ZeRO1 remain. Microbatch1 × accumulation16 keeps the released effective batch16 on one GPU. As in the original DS JSON, effective gradient clipping is0 even though the Hydra `max_grad_norm` field is1; no clipping was added.

The only modification to an existing official source file is the3-line optional optimizer-group hook in `trainer.py`. The factory attaches dtype-preserving, stable groups with at most1,000,000,000 elements to prevent the previously reproduced Torch2.7.1 oversized-flat foreach integer overflow. Every original trainable parameter is covered once; no parameter/dtype/hyperparameter is changed. CPU8-update checks across warmup show bitwise identical parameters and Adam states to the official ungrouped optimizer. The seed is set before the original factory initializes task heads; full release weights subsequently replace them.

Exact trainable parameters:6,020,710,599 (video4,999,787,712 + action1,020,886,023 + proprio36,864). BF16 weights/gradients and FP32 master/m/v alone require approximately89.716GiB, before activations, VAE and workspaces. Use one H200; full-model two-update preflight remains necessary. No GPU training has run during this local preparation.

Default planned target2176 updates is the official `ceil(34808/16)` count for one anchor pass. The last8-microbatch boundary and actual data exposure must be checked in the real run. Reader retries can substitute a training sample, so a numerical pass count is not proof that every distinct anchor was successfully read. A full pass is also not proof of overfitting. If measured throughput cannot fit this target plus save/eval in five hours, freeze a smaller target **before** formal training and rerun its matching schedule probe.512 updates represents8192 nominal draws, only23.53% of an anchor pass.

## Prepare before allocating GPU

Reuse the existing platform venv/data/T5 cache/VAE through absolute paths; never edit the active experiment's source or overwrite its data/checkpoints. Download the pinned12GB full FastWAM release into a separate assets directory with size/SHA256 verification. Then run this CPU-only checker; it memory-maps the checkpoint to verify every model tensor key/shape and hashes all converted data, shared stats, VAE and text cache:

```bash
PATH="$SHARED_REPO/.venv/bin:$PATH" PYTHONPATH=src \
 python scripts/baseline/validate_assets_cpu.py \
 --shared-repo "$SHARED_REPO" --pretrained-checkpoint "$FULL_FASTWAM" \
 --output runs/preparation/baseline_assets_cpu.json
```

The two-update GPU probe must use the **same planned target** as formal training, so its LR/warmup are real. It builds original Trainer/Adam/ZeRO and performs32 microbatches /2 actual engine updates, measures both update intervals, scans every trainable parameter for finite values and saves no weights/state:

```bash
DIFFSYNTH_MODEL_BASE_PATH="$SHARED_REPO/checkpoints" DIFFSYNTH_SKIP_DOWNLOAD=true \
PYTHONPATH=src PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
 accelerate launch --config_file scripts/accelerate_configs/accelerate_zero1_ds.yaml \
 --num_processes 1 scripts/baseline/preflight_gpu.py \
 --output runs/preparation/baseline_2176_two_updates --target-steps 2176 \
 --override "data.train.dataset_dirs=[$SHARED_REPO/data/forcevla_fastwam/wipe]" \
 --override "data.train.pretrained_norm_stats=$SHARED_REPO/data/forcevla_fastwam/shared_stats/dataset_stats.json" \
 --override "data.train.text_embedding_cache_dir=$SHARED_REPO/data/text_embeds_cache/forcevla" \
 --override "model.action_dit_pretrained_path=$SHARED_REPO/checkpoints/ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt" \
 --override "model.pretrained_fastwam_checkpoint=$FULL_FASTWAM"
```

After the probe passes and its measured timing justifies the frozen target, formal training starts fresh seed42. Set the platform automatic five-hour stop as well. The runner preserves the source/config/stats/data-manifest/full-policy identity, invokes official `scripts/train.py`, guards nonfinite logged loss and terminates its entire Accelerate process group on budget/failure. It refuses stale output and mismatching preflight schedules. Default `save_every=0` saves once at the end with the official weights+full optimizer state; `--save-every1024` is available if the plan deliberately reserves extra checkpoint I/O and space.

```bash
PYTHONPATH=src python scripts/baseline/run_fit.py \
 --shared-repo "$SHARED_REPO" --pretrained-checkpoint "$FULL_FASTWAM" \
 --preflight runs/preparation/baseline_2176_two_updates/preflight.json \
 --output runs/fastwam_original_wipe2176 --target-steps 2176 \
 --budget-seconds 18000 --train-seconds 16200 --eval-seconds 1200 \
 --eval-mode sequence
```

The runner's budget starts with formal launch; previously spent container/probe time must be deducted by the caller. The platform stop is the outer cap, and completed checkpoint/evaluation outputs are required before reporting success.

## Five-episode curve protocol

Freeze episodes0,12,24,36,49 before examining predictions. `middle` uses the identical complete middle windows as the ongoing comparison. `sequence` predicts nonoverlapping32-step chunks across these episodes and discloses uncovered boundary frames; every chunk is conditioned on the actual current observation. These are open-loop forecasts with GT observations at replan anchors, not a closed-loop trajectory or generalization/robot success-rate test.

All7 curves show unclipped source GT, inverse-normalized predictions and a physical hold-current-state baseline. Rotation conventions/control latency remain unverified, so the script reports source-axis differences without claiming SO(3) error. Source frame indices and30Hz times are saved.9-frame GT/predicted video strips show the initial conditioning frame separately; PSNR excludes it and is compared with a static-video baseline. Raw predictions, metrics, fixed selection, source/data/config/stats/checkpoint hashes and resumable per-window payload hashes are recorded. Any substituted reader index or incomplete window is rejected.

After downloading only small numeric/image evidence, render curves and video strips:

```bash
PYTHONPATH=src python scripts/baseline/render.py \
 --evaluation /absolute/path/evaluation --output /absolute/path/plots
```

Validation: local CPU tests cover the original attention path, no future inputs to inference, full-model meta count/group limits, bitwise AdamW equivalence, fixed window selection, raw physical metrics, strict checkpoint shape/step/SHA handling, and normalization-before-padding. Full release payload/download and GPU training status must be verified on the platform before launch claims.
