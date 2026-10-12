# New stock-equivalent pixel cache

This branch builds a new FP32 cache of the 34,808 recorded RGB frames. It preserves the original PIL/HF conversion, multiply-255/uint8 cast, FP32 division, per-camera resize, camera order, final resize/crop/normalization, full nine-frame window strides, numeric normalization, padding, prompts and one NumPy draw per getter. The existing source checkout and old latent caches are separate inputs and are never promoted into this cache.

The exhaustive CPU audit independently invokes the actual stock formatter, Base image cast, instantiated processor and final operators in batches of 33. It checks each unique frame's FP32 and BF16 bits and each original anchor's full nine-frame video, strides, numeric/raw labels, masks and stock prompt association. A sliding reference buffer avoids repeated stock HF window reads. Its passed receipt binds the exact manifest, source conversion, stats, numeric inventory, code and runtime. It does not establish hardware VAE parity.

All CLIs default to planning only and reject abbreviated execution switches. These commands use the paths supplied by the platform inventory; run from the newly deployed source checkout. No new packages, downloads, policy, optimizer or training are involved.

```bash
task_parent=/inspire/hdd/project/long-range-embodied-collaborative-intelligence/ky26340
task_root="$task_parent/fastwam-pixel-cached-preparation"
task_source="$task_root/source"
task_python="$task_parent/realman-fastwam/.venv/bin/python"
task_dataset="$task_parent/fastwam-libero-cached-preparation/wipe"
task_stats="$task_parent/realman-fastwam/data/forcevla_fastwam/shared_stats/dataset_stats.json"
task_texts="$task_parent/realman-fastwam/data/text_embeds_cache/forcevla"
task_assets="$task_parent/fastwam-checkpoints/DiffSynth-Studio/Wan-Series-Converted-Safetensors"
task_pixels=/tmp/forcevla-pixel-cache-v2
task_proof="$task_root/pixel-cpu-proof-v2.json"
task_plan="$task_root/window-worker-plan-v2.json"
task_workers="$task_root/window-workers-v2"
cd "$task_source"
export PYTHONPATH=src
```

First measure a bounded CPU pixel pilot. The FP32 full cache occupies 41,916,628,992 tensor bytes, approximately 39.04 GiB. The pixel writer has one exclusive external lock; a stale lock or orphan payload requires explicit recovery.

```bash
CUDA_VISIBLE_DEVICES='' "$task_python" scripts/forcevla/prepare_pixel_cache.py \
  --pixel-cache-root "$task_pixels" --dataset "$task_dataset" --stats "$task_stats" \
  --cpu-threads 2 --shard-size 128 --max-new-frames 256 --execute-pixels
```

Exit 2 means a valid partial cache. Complete it with the same frozen source and shard size; committed unique frames are not processed again. Then run the mandatory exhaustive audit with a fresh external receipt path.

```bash
CUDA_VISIBLE_DEVICES='' "$task_python" scripts/forcevla/prepare_pixel_cache.py \
  --pixel-cache-root "$task_pixels" --dataset "$task_dataset" --stats "$task_stats" \
  --cpu-threads 2 --shard-size 128 --resume --execute-pixels

CUDA_VISIBLE_DEVICES='' "$task_python" scripts/forcevla/prepare_pixel_cache.py \
  --pixel-cache-root "$task_pixels" --dataset "$task_dataset" --stats "$task_stats" \
  --cpu-threads 2 --receipt "$task_proof" --audit-pixels
```

Prepare a new four-rank plan using the verified local VAE/T5 assets and stock text payloads. The full Base checkpoint is not loaded by this offline component-only pipeline.

```bash
CUDA_VISIBLE_DEVICES='' "$task_python" scripts/forcevla/precompute_pixel_cache_distributed.py \
  --config configs/data/forcevla_libero_2cam.yaml --raw-config configs/data/forcevla_libero_2cam.yaml \
  --dataset "$task_dataset" --stats "$task_stats" --text-cache "$task_texts" \
  --pixel-cache "$task_pixels" --pixel-audit-receipt "$task_proof" \
  --vae-weights "$task_assets/Wan2.2_VAE.safetensors" \
  --t5-weights "$task_assets/models_t5_umt5-xxl-enc-bf16.safetensors" \
  --plan "$task_plan" --workers-root "$task_workers" --world-size 4 \
  --physical-devices 0,1,2,3 --vae-backend cudagraphs --prepare-plan
```

Launch a bounded worker independently for each rank. Replace the rank value with 0, 1, 2 and 3 in four separate processes. Each rank has one visible physical device, its own immutable output and complete assigned episodes. Each encoder call receives exactly one independently assembled `[1,3,9,224,448]` window, then the unchanged stock device BF16 cast, autocast and full-window VAE encode. Latent slicing is forbidden.

```bash
task_rank=0
CUDA_VISIBLE_DEVICES="$task_rank" OMP_NUM_THREADS=2 "$task_python" scripts/forcevla/precompute_pixel_cache_distributed.py \
  --config configs/data/forcevla_libero_2cam.yaml --raw-config configs/data/forcevla_libero_2cam.yaml \
  --dataset "$task_dataset" --stats "$task_stats" --text-cache "$task_texts" \
  --pixel-cache "$task_pixels" --pixel-audit-receipt "$task_proof" \
  --vae-weights "$task_assets/Wan2.2_VAE.safetensors" \
  --t5-weights "$task_assets/models_t5_umt5-xxl-enc-bf16.safetensors" \
  --plan "$task_plan" --workers-root "$task_workers" --world-size 4 \
  --rank "$task_rank" --device cuda:0 --physical-device "$task_rank" \
  --vae-backend cudagraphs --shard-size 128 --max-new-samples 3 \
  --max-worker-seconds 180 --execute-worker
```

Exit 2 is a valid bounded partial worker. After reviewing pilot timing and hardware comparisons, resume each rank using `--resume --complete-assigned-episodes --max-worker-seconds <explicit remaining bound>`, removing `--max-new-samples 3`. The time limit is checked between complete windows, so a cold encode may exceed the limit. No worker is allowed to share a mutable manifest.

Only complete workers may merge into a fresh output. Merge rechecks rank, device, plan, source, raw/numeric label, task/text, payload hash and global-row correspondence; copies immutable contiguous shards without re-encoding or repacking; validates staging; publishes the complete standard cache; then emits the external receipt.

```bash
CUDA_VISIBLE_DEVICES='' "$task_python" scripts/forcevla/precompute_pixel_cache_distributed.py \
  --config configs/data/forcevla_libero_2cam.yaml --raw-config configs/data/forcevla_libero_2cam.yaml \
  --dataset "$task_dataset" --stats "$task_stats" --text-cache "$task_texts" \
  --pixel-cache "$task_pixels" --pixel-audit-receipt "$task_proof" \
  --vae-weights "$task_assets/Wan2.2_VAE.safetensors" \
  --t5-weights "$task_assets/models_t5_umt5-xxl-enc-bf16.safetensors" \
  --plan "$task_plan" --workers-root "$task_workers" --world-size 4 \
  --vae-backend cudagraphs --output "$task_root/full-window-latents-v2" \
  --merge-receipt "$task_root/full-window-merge-v2.json" --execute-merge
```
