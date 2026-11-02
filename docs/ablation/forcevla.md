# ForceVLA on RM71D78

This adapter runs the real [official ForceVLA](https://github.com/ft-robotic/ForceVLA)
at commit `79de88fdd0651d957a2b5212cae8995159c5ad65`, linked by the
[authors' project](https://sites.google.com/view/forcevla2025). It retains their
pi0 model, `force_in_proj`, four-expert/top-1 LIMoE, flow-matching objective,
official optimizer, JAX training loop, and Orbax checkpoint/resume machinery.
The published recipe initializes pi0 from `gs://openpi-assets/checkpoints/pi0_base/params`
and initializes the added force/LIMoE and LoRA parameters. This is not a claim
that a full pretrained ForceVLA policy checkpoint was released or downloaded.

The default is the official LoRA method, 50,000 updates, global batch 4,
seed 42, checkpoint every 2,000 updates, retained every 10,000, no EMA.
`training_method=full` instead uses unmodified full Gemma variants and EMA 0.99.
The upstream table estimates >22.5 GB for LoRA and >70 GB for full fine-tuning;
plan a Linux GPU worker with at least 48 GB for LoRA until an actual memory
preflight establishes a smaller budget. FSDP is selected with `fsdp_devices`.
These are planning estimates, not measurements on this dataset.

## Compatibility changes

The official source has two issues relevant to this robot. Changes are applied
to an in-memory copy of the pinned module, guarded by its normalized-LF SHA256
`15ac88d43012fce1fe64b0b59c74f08c704b0aefe58fb3c7dad63f386b697aad`:

1. The original state slice `[:7]` and force slice `[7:13]` encode a seven-value
   pose/gripper robot. The RealMan adapter uses `[:8]` and `[8:14]`, retaining
   all seven joints **and** the gripper. The 32-channel pretrained projections
   retain their original shape. It trains a new joint-space policy; no IK or
   guessed conversion from Cartesian predictions is used.
2. Original `sample_actions` asserts that the cached call returns
   `prefix_out is None`, then concatenates that None with force tokens. The
   adapter uses the already computed `prefix_out_fix` tensor for this operation.
   Training loss and the actual MoE computations are retained.

No source files in the upstream checkout are edited. The adapter refuses an
unreviewed commit, a modified tracked source tree or a different model source.

Inputs are the shared export's `observation.state[8]`, `observation.force[6]`,
two images (`global`, `wrist`) and the task prompt. The force values are already
sensor-calibrated/causally filtered by the common data pipeline; this adapter
does not filter them again. State plus force is padded to 32 channels, normalized
with the training statistics, and sliced into separate proprioception and force
projections. Both cameras are resized with the official 224-square pad transform.
The absent right wrist view is masked. The official delta-action method is used
for seven joints; gripper stays absolute. Inference reverses normalization and
the joint deltas, then returns exactly 32 rows of absolute joint8 actions.

## Separate Linux environment

ForceVLA and current openpi both import as `openpi` but require different JAX/
Transformers versions. Install them in separate environments and processes.
Use Python 3.11 and an appropriate CUDA driver. Prepare the pinned source and
the LeRobot commit recorded by the official repository:

```bash
git clone https://github.com/ft-robotic/ForceVLA.git /work/upstream/ForceVLA
git -C /work/upstream/ForceVLA checkout 79de88fdd0651d957a2b5212cae8995159c5ad65
python3.11 -m venv /work/envs/forcevla
source /work/envs/forcevla/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install \
  -e /work/upstream/ForceVLA \
  -e /work/upstream/ForceVLA/flaxformer \
  -e /work/upstream/ForceVLA/packages/openpi-client \
  'lerobot @ git+https://github.com/huggingface/lerobot@0cf864870cf29f4738d3ade893e6fd13fbd7cdb5'
python -m pip check
python -m pip freeze > /work/envs/forcevla/installed-requirements.txt
```

The upstream `lerobot` gitlink is absent from `.gitmodules`; recursively cloning
ForceVLA alone does not install it. The explicit pinned installation above is
required and is checked by the training entrypoint. Install the four packages
in one resolver transaction so the upstream `jax[cuda12]==0.5.3`, `flax==0.10.2`,
`transformers==4.48.1`, `orbax-checkpoint==0.11.13` and `numpy<2` constraints
remain in force when resolving Flaxformer dependencies. The installed freeze is
part of the eventual worker receipt; upstream leaves several transitive ranges
open. RLDS/dlimp/TensorFlow is not
needed for this LeRobot experiment. Do not install this repository's all-model
`pyproject.toml` into this environment; the wrapper adds only its `src` to
`PYTHONPATH`. The commands are source-reviewed and still require execution in
the chosen Linux CUDA worker; they are not an environment-installation receipt.

## Configuration and operations

Create a JSON configuration with absolute paths (relative paths otherwise
resolve from the working directory):

```json
{
  "upstream_root": "/work/upstream/ForceVLA",
  "dataset_root": "/work/exports/rm71d78",
  "dataset_manifest_sha256": "COPY_IDENTITY_SHA256_FROM_EXPORT_MANIFEST",
  "run_root": "/work/runs/forcevla",
  "normalization_path": "/work/runs/forcevla/norm/norm_stats.json",
  "normalization_sha256": "FILL_AFTER_STATS",
  "base_params_path": "/work/weights/pi0_base/params",
  "base_params_sha256": "COPY_PATH_SHA256_FROM_ASSET_AUDIT",
  "training_method": "lora",
  "experiment_name": "rm71d78",
  "num_train_steps": 50000,
  "batch_size": 4,
  "save_interval": 2000,
  "keep_period": 10000,
  "fsdp_devices": 1,
  "seed": 42,
  "num_inference_steps": 10
}
```

`dataset_root` is the **common verified LeRobot v2.1 export**, not the raw v3
source. The script sets `HF_LEROBOT_HOME=dataset_root.parent` before importing
LeRobot, and uses `dataset_root.name` as the local repo ID. All 78 original
demonstrations contribute; continuous valid segments are separate episodes.
Statistics use every exported sample and its official 32-frame action window,
including the loader's end-of-episode padding. Replay is training-fit evaluation.

Run from this repository using the ForceVLA environment:

```bash
python scripts/ablation/forcevla.py --config /work/forcevla.json
python scripts/ablation/forcevla.py --config /work/forcevla.json --operation stats
# Copy the returned normalization_sha256 into forcevla.json.
# Download/audit official base weights separately and bind their path hash.
# Only the following explicit operations start or resume training:
python scripts/ablation/forcevla.py --config /work/forcevla.json --operation train
python scripts/ablation/forcevla.py --config /work/forcevla.json --operation resume
```

The default operation is `plan`: no data loading, weight downloading, GPU
initialization or training. `stats` reads data and computes statistics but does
not train. `norm_stats.binding.json` binds the normalization to the exact dataset.
`forcevla_training_config.json` is written exclusively before training and is
required unchanged on resume; overwrite is disabled. Official saved checkpoints
include both optimizer/train state and standalone inference params plus stats.
The shared `bind_checkpoint_assets` wrapper also writes this frozen training
binding inside each checkpoint's atomic Orbax `assets` save. The original
normalization save executes first; all optimizer/model checkpoint code is unchanged.

## Server

Use the common `scripts/ablation/serve.py` and deployment export instructions.
The model backend is `fastwam.ablation.forcevla.load_predictor`. Runtime config
must include `training_binding_path` pointing to a copy of
`run_root/forcevla_training_config.json`, `checkpoint_path` pointing to the saved
step directory containing `params/` and `assets/`, the matching copied
`normalization_path` plus its SHA256, `dataset_manifest_sha256`, and the pinned
`upstream_root`. The shared exporter/server binds the checkpoint directory hash.
The adapter also compares the checkpoint's own stored stats with the supplied
stats and requires `checkpoint/assets/ablation_training_binding.json` to equal
the external training binding. Two runs with identical normalization therefore
cannot silently exchange checkpoints or switch LoRA/full configurations.
Serving does not need the training dataset or the base pi0 weights.

Tests validate channel order, gripper preservation, image conventions, finite
values, action shape, source drift rejection and the three source edits. They
do not establish real weight loading, GPU forward/backward, optimization,
checkpoint resume, memory use or inference latency. Those receipts remain to be
collected on the actual data/weights; no training was started in this delivery.
