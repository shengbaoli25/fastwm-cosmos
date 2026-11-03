# ForceVLA on RM71D78

This adapter runs the real [official ForceVLA](https://github.com/ft-robotic/ForceVLA)
at commit `79de88fdd0651d957a2b5212cae8995159c5ad65`, linked by the
[authors' project](https://sites.google.com/view/forcevla2025). It retains their
pi0 model, `force_in_proj`, four-expert/top-1 LIMoE, flow-matching objective,
official optimizer, JAX training loop, and Orbax checkpoint/resume machinery.
The published recipe initializes pi0 from `gs://openpi-assets/checkpoints/pi0_base/params`
and initializes the added force/LIMoE and LoRA parameters. This is not a claim
that a full pretrained ForceVLA policy checkpoint was released or downloaded.

The requested run uses the official LoRA method, **30,000 updates, four H200 GPUs,
64 samples per GPU, global batch 256**, a single JAX process and FSDP across four devices,
seed 42, checkpoint every 2,000 updates, retained every 10,000, no EMA.
`training_method=full` instead uses unmodified full Gemma variants and EMA 0.99.
The pinned trainer creates mesh `(device_count / fsdp_devices, fsdp_devices)`.
For this run `fsdp_devices=4` gives mesh `(1,4)`. Its data PartitionSpec uses
**both** `batch` and `fsdp` axes, so the actual batch on each device is 256/4=64.
This is not gradient accumulation. FSDP reduces parameter-state memory; fitting
the requested activation batch still needs measurement. The runner rejects
fewer/different GPUs and does not silently lower batch size after OOM.

## Compatibility changes

The robot channel adaptation and cached-prefix correction are applied to an
in-memory copy of the pinned module, guarded by its normalized-LF SHA256
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

Single-request serving also needs an inference batch adapter. On 2026-11-03 the
real smoke checkpoint reached HTTP inference but failed in the official LIMoE
reshape: its three image slots (including the masked slot), 48 prompt tokens and
one force token produce 817 tokens. At batch one the official grouping returns
817 groups, which cannot be reshaped across four experts. The successful global
training batch of 256 does not have this divisibility problem.

`load_predictor` now wraps only the policy's sampler boundary: it repeats the
already preprocessed observation to internal batch four, calls the unchanged
official JIT sampler once and selects sample zero. All image/force/prompt masks
are repeated together. The external request and result remain one observation
and 32x8 absolute joint8 actions. No averaging, prompt padding changes, new force
tokens, expert changes, training changes or checkpoint conversion are applied.
For this fixed sequence, the official batch-four layout has 3,268 groups of one
token, with minimum expert capacity four; the duplicates do not compete within
a routing group. This is compatibility with the official batched sampler, not
a claim of numerical equivalence to the original nonfunctional batch-one call.
It performs four internal model samples, so latency and allocator memory must
be measured again by the real-checkpoint HTTP gate. No performance estimate is
reported as a measurement, and the formal run must remain blocked until that
gate passes.

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
  -c /path/to/fastwm-cosmos/configs/ablation/forcevla-linux.constraints \
  -e /work/upstream/ForceVLA \
  -e /work/upstream/ForceVLA/flaxformer \
  -e /work/upstream/ForceVLA/packages/openpi-client \
  'aqtp==0.9.0' 'tensorflow-cpu==2.17.1' \
  'lerobot @ git+https://github.com/huggingface/lerobot@0cf864870cf29f4738d3ade893e6fd13fbd7cdb5'
python -m pip check
python -c 'from tensorflow.compiler.xla.service import hlo_pb2; from aqt.jax_legacy.jax import flax_layers; import openpi.models.pi0_force'
python -m pip freeze > /work/envs/forcevla/installed-requirements.txt
```

The upstream `lerobot` gitlink is absent from `.gitmodules`; recursively cloning
ForceVLA alone does not install it. The explicit pinned installation above is
required and is checked by the training entrypoint. Install these packages
in one resolver transaction so the upstream `jax[cuda12]==0.5.3`, `flax==0.10.2`,
`transformers==4.48.1`, `orbax-checkpoint==0.11.13` and `numpy<2` constraints
remain in force when resolving Flaxformer dependencies. The installed freeze is
part of the eventual worker receipt; upstream leaves several transitive ranges
open. RLDS/dlimp is not needed for this LeRobot experiment, but vendored
Flaxformer imports AQT's legacy HLO helpers, which import TensorFlow's
`hlo_pb2`. AQT does not declare TensorFlow as a package dependency. The explicit
compatibility additions are `aqtp==0.9.0`, `tensorflow-cpu==2.17.1`, and
`keras==3.6.0`; these are not entries from the ForceVLA lock file. TensorFlow
2.17.1 requires `ml-dtypes>=0.3.1,<0.5` and NumPy below 2, compatible with the
pinned `ml-dtypes==0.4.1` and `numpy==1.26.4`. The upstream optional RLDS
TensorFlow 2.15 dependency instead requires `ml-dtypes~=0.2` and relies on an
`uv` override, so it cannot be copied into this ordinary pip resolver command.
The CPU TensorFlow package supplies these imports; model training remains JAX.
Sources: [TensorFlow 2.17.1 package metadata](https://pypi.org/pypi/tensorflow-cpu/2.17.1/json),
[AQT 0.9.0 package metadata](https://pypi.org/pypi/aqtp/0.9.0/json).
Do not install this repository's all-model
`pyproject.toml` into this environment; the wrapper adds only its `src` to
`PYTHONPATH`. The commands are source-reviewed and still require execution in
the chosen Linux CUDA worker; they are not an environment-installation receipt.

Alternatively, from this repository run
`bash scripts/ablation/bootstrap_forcevla.sh /work/upstream/ForceVLA /work/envs/forcevla`.
It checks out the exact source, installs the separately pinned LeRobot gitlink,
uses core package versions from the official `uv.lock` plus the documented AQT
import dependencies, checks imports/dependencies,
and records the installed package freeze. It does not train or download model weights.

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
  "num_train_steps": 30000,
  "batch_size": 256,
  "gpu_count": 4,
  "gpu_model": "H200",
  "per_gpu_batch_size": 64,
  "save_interval": 2000,
  "keep_period": 10000,
  "fsdp_devices": 4,
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

For the requested four-H200 job, the dedicated Linux launcher also checks the
exact four-H200/256/64/FSDP4 allocation before importing or initializing the model.
The example config selects 30k updates; a separate explicit smoke config may use
a shorter positive target without relaxing the device or batch contract:

```bash
bash scripts/ablation/launch_forcevla_h200.sh /work/envs/forcevla/bin/python /work/forcevla.json train
# Restart from the same frozen configuration and real optimizer checkpoint:
bash scripts/ablation/launch_forcevla_h200.sh /work/envs/forcevla/bin/python /work/forcevla.json resume
```

Launch one process on one host; do not use `torchrun`. The scheduler must expose
exactly four H200s through `CUDA_VISIBLE_DEVICES`. A launch receipt records actual
JAX device kinds, batch size, mesh and data identity before entering the official
training loop. It is not evidence that an optimizer update completed.

Before the formal run, use a **separate smoke run directory and configuration**
with the same 4 H200s/global256/per-GPU64/FSDP4, but `num_train_steps=3`,
`save_interval=1`, `keep_period=1`, and `log_interval=1`. Prepare its own statistics
and hashes; start from [forcevla.smoke.example.json](../../configs/ablation/forcevla.smoke.example.json).
The optional invocation bound preserves that frozen three-step
configuration and the official learning-rate schedule:

```bash
python scripts/ablation/forcevla.py --config /work/forcevla-smoke.json --operation train --stop-after-steps 2
python scripts/ablation/forcevla.py --config /work/forcevla-smoke.json --operation resume
```

The first invocation finishes normally after two updates and saves official
checkpoint step 1 with optimizer `train_state.step=2`. Resuming the unchanged
three-step config restores that state and performs the third update, producing
checkpoint step 2. The launch receipt records the invocation bound separately
from the configured target. Do not reuse the smoke run's binding/checkpoints for
the formal 30k experiment; keep the formal config and run directory separate.

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

CPU tests validate channel order, gripper preservation, image conventions,
finite values, action shape, source drift rejection, the three source edits,
sampler batch repetition/selection and the pinned expert layout. The layout
test uses NumPy shims and toy experts and does not validate JAX or trained model
outputs. The initial local validation record is retained as historical evidence;
the batch-four fix has its own [CPU validation record](validation-batch4-20261103.json). The actual smoke training
and restore receipts remain separate from the failed HTTP receipt. After this
fix, real-checkpoint HTTP inference, latency and memory still require a new
successful worker receipt before formal training.
