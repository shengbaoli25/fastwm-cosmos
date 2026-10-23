# RealMan joint-model deployment

This deployment app runs separately from the teleoperation project. The web
server, sensor reader, model inference loop and 30 Hz arm command loop use
separate threads and bounded/latest-only buffers. For GPU/ARM64 migration,
asset staging, `device=cuda:N` checks and physical bring-up, follow the
[Chinese migration guide](../docs/deploy/MIGRATION_zh.md). The read-only
`deploy/doctor.py` never connects hardware; `deploy/benchmark.py` runs synthetic
offline inference only.

## Matched force preprocessing (2026-10-19)

New `realman_force_v4` datasets use `observation.force_filtered`, also stored in
`observation.state[14:20]`, as the model force input. `observation.force` keeps
the calibrated, unfiltered diagnostic channel. `RealManSource` validates the
field correspondence, per-episode calibration and timestamps before training.
Feature cache identity includes the preprocessing contract and its statistics;
the older raw-force cache cannot be reused for this dataset.

`src/realman_force` is byte-identical to the collection repository's
`scripts/realman_force`. Each UDP packet is calibrated using its own TCP pose,
then passed through the same six independent random-walk Kalman filters.
The selected per-step Q/R is **1/65536 = 0.0000152587890625** at 200 Hz:

- Measurement variance R: `[0.25, 0.25, 0.25, 0.0004, 0.0004, 0.0004]`.
- Process variance per second: `[0.000762939453125] * 3 + [0.000001220703125] * 3`.
- Actual positive UDP time interval scales Q. Warmup is at least 0.5 s and
  100 packets. Gaps greater than 30 ms invalidate and reset the filter.
- On a fixed 30 Hz observation grid, select the latest processed packet at or
  before the grid timestamp, no more than 30 ms old. No future interpolation or
  zero-filled gaps. TCP pose, joints and force use the same source packet.
- The UDP callback updates the filter while model inference runs or pauses;
  sampling 30 Hz never reduces the filter's native update frequency.

The strong filter has about a 2.95 s nominal 90% step response. Both collection
and deployment keep this causal delay; timestamps are not shifted to conceal it.
`force_filter.json` contains the selected settings. `force_calibration.json`
is the current RM75 calibration snapshot, not a generic calibration for other
robots or tool assemblies. After recalibration, use matching data and model
artifacts; changed calibration IDs intentionally fail compatibility checks.

Both RealMan training entry points export `<work>/normalization.json` containing
the statistics actually used and the force preprocessing identity. After
training, package the matching checkpoint:

```sh
python scripts/cosmos/export_realman_deployment.py \
  --checkpoint /path/to/trained.pt \
  --normalization /path/to/training-work/normalization.json
```

Copy the generated `trained.pt.force.json` alongside the checkpoint. Point
`normalization_file` in the runtime config at that training export, and configure
`force_filter_file` and `force_calibration_file` in the hardware config. The
checkpoint loader verifies the binding before loading CUDA weights. The runtime
also compares hardware preprocessing, model statistics and per-observation
identity. Existing raw-force checkpoints and `normalization.example.json` are
legacy artifacts and cannot be used with the new filtered hardware pipeline.
No new model has been trained by this code change.

The recorded 72.875 s / 14,563-packet operation was replayed through both paths:
all 2,171 post-warmup 30 Hz rows and their FP32 normalized force inputs matched
exactly. Six 21–29 ms source intervals caused no reset. The audit is included in
`deploy/validation/force_pipeline_20261019.json`; it is an offline replay, not a
physical policy execution.

## No-reference joint diffusion deployment

New checkpoints with schema `cosmos-noaction-vfa-diffusion28-v1` use the same
`infer_joint` core as training, with request-local static O/H/text K/V caches.
The `4416` inference profile selects Video/Force/Action = 4/4/16, the measured
latency/accuracy balance, without retraining or changing checkpoint identity.
It can run `synchronous` or `async_relaxed`; async overlaps execution of an old
action chunk with the next inference and therefore accepts delayed observations.
`161616` remains the standard comparison. These schedules are runtime settings;
they do not rewrite the checkpoint's training-step manifest. The runtime neither
supplies nor updates rolling AE references for these checkpoints. Legacy
checkpoints retain their existing behavior.

Generate the runtime JSON from the resolved training config and exact training
normalization using `scripts/cosmos/export_noaction_deployment.py`; see the
[no-reference guide](../docs/cosmos/noaction-joint-diffusion.md) for arguments.
The loader verifies normalization SHA256, model dimensions, text/data identities,
checkpoint schema and sampling steps. It rejects the old `video_isolated_reuse`
backend for the new model. The old CUDA Graph/compiled serial runner is not used
by the new joint sampler; its full-model latency remains to be measured.

## Legacy rolling model input

The trained `JointCosmosForceWM.infer_forcewm` method takes one required clean
`Action32x7` condition. Both the video/force world-model branches receive the
same condition; the AE output is still sampled from Gaussian noise. At start,
the condition is the measured current TCP pose and gripper opening repeated 32
times. Once a valid AE result is produced, that result becomes the next round's
condition. The reference is shifted by elapsed 30 Hz slots and padded with the
last pose. The condition used here is a deployment approximation: training used
recorded clean actions.

The runtime replaces the action horizon at each valid inference result. A
pause/stop epoch invalidates both in-flight results and queued actions in every
execution mode.

## Execution timing

`execution_mode` selects the deployment schedule without changing model weights,
architecture or the Video 16 + AE 16 sampling schedule:

- `timestamped` (default): anchor actions to the input observation, drop elapsed
  slots and keep any valid old prefix until the new horizon begins. An expired
  result or exhausted action horizon stops the run.
- `async_relaxed`: anchor the returned chunk to its execution start. Replace
  remaining queued actions with the newest result, and issue at most one action
  per 30 Hz control slot without a catch-up burst. Inference continues on the
  newest available observation while the previous chunk executes. When actions
  run out, slow-stop once and wait; the next result re-arms and resumes execution.
  The rolling AE reference uses the new execution anchor. The original input
  timestamp is retained, and the dashboard reports both inference latency and
  input age at return.
- `synchronous`: execute the complete 32-action chunk, finish its last control
  slot, then slow-stop. Only after the stop acknowledgement and a newer sensor
  observation may the next inference begin. Hold throughout inference and start
  the next full chunk after it returns. Sensor acquisition and the web UI remain
  responsive, but model computation never overlaps commanded motion.

The local RTX 3080 configuration now selects `synchronous` to diagnose the
observed retractions. A cycle takes about 1.5 seconds of stationary inference
plus 1.07 seconds of nominal action execution and stop/acquisition overhead.
This removes inference based on observations captured while the previous chunk
is moving. It does not ensure that the model itself predicts monotonic motion.
The local hardware config enables `log_motion_trace`; each successful command
logs its measured TCP pose, requested model pose, rate-limited sent pose, feedback
timestamp and IK difference to distinguish predicted reversal from tracking.

With `async_relaxed`, at about 1.5 seconds
per inference and 32/30 seconds per action chunk, execution and inference overlap,
with roughly 0.4–0.5 seconds of waiting between chunks under ideal scheduling.
The first chunk requires one full inference. Actual waiting also depends on
sensor, SDK and scheduling delays. This mode accepts delayed predictions; it
does not reduce model latency or provide strict real-time alignment. It enables
evaluation of task behavior with delayed asynchronous control. Fresh live sensor
checks, pause, emergency stop and hardware motion checks remain active.

## Local setup

1. Copy the training checkpoint into `~/桌面/checkpoints/<name>.pt`.
2. Put the official Cosmos tokenizer, official action-conditioned checkpoint,
   Reason1 text cache and pinned Cosmos source at the paths in
   `realman_runtime.example.json`, or edit a copied config. The text cache must
   be the complete deployment cache whose identity matches the config.
3. Set `normalization_file` to the matching training run's export and package
   its checkpoint binding as above. The shipped `normalization.example.json`
   describes the historical `realman-1018-test-cup-force-v1` raw-force model;
   it is retained for reproducibility and will not pass the new force contract.
4. Review `realman_hardware.example.json` and match the attached RM75, force
   sensor, CTAG gripper and two camera serials. Defaults reflect the RealMan
   setup used during data collection; they are not a substitute for checking
   the actual robot before enabling motion.
5. Follow the portable [migration guide](../docs/deploy/MIGRATION_zh.md) to select matching CUDA wheels or retain a vendor ARM64 environment, install optional hardware bindings, export a relocatable config and check `cuda:N`. Then start with:

   ```sh
   python deploy/serve_realman.py --config /path/to/runtime.json
   ```

Open `http://127.0.0.1:8765`. Selecting a checkpoint loads the model in a worker
thread. Image-library imports and two synthetic initialization passes complete
before the model becomes ready. These passes do not connect hardware or publish
actions, and the first live inference retains its original sampling seed. Their
timings appear separately as warmup measurements in the dashboard. Connect
hardware explicitly, then start is enabled only after model,
CUDA, robot, two camera streams, force data and gripper feedback pass readiness
checks.

## Video-isolated inference reuse

The example config selects `inference_backend: "video_isolated_reuse"` for the
`orvfa-video-isolated-native-kv-generated-refresh-v3` checkpoint. It retains all
16 Video and 16 AE denoising steps. Within a request it caches text/condition
projections and the clean observed-frame KV, omits Force computation during the
isolated Video rollout, and splits attention into its exact dense visible
blocks. A final clean WM pass still produces Force and the native per-layer AE
prefix. Future Video computation uses `torch.compile` with
`max-autotune-no-cudagraphs`; two explicit CUDA Graphs replay the Video and
prefix/AE stages. Every request copies its current inputs into the graph buffers.
Outputs are cloned so later replays cannot overwrite a previously returned chunk.

Compilation, tuning and graph capture happen during model loading, before the
model becomes ready. A cold compiler cache can take several minutes; an already
compiled local cache took about 24 seconds for the first initialization pass in
this measurement. Compiler workers default to two. Errors are surfaced during
loading rather than silently falling back. To use the original implementation,
set `inference_backend` to `"reference"` and reload the model. To measure the
reuse path without compiler/graph acceleration, set `compile_future` and
`cuda_graph` to false under `inference_optimization`.

This preserves the mathematical attention visibility, weights and schedules,
but BF16 kernel shapes and fusion introduce floating-point differences. Three
recorded RealMan observations gave a maximum decoded TCP position difference of
0.51 mm, an Euler-axis difference of 1.30 degrees, and a gripper-opening difference
of 0.00389 on the [0, 1] scale versus the reference. These are implementation
comparisons, not measured task accuracy. AE compilation was excluded because
its additional speedup was small and its normalized output difference was larger.

## Joint8 no-reference inference optimization

For the independently selected `161616` / `4416` deployment profiles, use
`joint_expert_schedule` to run the same compiled Force/Action layers and complete
sampling-loop CUDA Graph used by the step10000 expert-grid evaluation:

```json
{
  "inference_backend": "joint_expert_schedule",
  "sampling_profile": "4416",
  "execution_mode": "synchronous",
  "inference_optimization": {
    "compile_layers": true,
    "cuda_graph": true,
    "compile_threads": 4
  }
}
```

Remove the legacy `inference_optimization.mode` field when selecting this backend;
it belongs to `joint_multirate` and is rejected here rather than silently ignored.
Existing configurations keep their backend. The settings API preserves the new
backend when switching profiles, unloads the previous policy, and requires a
fresh model load. Compilation and capture finish during the two loader warmups,
before readiness is published; failures are surfaced without an eager fallback.
The dashboard reports `编译＋完整采样图已启用` only after successful execution.

The compiler requires a C compiler and Python development headers matching the
interpreter. If headers are installed locally, expose their include directories
through `CPATH` in the service environment before launching Python. Startup
warmup timings are separate from accepted live-inference latency. See the
[deployment validation and limitations](../docs/deploy/EXPERT_SCHEDULE_zh.md).

For `cosmos-noaction-vfa-diffusion28-joint8-v2`, select
`inference_backend: "joint_optimized"` to enable the dedicated runner:

```json
{
  "inference_backend": "joint_optimized",
  "joint_steps": 16,
  "inference_optimization": {
    "compile_layers": true,
    "cuda_graph": true,
    "compile_threads": 4
  }
}
```

The runner reuses the trained parameters and 16-step schedule. It caches fixed
text projections/KV, timestep modulations and positions, recomputes observation,
history and proprio conditioning on each request, and replaces Action FP64
complex RoPE with FP32 real arithmetic. Weight updates, text updates, geometry
and schedule changes invalidate the relevant caches. Inference tensors without
a version counter are conservatively rebuilt. Requests must be serialized.

Force/Action blocks use full-graph `torch.compile` with BF16 rounding preserved.
Video uses native operators because compiling all Video reductions caused
larger accumulated differences in the local validation. CUDA Graph captures
the complete 16-step rollout, including Video, and copies all changing inputs
before replay. Returned outputs own their buffers. Loading performs compilation
and capture during warmup; their cost is excluded from steady-state timing.

On GB10/PyTorch 2.9.1+cu130, official Video/VAE weights with initialized full-size
Action/Force and synthetic inputs measured 2.399 s median for `joint_static_cache`
and 1.999 s for this backend (10 optimized trials, P95 2.014 s). Three input/seed
cases preserved Video exactly; maximum normalized Action/Force differences
were 0.046875/0.015625. This is implementation/runtime validation; validate the
trained checkpoint on recorded observations before using its outputs for control.
The synchronous robot scheduling policy is unchanged. Use `joint_static_cache`
to return to the original cache-enabled sampler. The legacy
`video_isolated_reuse` backend applies to a different checkpoint protocol.

## Container

Build from the repository root:

```sh
docker build -f deploy/Dockerfile -t fastwam-realman-deploy:local .
```

Example launch, adapting the model asset and checkpoint mounts to local paths:

```sh
docker run --rm --gpus all --network host --device=/dev/bus/usb \
  --mount type=bind,src="$HOME/桌面/checkpoints",dst=/root/桌面/checkpoints \
  --mount type=bind,src=/path/to/cosmos-assets,dst=/models/cosmos,readonly \
  fastwam-realman-deploy:local
```

The hardware button remains opt-in. The app listens on all interfaces in this
container example; use it only on a trusted local network. The default
non-container server listens on loopback.

## Controls and safety behavior

- **Pause / slow stop** invalidates actions and in-flight inference, clears the
  action horizon and asks the arm to stop. In the error state the same button
  reads **Reset error / remain stopped** and returns to a paused state.
- **Emergency stop** invalidates all outputs and latches the session. Restarting
  the app is required after checking/resetting the robot locally.
- Sensor age, camera pair skew, force coordinates, workspace radius, TCP speed,
  IK solution, joint soft limits and joint step are checked before commands.
- `max_joint_step_deg` in the hardware config limits the maximum difference
  between the target IK solution and current measured joint angles. An explicit
  `null` disables only this threshold; omitting the field keeps the 8-degree
  default. Hardware reconnection is required after editing it. The live hardware
  status reports the active value. IK validity, joint soft limits, workspace and
  speed checks, and emergency stop remain active.
- `workspace_radius_m` limits TCP distance from the configured workspace center
  (the connection-time TCP position by default). An explicit `null` disables this
  distance threshold; omission retains the 0.15 m default. Reconnect hardware
  after editing it and check the active value in hardware status. The separate
  orientation-deviation, speed, IK and joint-limit checks still apply.
- In `timestamped` mode, an exhausted horizon stops the run. In `async_relaxed`
  mode, an exhausted queue slow-stops and waits with sensors and inference still
  running; the next valid result resumes execution.
- Completed inference timings are retained even when a stop invalidates the
  result. Sensor errors include the last observation's age and sequence; action
  expiration errors include both inference time and the age of the input.
- Checkpoints are downloaded atomically into the configured checkpoints
  directory and catalog downloads require a pinned SHA256.

## Current validation boundary

CPU/mock tests cover the current-pose bootstrap, rolling AE reference, action
queue timing, strict-mode stale inference rejection, relaxed-mode delayed result
acceptance, wait/resume, concurrent control and 1.5-second inference, sensor expiry
while waiting, pause during re-arm, emergency-stop latch, backend selection and
seed preservation. All 35 deployment tests pass after the scheduling, trace and optional limit changes.
Synchronous tests verify full-chunk completion, post-stop observations, no
motion/inference overlap and cancelled results. The hardware tests use a fake SDK to verify that disabling only the joint-step
threshold preserves IK, soft-limit and emergency-stop checks.
Workspace tests verify that a disabled radius still preserves speed and
orientation checks.
The 36 model parity tests previously passed for masks, condition refresh, RNG
preservation and invalid inputs; this timing change does not modify model code.
CUDA/checkpoint loading and recorded-input GPU
inference were verified locally; Docker and a successful physical closed-loop
task have not been verified. The runtime blocks start when CUDA is unavailable.

On 2026-10-19, the local RM75 setup loaded the actual cup step-5000 model and
completed a live-input inference on an RTX 3080 20GB in 3.230 seconds. The runtime
discarded every action because the 32-step / 30 Hz horizon had already elapsed.
This run did not execute a model action. Model/hardware readiness alone therefore
does not establish the throughput required for continuous closed-loop control.

With inference reuse enabled on the same checkpoint and RTX 3080 20GB, the full
deployment adapter took 1.496–1.504 seconds (per-observation medians over three
runs) versus 3.144–3.155 seconds for the original adapter on the same three
recorded observations. The total includes preprocessing, tokenizer, both WMs and
AE. The measurements exclude camera acquisition and robot communication, and
are not live task runs. Model-only synthetic timing improved from 3.056 to
1.432 seconds. This remains too slow for the `timestamped` 32-step / 30 Hz horizon:
the first result needs an observation age below about 0.967 seconds; uninterrupted
serial refresh needs roughly `2 * latency <= 32 / 30` plus acquisition/scheduling
margin, so about 0.5 seconds per result is the practical target in that mode.
The local `async_relaxed` schedule removes this expiration requirement for model
results and waits between chunks as needed; strict-mode timing is still available
by changing `execution_mode` back to `timestamped` and restarting the service.
