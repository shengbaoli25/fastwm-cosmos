# RealMan cup joint training

This is an independent Video-isolated ORVFA joint run on `1018_test_cup_force`: 73 episodes,
35611 anchors, 30 Hz, no validation split, 30000 optimizer updates on four H200s.
The requested second architecture is inherited from `7f62575`: every Video
query rejects all Force-owned H/R/F keys and AE states. Force reads Video and
Force; AE reads both WMs and its own state. These branch rules intersect the
existing O/R/V/F/A mask. Native clean Action32 controls and text remain available
to Video. This is forward isolation only: Force/AE losses can still update Video.
No new visual queries are introduced.

## Data contract

The LeRobot v3 source is audited and SHA-bound without modifying source files.
`observation.images.global` maps to left/main, `observation.images.wrist` to
right/wrist. Decode RGB, divide by 255, resize each to 224x224 with torchvision
bilinear antialiasing, concatenate to 224x448 and normalize with mean/std 0.5.
Encode each complete 9-frame window independently using the pinned Cosmos
tokenizer; encode its single observed frame independently as before.

The action/state selection `[8,9,10,11,12,13,7]` gives absolute EEF XYZ (m),
XYZ Euler angles (rad), and gripper opening [0,1]. No joint-angle action,
delta conversion, angular unwrapping or extra filtering is applied. Proprio
is the normalized state7 plus zero; the model uses only its anchor row.
Normalize action, state and force separately using this dataset's global
population mean/std, then the inherited FP32 affine z-score and [-5,5] clip.
The saved statistics and extraction columns are part of cache identity.

Action offsets are [0..31], video [0,4,..32], history [-31..0], future force
[1..32]. Compute masks before clamping to episode boundaries. Padded history
and normalized force targets are zero with explicit validity masks. Future
labels are loss-only. Source force labels are float32 measurements promoted
to float64 for the raw cache field, not original float64 sensor precision.

Force uses `observation.force`: world/work-frame force/torque in N/Nm, identical
to state[14:20]. There is no added tare, gravity compensation or frame rotation.
Source timestamp checks require robot/force to share a callback and action,
gripper and camera arrival timestamps to be no later than the force anchor.
Software camera alignment is retained; it is not hardware synchronization.
The recorded task string is `1018_test_cup_force` and is encoded by Reason1.

## Initialization and execution

Use official Cosmos robot/action-cond weights and fresh AE/Force/adaptation
modules, optimizer and scheduler; no old ForceVLA trained weights or stats.
The existing config keeps B16/rank, global64, BF16/ZeRO1, lr1e-4, AdamW,
warmup1500, cosine30000, save5000, losses VideoFM/ActionFM/ForceMSE = 1/1/1.
WM conditions are clean real actions, so this run remains GT-conditioned
reconstruction rather than an evaluated GPT-reference refinement policy.

`scripts/cosmos/train_realman.py` audits source bytes, creates text features,
runs four independent VAE encoder processes, assembles the fully verified
cache, resolves the unchanged joint recipe and invokes `distributed_chunk.py`.
Two actual distributed Adam updates, full-state save and a fresh four-rank
restore are mandatory before the formal 30000-step run. Each new attempt
requires unused work/run directories; a failed encoder cannot enter training.

For the corrected second-architecture run, `scripts/cosmos/train_realman_cached.py`
reuses only the completed text/video/numeric caches from the first attempt. It
rechecks raw dataset identity, the complete cache lineage, every feature shard,
text pairing, official weights and upstream before dispatch. It requires the
Video-isolated protocol and a new work/run directory. The first attempt's model,
optimizer, scheduler and GPU-gate evidence are never reused. All 30000 formal
updates start fresh after new two-update and independent restore GPU gates.

## Validation

Local source audit: all 73 episodes and 35611 numeric rows passed; all 146
videos decoded, 71222 total camera frames, 30 Hz, 640x480. Two-view input was
rendered and visually checked. CPU regression: 34 tests passed across
`test_joint.py`, `test_cache.py`, and `test_realman.py`. These are not a substitute
for actual allocated GPU preflight evidence.

## Submitted run configuration

The running job uses source commit `bcb89b0` on
`codex/realman-cup-joint-30k`. Its byte-for-byte resolved configuration is saved in
[`configs/experiments/realman_cup_video_isolated_4h200_30k.resolved.yaml`](../../configs/experiments/realman_cup_video_isolated_4h200_30k.resolved.yaml),
including the verified RealMan feature/text cache identities and official weights.
The architecture mask is defined in the committed `joint.py`; it is not a YAML switch.

The [run record](experiments/realman-cup-video-isolated-20261018/run.json),
[original launch command](experiments/realman-cup-video-isolated-20261018/launch-command.txt),
and [cache reuse receipt](experiments/realman-cup-video-isolated-20261018/cache-reuse.json)
bind that snapshot to the actual four-H200 job. The dataset metadata identifies
the robot as `realman_rm75_ctag2f120`; this is the local RM75 dataset with six-axis
force/torque and two cameras. The saved paths describe the existing run. For a new
run, use unused work/run directories through `train_realman_cached.py`, which
revalidates the source and cache and creates the resolved configuration.
