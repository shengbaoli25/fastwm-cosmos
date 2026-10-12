# Cosmos + ForceWM: resume through a platform distributed training job

The current interactive run stays running. This entry is for the next start,
after the platform shows the previous allocation/job has terminated. Never
submit a concurrent copy. Do not use an interactive terminal to launch it.

Recipe stays four ranks, B16, accumulation 1, global64, 30000 total steps,
save5000, original BF16/ZeRO1/LR schedule and completed VAE features. An eight-GPU
node still runs four ranks on devices 0–3. No video decode or VAE recaching.

Deploy the repository to a new source directory; do not overwrite the source of
the running job. `frozen_code.json` binds the original model, trainer, sampler,
preflight and execution code. Only this independent resume entry is added.

## Before submission

1. Read the platform's actual STOPPED/FAILED/SUCCEEDED status for `dd805094-b321-4f0a-9793-8c5b585ff7a9`.
   Save a screenshot/page evidence and its SHA256. A stale `running` status file,
   elapsed 18 hours, or absent PIDs on a different node is not termination proof.
2. Create a stop-evidence JSON with `source: "platform_ui"`, actual `status`,
   `resource_id`, `observed_at_utc`, exact `prior_output`, `evidence_file` relative
   to the JSON, and `evidence_sha256`. Never manufacture this ahead of time.
3. Select one node in the platform **distributed training** UI. Four H200 are
   preferred; an eight-H200 allocation may run this four-rank recipe. Keep the
   outer command as one foreground process. Use the original runtime image and
   shared HDD/SSD mounts. Verify the page's actual maximum runtime setting.
   `TRAIN_JOB_ID` must come from the platform, not an exported notebook value.
4. Use the command below with the deployed source and actual stop-evidence path.
   A dry check omits `--execute`; it does not start CUDA training.

```bash
resume_hdd=/inspire/hdd/project/long-range-embodied-collaborative-intelligence/ky26340
resume_ssd=/inspire/ssd/project/long-range-embodied-collaborative-intelligence/ky26340
resume_source=/absolute/path/to/new/deployed/source
resume_stop=/absolute/path/to/observed-stop.json
: "${TRAIN_JOB_ID:?must be supplied by the platform training job}"
exec "$resume_hdd/realman-fastwam/.venv/bin/python" \
  "$resume_source/scripts/resume/distributed.py" \
  --source "$resume_source" \
  --prior-output "$resume_ssd/cosmos-dd805094-4rank-30k-v2" \
  --prior-log "$resume_ssd/cosmos-dd805094-launch-v2/formal-watch/training.log" \
  --stop-evidence "$resume_stop" \
  --output "$resume_ssd/cosmos-resume-${TRAIN_JOB_ID}" \
  --watch "$resume_ssd/cosmos-resume-${TRAIN_JOB_ID}-watch" \
  --native-python "$resume_hdd/realman-fastwam/.venv/bin/python" \
  --preflight "$resume_ssd/cosmos-dd805094-launch-v2/preflight/preflight.json" \
  --execute
```

This Cosmos experiment has no user-imposed total time limit. The controller stays in the foreground with full descendant cleanup; this is not a promise about the platform administrator’s policies.

For another interruption, point `--prior-output` at the last resumed output and
`--prior-log` at its watch/supervision/training.log. Record the actual stopped job
ID as the next evidence resource. Keep the original preflight receipt.
The lineage remains anchored to the same original experiment. Old output stays
read-only and each resumed job uses fresh output/watch directories.

## Restore and completion checks

The newest completed save is selected from its actual log and trainer state,
weights, scheduler, four RNG files, one replicated model and four ZeRO partitions.
Torch archive structure checks do not certify tensor contents. Native fresh-rank
loading must succeed; trainer/engine/Adam steps, parameter/master/moment finiteness,
full scheduler state, Python/NumPy/Torch/CUDA RNG and persistent Force memory RNG
are checked before any optimizer update. Failed checks stop the new job.

The resume-only loader skips already consumed **sharded batches** and preserves
the full original shuffle and end padding, including the next epoch. Cached
features are deterministic; rebuilding the iterator must not consume model RNG.
No change is made to the frozen native trainer or original sampler.

`origins/rank_000.json` through `rank_003.json` record actual `resumed` state.
Only all four `complete` origins, a final 30000 save (including the native final
second save), and unforced whole-process-tree exit produce `completion.json`.
CPU tests are not production GPU restore evidence. The first fresh four-rank
formal-checkpoint restore is still pending until the original allocation stops.
