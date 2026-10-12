# Cosmos four-GPU platform job

`distributed_fresh.py` runs in the platform **Distributed training** service,
in the foreground on one node with exactly four H200 GPUs. The saved environment
image is `dualwm:stopsave-cosmos-forcewm28-4h200-dcede-20261012230426`.
The platform supplies `TRAIN_JOB_ID`; do not manufacture a notebook allocation ID.
Configure one platform worker, disable automatic retries, and leave the platform
maximum runtime empty for the user's authorized unlimited Cosmos experiment.

The stopped interactive run ended at logged step 2070 and has no formal
checkpoint. This entry starts again from the official robot/action-cond Cosmos
weights and the original random initialization for AE/Force. It reuses the
complete 34808-window Cosmos feature cache. It does not run VAE/text encoding.
The only config change is a new output directory. Four ranks, B16, accumulation1,
global64, 30000 steps, save5000, generation16, and the original model math remain.

Supply an observed platform STOPPED screenshot and JSON receipt. The receipt
binds the prior allocation/output and original config SHA256. The entry rejects
an existing formal checkpoint, an existing new run directory, wrong GPUs,
multi-node jobs, and notebook identity. Keep old source/output/cache read-only.

Submit with the original `.venv/bin/python` invocation:

```bash
/inspire/hdd/project/long-range-embodied-collaborative-intelligence/ky26340/realman-fastwam/.venv/bin/python -u /PATH/TO/THIS/SOURCE/scripts/cosmos/distributed_fresh.py --config /inspire/ssd/project/long-range-embodied-collaborative-intelligence/ky26340/cosmos-dd805094-4rank-30k-v2/config.yaml --stop-evidence /PATH/TO/stop.json --run-root /inspire/ssd/project/long-range-embodied-collaborative-intelligence/ky26340/cosmos-distributed-fresh-4h200-v1 --native-python /inspire/hdd/project/long-range-embodied-collaborative-intelligence/ky26340/realman-fastwam/.venv/bin/python --execute
```

The first two phases each have a 3600-second diagnostic timeout: real four-rank
two-Adam updates with full-state saving, then a fresh four-rank reload with
model/optimizer/scheduler/trainer/RNG fingerprint comparison. Only unforced
successful process-tree exits permit the next phase. Formal training has no
user time cap and is supervised in the foreground. Status and logs persist under
`run-root/dispatch.json`, `preflight-watch`, `restore-watch`, and `formal-watch`.
The controller's final success also requires the final complete 30000-step state
and all four completed origins.

The older `scripts/resume` is pinned to the old interactive experiment. Its
frozen-code check intentionally rejects this fresh-job identity change; the
previous deployed `source-a6e5c73` remains intact. Do not use that old continuation
entry for this new run without binding its new job lineage and checkpoints.

CPU dispatch tests validate launch sequencing, failure gates, identity, evidence,
and the default inert plan mode. They do not establish GPU training success.
