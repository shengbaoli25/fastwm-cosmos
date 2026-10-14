# Clean-action direct Force chunk

Parent: 4228c8f. Branch: codex/clean-action-force-chunk.

The user approved a separate distributed experiment with the original four-H200 hyperparameters and three losses weighted 1/1/1. Existing AR experiments and their outputs remain independent. Initialize Cosmos from the complete official robot/action-cond checkpoint and the other branches from their original seeds. No AR checkpoint migration.

Both WMs receive the same normalized clean action32x7 in training and evaluation. Force history32x13 includes the measured current wrench; learned queries predict the whole future32x6. Historical validity, original z-score/clipping, immutable numeric sidecar +1 alignment and complete latent/text caches are unchanged. Future video/force labels are loss-only. No filtering is added.

Only Action Expert uses noisy A_sigma and its clock. Video/Force memory is computed once for the supplied clean condition, then detached for Action FM. A GT-conditioned Action metric must be reported as conditional reconstruction. Future reference-action experiments must explicitly supply clean reference actions and document their provenance; this branch invents no reference pairs.

Inference: `model.infer_world_model(sample, action_condition=normalized_clean_actions)` returns force and video latents. `infer_forcewm` additionally generates conditional Action, default16 steps. The explicit video_seed is independent of the Action sampling seed; intermediate Action states never reach a WM. Old eval callers that omit action_condition fail instead of silently changing semantics.

## Distributed execution

Use original `.venv/bin/python -u scripts/cosmos/distributed_chunk.py --config <prior resolved.yaml> --run-root <new absolute SSD directory> --native-python <original .venv/bin/python> --execute` through the platform distributed job UI. Without --execute it is inert. The driver changes only output_dir in the inherited recipe, refuses output reuse and requires TRAIN_JOB_ID. It preserves 4 ranks, B16, accum1, global64, 30000 steps, save5000, lr1e-4, warmup1500, cosine, BF16/ZeRO1, model dimensions, generation16, all seeds and three unit loss weights.

Actual GPU stages are two Adam updates with all32 query-gradient checks and complete native save, a fresh four-rank full-state restore/fingerprint match, then fresh official-initialized formal training. CPU tests cannot replace these gates. Every stage is foreground and requires normal process-tree closure; failures stop progression. Actual receipts determine completion and throughput, and no old probe is reused.
