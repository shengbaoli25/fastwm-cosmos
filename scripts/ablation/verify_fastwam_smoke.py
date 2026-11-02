"""Validate real four-rank smoke receipts before allowing the separate 30k run."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from fastwam.ablation.contracts import file_sha256, identity_sha256, load_dataset_manifest, write_json
from fastwam.ablation.training_audit import SCHEMA, assert_restored, tensor_fingerprint


def validate_receipt(receipt, rank, start, end, binding_hash):
    if (receipt.get("schema") != SCHEMA or receipt.get("completed") is not True
            or receipt.get("rank") != rank or receipt.get("binding_sha256") != binding_hash
            or (receipt.get("start_step"), receipt.get("end_step")) != (start, end)
            or receipt.get("restored_checkpoint_verified") != bool(start)
            or receipt.get("prepared_batch") != {"num_processes": 4, "per_gpu_batch_size": 64,
                                                   "gradient_accumulation_steps": 1, "global_batch_size": 256}
            or not receipt.get("changed_policy_samples")):
        raise ValueError("Missing, incomplete or mismatched real smoke receipt")
    for state, step in ((receipt["initial"], start), (receipt["final"], end)):
        if state["global_step"] != step or state["engine_global_steps"] != step:
            raise ValueError("Trainer/DeepSpeed progress does not match the audited invocation")
        if not state["optimizer"]["partitions"] or {p["step"] for p in state["optimizer"]["partitions"]} != {float(step)}:
            raise ValueError("Actual Adam counters do not match the audited invocation")
    if [item["step"] for item in receipt["steps"]] != list(range(start + 1, end + 1)):
        raise ValueError("Missing actual optimizer-step evidence")
    for item in receipt["steps"]:
        if (not math.isfinite(item["loss"]) or not math.isfinite(item["global_grad_norm"])
                or item["global_grad_norm"] <= 0 or not item["changed_master_partitions"]):
            raise ValueError("Invalid real loss/gradient/parameter-update evidence")
        before, after = item["optimizer_before"]["partitions"], item["optimizer_after"]["partitions"]
        if ({p["step"] for p in before} != {float(item["step"] - 1)}
                or {p["step"] for p in after} != {float(item["step"])}
                or len(before) != len(after)
                or not any(a["master"]["sha256"] != b["master"]["sha256"] for a, b in zip(before, after))):
            raise ValueError("Adam master samples/counters do not demonstrate an actual update")
    if not any(receipt["initial"]["policy_samples"][key]["sha256"] != receipt["final"]["policy_samples"][key]["sha256"]
               for key in receipt["changed_policy_samples"]):
        raise ValueError("Policy parameter samples did not change")
    memory = receipt["cuda_memory"]
    if not (memory["peak_reserved_bytes"] >= memory["peak_allocated_bytes"] > 0
            and memory["peak_allocated_bytes"] >= memory["allocated_bytes"] > 0):
        raise ValueError("Missing actual CUDA memory measurements")


def verify_configs(smoke_path, formal_path):
    from fastwam.ablation.fastwam import execution_profile, make_training_config, semantic_recipe
    smoke, formal = make_training_config(smoke_path), make_training_config(formal_path)
    for cfg, steps in ((smoke, 3), (formal, 30000)):
        execution_profile(cfg)
        if int(cfg.max_steps) != steps or cfg.resume:
            raise ValueError("Expected fresh 3-step smoke and 30000-step formal configurations")
    if int(smoke.save_every) != 1 or Path(smoke.output_dir) == Path(formal.output_dir):
        raise ValueError("Smoke must save each step into a separate run directory")
    left, right = semantic_recipe(smoke), semantic_recipe(formal)
    left["training"].pop("max_steps")
    right["training"].pop("max_steps")
    if left != right:
        raise ValueError("Smoke/formal training methods differ beyond the deliberate total-step schedule")
    for field in ("dataset_root", "normalization_path", "asset_root", "text_embedding_cache_dir", "pretrained_fastwam_checkpoint"):
        if getattr(smoke, field) != getattr(formal, field):
            raise ValueError(f"Smoke/formal assets differ: {field}")
    return smoke, formal


def verify_saved_weights(path, binding, expected_step, expected_samples):
    import torch
    # mmap validates/deserializes the real saved archive without allocating a
    # second full 12GB policy. Audited tensors are read from its actual bytes.
    payload = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    if payload.get("ablation") != binding or payload.get("step") != expected_step:
        raise ValueError("Saved deployable weights have a different binding or step")
    for record in expected_samples.values():
        name = record["name"]
        prefix, key = name.split(".", 1)
        if prefix not in {"mot", "proprio_encoder"}:
            raise ValueError(f"Unexpected audited policy parameter path: {name}")
        actual = {"name": name, **tensor_fingerprint(payload[prefix][key])}
        if actual != record:
            raise ValueError(f"Actual saved weight bytes differ from the runtime audit: {name}")


def verify_run(smoke, phase):
    from fastwam.ablation.fastwam import build_binding, semantic_recipe
    run = Path(smoke.output_dir)
    binding = json.loads((run / "fastwam_binding.json").read_text())
    manifest = load_dataset_manifest(smoke.dataset_root, verify_files=False)
    if (binding["dataset_identity"] != manifest["identity_sha256"]
            or binding["normalization_sha256"] != file_sha256(smoke.normalization_path)):
        raise ValueError("Current data/statistics identity differs from the audited smoke run")
    # Reusing prior evidence must not approve changed LR/seed/model transforms or
    # VAE/T5 bytes. The manifest identity was checked above; avoid reading the
    # entire image dataset again here (the actual train launcher verifies it).
    build_binding(smoke.dataset_root, smoke.normalization_path, smoke.asset_root, smoke.text_embedding_cache_dir,
                  training_recipe=semantic_recipe(smoke), deployment_binding=binding)
    binding_hash = identity_sha256(binding)
    hardware = json.loads((run / "training-hardware.json").read_text())
    if ((hardware["num_processes"], hardware["per_gpu_batch_size"], hardware["global_batch_size"], hardware["gradient_accumulation_steps"]) != (4, 64, 256, 1)
            or len(hardware["device_names"]) != 4 or not all("H200" in name.upper() for name in hardware["device_names"])):
        raise ValueError("Smoke did not run with four H200 GPUs and physical batch 64")
    evidence = []
    reference_final = None
    final_step = 2 if phase == "saved2" else 3
    for rank in range(4):
        first_path = run / "smoke-audit" / f"from_000000-to_000002.rank_{rank:03d}.json"
        first = json.loads(first_path.read_text())
        validate_receipt(first, rank, 0, 2, binding_hash)
        saved_path = run / "checkpoints/state/step_000002" / f"smoke-audit-rank-{rank:03d}.json"
        saved = json.loads(saved_path.read_text())
        if saved["schema"] != SCHEMA or saved["rank"] != rank or saved["binding_sha256"] != binding_hash:
            raise ValueError("Saved checkpoint audit identity differs")
        assert_restored(first["final"], saved["state"])
        latest = first
        evidence.extend([first_path, saved_path])
        if phase == "resumed3":
            resumed_path = run / "smoke-audit" / f"from_000002-to_000003.rank_{rank:03d}.json"
            latest = json.loads(resumed_path.read_text())
            validate_receipt(latest, rank, 2, 3, binding_hash)
            assert_restored(latest["initial"], saved["state"])
            final_path = run / "checkpoints/state/step_000003" / f"smoke-audit-rank-{rank:03d}.json"
            final_saved = json.loads(final_path.read_text())
            if final_saved["binding_sha256"] != binding_hash or final_saved["rank"] != rank or final_saved["schema"] != SCHEMA:
                raise ValueError("Resumed checkpoint audit identity differs")
            assert_restored(latest["final"], final_saved["state"])
            evidence.extend([resumed_path, final_path])
        if reference_final is None:
            reference_final = latest["final"]["policy_samples"]
        elif reference_final != latest["final"]["policy_samples"]:
            raise ValueError("Policy parameters differ across real distributed ranks")
    progress = json.loads((run / f"checkpoints/state/step_{final_step:06d}/trainer_state.json").read_text())
    if any(progress[key] != latest["final"][key] for key in ("global_step", "epoch", "batch_in_epoch")):
        raise ValueError("Actual saved trainer progress differs from runtime evidence")
    weights = run / f"checkpoints/weights/step_{final_step:06d}.pt"
    verify_saved_weights(weights, binding, final_step, reference_final)
    result = {"schema": SCHEMA, "phase": phase, "accepted": True, "binding_sha256": binding_hash,
              "evidence_sha256": {str(path.relative_to(run)): file_sha256(path) for path in evidence},
              "saved_weights_path": str(weights), "saved_weight_samples_verified": True,
              "all_four_rank_receipts_verified": True, "real_checkpoint_restore_verified": phase == "resumed3",
              "formal_training_permitted": phase == "resumed3"}
    write_json(run / "smoke-audit" / f"{phase}-verified.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke-config", required=True)
    parser.add_argument("--formal-config", required=True)
    parser.add_argument("--phase", choices=("configs", "saved2", "resumed3"), required=True)
    args = parser.parse_args()
    smoke, _ = verify_configs(args.smoke_config, args.formal_config)
    print(json.dumps({"configs_verified": True} if args.phase == "configs" else verify_run(smoke, args.phase), indent=2))


if __name__ == "__main__":
    main()
