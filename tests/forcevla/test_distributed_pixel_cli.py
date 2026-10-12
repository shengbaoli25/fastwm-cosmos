"""Explicit-action and CPU-only default gates; no model/data/GPU execution."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/forcevla/precompute_pixel_cache_distributed.py"
SPEC = importlib.util.spec_from_file_location("distributed_pixel_cli_test", SCRIPT)
CLI = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CLI)


def arguments(mode="--execute-worker"):
    return [mode, "--plan", "/TEST-only/plan.json", "--pixel-cache", "/TEST-only/pixels",
            "--pixel-audit-receipt", "/TEST-only/audit.json", "--workers-root", "/TEST-only/workers",
            "--vae-weights", "/TEST-only/vae", "--t5-weights", "/TEST-only/t5", "--world-size", "4"]


def worker_arguments():
    return arguments() + ["--rank", "2", "--device", "cuda:0", "--physical-device", "2"]


def test_default_plan_constructs_no_dataset_or_model_and_imports_no_torch(capsys, monkeypatch):
    monkeypatch.setattr(CLI, "execute", lambda *args: pytest.fail("Default must never delegate execution"))
    assert CLI.main([]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "planning_only" and not result["model_loaded"] and not result["dataset_constructed"]
    assert not result["formal_training_started"] and result["default_worker_bounds"]["new_samples"] == 3
    code = "import runpy,sys; sys.argv=['" + str(SCRIPT) + "']; runpy.run_path('" + str(SCRIPT) + "'); assert 'torch' not in sys.modules"
    result = subprocess.run([sys.executable, "-c", code], text=True, capture_output=True)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("flag", ["--execute-w", "--execute", "--exec", "--execute-m", "--prepare"])
def test_execution_and_preparation_abbreviations_are_never_accepted(flag):
    with pytest.raises(SystemExit): CLI.parser().parse_args([flag])


@pytest.mark.parametrize("option", ["--plan", "--pixel-cache", "--pixel-audit-receipt", "--workers-root", "--vae-weights", "--t5-weights", "--world-size"])
def test_every_action_requires_explicit_paths_and_world(option):
    argv = worker_arguments()
    position = argv.index(option)
    del argv[position:position + 2]
    with pytest.raises(ValueError): CLI.action_arguments(CLI.parser().parse_args(argv))


@pytest.mark.parametrize("change", ["wrong_visibility", "distributed_launcher", "missing_rank", "wrong_device", "missing_physical", "rank_outside4", "no_full_time_bound", "negative_bound", "nan_time", "two_bound_modes"])
def test_worker_rank_device_and_bound_gate_precedes_model_loading(monkeypatch, change):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2")
    monkeypatch.setenv("WORLD_SIZE", "1")
    argv = worker_arguments()
    if change == "wrong_visibility": monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1")
    elif change == "distributed_launcher": monkeypatch.setenv("WORLD_SIZE", "4")
    elif change == "missing_rank": position = argv.index("--rank"); del argv[position:position + 2]
    elif change == "missing_physical": position = argv.index("--physical-device"); del argv[position:position + 2]
    elif change == "wrong_device": argv[argv.index("--device") + 1] = "cuda:1"
    elif change == "rank_outside4": argv[argv.index("--rank") + 1] = "4"
    elif change == "no_full_time_bound": argv += ["--complete-assigned-episodes"]
    elif change == "negative_bound": argv += ["--max-new-samples", "0"]
    elif change == "nan_time": argv += ["--max-worker-seconds", "nan"]
    elif change == "two_bound_modes": argv += ["--complete-assigned-episodes", "--max-new-samples", "3", "--max-worker-seconds", "120"]
    with pytest.raises(ValueError): CLI.action_arguments(CLI.parser().parse_args(argv))


def test_full_worker_still_requires_positive_explicit_time_bound(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2")
    monkeypatch.setenv("WORLD_SIZE", "1")
    parsed = CLI.parser().parse_args(worker_arguments() + ["--complete-assigned-episodes", "--max-worker-seconds", "60"])
    assert CLI.action_arguments(parsed) is None and parsed.max_new_samples is None


def test_prepare_declares_four_unique_physical_devices_without_gpu_queries():
    parsed = CLI.parser().parse_args(arguments("--prepare-plan") + ["--physical-devices", "0,1,2,3"])
    bindings = CLI.action_arguments(parsed)
    assert bindings == [{"device": "cuda:0", "physical_device": str(rank), "visible_devices": str(rank)} for rank in range(4)]
    parsed.physical_devices = "0,0,1,2"
    with pytest.raises(ValueError): CLI.action_arguments(parsed)


def test_merge_requires_fresh_output_flag_and_never_selects_worker_mode():
    parsed = CLI.parser().parse_args(arguments("--execute-merge"))
    with pytest.raises(ValueError, match="output"): CLI.action_arguments(parsed)
    parsed.output = "/TEST-only/merged"
    assert CLI.action_arguments(parsed) is None and not parsed.execute_worker


def test_partial_worker_result_propagates_exit2_and_merge_result_exit0(capsys, monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2")
    monkeypatch.setenv("WORLD_SIZE", "1")
    monkeypatch.setattr(CLI, "execute", lambda *args: {"status": "partial", "formal_training_started": False})
    assert CLI.main(worker_arguments()) == 2
    monkeypatch.setattr(CLI, "execute", lambda *args: {"status": "complete", "formal_training_started": False})
    assert CLI.main(arguments("--execute-merge") + ["--output", "/TEST-only/merged"]) == 0


def test_identity_inventory_includes_new_worker_pixel_audit_and_original_vae_sources():
    assert len(CLI.SOURCE_FILES) == len(set(CLI.SOURCE_FILES))
    assert {"src/fastwam/cache/distributed.py", "src/fastwam/cache/pixels.py", "src/fastwam/cache/deduplicated_source.py",
            "src/fastwam/cache/pixel_audit.py", "src/fastwam/models/wan22/wan_video_vae.py",
            "scripts/forcevla/precompute_pixel_cache_distributed.py"} <= set(CLI.SOURCE_FILES)


@pytest.mark.parametrize("option", ["--workers-root", "--plan", "--pixel-audit-receipt", "--output", "--merge-receipt"])
def test_mutating_worker_merge_and_receipt_paths_cannot_overlap_pixel_cache(option):
    argv = arguments("--execute-merge") + ["--output", "/TEST-only/merged", "--merge-receipt", "/TEST-only/merge.json"]
    argv[argv.index(option) + 1] = "/TEST-only/pixels/forbidden"
    with pytest.raises(ValueError, match="immutable pixel cache"):
        CLI.action_arguments(CLI.parser().parse_args(argv))


def test_merge_output_cannot_be_an_ancestor_of_pixel_cache():
    argv = arguments("--execute-merge") + ["--output", "/TEST-only"]
    with pytest.raises(ValueError, match="immutable pixel cache"):
        CLI.action_arguments(CLI.parser().parse_args(argv))


def test_individual_rank_output_cannot_contain_pixel_cache():
    argv = arguments("--prepare-plan") + ["--physical-devices", "0,1,2,3"]
    argv[argv.index("--pixel-cache") + 1] = "/TEST-only/workers/rank_000/pixels"
    with pytest.raises(ValueError, match="immutable pixel cache"):
        CLI.action_arguments(CLI.parser().parse_args(argv))


@pytest.mark.parametrize("bad_key", ["numeric_reference_sha256", "dataset_stats_sha256", "conversion_manifest_sha256", "receipt_drift"])
def test_fresh_raw_audit_binding_prevents_asset_or_model_load(monkeypatch, bad_key):
    import fastwam.cache.deduplicated_source as source_module
    import fastwam.cache.pixel_audit as audit_module
    import fastwam.cache.latent as latent
    provenance = {key: str(index) * 64 for index, key in enumerate(
        ("numeric_reference_sha256", "dataset_stats_sha256", "conversion_manifest_sha256"), start=1)}
    source = SimpleNamespace(pixel_manifest={"identity_sha256": "a" * 64}, source_provenance=provenance)
    raw = object()
    stock = SimpleNamespace(load_config=lambda args: (None, raw),
                            _task_texts=lambda args: pytest.fail("Audit rejection must precede T5/asset/model work"))
    monkeypatch.setattr(CLI, "_stock_cli", lambda: stock)
    monkeypatch.setattr(source_module, "build_deduplicated_source", lambda *args: source)
    audit = {"source_provenance": dict(provenance)}
    if bad_key != "receipt_drift": audit["source_provenance"][bad_key] = "f" * 64
    monkeypatch.setattr(audit_module, "validate_pixel_audit_receipt", lambda *args: audit)
    hashes = iter(["b" * 64, "c" * 64] if bad_key == "receipt_drift" else ["b" * 64, "b" * 64])
    monkeypatch.setattr(latent, "file_sha256", lambda path: next(hashes))
    with pytest.raises(latent.CacheIntegrityError, match="receipt changed|provenance differs"):
        CLI.inputs(CLI.parser().parse_args(worker_arguments()))
