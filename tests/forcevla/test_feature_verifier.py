"""CPU verification with tiny marked stock-reader fixtures; no trained VAE."""
from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf
from safetensors.torch import load_file, save_file

from fastwam.cache import CacheIntegrityError, LatentCacheDataset, make_cache_identity, write_window_cache
from fastwam.cache.latent import LATENT_SHAPE, file_sha256
from fastwam.cache.source import build_index_rows, build_strict_source
from fastwam.datasets.lerobot.utils.normalizer import save_dataset_stats_to_json


ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("feature_verifier_test_module", ROOT / "scripts/forcevla/verify_cached_features.py")
verifier = importlib.util.module_from_spec(spec)
spec.loader.exec_module(verifier)


@pytest.fixture(scope="module", autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


@pytest.fixture
def fixture(tmp_path, request):
    conversion_spec = importlib.util.spec_from_file_location("feature_verifier_conversion_fixture", ROOT / "tests/forcevla/test_conversion.py")
    conversion = importlib.util.module_from_spec(conversion_spec)
    conversion_spec.loader.exec_module(conversion)
    source, converted = tmp_path / "TEST-only-source", tmp_path / "converted"
    conversion._write_source(source)
    conversion.converter.convert_dataset(source, converted, test_fixture=True)
    stats = tmp_path / "stats.json"
    # Nontrivial FP32 normalization and clipping expose a changed formula,
    # pre-normalization cast, temporal shift or copied unnormalized labels.
    save_dataset_stats_to_json({group: {"default": {"global_mean": torch.linspace(-.2, .3, 7),
                                                   "global_std": torch.linspace(.1, .8, 7)}}
                               for group in ("action", "state")}, stats)
    wrapper = OmegaConf.create({"data": OmegaConf.to_container(OmegaConf.load(ROOT / "configs/data/forcevla_libero_2cam.yaml"), resolve=False)})
    raw = wrapper.data.train
    raw.dataset_dirs = [str(converted)]
    raw.pretrained_norm_stats = str(stats)
    raw.text_embedding_cache_dir = str(tmp_path / "texts")
    Path(raw.text_embedding_cache_dir).mkdir()
    task = json.loads((converted / "meta/tasks.jsonl").read_text())["task"]
    from fastwam.datasets.lerobot.robot_video_dataset import DEFAULT_PROMPT
    prompt = DEFAULT_PROMPT.format(task=task)
    source_context = torch.arange(128, dtype=torch.bfloat16)[:, None].expand(128, 4096).clone()
    source_mask = torch.arange(128) < 9
    text_path = Path(raw.text_embedding_cache_dir) / (hashlib.sha256(prompt.encode()).hexdigest() + ".t5_len128.wan22ti2v5b.pt")
    torch.save({"context": source_context, "mask": source_mask}, text_path)
    helper = verifier._precompute_helpers()
    texts, hashes = helper._task_texts(raw)
    task_prompts = verifier.task_prompts_from_source(converted)
    if getattr(request, "param", None) == "two_texts":
        alternate = DEFAULT_PROMPT.format(task="TEST-only alternate task.")
        texts[alternate] = {"context": texts[prompt]["context"].clone() + 1,
                            "context_mask": texts[prompt]["context_mask"].clone()}
        hashes[alternate] = "d" * 64
    rows = build_index_rows(converted)
    getter = build_strict_source(raw, rows, tmp_path / "runtime")
    reference, lengths = verifier.build_numeric_reference(raw)
    identity = make_cache_identity(data_conversion_sha256=file_sha256(converted / "conversion_manifest.json"),
        dataset_stats_sha256=file_sha256(stats), vae_weights_sha256="a" * 64,
        text_encoder_weights_sha256="b" * 64, preprocessing={"TEST-only": True},
        source_code_sha256={"TEST-only fixture": "c" * 64}, text_payload_sha256=hashes,
        episode_lengths=lengths, test_fixture=True, encoding_backend="TEST-only fake encoder")
    calls = []
    def encoder(video):
        calls.append(tuple(video.shape))
        return torch.full((1, *LATENT_SHAPE), video.mean().item(), dtype=torch.bfloat16)
    cache_root = tmp_path / "cache"
    write_window_cache(cache_root, identity, getter, rows, encoder, texts, shard_size=3)
    cache = LatentCacheDataset(cache_root, require_production=False)
    return {"cache": cache, "raw": raw, "reference": reference, "texts": texts,
            "hashes": hashes, "getter": getter, "encoder": encoder, "calls": calls,
            "task_prompts": task_prompts}


def audit(fixture):
    return verifier.verify_cpu_payloads(fixture["cache"], fixture["reference"], fixture["texts"],
                                       fixture["hashes"], fixture["getter"],
                                       task_prompts=fixture["task_prompts"], selected_episodes=(0, 1))


def test_all_labels_padding_t5_and_full_stock_windows_exact(fixture):
    fixture["calls"].clear()
    result = audit(fixture)
    assert result["payload_samples_checked"] == 9
    assert result["all_labels_exact"] and result["all_padding_exact"] and result["all_texts_exact"]
    assert result["all_task_text_associations_exact"]
    assert [row["length"] for row in result["episode_boundaries"]] == [4, 5]
    assert [row["sample_index"] for row in result["strict_stock_windows"]] == [0, 2, 3, 4, 6, 8]
    assert fixture["reference"]["action"].dtype == torch.float32
    assert fixture["reference"]["raw_current_state"].dtype == torch.float64
    assert fixture["calls"] == []  # CPU acceptance never calls even a fixture encoder.


def test_text_zero_rows_and_all_true_mask_checked_against_original_payload(fixture):
    payload = next(iter(fixture["texts"].values()))
    assert payload["context_mask"].all() and not payload["context"][9:].any()
    changed = copy.deepcopy(fixture["texts"])
    next(iter(changed.values()))["context"][10, 0] = 1
    with pytest.raises(CacheIntegrityError, match="T5 context"):
        verifier.verify_cpu_payloads(fixture["cache"], fixture["reference"], changed,
                                    fixture["hashes"], fixture["getter"],
                                    task_prompts=fixture["task_prompts"], selected_episodes=(0, 1))


def test_rejects_numerically_wrong_labels_even_when_payload_hash_is_updated(fixture):
    cache = fixture["cache"]
    entry = cache.manifest["shards"][0]
    path = cache.root / entry["path"]
    payload = load_file(str(path))
    payload["action"][1, 2, 3] += .25
    save_file(payload, str(path))
    entry["sha256"] = file_sha256(path)
    with pytest.raises(CacheIntegrityError, match="action shard"):
        audit(fixture)


def test_rejects_correct_labels_with_wrong_episode_end_padding(fixture):
    cache = fixture["cache"]
    entry = cache.manifest["shards"][-1]
    path = cache.root / entry["path"]
    payload = load_file(str(path))
    payload["proprio_is_pad"][-1, -1] = False
    save_file(payload, str(path))
    entry["sha256"] = file_sha256(path)
    with pytest.raises(CacheIntegrityError, match="proprio_is_pad shard"):
        audit(fixture)


@pytest.mark.parametrize("fixture", ["two_texts"], indirect=True)
def test_rejects_wrong_legal_text_at_an_unselected_anchor(fixture):
    cache = fixture["cache"]
    assert audit(fixture)["all_task_text_associations_exact"]
    entry = cache.manifest["shards"][0]
    path = cache.root / entry["path"]
    payload = load_file(str(path))
    assert payload["sample_index"][1] == 1  # Full getter only samples 0, 2, 3 here.
    payload["text_index"][1] = 1 - payload["text_index"][1]
    save_file(payload, str(path))
    entry["sha256"] = file_sha256(path)
    with pytest.raises(CacheIntegrityError, match="task text association shard"):
        audit(fixture)


def test_rejects_full_getter_substitution_and_task_change(fixture):
    original = fixture["getter"]
    def replaced(index):
        sample = original(index)
        sample["sample_index"] = index + 1
        return sample
    with pytest.raises(CacheIntegrityError, match="substituted"):
        verifier.verify_cpu_payloads(fixture["cache"], fixture["reference"], fixture["texts"],
                                    fixture["hashes"], replaced,
                                    task_prompts=fixture["task_prompts"], selected_episodes=(0, 1))


def test_vae_measurements_report_error_without_batch16_bitwise_gate(fixture):
    result = verifier.vae_measurements(fixture["cache"], fixture["getter"], fixture["encoder"], episodes=(0, 1))
    assert result["all_torch_equal"] and result["sample_count"] == 6
    assert result["batch_size"] == 1 and result["bitwise_batch16_hardware_gate"] is False
    assert result["numeric_tolerance_gate"] is None
    def changed(video):
        return fixture["encoder"](video) + torch.tensor(.5, dtype=torch.bfloat16)
    result = verifier.vae_measurements(fixture["cache"], fixture["getter"], changed, episodes=(0, 1))
    assert not result["all_torch_equal"]
    assert all(row["max_abs_error"] > 0 for row in result["comparisons"])


def test_cpu_runtime_receipt_never_queries_cuda(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("CPU verification must not query a CUDA device")
    monkeypatch.setattr(torch.cuda, "get_device_properties", forbidden)
    monkeypatch.setattr(torch.cuda, "is_available", forbidden)
    result = verifier.runtime_versions()
    assert result["cuda_device_queried"] is False and result["device"] is None
    assert result["packages"]["torch"]


def test_cli_failure_receipt_is_outside_cache_and_never_claims_success(tmp_path):
    receipt = tmp_path / "receipt.json"
    result = subprocess.run([sys.executable, str(ROOT / "scripts/forcevla/verify_cached_features.py"),
                             "--cache-root", str(tmp_path / "missing-cache"), "--output-receipt", str(receipt)],
                            cwd=ROOT, capture_output=True, text=True)
    assert result.returncode == 1, result.stderr
    payload = json.loads(receipt.read_text())
    assert payload["status"] == "failed" and payload["training_started"] is False
    assert payload["optimizer_constructed"] is False and payload["check_vae_requested"] is False
    assert payload["verifier_sha256"] == file_sha256(ROOT / "scripts/forcevla/verify_cached_features.py")
    assert not (tmp_path / "missing-cache").exists()
