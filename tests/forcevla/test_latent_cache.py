"""CPU TEST-only fixtures: no trained VAE, RGB decoding, GPU or training.

Fake encoder call counts prove per-window writes and feature-only repeated
reads. They do not establish real VAE numerical equivalence on H100/H200.
"""
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
from safetensors.torch import load_file, save_file

from fastwam.cache import (CacheIntegrityError, LatentCacheDataset, cache_identity_sha256,
                          make_cache_identity, validate_cache, write_window_cache)
from fastwam.cache.latent import LATENT_SHAPE, file_sha256
from fastwam.cache.source import _AbortRawRead, _StrictReadProxy


@pytest.fixture(scope="module", autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


def _hash(value):
    return hashlib.sha256(value.encode()).hexdigest()


@pytest.fixture
def fixture():
    texts = {prompt: {"context": torch.full((128, 4096), number / 4, dtype=torch.bfloat16),
                      "context_mask": torch.ones(128, dtype=torch.bool)}
             for number, prompt in enumerate(("TEST-only task A", "TEST-only task B"))}
    texts["TEST-only task A"]["context"][7:] = 0  # Stock padded rows remain zero despite all-true mask.
    identity = make_cache_identity(data_conversion_sha256=_hash("fixture conversion"),
        dataset_stats_sha256=_hash("fixture stats"), vae_weights_sha256=_hash("fixture fakeencoder"),
        text_encoder_weights_sha256=_hash("fixture t5"), preprocessing={"TEST-only": True, "image_offsets": list(range(0, 33, 4))},
        source_code_sha256={"fixture.py": _hash("fixturecode")},
        text_payload_sha256={prompt: _hash(prompt) for prompt in texts}, episode_lengths=[3, 4], test_fixture=True,
        encoding_backend="TEST-only fakeencoder")
    rows = [{"sample_index": index, "episode_index": 0 if index < 3 else 1,
             "frame_index": index if index < 3 else index - 3, "task_index": 0 if index < 3 else 1}
            for index in range(7)]
    calls = {"get": [], "encode": []}

    def sample(index):
        calls["get"].append(index)
        row = rows[index]
        length = identity["episode_lengths"][row["episode_index"]]
        frame = row["frame_index"]
        future = torch.arange(33) + frame
        # Frame-dependent contents distinguish whole-window encoding and anchors.
        video = (torch.arange(9, dtype=torch.float32) + index).view(1, 9, 1, 1).expand(3, 9, 224, 448)
        return {"sample_index": index, "video": video,
                "action": torch.arange(32 * 7, dtype=torch.float32).reshape(32, 7) / 1000000007 + index,
                "proprio": torch.arange(32 * 8, dtype=torch.float32).reshape(32, 8) / 1000000009 + index,
                "image_is_pad": future[::4] >= length,
                "action_is_pad": future[:-1] >= length,
                "proprio_is_pad": future >= length,
                "prompt": "TEST-only task A" if index < 3 else "TEST-only task B",
                "raw_current_state": torch.arange(7, dtype=torch.float64) / 1000000033 + index,
                "raw_current_wrench": torch.arange(6, dtype=torch.float64) / 1000000087 + index}

    def encoder(video):
        assert tuple(video.shape) == (1, 3, 9, 224, 448)
        calls["encode"].append(video[0, 0, :, 0, 0].tolist())
        # Every temporal group uses the complete supplied window. No episode slicing.
        value = video[0, 0, :, 0, 0].sum().item()
        return torch.full((1, *LATENT_SHAPE), value, dtype=torch.bfloat16)

    return identity, rows, sample, encoder, texts, calls


def _write(root, fixture, **kwargs):
    identity, rows, sample, encoder, texts, calls = fixture
    return write_window_cache(root, identity, sample, rows, encoder, texts, shard_size=2, **kwargs)


def test_full_independent_windows_once_and_training_repeated_reads_encode_zero(tmp_path, fixture):
    manifest = _write(tmp_path / "cache", fixture)
    identity, rows, getter, encoder, texts, calls = fixture
    assert manifest["status"] == "complete" and manifest["completed_samples"] == 7
    assert calls["get"] == list(range(7)) and len(calls["encode"]) == 7
    assert calls["encode"] == [[index + frame for frame in range(9)] for index in range(7)]
    assert len(manifest["texts"]) == 2 and len(manifest["shards"]) == 4
    assert len(list((tmp_path / "cache").rglob("*.safetensors"))) == 6
    dataset = LatentCacheDataset(tmp_path / "cache", require_production=False)
    calls["get"].clear()
    calls["encode"].clear()
    for epoch in range(3):
        for index in range(7):
            cached = dataset[index]
            assert "video" not in cached and cached["video_num_frames"] == 9
            assert tuple(cached["video_latents"].shape) == LATENT_SHAPE
            assert cached["video_latents"].dtype == torch.bfloat16
            assert bool((cached["video_latents"] == 9 * index + 36).all())
            assert int(cached["sample_index"]) == index
            assert int(cached["episode_index"]) == rows[index]["episode_index"]
            assert int(cached["frame_index"]) == rows[index]["frame_index"]
            assert cached["proprio_is_pad"].numel() == 33 and cached["proprio"].shape == (32, 8)
            assert cached["action"].dtype == cached["proprio"].dtype == torch.float32
            assert bool(cached["context_mask"].all())
            if index < 3:
                assert bool((cached["context"][7:] == 0).all())
    assert calls == {"get": [], "encode": []}
    assert len(dataset._shards) <= 2
    _write(tmp_path / "cache", fixture, resume=True)
    assert calls == {"get": [], "encode": []}


def test_cache_numeric_values_padding_and_current_state_are_exact_and_immutable(tmp_path, fixture):
    _write(tmp_path / "cache", fixture)
    dataset = LatentCacheDataset(tmp_path / "cache", require_production=False)
    for index in (0, 2, 3, 6):
        expected = fixture[2](index)
        actual = dataset[index]
        for key in ("action", "proprio", "image_is_pad", "action_is_pad", "proprio_is_pad", "raw_current_state", "raw_current_wrench"):
            torch.testing.assert_close(actual[key], expected[key], atol=0, rtol=0)
        torch.testing.assert_close(actual["context"], fixture[4][actual["prompt"]]["context"], atol=0, rtol=0)
        actual["video_latents"].zero_()
        actual["action"].zero_()
        actual["context"].zero_()
        assert bool(dataset[index]["video_latents"].any())
        torch.testing.assert_close(dataset[index]["action"], expected["action"], atol=0, rtol=0)
    batch = torch.utils.data.default_collate([dataset[0], dataset[6]])
    assert batch["video_latents"].shape == (2, *LATENT_SHAPE)
    assert batch["video_num_frames"].tolist() == [9, 9]


def test_partial_resume_only_committed_windows_skipped_and_incomplete_refused(tmp_path, fixture):
    identity, rows, getter, encoder, texts, calls = fixture
    def interrupted(index):
        if index == 3:
            raise RuntimeError("TEST-only interrupted encoding")
        return getter(index)
    with pytest.raises(RuntimeError, match="interrupted"):
        write_window_cache(tmp_path / "cache", identity, interrupted, rows, encoder, texts, shard_size=2)
    manifest = validate_cache(tmp_path / "cache", require_complete=False)
    assert manifest["status"] == "partial" and manifest["completed_samples"] == 2
    with pytest.raises(CacheIntegrityError, match="Incomplete"):
        LatentCacheDataset(tmp_path / "cache", require_production=False)
    calls["get"].clear()
    calls["encode"].clear()
    _write(tmp_path / "cache", fixture, resume=True)
    assert calls["get"] == [2, 3, 4, 5, 6]  # Uncommitted index2 is encoded again; committed0/1 are skipped.
    assert len(calls["encode"]) == 5


def test_bounded_preprocess_pilot_commits_all_completed_windows_then_resume(tmp_path, fixture):
    commits = []
    manifest = _write(tmp_path / "cache", fixture, max_new_samples=3,
                      on_shard_commit=lambda manifest, entry: commits.append(manifest["completed_samples"]))
    assert manifest["status"] == "partial" and manifest["completed_samples"] == 3
    assert manifest["stopped_reason"] == "max_new_samples" and manifest["preprocessing_seconds"] >= 0
    assert fixture[5]["get"] == [0, 1, 2]
    assert len(fixture[5]["encode"]) == 3
    assert commits == [2, 3]
    fixture[5]["get"].clear()
    _write(tmp_path / "cache", fixture, resume=True)
    assert fixture[5]["get"] == [3, 4, 5, 6]
    assert validate_cache(tmp_path / "cache")["status"] == "complete"


def test_time_limit_flushes_valid_partial_shard_and_resume_has_no_index_gap(tmp_path, fixture, monkeypatch):
    import fastwam.cache.latent as latent_module
    clock = {"seconds": 0.0}
    monkeypatch.setattr(latent_module.time, "monotonic", lambda: clock["seconds"])
    original = fixture[2]
    def timed_getter(index):
        clock["seconds"] += 1.0
        return original(index)
    manifest = write_window_cache(tmp_path / "cache", fixture[0], timed_getter, fixture[1], fixture[3], fixture[4],
                                  shard_size=5, max_seconds=3.0)
    assert manifest["status"] == "partial" and manifest["completed_samples"] == 3
    assert manifest["stopped_reason"] == "max_seconds" and fixture[5]["get"] == [0, 1, 2]
    assert manifest["shards"][0]["start"] == 0 and manifest["shards"][0]["stop"] == 3
    fixture[5]["get"].clear()
    write_window_cache(tmp_path / "cache", fixture[0], original, fixture[1], fixture[3], fixture[4], shard_size=5, resume=True)
    assert fixture[5]["get"] == [3, 4, 5, 6]


@pytest.mark.parametrize("budget", ["samples", "seconds"])
def test_budget_inside_final_shard_returns_valid_partial_then_resumes(tmp_path, fixture, monkeypatch, budget):
    import fastwam.cache.latent as latent_module
    identity, rows, getter, encoder, texts, calls = fixture
    root = tmp_path / "cache"
    first = write_window_cache(root, identity, getter, rows, encoder, texts, shard_size=5, max_new_samples=5)
    assert first["status"] == "partial" and first["completed_samples"] == 5
    calls["get"].clear()
    clock = {"seconds": 0.0}
    monkeypatch.setattr(latent_module.time, "monotonic", lambda: clock["seconds"])
    def timed_getter(index):
        clock["seconds"] += 1.0
        return getter(index)
    limits = {"max_new_samples": 1} if budget == "samples" else {"max_seconds": 1.0}
    partial = write_window_cache(root, identity, timed_getter, rows, encoder, texts,
                                 shard_size=5, resume=True, **limits)
    assert partial["status"] == "partial" and partial["completed_samples"] == 6
    assert partial["stopped_reason"] == ("max_new_samples" if budget == "samples" else "max_seconds")
    assert calls["get"] == [5]
    assert validate_cache(root, require_complete=False)["completed_samples"] == 6
    with pytest.raises(CacheIntegrityError, match="Incomplete"):
        LatentCacheDataset(root, require_production=False)
    calls["get"].clear()
    complete = write_window_cache(root, identity, getter, rows, encoder, texts, shard_size=5, resume=True)
    assert complete["status"] == "complete" and complete["completed_samples"] == 7
    assert "stopped_reason" not in complete and calls["get"] == [6]


@pytest.mark.parametrize("field", ["dataset_stats_sha256", "data_conversion_sha256", "vae_weights_sha256", "text_encoder_weights_sha256", "preprocessing"])
def test_identity_changed_resume_rejected_without_source_reads(tmp_path, fixture, field):
    _write(tmp_path / "cache", fixture)
    identity = copy.deepcopy(fixture[0])
    identity[field] = {"changed": True} if field == "preprocessing" else _hash("changed")
    fixture[5]["get"].clear()
    with pytest.raises(CacheIntegrityError, match="identity changed"):
        write_window_cache(tmp_path / "cache", identity, fixture[2], fixture[1], fixture[3], fixture[4], resume=True)
    assert fixture[5]["get"] == []


@pytest.mark.parametrize("kind", ["shard", "text"])
def test_payload_tamper_or_missing_is_never_silent_fallback(tmp_path, fixture, kind):
    manifest = _write(tmp_path / "cache", fixture)
    row = manifest["shards" if kind == "shard" else "texts"][0]
    path = tmp_path / "cache" / row["path"]
    original = path.read_bytes()
    path.write_bytes(original[:-1] + bytes([original[-1] ^ 1]))
    with pytest.raises(CacheIntegrityError, match="SHA256"):
        LatentCacheDataset(tmp_path / "cache", require_production=False)
    path.unlink()
    with pytest.raises(CacheIntegrityError, match="missing"):
        validate_cache(tmp_path / "cache")


def test_tensor_shapes_and_source_indices_checked_even_when_manifest_hash_updated(tmp_path, fixture):
    _write(tmp_path / "cache", fixture)
    root = tmp_path / "cache"
    manifest = json.loads((root / "manifest.json").read_text())
    entry = manifest["shards"][0]
    path = root / entry["path"]
    payload = load_file(str(path))
    payload["sample_index"][0] = 5
    save_file(payload, str(path))
    entry["sha256"] = file_sha256(path)
    (root / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(CacheIntegrityError, match="source indices"):
        validate_cache(root)
    payload["sample_index"][0] = 0
    payload["video_latents"] = payload["video_latents"][:, :, :, :, :27].contiguous()
    save_file(payload, str(path))
    entry["sha256"] = file_sha256(path)
    entry["tensor_shapes"]["video_latents"][-1] = 27
    (root / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(CacheIntegrityError, match="shape mismatch"):
        validate_cache(root)


def test_requires_all_indices_no_replacement_and_no_overwrite(tmp_path, fixture):
    identity, rows, getter, encoder, texts, calls = fixture
    with pytest.raises(CacheIntegrityError, match="every original"):
        write_window_cache(tmp_path / "missing", identity, getter, rows[:-1], encoder, texts)
    def replaced(index):
        sample = getter(index)
        sample["sample_index"] = index + 1
        return sample
    with pytest.raises(CacheIntegrityError, match="substituted"):
        write_window_cache(tmp_path / "replacement", identity, replaced, rows, encoder, texts)
    assert calls["encode"] == []
    _write(tmp_path / "cache", fixture)
    with pytest.raises(FileExistsError):
        _write(tmp_path / "cache", fixture)
    with pytest.raises(CacheIntegrityError, match="TEST-only"):
        LatentCacheDataset(tmp_path / "cache")


def test_strict_raw_proxy_prevents_random_retry_replacement():
    class Reader:
        def __getitem__(self, index):
            raise OSError("TEST-only corrupt RGB")
    proxy = _StrictReadProxy(Reader(), 8)
    # Upstream except Exception cannot catch this internal control-flow guard.
    with pytest.raises(_AbortRawRead):
        try:
            proxy[8]
        except Exception:
            pytest.fail("Raw RGB failure reached the upstream random-replacement handler")
    with pytest.raises(_AbortRawRead, match="replacement"):
        proxy[1]


def test_wrong_label_shape_fails_before_encoder_and_padding_tamper_rejected(tmp_path, fixture):
    identity, rows, getter, encoder, texts, calls = fixture
    def wrong_shape(index):
        sample = getter(index)
        sample["action"] = sample["action"][:, :6]
        return sample
    with pytest.raises(CacheIntegrityError, match="numeric tensor"):
        write_window_cache(tmp_path / "bad_shape", identity, wrong_shape, rows, encoder, texts)
    assert calls["encode"] == []
    _write(tmp_path / "cache", fixture)
    root = tmp_path / "cache"
    manifest = json.loads((root / "manifest.json").read_text())
    entry = manifest["shards"][0]
    path = root / entry["path"]
    payload = load_file(str(path))
    payload["image_is_pad"][0, 8] = False
    save_file(payload, str(path))
    entry["sha256"] = file_sha256(path)
    (root / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(CacheIntegrityError, match="episode-end padding"):
        validate_cache(root)


def test_manifest_prompt_identity_and_ambiguous_untracked_resume_rejected(tmp_path, fixture):
    _write(tmp_path / "cache", fixture)
    root = tmp_path / "cache"
    manifest = json.loads((root / "manifest.json").read_text())
    manifest["texts"][0]["prompt"] = "different task"
    (root / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(CacheIntegrityError, match="task prompts"):
        validate_cache(root)
    manifest["texts"][0]["prompt"] = "TEST-only task A"
    (root / "manifest.json").write_text(json.dumps(manifest))
    (root / "shards/interrupted.tmp").write_bytes(b"TEST-only ambiguous partial write")
    with pytest.raises(CacheIntegrityError, match="untracked"):
        _write(root, fixture, resume=True)


def test_cli_dryrun_constructs_no_model_reader_or_gpu(tmp_path):
    root = Path(__file__).resolve().parents[2]
    script = root / "scripts/forcevla/precompute_vae_cache.py"
    # An existing original task works before the new cached task is committed.
    result = subprocess.run([sys.executable, str(script), "--task", "forcevla_libero_2cam_30k",
                             "--cache-root", str(tmp_path / "not-created")],
                            capture_output=True, text=True, cwd=root)
    assert result.returncode == 0, result.stderr
    planned = json.loads(result.stdout)
    assert planned["execute_cache"] is False and planned["training_started"] is False
    assert planned["samples"] == 34808 and planned["window_latent_shape"] == [1, 48, 3, 14, 28]
    assert not (tmp_path / "not-created").exists()


def test_feature_only_import_has_no_raw_reader_or_vae_side_effects():
    result = subprocess.run([sys.executable, "-c",
        "import sys; import fastwam.cache; assert 'fastwam.datasets.lerobot.robot_video_dataset' not in sys.modules; "
        "assert 'fastwam.models.wan22.wan_video_vae' not in sys.modules"], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_strict_source_uses_actual_stock_pipeline_and_end_padding(tmp_path):
    """Tiny marked parquet fixtures run through the real reader and transforms."""
    from omegaconf import OmegaConf
    from fastwam.cache.source import build_index_rows, build_strict_source
    from fastwam.datasets.lerobot.utils.normalizer import save_dataset_stats_to_json

    root = Path(__file__).resolve().parents[2]
    spec = importlib.util.spec_from_file_location("cache_test_conversion_fixture", root / "tests/forcevla/test_conversion.py")
    fixture_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fixture_module)
    source = tmp_path / "TEST-only-source"
    fixture_module._write_source(source)
    converted = tmp_path / "converted"
    fixture_module.converter.convert_dataset(source, converted, test_fixture=True)
    stats_path = tmp_path / "stats.json"
    save_dataset_stats_to_json({group: {"default": {"global_mean": torch.zeros(7), "global_std": torch.ones(7)}}
                               for group in ("action", "state")}, stats_path)
    wrapper = OmegaConf.create({"data": OmegaConf.to_container(OmegaConf.load(root / "configs/data/forcevla_libero_2cam.yaml"), resolve=False)})
    config = wrapper.data.train
    config.dataset_dirs = [str(converted)]
    config.pretrained_norm_stats = str(stats_path)
    rows = build_index_rows(converted)
    getter = build_strict_source(config, rows, tmp_path / "runtime")
    for index in (0, 3, 4, 8):
        sample = getter(index)
        assert sample["sample_index"] == index
        assert tuple(sample["video"].shape) == (3, 9, 224, 448)
        assert sample["action"].dtype == sample["proprio"].dtype == torch.float32
        assert sample["proprio"].shape == (32, 8) and sample["proprio_is_pad"].shape == (33,)
        torch.testing.assert_close(sample["raw_current_state"], rows[index]["raw_current_state"], atol=0, rtol=0)
        torch.testing.assert_close(sample["raw_current_wrench"], rows[index]["raw_current_wrench"], atol=0, rtol=0)
        assert bool((sample["proprio"][:, 7] == 0).all())
        if index in (3, 8):
            assert sample["image_is_pad"].tolist() == [False] + [True] * 8
            assert sample["action_is_pad"].tolist() == [False] + [True] * 31
