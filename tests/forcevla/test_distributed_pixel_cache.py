"""CPU-only distributed cache integrity checks; fake encoder, no model/GPU."""
from copy import deepcopy
import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file, save_file

import fastwam.cache.distributed as distributed
from fastwam.cache import CacheIntegrityError, LatentCacheDataset, make_cache_identity, validate_cache
from fastwam.cache.distributed import (episode_partition, expected_numeric_window, make_distributed_plan,
                                      merge_worker_caches, validate_distributed_plan, validate_worker_cache,
                                      write_worker_cache)
from fastwam.cache.latent import LATENT_SHAPE, cache_identity_sha256, file_sha256


@pytest.fixture(scope="module", autouse=True)
def threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


@pytest.fixture
def fixture():
    lengths = [3, 4, 5, 6]
    prompts = {0: "TEST-only task A", 1: "TEST-only task B"}
    texts = {prompt: {"context": torch.full((128, 4096), task / 4, dtype=torch.bfloat16),
                      "context_mask": torch.ones(128, dtype=torch.bool)} for task, prompt in prompts.items()}
    identity = make_cache_identity(data_conversion_sha256="a" * 64, dataset_stats_sha256="b" * 64,
        vae_weights_sha256="c" * 64, text_encoder_weights_sha256="d" * 64,
        preprocessing={"TEST-only": True}, source_code_sha256={"fixture.py": "e" * 64},
        text_payload_sha256={prompt: "f" * 64 for prompt in prompts.values()}, episode_lengths=lengths,
        test_fixture=True, encoding_backend="TEST-only independent fake encoder")
    rows = []
    for episode, length in enumerate(lengths):
        for frame in range(length):
            rows.append({"sample_index": len(rows), "episode_index": episode, "frame_index": frame, "task_index": episode % 2})
    count = len(rows)
    reference = {key: torch.arange(count * dim, dtype=dtype).reshape(count, dim) / 10000003
                 for key, dim, dtype in (("action", 7, torch.float32), ("proprio", 8, torch.float32),
                                        ("raw_action", 7, torch.float64), ("raw_state", 7, torch.float64),
                                        ("raw_wrench", 6, torch.float64))}
    reference["task_index"] = torch.tensor([row["task_index"] for row in rows], dtype=torch.int64)
    for row in rows:
        row["raw_current_state"] = reference["raw_state"][row["sample_index"]].clone()
        row["raw_current_wrench"] = reference["raw_wrench"][row["sample_index"]].clone()
    plan = make_distributed_plan(identity, "0" * 64, rows, reference, prompts, world_size=2,
        device_bindings=[{"device": "cpu", "physical_device": "TEST-only CPU rank0"},
                         {"device": "cpu", "physical_device": "TEST-only CPU rank1"}],
        source_provenance={"scope": "TEST-only numeric reference"})
    calls = {"get": [], "encode": []}

    def sample(index):
        calls["get"].append(index)
        row = rows[index]
        video = torch.empty((9, 3, 224, 448), dtype=torch.float32)
        for frame in range(9):
            video[frame].fill_(index / 100 + frame / 1000)
        return {"sample_index": index, "video": video.permute(1, 0, 2, 3),
                "prompt": prompts[row["task_index"]], **expected_numeric_window(identity, row, reference)}

    def encoder(video):
        assert tuple(video.shape) == (1, 3, 9, 224, 448) and video.stride() == (301056, 100352, 301056, 448, 1)
        calls["encode"].append(video[0, 0, :, 0, 0].tolist())
        return torch.full((1, *LATENT_SHAPE), video[0, 0, :, 0, 0].sum().item(), dtype=torch.bfloat16)

    return plan, rows, reference, sample, encoder, texts, calls


def write(root, rank, fixture, **kwargs):
    plan, rows, reference, sample, encoder, texts, _ = fixture
    return write_worker_cache(root, plan, rank, sample, rows, encoder, texts, reference, shard_size=2, **kwargs)


def workers(tmp_path, fixture):
    roots = [tmp_path / f"rank_{rank}" for rank in range(2)]
    for rank, root in enumerate(roots):
        write(root, rank, fixture)
    return roots


def save_manifest(root, manifest):
    (root / "manifest.json").write_text(json.dumps(manifest))


def rewrite_shard(root, edit):
    manifest = json.loads((root / "manifest.json").read_text())
    entry = manifest["shards"][0]
    path = root / entry["path"]
    payload = load_file(str(path))
    edit(payload)
    save_file(payload, str(path))
    entry["sha256"] = file_sha256(path)
    entry["tensor_shapes"] = {key: list(value.shape) for key, value in payload.items()}
    save_manifest(root, manifest)


def test_episode_partition_is_disjoint_balanced_and_deterministic(fixture):
    plan = fixture[0]
    assert plan["partition"] == [[0, 3], [1, 2]]
    assert episode_partition(plan["identity"], 2) == plan["partition"]
    assert sorted(episode for episodes in plan["partition"] for episode in episodes) == list(range(4))


def test_four_rank_production_partition_covers_all34808_and_rejects_other_world(fixture):
    identity = deepcopy(fixture[0]["identity"])
    identity.update(scope="production", episode_lengths=[696] * 49 + [704], sample_count=34808)
    partition = episode_partition(identity, 4)
    assert sorted(episode for episodes in partition for episode in episodes) == list(range(50))
    assert sum(sum(identity["episode_lengths"][episode] for episode in episodes) for episodes in partition) == 34808
    with pytest.raises(CacheIntegrityError, match="four"):
        episode_partition(identity, 2)


def test_independent_full_windows_immutable_merge_and_feature_reader(tmp_path, fixture):
    plan, rows, reference, _, _, _, calls = fixture
    roots = workers(tmp_path, fixture)
    assert sorted(calls["get"]) == list(range(18)) and len(calls["encode"]) == 18
    before = {str(path): file_sha256(path) for root in roots for path in root.rglob("*") if path.is_file()}
    receipt = merge_worker_caches(list(reversed(roots)), tmp_path / "merged", plan, rows, reference)
    manifest = validate_cache(tmp_path / "merged", plan["identity_sha256"])
    assert receipt["completed_samples"] == manifest["completed_samples"] == 18
    assert receipt["worker_ranks"] == [0, 1] and not receipt["hardware_equivalence_verified"]
    assert all(file_sha256(path) == digest for path, digest in before.items())
    dataset = LatentCacheDataset(tmp_path / "merged", require_production=False)
    for index in range(18):
        actual = dataset[index]
        for key, expected in expected_numeric_window(plan["identity"], rows[index], reference).items():
            assert torch.equal(actual[key], expected)
        assert actual["prompt"] == plan["task_prompts"][str(rows[index]["task_index"])]
    assert len(calls["encode"]) == 18
    first = manifest["shards"][0]["path"]
    assert (tmp_path / "merged" / first).stat().st_ino != (roots[0] / first).stat().st_ino
    with pytest.raises(CacheIntegrityError, match="fresh"):
        merge_worker_caches(roots, tmp_path / "merged", plan, rows, reference)


def test_partial_bound_resume_does_not_reencode_committed_windows(tmp_path, fixture):
    plan, rows, reference, _, _, _, calls = fixture
    root = tmp_path / "rank0"
    partial = write(root, 0, fixture, max_new_samples=2)
    assert partial["status"] == "partial" and partial["completed_samples"] == 2
    first = root / partial["shards"][0]["path"]
    digest = file_sha256(first)
    complete = write(root, 0, fixture, resume=True)
    assert complete["status"] == "complete" and complete["completed_samples"] == 9
    assert "stopped_reason" not in complete
    assert calls["get"] == [0, 1, 2, 12, 13, 14, 15, 16, 17] and file_sha256(first) == digest
    count = len(calls["encode"])
    write(root, 0, fixture, resume=True)
    assert len(calls["encode"]) == count
    validate_worker_cache(root, plan, rows, reference)


@pytest.mark.parametrize("field", ["action", "proprio", "raw_action", "raw_current_state", "raw_current_wrench", "raw_wrench_future"])
def test_source_numeric_drift_rejected_before_encoder(tmp_path, fixture, field):
    plan, rows, reference, sample, encoder, texts, calls = fixture
    def corrupt(index):
        value = sample(index)
        value[field] = value[field].clone()
        value[field].view(-1)[0] += 1
        return value
    with pytest.raises(CacheIntegrityError, match="numeric bytes"):
        write_worker_cache(tmp_path / "bad", plan, 0, corrupt, rows, encoder, texts, reference)
    assert not calls["encode"]
    assert json.loads((tmp_path / "bad/manifest.json").read_text())["status"] == "error"


def test_signed_zero_and_wrong_task_are_rejected(tmp_path, fixture):
    plan, rows, reference, sample, encoder, texts, _ = fixture
    def signed_zero(index):
        value = sample(index)
        value["action"] = value["action"].clone()
        value["action"][0, 0] = -0.
        return value
    with pytest.raises(CacheIntegrityError, match="numeric bytes"):
        write_worker_cache(tmp_path / "zero", plan, 0, signed_zero, rows, encoder, texts, reference)
    def wrong_task(index):
        value = sample(index); value["prompt"] = "TEST-only task B"; return value
    with pytest.raises(CacheIntegrityError, match="prompt"):
        write_worker_cache(tmp_path / "task", plan, 0, wrong_task, rows, encoder, texts, reference)


@pytest.mark.parametrize("kind", ["nan_video", "contiguous_video", "bf16_video", "nan_latent", "wrong_latent_shape"])
def test_changed_video_or_encoder_contract_rejected(tmp_path, fixture, kind):
    plan, rows, reference, sample, encoder, texts, _ = fixture
    def altered_sample(index):
        value = sample(index)
        if kind == "nan_video": value["video"][0, 0, 0, 0] = float("nan")
        elif kind == "contiguous_video": value["video"] = value["video"].contiguous()
        elif kind == "bf16_video": value["video"] = value["video"].to(torch.bfloat16)
        return value
    def altered_encoder(video):
        if kind == "nan_latent": return torch.full((1, *LATENT_SHAPE), float("nan"), dtype=torch.bfloat16)
        if kind == "wrong_latent_shape": return torch.zeros((1, 48, 1, 14, 28), dtype=torch.bfloat16)
        return encoder(video)
    with pytest.raises(CacheIntegrityError):
        write_worker_cache(tmp_path / "bad", plan, 0, altered_sample, rows, altered_encoder, texts, reference)


@pytest.mark.parametrize("field", ["action", "proprio", "raw_action", "raw_current_state", "raw_current_wrench", "raw_wrench_future", "image_is_pad", "task_index", "text_index", "sample_index"])
def test_rehashed_payload_corruption_is_rejected_by_numeric_task_and_index_oracle(tmp_path, fixture, field):
    plan, rows, reference, *_ = fixture
    roots = workers(tmp_path, fixture)
    def edit(payload):
        if payload[field].dtype == torch.bool: payload[field].view(-1)[0] = ~payload[field].view(-1)[0]
        else: payload[field].view(-1)[0] += 1
    rewrite_shard(roots[0], edit)
    with pytest.raises(CacheIntegrityError):
        merge_worker_caches(roots, tmp_path / "merged", plan, rows, reference)
    assert not (tmp_path / "merged").exists()


@pytest.mark.parametrize("kind", ["missing_rank", "duplicate_root", "duplicate_rank", "partial_rank", "error_rank", "wrong_device", "untracked_file", "wrong_pixel_plan", "wrong_worker_identity"])
def test_merge_rejects_invalid_worker_sets_before_publication(tmp_path, fixture, kind):
    plan, rows, reference, *_ = fixture
    roots = workers(tmp_path, fixture)
    if kind == "missing_rank": roots = roots[:1]
    elif kind == "duplicate_root": roots = [roots[0], roots[0]]
    elif kind == "untracked_file": (roots[0] / "untracked").write_text("ambiguous")
    else:
        manifest = json.loads((roots[1] / "manifest.json").read_text())
        if kind == "duplicate_rank": manifest["rank"] = 0
        elif kind == "partial_rank": manifest["status"] = "partial"
        elif kind == "error_rank": manifest.update(status="error", error="worker failure")
        elif kind == "wrong_device": manifest["device_binding"] = {"device": "cpu", "physical_device": "wrong"}
        elif kind == "wrong_pixel_plan": manifest["plan"]["pixel_identity_sha256"] = "9" * 64
        elif kind == "wrong_worker_identity": manifest["worker_identity"]["sha256"] = "9" * 64
        save_manifest(roots[1], manifest)
    with pytest.raises(CacheIntegrityError):
        merge_worker_caches(roots, tmp_path / "merged", plan, rows, reference)
    assert not (tmp_path / "merged").exists()


def test_reference_source_and_physical_device_changes_fail_plan_validation(fixture):
    plan, rows, reference, *_ = fixture
    changed = {key: value.clone() for key, value in reference.items()}
    changed["raw_action"][0, 0] += 1
    with pytest.raises(CacheIntegrityError, match="inventory"):
        validate_distributed_plan(plan, rows, changed)
    bad = deepcopy(plan)
    bad["device_bindings"][1] = bad["device_bindings"][0]
    bad["sha256"] = cache_identity_sha256({key: value for key, value in bad.items() if key != "sha256"})
    with pytest.raises(CacheIntegrityError, match="overlap"):
        validate_distributed_plan(bad, rows, reference)


def test_source_failure_preserves_exception_and_error_worker_cannot_resume(tmp_path, fixture):
    plan, rows, reference, _, encoder, texts, _ = fixture
    error = RuntimeError("TEST-only failed source")
    def failed(index): raise error
    with pytest.raises(RuntimeError) as captured:
        write_worker_cache(tmp_path / "worker", plan, 0, failed, rows, encoder, texts, reference)
    assert captured.value is error
    with pytest.raises(CacheIntegrityError, match="failed"):
        write(tmp_path / "worker", 0, fixture, resume=True)


def test_time_bound_commits_valid_short_shard_and_resumes_only_remaining(tmp_path, fixture, monkeypatch):
    plan, rows, reference, sample, encoder, texts, calls = fixture
    clock = [0.]
    monkeypatch.setattr(distributed.time, "monotonic", lambda: clock[0])
    def timed(index):
        value = sample(index)
        clock[0] += 1
        return value
    partial = write_worker_cache(tmp_path / "rank0", plan, 0, timed, rows, encoder, texts, reference,
                                 shard_size=2, max_seconds=2.5)
    assert partial["status"] == "partial" and partial["completed_samples"] == 3
    assert partial["stopped_reason"] == "max_seconds"
    complete = write(tmp_path / "rank0", 0, fixture, resume=True)
    assert complete["status"] == "complete" and len(calls["encode"]) == 9
    assert calls["get"] == [0, 1, 2, 12, 13, 14, 15, 16, 17]


@pytest.mark.parametrize("place", ["worker", "stage", "final"])
def test_merge_receipt_must_be_external_to_every_cache_and_staging(tmp_path, fixture, place):
    plan, rows, reference, *_ = fixture
    roots = workers(tmp_path, fixture)
    output = tmp_path / "merged"
    protected = roots[0] if place == "worker" else output if place == "final" else output.with_name(output.name + f".staging-{distributed.os.getpid()}")
    with pytest.raises(CacheIntegrityError, match="overlaps"):
        merge_worker_caches(roots, output, plan, rows, reference, receipt_path=protected / "receipt.json")
    assert not (protected / "receipt.json").exists() and not output.exists()
    validate_worker_cache(roots[0], plan, rows, reference)


def test_manifest_drift_after_validation_is_not_bound_as_audited(tmp_path, fixture, monkeypatch):
    plan, rows, reference, *_ = fixture
    roots = workers(tmp_path, fixture)
    original = distributed.validate_worker_cache
    changed = [False]
    def drift(root, *args, **kwargs):
        result = original(root, *args, **kwargs)
        if not changed[0]:
            changed[0] = True
            manifest = deepcopy(result); manifest["preprocessing_seconds"] += 1
            save_manifest(Path(root), manifest)
        return result
    monkeypatch.setattr(distributed, "validate_worker_cache", drift)
    with pytest.raises(CacheIntegrityError, match="drift"):
        merge_worker_caches(roots, tmp_path / "merged", plan, rows, reference)
    assert not (tmp_path / "merged").exists()


def test_new_untracked_worker_file_during_copy_prevents_publication(tmp_path, fixture, monkeypatch):
    plan, rows, reference, *_ = fixture
    roots = workers(tmp_path, fixture)
    original = distributed._copy_checked
    changed = [False]
    def drift(source, target, sha):
        original(source, target, sha)
        if not changed[0]:
            changed[0] = True
            (roots[0] / "unexpected").write_text("changed while merging")
    monkeypatch.setattr(distributed, "_copy_checked", drift)
    with pytest.raises(CacheIntegrityError, match="untracked"):
        merge_worker_caches(roots, tmp_path / "merged", plan, rows, reference)
    assert not (tmp_path / "merged").exists() and not (tmp_path / "merged_merge.json").exists()


def test_failed_publication_cannot_leave_complete_receipt(tmp_path, fixture, monkeypatch):
    plan, rows, reference, *_ = fixture
    roots = workers(tmp_path, fixture)
    original = Path.rename
    def failed(self, target):
        if ".staging-" in self.name: raise OSError("TEST-only publication failure")
        return original(self, target)
    monkeypatch.setattr(Path, "rename", failed)
    with pytest.raises(OSError, match="publication"):
        merge_worker_caches(roots, tmp_path / "merged", plan, rows, reference)
    assert not (tmp_path / "merged").exists() and not (tmp_path / "merged_merge.json").exists()


def test_same_rank_concurrent_writer_is_refused_without_touching_stale_lock(tmp_path, fixture):
    lock = tmp_path / "rank0_writer.lock"
    lock.write_text("TEST-only active or stale writer")
    with pytest.raises(FileExistsError):
        write(tmp_path / "rank0", 0, fixture)
    assert lock.read_text() == "TEST-only active or stale writer" and not (tmp_path / "rank0").exists()


def test_same_output_concurrent_merge_is_refused_without_touching_stale_lock(tmp_path, fixture):
    plan, rows, reference, *_ = fixture
    roots = workers(tmp_path, fixture)
    lock = tmp_path / "merged_merge.lock"
    lock.write_text("TEST-only active or stale merge")
    with pytest.raises(FileExistsError):
        merge_worker_caches(roots, tmp_path / "merged", plan, rows, reference)
    assert lock.read_text() == "TEST-only active or stale merge" and not (tmp_path / "merged").exists()


@pytest.mark.parametrize("kind", ["source_index_gap", "foreign_episode", "missing_raw_field", "bad_text"])
def test_worker_inventory_audits_reject_rehashed_structural_changes(tmp_path, fixture, kind):
    plan, rows, reference, *_ = fixture
    roots = workers(tmp_path, fixture)
    root = roots[0]
    manifest = json.loads((root / "manifest.json").read_text())
    if kind == "source_index_gap":
        manifest["shards"][0]["start"] += 1; save_manifest(root, manifest)
    elif kind == "foreign_episode":
        manifest["assigned_episodes"] = [1, 3]; save_manifest(root, manifest)
    elif kind == "missing_raw_field":
        rewrite_shard(root, lambda payload: payload.pop("raw_action"))
    elif kind == "bad_text":
        entry = manifest["texts"][0]
        payload = load_file(str(root / entry["path"]))
        payload["context"][0, 0] = float("nan")
        save_file(payload, str(root / entry["path"]))
        entry["sha256"] = file_sha256(root / entry["path"])
        save_manifest(root, manifest)
    with pytest.raises(CacheIntegrityError):
        merge_worker_caches(roots, tmp_path / "merged", plan, rows, reference)
