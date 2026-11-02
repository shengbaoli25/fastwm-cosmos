"""Synthetic capture/export and real HTTP tests, explicitly not model validation."""
import importlib
import json
from pathlib import Path
import threading
import tempfile
from urllib.request import Request, build_opener, ProxyHandler

import numpy as np
import pytest

from fastwam.ablation import contracts, data
from fastwam.ablation.runtime import AblationPolicy, export_deployment, load_deployment
from realman_force.contract import input_contract
from realman_sync.server import InferenceService, PolicyHTTPServer
from realman_sync.wire import PROTOCOL, encode_observation, validate_response

ROOT = Path(__file__).resolve().parents[2]


def capture_table(n=6, episode=0):
    stamp = 100 + np.arange(n) / 30
    state = np.arange(n * 20, dtype=np.float32).reshape(n, 20) / 100
    state[:, 19] = .5
    action = state[:, [0, 1, 2, 3, 4, 5, 6, 19]] + .01
    table = {"observation.state": state.tolist(), "action": action.tolist(),
             "observation.force_filtered": state[:, 13:19].tolist(),
             "frame_index": list(range(n)), "episode_index": [episode] * n,
             "index": list(range(episode * n, (episode + 1) * n)), "timestamp": (np.arange(n) / 30).tolist(),
             "observation.monotonic_time_s": stamp.tolist(),
             "observation.capture_fresh": [[1] * 4] * n, "action.applied": [1] * n,
             "observation.force_filter_input_count": [200] * n,
             "observation.capture_discontinuity": [0] * n, "observation.capture_gap_s": [0] * n,
             "observation.force_filter_epoch": [0] * n, "action.follow_session_id": [0] * n,
             "action.leader_anchor": [[0] * 7] * n, "action.follower_anchor": [[0] * 7] * n,
             "observation.camera.arrival_time_s": np.repeat((stamp - .01)[:, None], 2, axis=1).tolist()}
    for key in ("observation.force_timestamp_s", "observation.force_filter_timestamp_s", "observation.robot_timestamp_s",
                "action.timestamp_s", "observation.leader_timestamp_s", "observation.gripper_timestamp_s", "action.gripper_timestamp_s"):
        table[key] = stamp.tolist()
    return table


def test_gaps_and_references_never_bridge_training_windows():
    table = capture_table(12)
    table["observation.capture_discontinuity"][4] = 1
    table["action.follow_session_id"][8:] = [1] * 4
    assert data.continuous_segments(table) == [(0, 4), (4, 8), (8, 12)]
    table["action.applied"][2] = 0
    assert data.continuous_segments(table) == [(0, 2), (4, 8), (8, 12)]


def test_joint8_selects_actual_gripper_and_rejects_force_mismatch():
    table = capture_table()
    result, _ = data.numeric_episode(table, {"length": 6}, 0)
    np.testing.assert_array_equal(result["observation.state"][:, 7], .5)
    np.testing.assert_array_equal(result["action"][:, 7], np.float32(.51))
    table["observation.force_filtered"][0][0] += 1
    with pytest.raises(ValueError, match="Force columns"):
        data.numeric_episode(table, {"length": 6}, 0)


@pytest.fixture(scope="module")
def exported(tmp_path_factory):
    import av
    import pyarrow as pa
    import pyarrow.parquet as pq
    tmp = tmp_path_factory.mktemp("synthetic-capture")
    source, output = tmp / "source", tmp / "export"
    source.mkdir()
    calibration = contracts.read_json(ROOT / "deploy/force_calibration.json")
    contract = input_contract(calibration)
    contracts.write_json(source / "meta/force.json", {"schema": "realman_leader_force_gripper_v2", "model_input": contract})
    contracts.write_json(source / "meta/leader.json", {"training_demonstration": True, "mode": "execute"})
    for camera in contracts.CAMERAS:
        path = source / "videos" / f"{camera}.mp4"
        path.parent.mkdir(exist_ok=True)
        with av.open(str(path), "w") as container:
            stream = container.add_stream("libx264", rate=30)
            stream.width, stream.height, stream.pix_fmt = 640, 480, "yuv420p"
            for index in range(6):
                pixels = np.full((480, 640, 3), index * 20 + (30 if camera == "wrist" else 0), dtype=np.uint8)
                for packet in stream.encode(av.VideoFrame.from_ndarray(pixels, format="rgb24")):
                    container.mux(packet)
            for packet in stream.encode():
                container.mux(packet)
    # Both episodes point into their own camera file through a format placeholder.
    episodes, tables = [], []
    for ep in range(78):
        contracts.write_json(source / f"meta/force_sessions/episode_{ep:06d}.json", calibration)
        row = dict(episode_index=ep, length=6, tasks=[contracts.PROMPT], dataset_from_index=ep * 6,
                   dataset_to_index=(ep + 1) * 6, **{"data/chunk_index": 0, "data/file_index": 0})
        for camera in contracts.CAMERAS:
            key = "videos/observation.images." + camera + "/"
            row.update({key + "chunk_index": 0, key + "file_index": 0, key + "from_timestamp": 0., key + "to_timestamp": .2})
        episodes.append(row)
        tables.append(pa.table(capture_table(episode=ep)))
    (source / "data").mkdir()
    (source / "meta/episodes").mkdir()
    pq.write_table(pa.concat_tables(tables), source / "data/all.parquet")
    pq.write_table(pa.Table.from_pylist(episodes), source / "meta/episodes/all.parquet")
    # Names match the real source format, without relying on any real captured frames.
    for camera in contracts.CAMERAS:
        (source / "videos" / f"{camera}.mp4").rename(source / "videos" / f"observation.images.{camera}.mp4")
    contracts.write_json(source / "meta/info.json", dict(codebase_version="v3.0", fps=30, total_episodes=78,
                        total_frames=78 * 6, data_path="data/all.parquet", video_path="videos/{video_key}.mp4"))
    patch = pytest.MonkeyPatch()
    digest = contracts.identity_sha256(contracts.inventory(source))
    patch.setattr(data, "SOURCE_SHA256", digest)
    patch.setattr(contracts, "SOURCE_SHA256", digest)
    try:
        manifest = data.prepare_dataset(source, output)
        yield output, manifest
    finally:
        patch.undo()


def test_complete_export_readable_by_official_hf_loader(exported, tmp_path):
    from datasets import load_dataset
    from PIL import Image
    output, manifest = exported
    assert len(manifest["episodes"]) == 78
    assert manifest["source_episodes"] == 78
    assert contracts.load_dataset_manifest(output)["identity_sha256"] == manifest["identity_sha256"]
    # HF includes the cache path twice in its lock filename; keep it short on Windows.
    with tempfile.TemporaryDirectory(prefix="ahf-") as cache:
        dataset = load_dataset("parquet", data_files=str(output / "data/chunk-000/episode_000000.parquet"), split="train", cache_dir=cache)
        row = dataset[0]
        del dataset
    assert isinstance(row["observation.images.global"], Image.Image)
    assert row["observation.images.global"].size == (640, 480)
    assert np.asarray(row["observation.images.wrist"]).mean() > np.asarray(row["observation.images.global"]).mean()
    assert len(row["observation.state"]) == len(row["action"]) == 8


def test_actual_http_contract_with_injected_predictor(exported, tmp_path, monkeypatch):
    backend = "fastwam"
    output, manifest = exported
    for filename in ("weights", "stats", "binding"):
        (tmp_path / filename).write_text("test-only", encoding="utf-8")
    config = {"checkpoint_path": str(tmp_path / "weights"), "normalization_path": str(tmp_path / "stats"), "binding_path": str(tmp_path / "binding")}
    contracts.write_json(tmp_path / "binding", {"dataset_identity": manifest["identity_sha256"], "normalization_sha256": contracts.file_sha256(tmp_path / "stats")})
    for field in ("training_config_path", "asset_root", "text_embedding_cache_dir"):
        config[field] = str(tmp_path / "weights")
    deployment = tmp_path / "deployment.json"
    export_deployment(backend, config, output, deployment)
    observed = []
    def predictor(sample):
        observed.append(sample)
        return np.tile(sample["state"], (32, 1))
    monkeypatch.setattr(importlib.import_module("fastwam.ablation." + backend), "load_predictor", lambda config: predictor)
    policy = AblationPolicy(deployment)
    service = InferenceService(policy)
    server = PolicyHTTPServer(("127.0.0.1", 0), service)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    opener = build_opener(ProxyHandler({}))
    url = f"http://127.0.0.1:{server.server_port}"
    try:
        with opener.open(url + "/health") as response:
            assert json.load(response)["ready"]
        sample = dict(sequence=1, stamp=10., state=[.1] * 7 + [.5], force=[.2] * 6,
                      force_meta={"preprocessing_id": policy.metadata["force_preprocessing"]["id"], "epoch": 0, "source_time": 10.})
        observation = encode_observation(dict(sample, action_space="joint8", frame_meta={}, frames={
            name: dict(width=640, height=480, data=bytes(640 * 480 * 3)) for name in contracts.CAMERAS}))
        request = dict(protocol=PROTOCOL, session_id="test", request_id="1", observation=observation,
                       history=[sample], force_preprocessing=policy.metadata["force_preprocessing"])
        def post(endpoint, payload):
            with opener.open(Request(url + endpoint, json.dumps(payload).encode(), {"Content-Type": "application/json"})) as response:
                return json.load(response)
        result = post("/infer", request)
        assert len(validate_response(result, request, policy.metadata)) == 32
        assert observed[0]["images"]["wrist"].shape == (480, 640, 3)
        assert post("/session/release", dict(protocol=PROTOCOL, session_id="test"))["released"]
        changed = tmp_path / "weights"
        changed.write_text("changed", encoding="utf-8")
        with pytest.raises(ValueError, match="asset changed"):
            load_deployment(deployment)
    finally:
        server.shutdown()
        worker.join(timeout=5)
        server.server_close()
        service.close()
