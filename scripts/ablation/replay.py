"""Real-checkpoint HTTP replay and measured latency; never connects a robot."""
from __future__ import annotations

import argparse
import io
import json
from pathlib import Path
import sys
import threading
import time
from urllib.request import Request, build_opener, ProxyHandler

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "deploy")]
from fastwam.ablation.contracts import load_dataset_manifest, require, write_json
from fastwam.ablation.runtime import AblationPolicy
from realman_sync.server import InferenceService, PolicyHTTPServer
from realman_sync.wire import PROTOCOL, encode_observation, validate_response


def memory_report():
    """Read the allocator used by the already-loaded backend, without new GPU init."""
    if "jax" in sys.modules:
        stats = [d.memory_stats() for d in sys.modules["jax"].devices() if d.platform == "gpu"]
        return {"provider": "jax", "device_allocator_stats": stats}
    return {"provider": None, "reason": "Backend did not expose a GPU allocator"}


def payload_from_episode(table, index, metadata, request_id):
    from PIL import Image
    n = len(table["index"])
    require(0 <= index < n, "Replay frame outside episode")
    history = []
    for j in range(max(0, index - 31), index + 1):
        stamp = 10000 + j / 30
        history.append(dict(sequence=j, stamp=stamp, state=table["observation.state"][j],
                            force=table["observation.force"][j],
                            force_meta=dict(preprocessing_id=metadata["force_preprocessing"]["id"], epoch=0, source_time=stamp)))
    frames = {}
    for camera in metadata["camera_names"]:
        cell = table["observation.images." + camera][index]
        rgb = Image.open(io.BytesIO(cell["bytes"])).convert("RGB")
        frames[camera] = dict(width=rgb.width, height=rgb.height, data=rgb.tobytes())
    obs = encode_observation(dict(history[-1], action_space="joint8", frames=frames, frame_meta={}))
    return dict(protocol=PROTOCOL, session_id="offline-replay", request_id=str(request_id),
                observation=obs, history=history, force_preprocessing=metadata["force_preprocessing"])


def replay(deployment, dataset, count, warmup, report_path):
    import pyarrow.parquet as pq
    require(count > 0 and warmup >= 0, "Invalid benchmark sample count")
    manifest = load_dataset_manifest(dataset)
    started = time.perf_counter()
    policy = AblationPolicy(deployment)
    require(policy.config["dataset_identity_sha256"] == manifest["identity_sha256"], "Replay dataset differs from checkpoint")
    load_ms = (time.perf_counter() - started) * 1000
    service = InferenceService(policy)
    server = PolicyHTTPServer(("127.0.0.1", 0), service)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    opener = build_opener(ProxyHandler({}))
    url = f"http://127.0.0.1:{server.server_address[1]}"
    def post(endpoint, body):
        request = Request(url + endpoint, json.dumps(body, allow_nan=False).encode(), {"Content-Type": "application/json"})
        with opener.open(request, timeout=600) as response:
            return json.load(response)
    measured, errors, receipts = [], [], []
    try:
        with opener.open(url + "/health") as response:
            health = json.load(response)
        require(health["ready"] is True, "Server not ready")
        total = count + warmup
        selection = np.linspace(0, len(manifest["episodes"]) - 1, total).astype(int)
        for number, ep in enumerate(selection):
            path = Path(dataset) / f"data/chunk-{ep // 1000:03d}/episode_{ep:06d}.parquet"
            table = pq.read_table(path).to_pydict()
            anchor = min(31, len(table["index"]) - 1)
            payload = payload_from_episode(table, anchor, policy.metadata, number)
            before = time.perf_counter()
            response = post("/infer", payload)
            elapsed = (time.perf_counter() - before) * 1000
            actions = np.asarray(validate_response(response, payload, policy.metadata))
            post("/session/release", dict(protocol=PROTOCOL, session_id=payload["session_id"]))
            if number >= warmup:
                measured.append(elapsed)
                target = np.asarray(table["action"])[np.minimum(anchor + np.arange(32), len(table["action"]) - 1)]
                valid = anchor + np.arange(32) < len(table["action"])
                error = actions[valid] - target[valid]
                errors.append(error)
                receipts.append(dict(episode=int(ep), source_episode=manifest["episodes"][ep]["source_episode"],
                                     frame=anchor, inference_ms=response["inference_ms"], http_ms=elapsed,
                                     supervised_steps=int(valid.sum())))
        err = np.concatenate(errors)
        result = dict(schema="rm71d78-replay-v1", backend=policy.config["backend"], model_load_verified=True,
                      http_inference_verified=True, formal_training_launched_by_replay=False,
                      deployment_identity=policy.config["identity_sha256"], dataset_identity=manifest["identity_sha256"],
                      scope="training-fit replay; no held-out or physical robot success evidence",
                      timing_policy="source-valid segments at 30Hz, timestamps rebased for offline wire replay",
                      load_ms=load_ms, warmup_samples=warmup, measured_samples=count,
                      http_ms_p50=float(np.percentile(measured, 50)), http_ms_p95=float(np.percentile(measured, 95)),
                      joint_mae_rad=float(np.abs(err[:, :7]).mean()), gripper_mae=float(np.abs(err[:, 7]).mean()),
                      gpu_memory=memory_report(), samples=receipts)
        write_json(report_path, result)
        return result
    finally:
        server.shutdown()
        thread.join(timeout=5)
        server.server_close()
        service.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deployment", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--report", required=True)
    args = parser.parse_args()
    result = replay(args.deployment, args.dataset, args.samples, args.warmup, args.report)
    print(json.dumps({k: result[k] for k in ("backend", "model_load_verified", "http_ms_p50", "http_ms_p95", "gpu_memory")}, indent=2))


if __name__ == "__main__":
    main()
