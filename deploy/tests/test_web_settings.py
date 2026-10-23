"""Settings API persists inference schedules without changing model files."""
import json
from pathlib import Path
import sys
import tempfile
import threading
from urllib.error import HTTPError
from urllib.request import Request, urlopen

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from realman_deploy.normalization import DatasetNormalizer
from realman_deploy.runtime import DeploymentRuntime
from realman_deploy.web import DeploymentWeb


def test_settings_api_persists_4416_mode_and_rejects_invalid_combinations(tmp_path):
    normalizer = DatasetNormalizer({
        "state": {"mean": [0.] * 8, "std": [1.] * 8},
        "action": {"mean": [0.] * 8, "std": [1.] * 8},
        "force": {"mean": [0.] * 6, "std": [1.] * 6},
    }, action_space="joint8")
    config_path = tmp_path / "runtime.json"
    config_path.write_text(json.dumps({"device": "cuda:1", "sampling_profile": "161616",
        "execution_mode": "synchronous", "model_kwargs": {"marker": "preserved"}}))
    catalog_path = tmp_path / "catalog.json"
    catalog_path.write_text('{"checkpoints": []}')
    runtime = DeploymentRuntime(normalizer, require_cuda=False, execution_mode="synchronous",
                                cuda_device="cuda:1", sampling_profile="161616")
    web = DeploymentWeb(runtime, catalog_path=catalog_path, config_path=config_path, port=0)
    server_thread = threading.Thread(target=web.serve_forever, daemon=True)
    server_thread.start()
    base = f"http://127.0.0.1:{web.server.server_address[1]}"
    try:
        with urlopen(base + "/api/settings") as response:
            before = json.load(response)
        assert before["supported"] is True
        assert before["sampling_profile"] == "161616"
        payload = json.dumps({"sampling_profile": "4416", "execution_mode": "async_relaxed"}).encode()
        request = Request(base + "/api/settings", data=payload, headers={"Content-Type": "application/json"})
        with urlopen(request) as response:
            applied = json.load(response)
        assert applied["reload_required"] is True
        assert runtime.sampling_profile == "4416"
        assert runtime.execution_mode == "async_relaxed"
        saved = json.loads(config_path.read_text())
        assert saved["sampling_profile"] == "4416"
        assert saved["inference_backend"] == "joint_multirate"
        assert saved["device"] == "cuda:1" and saved["model_kwargs"]["marker"] == "preserved"

        bad = json.dumps({"sampling_profile": "161616", "execution_mode": "async_relaxed"}).encode()
        request = Request(base + "/api/settings", data=bad, headers={"Content-Type": "application/json"})
        try:
            urlopen(request)
        except HTTPError as error:
            assert error.code == 400
            assert "仅支持" in json.load(error)["error"]
        else:
            raise AssertionError("invalid standard-profile async setting was accepted")
        assert json.loads(config_path.read_text())["sampling_profile"] == "4416"
    finally:
        web.close()
        server_thread.join(timeout=2)
