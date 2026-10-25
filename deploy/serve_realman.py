#!/usr/bin/env python3
"""Start the local RealMan deployment dashboard. No hardware is opened at boot."""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "deploy"))
sys.path.insert(0, str(ROOT / "src"))

from realman_deploy.normalization import DatasetNormalizer
from realman_deploy.runtime import DeploymentRuntime
from realman_deploy.web import DeploymentWeb
from realman_deploy.sampling import updated_config
from realman_deploy.deployment_settings import deployment_settings


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "deploy/realman_runtime.example.json"))
    parser.add_argument("--catalog", default=str(ROOT / "deploy/checkpoint_catalog.json"))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--start-estopped", action="store_true",
                        help="Start with the software stop latched; no hardware is opened")
    args = parser.parse_args()
    config_path = Path(args.config).resolve()
    config = json.loads(config_path.read_text())
    normalizer_path = Path(config["normalization_file"]).expanduser()
    if not normalizer_path.is_absolute():
        normalizer_path = config_path.parent / normalizer_path
    normalizer = DatasetNormalizer.from_json(normalizer_path)
    if config.get("sampling_profile") is not None:
        # Reject invalid profile/mode combinations before creating the server.
        updated_config(config, config["sampling_profile"], config.get("execution_mode", "synchronous"))
    choices = deployment_settings(config)
    runtime = DeploymentRuntime(normalizer,
        execution_mode=config.get("execution_mode", "synchronous"),
        cuda_device=config.get("device", "cuda"),
        sampling_profile=config.get("sampling_profile"),
        inference_device=choices["inference_device"],
        max_result_age_s=choices["max_result_age_s"],
        action_smoothing_alpha=choices["action_smoothing_alpha"])
    if args.start_estopped:
        runtime.emergency_stop()
    web = DeploymentWeb(runtime, catalog_path=args.catalog, config_path=config_path,
                        hardware_factory=config.get("hardware_factory"),
                        host=args.host, port=args.port)
    print(f"RealMan deploy UI listening on http://{args.host}:{args.port}", flush=True)
    try:
        web.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        web.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
