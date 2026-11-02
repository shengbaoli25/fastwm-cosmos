"""Serve trained ForceVLA through the existing realman-sync-v1 protocol."""
import argparse
import logging
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "deploy")]
from fastwam.ablation.runtime import AblationPolicy
from realman_sync.server import InferenceService, PolicyHTTPServer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deployment", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8766)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    service = InferenceService(AblationPolicy(args.deployment))
    server = PolicyHTTPServer((args.host, args.port), service, token=os.environ.get("REALMAN_SERVER_TOKEN"))
    try:
        logging.info("Loaded trained %s; listening on %s:%d", service.metadata["backend"], args.host, args.port)
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        service.close()


if __name__ == "__main__":
    main()
