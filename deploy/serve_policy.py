#!/usr/bin/env python3
"""Start a synchronous inference-only server (no robot SDK or hardware access)."""
import argparse
import json
import logging
import os
from pathlib import Path
import signal
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "deploy")]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("cosmos", "dense", "mock"), required=True)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument("--token-env", default="REALMAN_SERVER_TOKEN")
    parser.add_argument("--session-timeout", type=float, default=300)
    parser.add_argument("--mock-video-context-frames", type=int, choices=(1, 2), default=1)
    args = parser.parse_args(argv)
    if args.backend != "mock" and (args.config is None or args.checkpoint is None):
        parser.error("--config and --checkpoint are required for a real policy")
    if not 0 <= args.port <= 65535 or args.session_timeout <= 0:
        parser.error("Invalid port or session timeout")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    token = os.environ.get(args.token_env)
    if args.backend == "mock":
        from realman_sync.mock_policy import MockPolicy
        policy = MockPolicy(video_context_frames=args.mock_video_context_frames)
    else:
        config_path = args.config.expanduser().resolve()
        config = json.loads(config_path.read_text())
        if args.backend == "cosmos":
            from realman_sync.cosmos_policy import CosmosPolicy as Policy
        else:
            from realman_sync.dense_policy import DensePolicy as Policy
        policy = Policy(config=config, config_path=config_path, checkpoint=args.checkpoint.expanduser().resolve())
    from realman_sync.server import InferenceService, PolicyHTTPServer
    server = None
    try:
        service = InferenceService(policy, session_timeout_s=args.session_timeout)
        server = PolicyHTTPServer((args.host, args.port), service, token=token)
        logging.info("Ready backend=%s address=%s:%s metadata=%s", args.backend, *server.server_address, service.metadata)
        def stop(signum, frame):
            raise KeyboardInterrupt
        signal.signal(signal.SIGTERM, stop)
        server.serve_forever(poll_interval=.1)
    except KeyboardInterrupt:
        pass
    finally:
        if server is not None:
            server.server_close()
        if "service" in locals():
            service.close()
        else:
            policy.close()


if __name__ == "__main__":
    main()
