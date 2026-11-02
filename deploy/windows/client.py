"""Run the copied independent client; all CLI options come from that client."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "client/src"))
from realman_sync_client.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
