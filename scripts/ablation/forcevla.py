#!/usr/bin/env python3
"""ForceVLA entrypoint. Defaults to a non-training plan."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from fastwam.ablation.forcevla import run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--operation", choices=("plan", "stats", "train", "resume"), default="plan")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    print(json.dumps(run(config, args.operation), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
