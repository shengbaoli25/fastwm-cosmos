#!/usr/bin/env python3
"""Explicit plan/stats/train/resume entry points for the official pi05 trainer."""
from pathlib import Path
import argparse
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from fastwam.ablation import pi05


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("plan", "stats", "train", "resume"), nargs="?", default="plan")
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if args.command == "plan":
        print(json.dumps(pi05.plan(config), indent=2, ensure_ascii=False))
    elif args.command == "stats":
        print(json.dumps(pi05.prepare_statistics(config), indent=2, ensure_ascii=False))
    else:
        pi05.train(config, resume=args.command == "resume")


if __name__ == "__main__":
    main()
