"""Bind an existing trained checkpoint and its assets; does not load or train it."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from fastwam.ablation.contracts import read_json
from fastwam.ablation.runtime import BACKENDS, export_deployment


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=BACKENDS, required=True)
    parser.add_argument("--model-config", required=True, help="JSON with real checkpoint/statistics/binding paths")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    export_deployment(args.backend, read_json(args.model_config), args.dataset, args.output)
    print("Exported deployment identity; model_load_verified=false until actual server loading/replay")


if __name__ == "__main__":
    main()
