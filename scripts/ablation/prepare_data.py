"""Audit the original 78 trajectories, optionally export a shared training view."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from fastwam.ablation.data import prepare_dataset
from fastwam.ablation.contracts import write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", help="New destination; omitted means audit only")
    parser.add_argument("--report", required=True)
    args = parser.parse_args()
    result = prepare_dataset(args.source, args.output)
    write_json(args.report, result)
    print(f"Audited {result['source_episodes']} source episodes, {result['retained_frames']} retained frames")


if __name__ == "__main__":
    main()
