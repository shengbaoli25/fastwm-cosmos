"""Package the matching training normalization with a checkpoint identity."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'deploy')]
from realman_deploy.preprocessing import binding


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--normalization', type=Path, required=True,
                        help='normalization.json exported by the corresponding training run')
    args = parser.parse_args()
    result = binding(args.checkpoint, args.normalization)
    output = Path(str(args.checkpoint)+'.force.json')
    if output.exists() and json.loads(output.read_text()) != result:
        raise ValueError('Existing checkpoint binding differs; use the correct checkpoint/normalization pair')
    output.write_text(json.dumps(result, indent=2)+'\n')
    print(output)


if __name__ == '__main__':
    main()
