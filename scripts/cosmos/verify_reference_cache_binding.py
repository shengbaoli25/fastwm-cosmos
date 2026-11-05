"""CPU-only provenance binding of a reference bank to a Cosmos cache manifest.

This verifies manifest lineage and episode/frame semantics, not cache tensor
payloads, GPU execution or model readiness. The receipt lives outside both banks.
"""
import argparse
import json
from pathlib import Path, PurePosixPath
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'src'))

from fastwam.cosmos.reference_bank import atomic_json, check
from fastwam.cosmos.reference_binding import PROFILE, REPRESENTATION, verify_binding


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bank', type=Path, required=True)
    parser.add_argument('--cache-manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--local-raw-inventory', type=Path,
                        help='Optional complete local path-to-SHA256 JSON for nonnumeric difference diagnosis')
    args = parser.parse_args(argv)
    out = args.output.resolve()
    for protected in (args.bank.resolve(), args.cache_manifest.resolve().parent):
        check(not out.is_relative_to(protected), 'Write receipt outside reference bank and cache directories')
    if args.local_raw_inventory:
        check(out != args.local_raw_inventory.resolve(), 'Receipt must not overwrite inventory evidence')
    check(not out.exists(), 'Receipt already exists; choose a fresh output path')
    receipt = verify_binding(args.bank, args.cache_manifest, local_raw_inventory=args.local_raw_inventory)
    out.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(out, receipt)
    print(json.dumps(dict(status=receipt['status'], output=str(out),
                          numeric_metadata_binding_verified=receipt['numeric_metadata_binding_verified'],
                          full_raw_inventory_equal=receipt['full_raw_inventory_equal'],
                          errors=receipt['errors']), ensure_ascii=False))
    return 0 if receipt['status'] == 'bound' else 2


if __name__ == '__main__':
    raise SystemExit(main())
