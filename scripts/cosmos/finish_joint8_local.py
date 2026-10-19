"""Install a downloaded step1000 weight file, export runtime and replay locally.

The download itself uses the authenticated in-app project-file UI. This command
only handles local files and never opens hardware or starts robot control.
"""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'deploy')]

from realman_deploy.preprocessing import file_sha256


def verified_copy(source, destination, expected_sha, expected_size):
    if source.stat().st_size != expected_size or file_sha256(source) != expected_sha:
        raise ValueError('Downloaded checkpoint differs from remote save receipt')
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if destination.stat().st_size != expected_size or file_sha256(destination) != expected_sha:
            raise ValueError('Refuse to overwrite a different installed checkpoint')
        return
    temporary = destination.with_suffix('.download')
    with source.open('rb') as src, temporary.open('xb') as dst:
        shutil.copyfileobj(src, dst, length=8 * 1024 * 1024)
    if file_sha256(temporary) != expected_sha:
        raise ValueError('Local copy hash differs')
    temporary.replace(destination)


def main():
    import torch
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('bundle', 'dataset', 'downloaded-checkpoint', 'remote-receipt', 'base-checkpoint', 'tokenizer'):
        p.add_argument('--' + key, type=Path, required=True)
    p.add_argument('--upstream', type=Path, default=ROOT / 'third_party/cosmos-predict2.5')
    p.add_argument('--check-uncached', action='store_true')
    args = p.parse_args()
    bundle = args.bundle.resolve()
    receipt = json.loads(args.remote_receipt.read_text())
    if (receipt.get('step') != 1000 or receipt.get('schema') != 'joint8-step1000-download-v1'
            or receipt.get('training_complete') is not True):
        raise ValueError('A completed step1000 remote save receipt is required')
    checkpoint = bundle / 'weights/step_001000.pt'
    verified_copy(args.downloaded_checkpoint, checkpoint, receipt['sha256'], receipt['size_bytes'])
    payload = torch.load(checkpoint, map_location='cpu', mmap=True, weights_only=True)
    from fastwam.cosmos.noaction import SCHEMA
    if payload.get('step') != 1000 or payload.get('schema') != SCHEMA:
        raise ValueError('Checkpoint step or model protocol differs')
    del payload
    local_receipt = bundle / 'checkpoint-download-receipt.json'
    local_receipt.write_text(json.dumps(receipt, indent=2) + '\n')
    runtime = bundle / 'runtime.local.json'
    command = [sys.executable, str(ROOT / 'scripts/cosmos/export_noaction_deployment.py'),
        '--training-config', str(bundle / 'short/resolved.yaml'), '--checkpoint', str(checkpoint),
        '--normalization', str(bundle / 'prepared/normalization.json'), '--tokenizer', str(args.tokenizer),
        '--hardware-config', str(bundle / 'hardware.local.json'), '--upstream', str(args.upstream),
        '--base-checkpoint', str(args.base_checkpoint), '--text-cache', str(bundle / 'prepared/text'),
        '--output', str(runtime)]
    if not runtime.exists():
        subprocess.run(command, cwd=ROOT, check=True)
    else:
        # The exporter refuses overwrites; compare a fresh temporary export.
        candidate = bundle / ('runtime-check-' + str(time.time_ns()) + '.json')
        command[-1] = str(candidate)
        subprocess.run(command, cwd=ROOT, check=True)
        if json.loads(candidate.read_text()) != json.loads(runtime.read_text()):
            raise ValueError('Existing runtime differs from current training-bound export')
        candidate.unlink()
    catalog = dict(checkpoints=[dict(id='step_001000', label='Joint8 merged50 · step1000 · no-reference',
        source=str(checkpoint), sha256=receipt['sha256'], size_bytes=receipt['size_bytes'])])
    (bundle / 'checkpoint_catalog.json').write_text(json.dumps(catalog, indent=2, ensure_ascii=False) + '\n')
    output = bundle / ('validation-' + time.strftime('%Y%m%d-%H%M%S'))
    command = [sys.executable, str(ROOT / 'scripts/cosmos/validate_joint8_local.py'),
        '--bundle', str(bundle), '--dataset', str(args.dataset), '--output', str(output),
        '--runtime', str(runtime), '--checkpoint', str(checkpoint)]
    if args.check_uncached:
        command.append('--check-uncached')
    subprocess.run(command, cwd=ROOT, check=True)
    (bundle / 'latest-validation.json').write_text(json.dumps(dict(
        report=str(output / 'report.json'), runtime=str(runtime), checkpoint=str(checkpoint)), indent=2) + '\n')
    print(output / 'report.json', flush=True)


if __name__ == '__main__':
    main()
