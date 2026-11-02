"""Read-only integrity and safe identity check for this exact ViT-B 20k run.

Never constructs a model, scans all tensor values, initializes CUDA, writes a
runtime configuration, starts a service, or accesses hardware. The subsequent
prepare_runtime.py check validates the full source model and finite tensors.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time
import zipfile


ROOT = Path(__file__).resolve().parent
SCHEMA = 'fastwam-jepa-video-context-cosmos-joint8-v1'
CONFIG_SHA = '031f6256d3697427430f76b30dba7798f83cea501068b715c722c70db7f33e45'
TEACHER_SHA = '848a77c33cc9e6649ed2119c9bea1e2c569bcdab9539ff3e7c02ccc2959ddf4d'
CACHE = '7019254c3a99f3859e0cf742a211a7fbd05201797c7e17066d799c41174fe0ce'
DATASET = 'b24b701239f54ad38d16c819b9ea9f6fad5f9374a683a84a68d2cd7ef27067a8'
ARCHITECTURE = dict(depth=28, feature_dim=768, hidden_dim=768, heads=12,
    view_grid=14, action_hidden=768, force_hidden=768, action_ffn=3072,
    force_ffn=2048, joint_heads=16, joint_head_dim=128, context_dim=1024)


class Incomplete(ValueError):
    pass


def fingerprint(path):
    stat = path.stat()
    return stat.st_size, stat.st_mtime_ns


def require_finished(path):
    if path.suffix.lower() != '.pt' or not path.is_file() or path.stat().st_size == 0:
        raise Incomplete('A nonempty final .pt is required; do not rename a partial')
    if any(Path(str(path)+suffix).exists()
           for suffix in ('.crswap', '.crdownload', '.part', '.partial', '.tmp')):
        raise Incomplete('Matching transfer temporary file is still present')


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def verify(path):
    require_finished(path)
    before = fingerprint(path)
    # A broad rejection bound only; the platform's rounded 2.09 GB is not an
    # expected exact size. Record the actual length and validate the identity.
    if not 2_000_000_000 <= before[0] <= 2_500_000_000:
        raise ValueError('Unexpected size for this exact 20k ViT-B weight file')
    time.sleep(2)
    if fingerprint(path) != before:
        raise Incomplete('Checkpoint is still changing')
    with zipfile.ZipFile(path) as archive:
        members = archive.infolist()
        if not members or sum(x.file_size for x in members) > 4 * 1024**3:
            raise ValueError('Unexpected checkpoint ZIP structure')
        bad_member = archive.testzip()
        if bad_member is not None:
            raise ValueError('Checkpoint ZIP CRC failed')
        archive_info = dict(format='torch_zip', members=len(members),
            uncompressed_bytes=sum(x.file_size for x in members), full_member_crc_verified=True)
    digest = sha256(path)
    config_path = ROOT / 'config/training.config.yaml'
    if sha256(config_path) != CONFIG_SHA:
        raise ValueError('Saved training config differs from the verified exact-run YAML')

    import torch
    from omegaconf import OmegaConf
    torch.set_num_threads(1)
    if torch.cuda.is_initialized():
        raise RuntimeError('Run this verifier in a fresh CPU-only process')
    config = OmegaConf.to_container(OmegaConf.load(config_path), resolve=False)
    payload = torch.load(path, weights_only=True, mmap=True, map_location='cpu')
    if (not isinstance(payload, dict) or payload.get('schema') != SCHEMA
            or type(payload.get('step')) is not int or payload['step'] != 20000):
        raise ValueError('Checkpoint schema or saved step differs from ViT-B 20k')
    manifest = payload.get('manifest')
    if not isinstance(manifest, dict):
        raise ValueError('Checkpoint manifest is missing')
    expected = config['model']['manifest']
    if any(manifest.get(key) != value for key, value in expected.items()):
        raise ValueError('Checkpoint base manifest differs from the saved training configuration')
    if (manifest.get('teacher', {}).get('name') != 'vjepa21'
            or manifest['teacher'].get('weight_sha256') != TEACHER_SHA
            or manifest.get('cache_identity_sha256') != CACHE
            or manifest.get('dataset_source_sha256') != DATASET
            or manifest.get('architecture') != ARCHITECTURE
            or manifest.get('num_inference_steps') != 16
            or manifest.get('action_space') != 'joint8'):
        raise ValueError('Teacher/cache/dataset/architecture/recipe identity differs')
    state = payload.get('model')
    if (not isinstance(state, dict) or not state
            or any(not isinstance(x, torch.Tensor) or x.device.type != 'cpu' or x.is_meta
                   for x in state.values())):
        raise ValueError('Expected a nonempty CPU tensor model state')
    if torch.cuda.is_initialized():
        raise RuntimeError('Unexpected CUDA initialization')
    require_finished(path)
    if fingerprint(path) != before:
        raise Incomplete('Checkpoint changed during validation')
    return dict(status='verified', checkpoint_path=str(path), bytes=before[0],
        mtime_ns=before[1], sha256=digest, sha256_kind='locally_computed_complete_file_fingerprint',
        upstream_expected_checkpoint_sha256=None, archive=archive_info,
        training_config_sha256=CONFIG_SHA, schema=SCHEMA, saved_step=20000,
        exact_training_manifest_fields_match=True, manifest=manifest,
        model_tensor_count=len(state), model_elements=sum(x.numel() for x in state.values()),
        deserialization=dict(weights_only=True, mmap=True, map_location='cpu', torch_version=torch.__version__),
        model_constructed=False, finite_tensor_scan=False, full_model_contract_checked=False,
        cuda_initialized=False, server_started=False, hardware_accessed=False,
        next_check='prepare_runtime.py (full original architecture/state/finite/normalization/force validation)')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, default=ROOT/'checkpoints/step_020000.pt')
    parser.add_argument('--report', type=Path, help='Optional NEW report path; existing reports are refused')
    args = parser.parse_args()
    output = args.report or ROOT/'diagnostics'/('checkpoint-verification-'+
        datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')+'.json')
    if output.exists():
        parser.error('Report already exists; select a new filename')
    report = dict(report_schema='vjepa-vitb20k-readonly-verification-v1',
                  checked_utc=datetime.now(timezone.utc).isoformat())
    try:
        report.update(verify(args.checkpoint.expanduser().resolve()))
        code = 0
    except Incomplete as error:
        report.update(status='missing_or_in_progress', error=str(error))
        code = 2
    except Exception as error:
        report.update(status='invalid_or_error', error_type=type(error).__name__, error=str(error))
        code = 1
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    # The complete manifest goes only to the local report, keeping progress short.
    print(json.dumps({key: value for key, value in report.items() if key != 'manifest'},
                     ensure_ascii=False, indent=2, allow_nan=False))
    print('report='+str(output.resolve()))
    return code


if __name__ == '__main__':
    raise SystemExit(main())
