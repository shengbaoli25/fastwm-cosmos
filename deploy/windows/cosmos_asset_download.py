"""Plan authenticated browser downloads; verify/import completed local files.

No network, browser profile, credentials, pickle, GPU, or hardware access.
The download itself remains an authenticated, manual platform browser action.
Exit codes: 0 verified/plan, 2 missing/in progress, 1 invalid/error.
"""
from __future__ import annotations

import argparse
from array import array
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import struct
import sys
import tempfile
import zipfile


HERE = Path(__file__).resolve().parent
TEMP_SUFFIXES = ('.crdownload', '.crswap', '.part', '.tmp', '.download')


class Incomplete(ValueError):
    pass


def local_path(value):
    if '://' in str(value):
        raise ValueError('Only local filesystem paths are accepted; URLs are not logged or fetched')
    return Path(value).expanduser().resolve()


def fingerprint(path):
    stat = path.stat()
    return stat.st_size, stat.st_mtime_ns


def reject_partial(path):
    if path.name.lower().endswith(TEMP_SUFFIXES):
        raise Incomplete('Temporary download filename is not accepted')
    if not path.is_file() or path.stat().st_size == 0:
        raise Incomplete('Missing or empty asset')
    if any(Path(str(path) + suffix).exists() for suffix in TEMP_SUFFIXES):
        raise Incomplete('A matching browser/transfer temporary file is still present')


def canonical_identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
        ensure_ascii=False, allow_nan=False).encode('utf-8')).hexdigest()


def inspect_context(path):
    # SafeTensors header and BF16 finiteness checked without torch/model imports.
    with path.open('rb') as stream:
        prefix = stream.read(8)
        if len(prefix) != 8:
            raise ValueError('Missing SafeTensors header')
        length = struct.unpack('<Q', prefix)[0]
        if not 0 < length <= 1024 * 1024:
            raise ValueError('Unexpected SafeTensors header length')
        header = json.loads(stream.read(length))
        context = header.get('context', {})
        expected_bytes = 512 * 100352 * 2
        start, end = context.get('data_offsets', (-1, -1))
        if (context.get('dtype') != 'BF16' or context.get('shape') != [512, 100352]
                or start < 0 or end-start != expected_bytes
                or 8+length+end > path.stat().st_size):
            raise ValueError('Context must be BF16 [512,100352] with valid offsets')
        stream.seek(8+length+start)
        remaining = expected_bytes
        while remaining:
            block = stream.read(min(1024*1024, remaining))
            if not block or len(block) % 2:
                raise ValueError('Incomplete BF16 context payload')
            words = array('H')
            words.frombytes(block)
            if sys.byteorder != 'little':
                words.byteswap()
            if any(word & 0x7f80 == 0x7f80 for word in words):
                raise ValueError('Context contains NaN or infinity')
            remaining -= len(block)
    return dict(shape=[512, 100352], dtype='BF16', finite=True)


def verify(path, asset, lock, *, crc=False, finite=False, partial_check=True):
    if partial_check:
        reject_partial(path)
    before = fingerprint(path)
    if before[0] != asset['bytes']:
        raise ValueError('File length differs from the verified asset lock')
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(4*1024*1024), b''):
            digest.update(block)
    if digest.hexdigest() != asset['sha256']:
        raise ValueError('SHA256 differs from the verified asset lock')
    result = dict(id=asset['id'], path=str(path), status='verified', bytes=before[0],
                  sha256=digest.hexdigest())
    if asset['id'] == 'text_manifest':
        manifest = json.loads(path.read_text('utf-8'))
        identity = canonical_identity(manifest)
        if identity != lock['text_identity']:
            raise ValueError('Text manifest identity differs from this checkpoint')
        rows = manifest.get('texts', [])
        expected = next(x for x in lock['assets'] if x['id'] == 'text_context')
        if (manifest.get('schema') != 'cosmos-reason1-full-concat-v1'
                or manifest.get('status') != 'complete' or len(rows) != 1
                or rows[0].get('path') != 'text_000.safetensors'
                or rows[0].get('sha256') != expected['sha256']):
            raise ValueError('Manifest context binding differs from the locked asset')
        result['canonical_identity'] = identity
    if asset['id'] == 'text_context' and finite:
        result['tensor'] = inspect_context(path)
    if asset['id'] == 'checkpoint' and crc:
        with zipfile.ZipFile(path) as archive:
            if len(archive.infolist()) != 24 or archive.testzip() is not None:
                raise ValueError('Checkpoint ZIP member count or full CRC differs')
        result['archive'] = dict(members=24, full_crc_verified=True, pickle_executed=False)
    if before != fingerprint(path):
        raise Incomplete('Asset changed during verification')
    if partial_check:
        reject_partial(path)
    result['mtime_ns'] = before[1]
    return result


def destination(root, asset):
    result = (root / asset['relative_path']).resolve()
    if not result.is_relative_to(root):
        raise ValueError('Destination escapes the inference root')
    return result


def import_verified(source, target, asset, lock, *, crc=False, finite=False):
    checked = verify(source, asset, lock, crc=crc, finite=finite)
    if target.exists():
        existing = verify(target, asset, lock, crc=crc, finite=finite)
        return dict(existing, copied=False, source_preserved=True, existing_identical=True)
    target.parent.mkdir(parents=True, exist_ok=True)
    temp = None
    try:
        with tempfile.NamedTemporaryFile(prefix='.'+target.name+'.', suffix='.part',
                dir=target.parent, delete=False) as output:
            temp = Path(output.name)
            with source.open('rb') as stream:
                for block in iter(lambda: stream.read(4*1024*1024), b''):
                    output.write(block)
            output.flush()
            os.fsync(output.fileno())
        if fingerprint(source) != (checked['bytes'], checked['mtime_ns']):
            raise Incomplete('Source changed during copy')
        reject_partial(source)
        verified = verify(temp, asset, lock, crc=crc, finite=finite, partial_check=False)
        # Same-directory hard-link publication is atomic and fails if target
        # appeared concurrently. Never replace an existing destination.
        os.link(temp, target)
        return dict(verified, path=str(target), source=str(source), copied=True,
                    source_preserved=True, publication='atomic_no_replace_hardlink')
    finally:
        if temp is not None:
            temp.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('plan', 'check', 'import'))
    parser.add_argument('--root', default=str(HERE.parent))
    parser.add_argument('--asset', default='all')
    parser.add_argument('--source', help='Completed local browser download; required by import')
    parser.add_argument('--crc', action='store_true', help='Also verify all checkpoint ZIP member CRCs')
    parser.add_argument('--finite', action='store_true', help='Also scan all BF16 text values on CPU')
    parser.add_argument('--report', help='Optional NEW report file; existing reports are never overwritten')
    args = parser.parse_args()
    lock = json.loads((HERE/'COSMOS_ASSETS.lock.json').read_text('utf-8'))
    root = local_path(args.root)
    selected = [x for x in lock['assets'] if args.asset in ('all', x['id'])]
    if not selected or (args.command == 'import' and (len(selected) != 1 or not args.source)):
        parser.error('Choose a known asset; import requires one --asset and one --source')
    report = dict(schema='cosmos20k-local-transfer-check-v1',
                  checked_utc=datetime.now(timezone.utc).isoformat(), command=args.command,
                  network_used=False, credentials_accessed=False, model_loaded=False,
                  hardware_accessed=False, results=[])
    code = 0
    for asset in selected:
        target = destination(root, asset)
        try:
            if args.command == 'plan':
                row = dict(asset, destination=str(target), status='manual_browser_download_required')
            elif args.command == 'import':
                row = import_verified(local_path(args.source), target, asset, lock,
                                      crc=args.crc, finite=args.finite)
            else:
                row = verify(target, asset, lock, crc=args.crc, finite=args.finite)
        except Incomplete as exc:
            row = dict(id=asset['id'], status='missing_or_in_progress', detail=str(exc))
            code = code or 2
        except (ValueError, OSError, zipfile.BadZipFile) as exc:
            # Do not print exception args: a caller may accidentally pass a URL
            # or a sensitive filename. Diagnostics use fixed stage/error types.
            row = dict(id=asset['id'], status='invalid_or_error', error_type=type(exc).__name__)
            code = 1
        report['results'].append(row)
    encoded = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)+'\n'
    if args.report:
        output = local_path(args.report)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open('x', encoding='utf-8') as stream:
            stream.write(encoded)
    print(encoded, end='')
    return code


if __name__ == '__main__':
    raise SystemExit(main())
