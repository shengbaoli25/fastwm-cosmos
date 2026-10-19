"""Wait for the final step1000 save and attest its stable bytes, using CPU only."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import time


def signature(path):
    stat = path.stat()
    return stat.st_ino, stat.st_size, stat.st_mtime_ns


def digest(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def final_save_completed(text):
    # Rich consumes the optional [done] markup in rendered training logs.
    return re.search(r'\bmax_steps\s+reached\s+step=1000\b', text) is not None


def capture(runbase, output, *, timeout=14400):
    runbase = Path(runbase).resolve(); output = Path(output).resolve()
    weight = runbase / 'short/training/checkpoints/weights/step_001000.pt'
    log = runbase / 'short/formal-watch/training.log'
    cached_signature = cached_sha = None
    started = time.monotonic()
    while time.monotonic() - started < timeout:
        exit_file = runbase / 'short_exit.json'
        if exit_file.is_file() and json.loads(exit_file.read_text()).get('returncode') != 0:
            raise RuntimeError('Short training exited unsuccessfully; no final receipt issued')
        if weight.is_file() and not weight.with_suffix('.pt.tmp').exists():
            before = signature(weight)
            if before != cached_signature:
                sha = digest(weight)
                if signature(weight) != before or weight.with_suffix('.pt.tmp').exists():
                    continue
                cached_signature, cached_sha = before, sha
                print('STABLE_WEIGHT_HASHED', before[1], sha, flush=True)
            # This marker follows the final weight AND full optimizer save.
            text = log.read_text() if log.is_file() else ''
            final = final_save_completed(text)
            if final and signature(weight) == cached_signature:
                receipt = dict(schema='joint8-step1000-download-v1', step=1000,
                    training_complete=True, size_bytes=cached_signature[1], sha256=cached_sha,
                    source_path=str(weight), final_log_sha256=digest(log),
                    verified_at=time.strftime('%Y-%m-%d %H:%M:%S %z'))
                output.parent.mkdir(parents=True, exist_ok=True)
                temporary = output.with_suffix('.tmp')
                temporary.write_text(json.dumps(receipt, indent=2) + '\n')
                temporary.replace(output)
                print('FINAL_DOWNLOAD_RECEIPT', str(output), flush=True)
                return receipt
        time.sleep(1)
    raise TimeoutError('Final step1000 save was not completed within the bounded wait')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runbase', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    capture(args.runbase, args.output)


if __name__ == '__main__':
    main()
