"""Resume completed evaluations and release the card if a checkpoint is absent."""
import argparse
import datetime
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import time


def completed(out, step):
    path = out / 'protocol.json'
    if not path.is_file():
        return False
    data = json.loads(path.read_text())
    if data.get('status') != 'complete':
        return False
    if data.get('step') != step or data.get('completed_windows') != data.get('windows'):
        raise ValueError('Inconsistent completed evaluation: ' + str(out))
    return True


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('root', 'training-run', 'dataset', 'source-sha', 'tokenizer'):
        p.add_argument('--' + key, required=True)
    p.add_argument('--steps', type=int, nargs='+', default=[5000, 15000])
    p.add_argument('--wait-existing-pid', type=int)
    p.add_argument('--no-preview', action='store_true')
    p.add_argument('--defer-render',action='store_true',help='Release GPU after raw exports; render on local CPU')
    p.add_argument('--status-name', default='driver-status.json')
    a = p.parse_args()
    root, train, scripts = Path(a.root), Path(a.training_run), Path(__file__).parent
    root.mkdir(parents=True, exist_ok=True)
    status = root / a.status_name

    def state(**kw):
        kw.update(updated_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  job_id=os.environ.get('TRAIN_JOB_ID'), steps=a.steps)
        temp = status.with_suffix('.tmp')
        temp.write_text(json.dumps(kw, indent=2))
        temp.replace(status)
        print(json.dumps(kw), flush=True)

    def run(cmd):
        subprocess.run([sys.executable, '-u', *map(str, cmd)], check=True)

    try:
        # Reuse the active first evaluation only after its complete receipt.
        if a.wait_existing_pid:
            first = root / f'step{a.steps[0]:06d}'
            deadline = time.monotonic() + 2 * 3600
            while not completed(first, a.steps[0]):
                os.kill(a.wait_existing_pid, 0)
                if time.monotonic() > deadline:
                    raise TimeoutError('Existing evaluation did not complete')
                state(stage='waiting_active_evaluation', step=a.steps[0], pid=a.wait_existing_pid)
                time.sleep(30)
        for step in a.steps:
            out = root / f'step{step:06d}'
            ckpt = train / 'long/training/checkpoints/weights' / f'step_{step:06d}.pt'
            if completed(out, step):
                state(stage='reusing_completed', step=step)
                continue
            if out.exists():
                raise FileExistsError('Refuse to overwrite incomplete evaluation ' + str(out))
            if not ckpt.is_file():
                state(stage='pending_checkpoint_card_released', step=step)
                return
            signature = (ckpt.stat().st_size, ckpt.stat().st_mtime_ns)
            time.sleep(5)
            if signature != (ckpt.stat().st_size, ckpt.stat().st_mtime_ns):
                raise RuntimeError('Checkpoint is still changing')
            if step == 20000:
                receipts = list((train / 'long/training/origins').glob('rank_*.json'))
                if len(receipts) != 4 or not all(json.loads(r.read_text()).get('status') == 'complete'
                        and json.loads(r.read_text()).get('global_step') == step for r in receipts):
                    state(stage='pending_final_receipts_card_released', step=step)
                    return
            state(stage='evaluating', step=step)
            cmd=[scripts / 'evaluate.py', '--config', train / 'inputs/resolved.yaml',
                 '--dataset', a.dataset, '--source-sha', a.source_sha, '--checkpoint', ckpt,
                 '--expected-step', step, '--tokenizer', a.tokenizer, '--output', out]
            if a.defer_render:cmd.append('--skip-plots')
            run(cmd)
            if not a.no_preview and not a.defer_render and step == a.steps[0]:
                state(stage='rendering_preview', step=step)
                run([scripts / 'compare.py', '--runs', out, '--output', root / f'preview{step}'])
        if not a.defer_render:
            state(stage='rendering_comparison')
            run([scripts / 'compare.py', '--runs', *[root / f'step{s:06d}' for s in a.steps],
                 '--output', root / 'comparison'])
        logs = root / 'logs'
        logs.mkdir(exist_ok=True)
        shutil.copy2(train / 'long/formal-watch/training.log', logs / 'training-until-evaluation.log')
        shutil.copy2(train / 'inputs/resolved.yaml', logs / 'resolved.yaml')
        name = 'results-' + '-'.join(str(s) for s in a.steps) + ('-predictions.tar.gz' if a.defer_render else '-complete.tar.gz')
        with tarfile.open(root / name, 'w:gz') as tar:
            for item in [*[f'step{s:06d}' for s in a.steps], *([] if a.defer_render else ['comparison']), 'logs']:
                tar.add(root / item, arcname=item)
        state(stage='predictions_complete' if a.defer_render else 'complete', archive=name)
    except BaseException as error:
        state(stage='failed', error=repr(error))
        raise


if __name__ == '__main__':
    main()
