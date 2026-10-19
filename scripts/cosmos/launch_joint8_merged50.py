"""Foreground platform launcher for the user's independent 1k and 20k runs."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
BASE=Path('/inspire/hdd/project/long-range-embodied-collaborative-intelligence/ky26340')
SHARED=BASE/'realman-noaction-joint8'
SOURCE_SHA='e4feb413686fdcc44e1bc046e4510250d187ef0bd6ae1dc6c1c32e1aad02d60d'
WEIGHT=SHARED/'assets/base/base/post-trained/81edfebe-bd6a-4039-8c1d-737df1a790bf_ema_bf16.pt'


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--mode',choices=['short','long'],required=True)
    p.add_argument('--run-tag',required=True);p.add_argument('--batch-size',type=int,choices=[16,32,64],default=64)
    a=p.parse_args()
    if not os.environ.get('TRAIN_JOB_ID'):raise ValueError('Use the platform training job')
    if not a.run_tag.replace('-','').isalnum():raise ValueError('Simple unique run tag required')
    runbase=SHARED/'runs'/a.run_tag;runbase.mkdir(parents=True,exist_ok=True)
    prepared=runbase/'prepared';run=runbase/a.mode;exit_file=runbase/'short_exit.json'
    common=['--dataset',str(SHARED/'data/1020_cup_merged50'),'--source-sha',SOURCE_SHA,
            '--checkpoint',str(WEIGHT),'--batch-size',str(a.batch_size),'--run-root',str(run)]
    if a.mode=='short':
        command=[sys.executable,'-u',str(ROOT/'scripts/cosmos/train_realman.py'),*common,
            '--max-steps','1000','--work',str(prepared),
            '--reason1-checkpoint',str(BASE/'cosmos-reason1-upload-20261012'),
            '--tokenizer',str(BASE/'realman-dualwm/checkpoints/tokenizer.pth')]
    else:
        started=time.monotonic()
        while not (prepared/'prepared.json').is_file():
            if exit_file.is_file() and json.loads(exit_file.read_text())['returncode'] != 0:
                raise RuntimeError('Shared cache preparation failed; inspect the short job')
            if time.monotonic()-started>4*3600:raise TimeoutError('Shared cache was not prepared within four hours')
            print('WAITING_FOR_VERIFIED_SHARED_CACHE',str(prepared),flush=True);time.sleep(30)
        receipt=json.loads((prepared/'prepared.json').read_text())
        if receipt.get('status')!='complete' or receipt.get('source_sha256')!=SOURCE_SHA:
            raise ValueError('Shared cache preparation receipt differs')
        command=[sys.executable,'-u',str(ROOT/'scripts/cosmos/train_realman_cached.py'),*common,
            '--max-steps','20000','--prepared',str(prepared),'--work',str(runbase/'long-inputs')]
    print(json.dumps(dict(mode=a.mode,batch_per_rank=a.batch_size,global_batch=4*a.batch_size,
                         argv=command,source_sha256=SOURCE_SHA)),flush=True)
    result=subprocess.run(command,cwd=ROOT)
    if a.mode=='short':
        temp=exit_file.with_suffix('.tmp');temp.write_text(json.dumps(dict(returncode=result.returncode)))
        temp.replace(exit_file)
    raise SystemExit(result.returncode)


if __name__=='__main__':main()
