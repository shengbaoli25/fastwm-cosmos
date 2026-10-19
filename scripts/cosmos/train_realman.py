"""New real-robot run: verify source, encode caches, gate, train 30000 steps."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts/cosmos')]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('dataset','source-sha','work','run-root','reason1-checkpoint','checkpoint','tokenizer'):
        p.add_argument('--'+k,required=True)
    a=p.parse_args()
    from distributed_fresh import clean_environment
    env=clean_environment(os.environ,ROOT)
    os.environ.clear();os.environ.update(env)
    require=lambda ok,msg: None if ok else (_ for _ in ()).throw(ValueError(msg))
    require(bool(os.environ.get('TRAIN_JOB_ID')), 'Platform distributed training job required')
    work=Path(a.work);work.mkdir(parents=True,exist_ok=False)
    upstream=ROOT/'third_party/cosmos-predict2.5'
    from fastwam.cosmos.realman import RealManSource
    from fastwam.cosmos.cache import _atomic
    import torch
    torch.set_num_threads(4)
    s=RealManSource(a.dataset,a.source_sha)
    _atomic(work/'source_audit.json',dict(identity=s.identity,timing=s.timing))
    s.export_normalization(work/'normalization.json')
    _atomic(work/'prompts.json',sorted(set(s.prompts))); del s
    def run(phase,args,extra=None):
        _atomic(work/'status.json',dict(phase=phase,status='running',time=time.time()))
        print('REALMAN_PHASE='+phase,flush=True)
        e=dict(os.environ);e.update(extra or {})
        with (work/(phase+'.log')).open('w') as log:
            subprocess.run([sys.executable,'-u',*map(str,args)],env=e,check=True,stdout=log,stderr=subprocess.STDOUT,cwd=ROOT)
    run('text',[ROOT/'scripts/cosmos/precompute_text_sdpa.py','--prompts',work/'prompts.json',
        '--reason1-checkpoint',a.reason1_checkpoint,'--upstream',upstream,'--output',work/'text','--execute'],
        {'CUDA_VISIBLE_DEVICES':'0'})
    args=[ROOT/'scripts/cosmos/precompute_realman.py','--dataset',a.dataset,'--source-sha',a.source_sha,
        '--text-cache',work/'text','--tokenizer',a.tokenizer,'--upstream',upstream,
        '--output',work/'features','--stage',work/'encoder-stage']
    _atomic(work/'status.json',dict(phase='four_gpu_encoding',status='running',time=time.time()))
    workers=[]; logs=[]
    try:
        for rank in range(4):
            log=(work/f'encode_rank{rank}.log').open('w');logs.append(log)
            e=dict(os.environ,CUDA_VISIBLE_DEVICES=str(rank))
            workers.append(subprocess.Popen([sys.executable,'-u',*map(str,args),'--rank',str(rank)],
                env=e,stdout=log,stderr=subprocess.STDOUT,cwd=ROOT))
        while any(proc.poll() is None for proc in workers):
            require(all(proc.poll() in (None,0) for proc in workers),'A feature encoder failed; see rank logs')
            time.sleep(2)
        require(all(proc.returncode==0 for proc in workers),'Feature encoding failed')
    finally:
        for proc in workers:
            if proc.poll() is None:proc.terminate()
        for proc in workers:proc.wait()
        for log in logs:log.close()
    run('assemble',[*args,'--assemble'])
    run('resolve',[ROOT/'scripts/cosmos/resolve_config.py','--task','cosmos_noaction_joint_4gpu_30k',
        '--normalization',work/'normalization.json',
        '--text-cache',work/'text','--feature-cache',work/'features','--checkpoint',a.checkpoint,
        '--upstream',upstream,'--output-dir',work/'config-placeholder',
        '--config-output',work/'resolved.yaml'])
    # The task config selects official Cosmos initialization and fresh heads,
    # optimizer and schedule. No old-data checkpoint or normalization is reused.
    _atomic(work/'status.json',dict(phase='distributed_gate_and_train',status='running',time=time.time()))
    os.execv(sys.executable,[sys.executable,'-u',str(ROOT/'scripts/cosmos/distributed_chunk.py'),
        '--config',str(work/'resolved.yaml'),'--run-root',a.run_root,
        '--native-python',sys.executable,'--execute'])


if __name__=='__main__':main()
