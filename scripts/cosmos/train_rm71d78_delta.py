"""All 78 demonstrations, triangular V/F/A attention, current-state joint deltas.

Reuse verified official assets and the existing text bank. Build new all-episode
video/numeric caches and statistics, then run the existing four-rank update and
full-state restore gates before a fresh 30,000-update run. The learning-rate
schedule retains the previous 50,000-step horizon. No old trained weights or
optimizer states initialize this experiment.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
SSD = Path('/inspire/ssd/project/long-range-embodied-collaborative-intelligence/ky26340')
HDD = Path('/inspire/hdd/project/long-range-embodied-collaborative-intelligence/ky26340')
OLD = SSD / 'rm71d-jepa-cosmos-50k-20261029'
OLD_RUN = OLD / 'cosmos/online/job-187fc8ee-5443-4786-963b-d48882d3f8ff'
RAW_SHA = 'b24b701239f54ad38d16c819b9ea9f6fad5f9374a683a84a68d2cd7ef27067a8'
REPRESENTATION = 'delta_current_joint7'


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--dataset', type=Path, default=OLD/'data/rm71d_plug_transfer_20261029')
    parser.add_argument('--text-cache', type=Path, default=OLD_RUN/'prepared/text')
    parser.add_argument('--original-config', type=Path, default=OLD_RUN/'runs/long/training/config.yaml')
    parser.add_argument('--upstream', type=Path, default=HDD/'realman-noaction-joint8/code-82cb9d2/third_party/cosmos-predict2.5')
    parser.add_argument('--tokenizer', type=Path, default=HDD/'realman-dualwm/checkpoints/tokenizer.pth')
    parser.add_argument('--checkpoint', type=Path, default=HDD/'realman-noaction-joint8/assets/base/base/post-trained/81edfebe-bd6a-4039-8c1d-737df1a790bf_ema_bf16.pt')
    parser.add_argument('--execute', action='store_true')
    return parser.parse_args()


def verify_original_recipe(cfg):
    expected = dict(batch_size=64, num_workers=8, learning_rate=1e-4,
                    weight_decay=.01, max_steps=50000, lr_scheduler_type='cosine',
                    gradient_accumulation_steps=1, mixed_precision='bf16',
                    seed=42, save_every=5000, log_every=10, eval_every=0,
                    max_grad_norm=1., resume=None)
    if any(cfg.get(key) != value for key, value in expected.items()):
        raise ValueError('The previous 4-H200 B64/50k recipe has changed')
    expected_model = dict(_target_='fastwam.cosmos.assets.create_noaction_joint_cosmos',
                          action_hidden=768, action_ffn=3072, force_hidden=768,
                          force_ffn=2048, force_heads=16, force_shift=1.,
                          action_seed=42, force_seed=43, memory_seed=44,
                          action_space='joint8', fps=7.5, num_inference_steps=16,
                          gradient_checkpointing=True)
    if any(cfg['model'].get(key) != value for key, value in expected_model.items()):
        raise ValueError('The previous model hyperparameters have changed')


def main():
    args = arguments()
    plan = dict(source=str(ROOT), dataset=str(args.dataset), source_sha256=RAW_SHA,
                source_episodes=78, split_counts=dict(train=78, val=0, test=0),
                action_representation=REPRESENTATION, branch_reads=[[1,0,0],[1,1,0],[1,1,1]],
                batch_per_rank=64, world_size=4, global_batch=256,
                max_steps=30000, scheduler_total_steps=50000,
                warmup_steps=2500, cosine_steps=47500,
                loss_weights=dict(video=1., force=1., action=1.),
                official_initialization=True, optimizer_resume=False,
                existing_text=str(args.text_cache), output_root=str(args.output_root))
    print(json.dumps(plan, indent=2), flush=True)
    if not args.execute:
        return
    if not os.environ.get('TRAIN_JOB_ID'):
        raise ValueError('Run in a new four-H200 platform training job')
    if not args.output_root.is_absolute() or args.output_root.exists():
        raise ValueError('A fresh absolute output root is required')
    sys.path[:0] = [str(ROOT/'src'), str(ROOT), str(ROOT/'scripts/cosmos')]
    from distributed_fresh import clean_environment
    env = clean_environment(os.environ, ROOT)
    env['NCCL_NVLS_ENABLE'] = '0'
    os.environ.clear()
    os.environ.update(env)
    import torch
    from omegaconf import OmegaConf
    from fastwam.cache.latent import file_sha256
    from fastwam.cosmos.assets import checked, TOKENIZER_SHA, NONACTION_WEIGHT_SHA, read_text_bank
    from fastwam.cosmos.cache import _atomic
    from fastwam.cosmos.noaction import BRANCH_READS
    from fastwam.cosmos.realman_joint import open_realman_source
    from fastwam.cosmos.native import verify_upstream
    from fastwam.cosmos.persistence import resource_identity
    torch.set_num_threads(4)
    if BRANCH_READS != ((1,0,0),(1,1,0),(1,1,1)):
        raise ValueError('Triangular no-action-conditioned V/F/A attention required')
    resource = resource_identity()
    verify_original_recipe(OmegaConf.to_container(OmegaConf.load(args.original_config), resolve=True))
    verify_upstream(args.upstream)
    checked(args.tokenizer, TOKENIZER_SHA)
    checked(args.checkpoint, NONACTION_WEIGHT_SHA)
    args.output_root.mkdir(parents=True, exist_ok=False)
    prepared = args.output_root/'prepared'
    prepared.mkdir()
    _atomic(args.output_root/'experiment.json', dict(plan, resource=resource,
            original_config_sha256=file_sha256(args.original_config)))

    def status(phase, state='running', **extra):
        receipt = dict(phase=phase, status=state, updated_at=time.time(), **extra)
        _atomic(args.output_root/'pipeline-status.json', receipt)
        print(json.dumps(receipt), flush=True)

    status('audit_all78_delta_source')
    source = open_realman_source(args.dataset, RAW_SHA, action_representation=REPRESENTATION)
    if (len(source.episodes) != 78 or source.identity['source_episode_indices'] != list(range(78))
            or source.identity['split_counts'] != dict(train=78, val=0, test=0)):
        raise ValueError('All 78 original demonstrations must be training sources')
    bank, text_manifest, _ = read_text_bank(args.text_cache)
    if [x['prompt'] for x in text_manifest['texts']] != sorted(set(source.prompts)):
        raise ValueError('Reused text bank does not cover the recorded prompts')
    del bank
    _atomic(prepared/'source_audit.json', dict(identity=source.identity, timing=source.timing,
                                             training_anchors=len(source)))
    source.export_normalization(prepared/'normalization.json')
    del source
    (prepared/'text').symlink_to(args.text_cache.resolve(), target_is_directory=True)

    encoder_args = [str(ROOT/'scripts/cosmos/precompute_realman.py'), '--dataset',str(args.dataset),
                    '--source-sha',RAW_SHA, '--text-cache',str(prepared/'text'),
                    '--tokenizer',str(args.tokenizer), '--upstream',str(args.upstream),
                    '--output',str(prepared/'features'), '--stage',str(prepared/'encoder-stage'),
                    '--action-representation',REPRESENTATION, '--expected-episodes','78']
    workers, logs = [], []
    status('encode_all78_on_four_gpus')
    try:
        for rank in range(4):
            log = (prepared/f'encode_rank{rank}.log').open('w')
            logs.append(log)
            workers.append(subprocess.Popen([sys.executable,'-u',*encoder_args,'--rank',str(rank)],
                env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(rank)), stdout=log,
                stderr=subprocess.STDOUT,cwd=ROOT))
        while any(proc.poll() is None for proc in workers):
            if any(proc.poll() not in (None,0) for proc in workers):
                raise RuntimeError('An encoder failed; see encode_rank logs')
            time.sleep(2)
        if any(proc.returncode != 0 for proc in workers):
            raise RuntimeError('Feature encoding failed')
    except BaseException as error:
        status('encode_all78_on_four_gpus','failed',error=str(error))
        raise
    finally:
        for proc in workers:
            if proc.poll() is None:
                proc.terminate()
        for proc in workers:
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
        for log in logs:
            log.close()
    status('assemble_and_verify_all78_cache')
    with (prepared/'assemble.log').open('w') as log:
        subprocess.run([sys.executable,'-u',*encoder_args,'--assemble'], cwd=ROOT,
                       stdout=log,stderr=subprocess.STDOUT,check=True)
    status('gpu_update_restore_gates_then_fresh_training')
    os.execv(sys.executable,[sys.executable,'-u',str(ROOT/'scripts/cosmos/train_realman_cached.py'),
        '--dataset',str(args.dataset),'--source-sha',RAW_SHA,'--prepared',str(prepared),
        '--work',str(args.output_root/'resolve'),'--run-root',str(args.output_root/'run'),
        '--checkpoint',str(args.checkpoint),'--upstream',str(args.upstream),
        '--batch-size','64','--max-steps','30000','--scheduler-total-steps','50000',
        '--action-representation',REPRESENTATION,'--expected-episodes','78'])


if __name__ == '__main__':
    main()
