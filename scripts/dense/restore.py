"""Fresh four-rank full-state restore check for the dense Joint8 preflight."""
import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT), str(Path(__file__).parent)]
from runtime import fingerprint, load_run, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('config', 'run-root', 'dataset', 'source-sha', 'cache-root', 'teacher',
                 'teacher-source', 'weights', 'weight-sha'):
        parser.add_argument('--' + name, required=True)
    args = parser.parse_args()
    import torch
    from accelerate import Accelerator
    from fastwam.trainer import Wan22Trainer
    from fastwam.utils import misc
    from fastwam.utils.pytorch_utils import set_global_seed
    from fastwam.cosmos.persistence import resource_identity

    preflight_path = Path(args.run_root).resolve() / 'preflight.json'
    preflight = json.loads(preflight_path.read_text())
    if preflight.get('status') != 'two_update_checkpoint_verified':
        raise ValueError('Successful two-update preflight required')
    torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
    accelerator = Accelerator(gradient_accumulation_steps=1, mixed_precision='bf16',
                              step_scheduler_with_optimizer=False)
    if accelerator.num_processes != 4:
        raise RuntimeError('Four-rank restore required')
    cfg, model, dataset, cache = load_run(args.config, args.run_root, args.dataset,
        args.source_sha, args.cache_root, args.teacher, args.teacher_source,
        args.weights, args.weight_sha)
    rank = accelerator.process_index
    cfg.output_dir = str(Path(args.run_root).resolve() / 'restore' / f'rank_{rank:03d}')
    Path(cfg.output_dir).mkdir(parents=True, exist_ok=False)
    misc.register_work_dir(cfg.output_dir)
    set_global_seed(42)
    trainer = Wan22Trainer(model, dataset, None, cfg=cfg)
    expected = preflight['ranks'][rank]
    if expected['resource'] != resource_identity() or cache['identity_sha256'] != preflight['cache_identity']:
        raise ValueError('Restore resource or cache differs from preflight')
    trainer.load_training_state(expected['checkpoint']['state_path'])
    actual = fingerprint(trainer, model)
    report = dict(rank=rank, accepted=actual == expected['fingerprints'], actual=actual,
                  expected=expected['fingerprints'])
    write_json(Path(args.run_root).resolve() / 'restore' / f'rank_{rank:03d}.json', report)
    ranks = [None] * 4
    torch.distributed.all_gather_object(ranks, report)
    if accelerator.is_main_process:
        passed = all(item['accepted'] for item in ranks)
        write_json(Path(args.run_root).resolve() / 'restore' / 'restore.json', dict(
            status='fresh_four_rank_restore_verified' if passed else 'failed', ranks=ranks,
            cache_identity=cache['identity_sha256'], resource=resource_identity()))
        print('DENSE_JOINT8_RESTORE_' + ('PASSED' if passed else 'FAILED'), flush=True)
    accelerator.wait_for_everyone()
    torch.distributed.destroy_process_group()
    if not all(item['accepted'] for item in ranks):
        raise SystemExit(2)


if __name__ == '__main__':
    main()
