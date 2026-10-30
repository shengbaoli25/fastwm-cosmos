"""Start the configured fresh BF16 Joint8 run after all gates pass."""
import argparse
import json
import logging
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT), str(Path(__file__).parent)]
from runtime import load_run, run_contract, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('config', 'run-root', 'dataset', 'source-sha', 'cache-root', 'teacher',
                 'teacher-source', 'weights', 'weight-sha'):
        parser.add_argument('--' + name, required=True)
    args = parser.parse_args()
    import torch
    from accelerate import Accelerator
    from fastwam.cosmos.jepa_trainer import JepaVideoTrainer as Wan22Trainer
    from fastwam.utils import misc
    from fastwam.utils.logging_config import setup_logging
    from fastwam.utils.pytorch_utils import set_global_seed
    from fastwam.cosmos.persistence import resource_identity

    run_root = Path(args.run_root).resolve()
    gate = json.loads((run_root / 'preflight.json').read_text())
    restore = json.loads((run_root / 'restore/restore.json').read_text())
    if gate.get('status') != 'two_update_checkpoint_verified' or restore.get('status') != 'fresh_four_rank_restore_verified':
        raise ValueError('Both four-rank gates must pass before formal training')
    torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
    accelerator = Accelerator(gradient_accumulation_steps=1, mixed_precision='bf16',
                              step_scheduler_with_optimizer=False)
    if accelerator.num_processes != 4 or torch.cuda.device_count() != 4:
        raise RuntimeError('Exactly four GPU ranks required')
    cfg, model, dataset, cache = load_run(args.config, run_root, args.dataset,
        args.source_sha, args.cache_root, args.teacher, args.teacher_source,
        args.weights, args.weight_sha)
    if (cache['identity_sha256'] != gate['cache_identity']
            or restore['cache_identity'] != gate['cache_identity']
            or any(item.get('run_contract') != run_contract(cfg, cache) for item in gate['ranks'])):
        raise ValueError('Formal training config or cache differs from the gate')
    rank = accelerator.process_index
    cfg.output_dir = str(run_root / 'training')
    Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)
    misc.register_work_dir(cfg.output_dir)
    set_global_seed(42)
    setup_logging(log_level=logging.INFO, is_main_process=rank == 0)
    trainer = Wan22Trainer(model, dataset, None, cfg=cfg)
    if trainer.max_steps != int(cfg.max_steps) or trainer.batch_size != 64 or trainer.eval_every != 0:
        raise ValueError('Formal training must be B64/rank for the configured number of updates')
    report = dict(status='formal_training_starting', teacher=args.teacher,
        source_sha256=args.source_sha, cache_identity=cache['identity_sha256'],
        resource=resource_identity(), config=cfg.output_dir, max_steps=trainer.max_steps,
        batch_per_rank=trainer.batch_size, global_batch=trainer.batch_size*accelerator.num_processes,
        fresh_initialization=True, optimizer_resume=False, model_warm_start=False)
    write_json(run_root / 'training' / f'origin_rank_{rank:03d}.json', report)
    if rank == 0:
        from omegaconf import OmegaConf
        OmegaConf.save(cfg, str(run_root / 'training' / 'config.yaml'))
        write_json(run_root / 'training' / 'start.json', report)
    accelerator.wait_for_everyone()
    trainer.train()
    if trainer.global_step != int(cfg.max_steps):
        raise RuntimeError('Formal run ended before the configured update count')
    report.update(status='complete', global_step=trainer.global_step)
    write_json(run_root / 'training' / f'origin_rank_{rank:03d}.json', report)
    accelerator.wait_for_everyone()
    torch.distributed.destroy_process_group()


if __name__ == '__main__':
    main()
