"""Four-H200, two-update, save and complete-state preflight for dense Joint8."""
import argparse
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT), str(Path(__file__).parent)]
from runtime import ExpertGradientProbe, fingerprint, load_run, run_contract, write_json


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
    from fastwam.utils.pytorch_utils import set_global_seed
    from fastwam.cosmos.persistence import resource_identity

    if not torch.cuda.is_available() or torch.cuda.device_count() != 4:
        raise RuntimeError('Exactly four H200 devices required')
    torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
    accelerator = Accelerator(gradient_accumulation_steps=1, mixed_precision='bf16',
                              step_scheduler_with_optimizer=False)
    if accelerator.num_processes != 4 or accelerator.state.deepspeed_plugin is None:
        raise RuntimeError('Four-rank DeepSpeed is required')
    cfg, model, dataset, cache = load_run(args.config, args.run_root, args.dataset,
        args.source_sha, args.cache_root, args.teacher, args.teacher_source,
        args.weights, args.weight_sha)
    rank = accelerator.process_index
    cfg.output_dir = str(Path(args.run_root).resolve() / f'rank_{rank:03d}')
    Path(cfg.output_dir).mkdir(parents=True, exist_ok=False)
    misc.register_work_dir(cfg.output_dir)
    set_global_seed(42)
    trainer = Wan22Trainer(model, dataset, None, cfg=cfg)
    engine = trainer.model
    zero = trainer.accelerator.state.deepspeed_plugin.deepspeed_config['zero_optimization']
    if (zero['stage'] != 1 or engine.train_micro_batch_size_per_gpu() != 64
            or engine.gradient_accumulation_steps() != 1 or engine.train_batch_size() != 256
            or trainer.max_steps != int(cfg.max_steps) or trainer.eval_every != 0):
        raise RuntimeError('Actual engine differs from configured four-H200 B64/rank recipe')
    if torch.cuda.get_device_name() != 'NVIDIA H200' and 'H200' not in torch.cuda.get_device_name():
        raise RuntimeError('Allocated GPU is not H200')
    started = time.monotonic()
    updates = []
    iterator = iter(trainer.train_loader)
    for _ in range(2):
        sample = next(iterator)
        trainer.batch_in_epoch += 1
        with trainer.accelerator.accumulate(trainer.model):
            with trainer.accelerator.autocast():
                loss, components = trainer.model.training_loss(sample)
            if not bool(torch.isfinite(loss.detach()).all()):
                raise FloatingPointError('Nonfinite dense Joint8 loss')
            base_model = trainer.accelerator.unwrap_model(trainer.model)
            # Accelerate calls DeepSpeed engine.step inside backward at an
            # accumulation boundary.  Hooks capture gradients before ZeRO
            # reduces/clears them, without changing the stock update sequence.
            gradient_probe = ExpertGradientProbe(base_model)
            try:
                trainer.accelerator.backward(loss)
                expert_grad_norms = gradient_probe.norms()
            finally:
                gradient_probe.close()
            if not trainer.accelerator.sync_gradients:
                raise RuntimeError('Accumulation must be one')
            grad_norm = trainer.accelerator.clip_grad_norm_(trainer.model.parameters(), trainer.max_grad_norm)
            trainer.optimizer.step()
            skipped = bool(trainer.accelerator.optimizer_step_was_skipped)
            if not skipped:
                trainer.scheduler.step()
            trainer.optimizer.zero_grad(set_to_none=True)
            trainer.global_step += 1
            engine_steps = int(engine.global_steps)
            if engine_steps != trainer.global_step:
                raise RuntimeError(
                    f'DeepSpeed engine/global trainer steps diverged: {engine_steps} != {trainer.global_step}')
        torch.cuda.synchronize()
        updates.append(dict(step=trainer.global_step, loss=float(loss.detach().float()),
            components=components, grad_norm=float(grad_norm),
            expert_grad_norms=expert_grad_norms, skipped=skipped,
            seconds=time.monotonic()-started, sample_index=sample['sample_index'].cpu().tolist()))
    if any(update['skipped'] for update in updates):
        raise RuntimeError('AdamW skipped a preflight update')
    trainer.checkpoint_root = str(Path(args.run_root).resolve() / 'checkpoint')
    trainer.weights_dir = str(Path(trainer.checkpoint_root) / 'weights')
    trainer.state_dir = str(Path(trainer.checkpoint_root) / 'state')
    Path(trainer.weights_dir).mkdir(parents=True, exist_ok=True)
    Path(trainer.state_dir).mkdir(parents=True, exist_ok=True)
    saved = trainer.save_checkpoint()
    report = dict(rank=rank, resource=resource_identity(), updates=updates,
        actual_batch_per_rank=engine.train_micro_batch_size_per_gpu(),
        actual_global_batch=engine.train_batch_size(), max_steps=trainer.max_steps,
        checkpoint=saved, fingerprints=fingerprint(trainer, model),
        dataset_source_sha256=args.source_sha, feature_cache_identity=cache['identity_sha256'],
        teacher=args.teacher, run_contract=run_contract(cfg, cache),
        architecture_counts={name: sum(p.numel() for p in module.parameters())
                             for name, module in model.dit.items()},
        trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
        gpu=torch.cuda.get_device_name(), peak_allocated_bytes=torch.cuda.max_memory_allocated())
    write_json(Path(args.run_root) / f'rank_{rank:03d}.json', report)
    ranks = [None] * 4
    torch.distributed.all_gather_object(ranks, report)
    if accelerator.is_main_process:
        same_resource = all(item['resource'] == ranks[0]['resource'] for item in ranks)
        same_data = all(item['feature_cache_identity'] == ranks[0]['feature_cache_identity'] for item in ranks)
        passed = same_resource and same_data and all(len(item['updates']) == 2 for item in ranks)
        write_json(Path(args.run_root) / 'preflight.json', dict(
            status='two_update_checkpoint_verified' if passed else 'failed', ranks=ranks,
            formal_training_started=False, cache_identity=ranks[0]['feature_cache_identity']))
        print(json.dumps({'status': 'two_update_checkpoint_verified' if passed else 'failed',
                          'seconds_per_update': max(x['updates'][-1]['seconds'] for x in ranks)},
                         ensure_ascii=True), flush=True)
    accelerator.wait_for_everyone()
    torch.distributed.destroy_process_group()


if __name__ == '__main__':
    main()
