"""Actual four-rank configured-batch two-Adam probe; no tiny-model substitute."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import time
from fastwam.utils.training_schedule import resolve_scheduler_total_steps

ROOT=Path(__file__).resolve().parents[3]
REFERENCE_TARGET = 'fastwam.cosmos.reference_model.create_reference_joint_cosmos'
REFERENCE_DATASET = 'fastwam.cosmos.reference_dataset.ReferenceCachedDataset'


def validate_config(cfg):
    noaction = cfg['model']['_target_'] == 'fastwam.cosmos.assets.create_noaction_joint_cosmos'
    reference = cfg['model']['_target_'] == REFERENCE_TARGET
    joint8 = noaction or reference
    batch,steps = cfg.get('batch_size'),cfg.get('max_steps')
    if noaction and (batch not in (16,32,64) or steps not in (1000,20000,30000)):
        raise ValueError('Joint8 requires an authorized batch16/32/64 and 1000/20000/30000 schedule')
    if reference and (batch != 64 or steps != 30000 or cfg.get('scheduler_total_steps') != 50000):
        raise ValueError('Reference experiment requires batch64, stop30000 and original scheduler50000')
    expected={'batch_size':batch if joint8 else 16,'gradient_accumulation_steps':1,
        'max_steps':steps if joint8 else 30000,'save_every':1000 if noaction and steps == 1000 else 5000,
        'learning_rate':1e-4,'weight_decay':.01,'mixed_precision':'bf16',
        'seed':42,'resume':None,'lr_scheduler_type':'cosine','eval_every':0}
    if any(cfg.get(k)!=v for k,v in expected.items()):
        raise ValueError('Exact configured batch/accum1/schedule/save recipe required')
    if cfg['data'].get('val') is not None:
        raise ValueError('The inherited recipe has no separate validation split')
    m=cfg['model'];d=cfg['data']['train']
    horizon = resolve_scheduler_total_steps(steps, cfg.get('scheduler_total_steps'))
    if horizon != steps and (not joint8 or horizon != 50000):
        raise ValueError('Only the baseline-derived Joint8 models may preserve the original 50000-step scheduler')
    if cfg.get('scheduler_total_steps') is not None:
        recipe = m.get('training_recipe', {})
        expected_recipe = dict(batch_per_rank=batch, max_steps=steps, scheduler_total_steps=horizon)
        if any(recipe.get(key) != value for key, value in expected_recipe.items()):
            raise ValueError('Model training_recipe must bind batch, stop max_steps and scheduler_total_steps')
    else:
        recipe_horizon = m.get('training_recipe', {}).get('scheduler_total_steps')
        if recipe_horizon is not None and recipe_horizon != horizon:
            raise ValueError('Model training_recipe scheduler horizon differs from the trainer')
    if joint8:
        required = dict(action_hidden=768, action_ffn=3072, force_hidden=768, force_ffn=2048,
                        force_heads=16, force_shift=1., num_inference_steps=16,
                        action_seed=42, force_seed=43, memory_seed=44, action_space='joint8')
        if (any(m.get(k) != v for k, v in required.items())
                or any(k in m for k in ('control_seed', 'initial_checkpoint', 'action_condition'))
                or d['_target_'] != (REFERENCE_DATASET if reference else 'fastwam.cosmos.cache.CosmosCachedDataset')
                or len(d.get('expected_identity_sha256', '')) != 64
                or len(m.get('expected_text_identity', '')) != 64 or not d.get('require_production')
                or m.get('expected_data_identity') != d['expected_identity_sha256']):
            raise ValueError('Bound baseline-derived Cosmos/AE768/Force768 diffusion configuration required')
        if reference:
            required_reference = dict(reference_hidden=128, reference_heads=4, reference_layers=2,
                                      reference_seed=46, alignment_weight=.1,
                                      action_representation='delta_current_joint7')
            if (any(m.get(k) != v for k, v in required_reference.items())
                    or d.get('expected_action_representation') != 'delta_current_joint7'
                    or len(m.get('expected_reference_identity', '')) != 64
                    or m['expected_reference_identity'] != d.get('expected_reference_identity')
                    or len(m.get('expected_normalization_sha256', '')) != 64
                    or m['expected_text_identity'] != d.get('expected_text_identity')):
                raise ValueError('Bound observation-only variable-plan/local32 reference configuration required')
        return
    if (m['_target_'] not in ('fastwam.cosmos.assets.create_cosmos_forcewm',
                              'fastwam.cosmos.assets.create_joint_cosmos_forcewm')
            or d['_target_']!='fastwam.cosmos.cache.CosmosCachedDataset'
            or len(d.get('expected_identity_sha256',''))!=64
            or not d.get('require_production') or len(m.get('expected_text_identity',''))!=64
            or m['generated_video_steps']!=16 or m['force_hidden']!=1280
            or m['force_ffn']!=4608 or m['force_heads']!=16
            or m['action_hidden']!=1024 or m['action_ffn']!=4096
            or m['memory_width']!=512 or m['memory_heads']!=8
            or m.get('control_seed')!=45
            or (m['action_seed'],m['force_seed'],m['memory_seed'])!=(42,43,44)):
        raise ValueError('Bound production Cosmos28/AE28/Force28 configuration required')



def helpers():
    spec=importlib.util.spec_from_file_location('baseline_probe_helpers',ROOT/'scripts/forcevla/preflight_cached_gpu.py')
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    return mod


def build_probe_datasets(data_cfg):
    """Preserve the stock no-split contract: val aliases the training object."""
    from fastwam import runtime
    if data_cfg.get('val') is not None:
        raise ValueError('The inherited recipe has no separate validation split')
    train_ds, val_ds = runtime.build_datasets(data_cfg)
    if val_ds is not train_ds:
        raise RuntimeError('Stock no-split validation must alias the training dataset')
    return train_ds, val_ds


def verify_trainer_schedule(trainer, cfg, *, step=0):
    """Verify actual scheduler geometry at preflight, formal start and restore."""
    stop = int(cfg['max_steps'])
    horizon = resolve_scheduler_total_steps(stop, cfg.get('scheduler_total_steps'))
    warmup = int(horizon * .05)
    expected = dict(last_epoch=step, warmup_milestones=[warmup], cosine_steps=horizon-warmup)
    actual = helpers().inspect_scheduler(trainer.scheduler)
    if (trainer.max_steps != stop or trainer.scheduler_total_steps != horizon or actual != expected):
        raise RuntimeError('Actual stop limit or learning rate schedule differs from configured horizons')
    return dict(max_steps=stop, scheduler_total_steps=horizon, scheduler=actual)


def execute(config_path,output):
    import torch
    from accelerate import Accelerator
    from hydra.utils import instantiate
    from omegaconf import OmegaConf
    from fastwam import runtime
    from fastwam.trainer import Wan22Trainer
    from fastwam.utils import misc
    from fastwam.utils.pytorch_utils import set_global_seed

    start=time.monotonic(); cfg=OmegaConf.load(config_path)
    validate_config(OmegaConf.to_container(cfg,resolve=True))
    if not torch.cuda.is_available():raise RuntimeError('Actual CUDA hardware required')
    torch.cuda.set_device(int(os.environ.get('LOCAL_RANK','0')))
    torch.cuda.reset_peak_memory_stats()
    accelerator=Accelerator(gradient_accumulation_steps=1,mixed_precision='bf16',step_scheduler_with_optimizer=False)
    if accelerator.num_processes!=4 or accelerator.state.deepspeed_plugin is None:
        raise RuntimeError('Actual four DeepSpeed ranks required')
    output=Path(output)
    if accelerator.is_main_process:output.mkdir(parents=True,exist_ok=False)
    accelerator.wait_for_everyone()
    rank=accelerator.process_index;rank_dir=output/f'rank_{rank:03d}';rank_dir.mkdir()
    cfg.output_dir=str(rank_dir);misc.register_work_dir(str(rank_dir));set_global_seed(42)
    cfgfile=OmegaConf.to_container(cfg,resolve=True)
    (rank_dir/'config.json').write_text(json.dumps(cfgfile,indent=2))
    model=instantiate(cfg.model,device=runtime._resolve_train_device())
    if len(model.video.blocks)!=28 or len(model.dit['action'].blocks)!=28 or len(model.dit['force'].blocks)!=28:
        raise ValueError('Actual Cosmos/AE/Force depth must all be28')
    if hasattr(model,'vae'):raise ValueError('Training model must have no VAE')
    train_ds,val_ds=build_probe_datasets(cfg.data)
    from .cache import validate
    dataset_receipt=validate(train_ds.root,train_ds.identity_sha256)
    reference = cfg.model._target_ == REFERENCE_TARGET
    if reference:
        receipt = train_ds.reference_binding
        if (receipt.get('status') != 'bound'
                or receipt.get('reference_identity_sha256') != cfg.model.expected_reference_identity
                or receipt.get('cache_identity_sha256') != train_ds.identity_sha256
                or model.reference_mode != 'obs_only_full_plan_soft_local32'
                or model.dit['reference_auditor'].local_tokens != 32):
            raise ValueError('Actual reference/cache binding or observation-only local32 model differs')
    trainer=Wan22Trainer(model,train_ds,val_ds,cfg=cfg)
    engine=trainer.model
    zero=trainer.accelerator.state.deepspeed_plugin.deepspeed_config['zero_optimization']
    batch,steps=int(cfg.batch_size),int(cfg.max_steps)
    if (zero['stage']!=1 or engine.train_micro_batch_size_per_gpu()!=batch
            or engine.gradient_accumulation_steps()!=1 or engine.train_batch_size()!=4*batch
            or any(zero.get(k,{}).get('device','none')!='none' for k in ('offload_optimizer','offload_param'))):
        raise RuntimeError('Actual engine differs from configured four-rank batch and ZeRO1')
    h=helpers();adam=h.unwrap_optimizer(trainer.optimizer)
    if not isinstance(adam,torch.optim.AdamW):raise RuntimeError('Actual AdamW required')
    schedule = verify_trainer_schedule(trainer, cfg)
    layer_calls=[0]*28;handles=[];query_gradients=[];reference_gradients={}
    def record_query_gradient(gradient):
        query_gradients.append({'finite':bool(torch.isfinite(gradient).all()),
                                'all_queries_nonzero':bool(gradient.float().abs().sum(-1).gt(0).all())})
        return gradient
    force_probe = (model.dit['force'].force_encoder.weight if hasattr(model.dit['force'], 'force_encoder')
                   else model.dit['force'].queries)
    handles.append(force_probe.register_hook(record_query_gradient))
    if reference:
        # Probe both the auditor and its AE adapter: alignment CE alone could
        # otherwise hide a disconnected action-conditioning path.
        probes = dict(auditor=model.dit['reference_auditor'].reference_embedding[0].weight,
                      action_condition=model.dit['reference_condition'][-1].weight)
        for name, parameter in probes.items():
            reference_gradients[name] = []
            def record_reference_gradient(gradient, name=name):
                reference_gradients[name].append(dict(finite=bool(torch.isfinite(gradient).all()),
                    nonzero=bool(gradient.float().abs().sum() > 0)))
                return gradient
            handles.append(parameter.register_hook(record_reference_gradient))
    for i,block in enumerate(model.dit['force'].blocks):
        def count(module,args,out,i=i):layer_calls[i]+=1
        handles.append(block.register_forward_hook(count))
    torch.cuda.synchronize();startup=time.monotonic()-start
    from .probe_updates import run_two_updates
    losses,updates=run_two_updates(trainer,synchronize=torch.cuda.synchronize)
    for handle in handles:handle.remove()
    report={'rank':rank,'world_size':4,'batch_per_rank':batch,'global_batch':4*batch,'max_steps':steps,
        'scheduler_total_steps':schedule['scheduler_total_steps'],
        'actual_engine_updates':int(engine.global_steps),'scheduler':h.inspect_scheduler(trainer.scheduler),
        **h.inspect_adam(adam),'losses':losses,'updates':updates,'chunk_layer_calls':layer_calls,'query_gradients':query_gradients,
        'force_gradient_target':'input_encoder' if hasattr(model.dit['force'], 'force_encoder') else 'learned_queries',
        'all_parameters_finite':all(h._finite(p) for p in model.parameters()),
        'startup_seconds':startup,'peak_allocated_gib':torch.cuda.max_memory_allocated()/2**30,
        'peak_reserved_gib':torch.cuda.max_memory_reserved()/2**30,
        'gpu':torch.cuda.get_device_name(),'architecture_counts':{k:sum(p.numel() for p in v.parameters()) for k,v in model.dit.items()},
        'dataset':dataset_receipt,'official_initialization':model.manifest,
        'memory_rng_state':model.dit['runtime'].get_state().tolist(),
        'formal_training_started':False,'checkpoint_saved':False,'distributed_resume_verified':False}
    if reference:
        report.update(reference_binding=train_ds.reference_binding,
                      reference_gradients=reference_gradients,
                      reference_augmentation=train_ds.reference_augmentation_config)
    accepted=(report['actual_engine_updates']==2 and report['adam_state_steps']
        and all(x==2 for x in report['adam_state_steps']) and report['all_parameters_finite']
        and report['adam_master_and_moments_finite'] and min(layer_calls)>=2
        and len(query_gradients)==2 and all(g['finite'] and g['all_queries_nonzero'] for g in query_gradients)
        and all(not u['optimizer_step_was_skipped'] for u in updates))
    if reference:
        accepted = accepted and all(len(values) == 2 and all(g['finite'] and g['nonzero'] for g in values)
                                    for values in reference_gradients.values())
    from .persistence import resource_identity,training_state_fingerprints
    report['resource']=resource_identity()
    trainer.checkpoint_root=str(output/'checkpoint')
    trainer.weights_dir=str(output/'checkpoint/weights')
    trainer.state_dir=str(output/'checkpoint/state')
    if accelerator.is_main_process:
        Path(trainer.weights_dir).mkdir(parents=True)
        Path(trainer.state_dir).mkdir(parents=True)
    accelerator.wait_for_everyone();save_start=time.monotonic()
    saved=trainer.save_checkpoint();torch.cuda.synchronize()
    report['checkpoint_saved']=True
    report['save_seconds']=time.monotonic()-save_start
    report['state_path']=saved['state_path']
    report['state_fingerprints']=training_state_fingerprints(trainer,model)
    report['accepted']=bool(accepted)
    from .binding import binding
    report['binding']=binding(config_path)
    (rank_dir/'preflight.json').write_text(json.dumps(report,indent=2,allow_nan=False))
    ranks=[None]*4;torch.distributed.all_gather_object(ranks,report)
    if accelerator.is_main_process:
        accepted=all(r['accepted'] for r in ranks) and all(r['memory_rng_state']==ranks[0]['memory_rng_state'] for r in ranks)
        warm=max(r['updates'][-1]['seconds'] for r in ranks)
        result={'binding':binding(config_path),'resource':report['resource'],'status':'two_adam_verified' if accepted else 'failed','ranks':ranks,
            'warm_seconds_per_step':warm,'training_seconds_estimate':warm*steps,
            'save_eval_budget_verified':False,'formal_training_started':False,
            'checkpoint_saved':True,'distributed_resume_verified':False}
        result['checkpoint_bytes']=sum(p.stat().st_size for p in (output/'checkpoint').rglob('*') if p.is_file())
        result['max_save_seconds']=max(r['save_seconds'] for r in ranks)
        (output/'preflight.json').write_text(json.dumps(result,indent=2,allow_nan=False))
        print(json.dumps({k:v for k,v in result.items() if k!='ranks'}),flush=True)
    accelerator.wait_for_everyone()
    ok=json.loads((output/'preflight.json').read_text())['status']=='two_adam_verified'
    torch.distributed.destroy_process_group()
    return ok


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True);parser.add_argument('--output',required=True)
    parser.add_argument('--execute',action='store_true');args=parser.parse_args()
    if not args.execute:
        from omegaconf import OmegaConf
        cfg = OmegaConf.to_container(OmegaConf.load(args.config), resolve=True)
        validate_config(cfg)
        print(json.dumps({'mode':'plan_only','world_size':4,'B':cfg['batch_size'],
                          'global_batch':4*cfg['batch_size'],'schedule':cfg['max_steps'],'probe_updates':2}));return
    try:ok=execute(args.config,args.output)
    except Exception as e:
        import sys
        result={'status':'failed','error_type':type(e).__name__,'error':str(e),'rank':int(os.environ.get('RANK',0))}
        t=sys.modules.get('torch')
        if t is not None and t.cuda.is_initialized():result['peak_allocated_gib']=t.cuda.max_memory_allocated()/2**30
        print(json.dumps(result),flush=True)
        if Path(args.output).is_dir():(Path(args.output)/f'failure_rank_{result["rank"]}.json').write_text(json.dumps(result,indent=2))
        raise
    raise SystemExit(0 if ok else 2)


if __name__=='__main__':main()
