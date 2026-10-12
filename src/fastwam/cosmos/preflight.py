"""Actual four-rank B16 two-Adam probe; never substitutes tiny-model tests."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import time

ROOT=Path(__file__).resolve().parents[3]


def validate_config(cfg):
    expected={'batch_size':16,'gradient_accumulation_steps':1,'max_steps':30000,
        'save_every':5000,'learning_rate':1e-4,'weight_decay':.01,'mixed_precision':'bf16',
        'seed':42,'resume':None,'lr_scheduler_type':'cosine','eval_every':0}
    if any(cfg.get(k)!=v for k,v in expected.items()):
        raise ValueError('Exact B16/accum1/30000 original recipe required')
    if cfg['data'].get('val') is not None:
        raise ValueError('The inherited recipe has no separate validation split')
    m=cfg['model'];d=cfg['data']['train']
    if (m['_target_']!='fastwam.cosmos.assets.create_cosmos_forcewm'
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
    trainer=Wan22Trainer(model,train_ds,val_ds,cfg=cfg)
    engine=trainer.model
    zero=trainer.accelerator.state.deepspeed_plugin.deepspeed_config['zero_optimization']
    if (zero['stage']!=1 or engine.train_micro_batch_size_per_gpu()!=16
            or engine.gradient_accumulation_steps()!=1 or engine.train_batch_size()!=64
            or any(zero.get(k,{}).get('device','none')!='none' for k in ('offload_optimizer','offload_param'))):
        raise RuntimeError('Actual engine differs from four-rank global64 original ZeRO1')
    h=helpers();adam=h.unwrap_optimizer(trainer.optimizer)
    if not isinstance(adam,torch.optim.AdamW):raise RuntimeError('Actual AdamW required')
    if h.inspect_scheduler(trainer.scheduler)!={'last_epoch':0,'warmup_milestones':[1500],'cosine_steps':28500}:
        raise RuntimeError('Complete 30k learning rate schedule required')
    layer_calls=[0]*28;handles=[]
    for i,block in enumerate(model.dit['force'].blocks):
        def count(module,args,out,i=i):layer_calls[i]+=1
        handles.append(block.register_forward_hook(count))
    torch.cuda.synchronize();startup=time.monotonic()-start
    from .probe_updates import run_two_updates
    losses,updates=run_two_updates(trainer,synchronize=torch.cuda.synchronize)
    for handle in handles:handle.remove()
    report={'rank':rank,'world_size':4,'batch_per_rank':16,'global_batch':64,
        'actual_engine_updates':int(engine.global_steps),'scheduler':h.inspect_scheduler(trainer.scheduler),
        **h.inspect_adam(adam),'losses':losses,'updates':updates,'teacher_layer_calls':layer_calls,
        'all_parameters_finite':all(h._finite(p) for p in model.parameters()),
        'startup_seconds':startup,'peak_allocated_gib':torch.cuda.max_memory_allocated()/2**30,
        'peak_reserved_gib':torch.cuda.max_memory_reserved()/2**30,
        'gpu':torch.cuda.get_device_name(),'architecture_counts':{k:sum(p.numel() for p in v.parameters()) for k,v in model.dit.items()},
        'dataset':dataset_receipt,'official_initialization':model.manifest,
        'memory_rng_state':model.dit['runtime'].get_state().tolist(),
        'formal_training_started':False,'checkpoint_saved':False,'distributed_resume_verified':False}
    accepted=(report['actual_engine_updates']==2 and report['adam_state_steps']
        and all(x==2 for x in report['adam_state_steps']) and report['all_parameters_finite']
        and report['adam_master_and_moments_finite'] and min(layer_calls)>=2
        and all(not u['optimizer_step_was_skipped'] for u in updates))
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
            'warm_seconds_per_step':warm,'training_only_30000_seconds_estimate':warm*30000,
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
        print(json.dumps({'mode':'plan_only','world_size':4,'B':16,'global_batch':64,'schedule':30000,'probe_updates':2}));return
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
