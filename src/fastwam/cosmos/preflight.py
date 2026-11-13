"""Actual four-rank B16 two-Adam probe; never substitutes tiny-model tests."""
import argparse
import importlib.util
import json
import math
import os
from pathlib import Path
import time
from fastwam.utils.training_schedule import resolve_scheduler_total_steps

ROOT=Path(__file__).resolve().parents[3]
JEPA_TARGET = 'fastwam.cosmos.jepa_force_latent.create_jepa_force_latent_cosmos'
JEPA_DATASET = 'fastwam.cosmos.jepa_training_data.JepaForceTrainingDataset'


def validate_config(cfg):
    jepa = cfg['model']['_target_'] == JEPA_TARGET
    noaction = jepa or cfg['model']['_target_'] == 'fastwam.cosmos.assets.create_noaction_joint_cosmos'
    batch,steps = cfg.get('batch_size'),cfg.get('max_steps')
    if noaction and (batch not in (16,32,64) or steps not in (1000,20000,30000)):
        raise ValueError('Joint8 requires an authorized batch16/32/64 and 1000/20000/30000 schedule')
    expected={'batch_size':batch if noaction else 16,'gradient_accumulation_steps':1,
        'max_steps':steps if noaction else 30000,'save_every':1000 if noaction and steps == 1000 else 5000,
        'learning_rate':1e-4,'weight_decay':.01,'mixed_precision':'bf16',
        'seed':42,'resume':None,'lr_scheduler_type':'cosine','eval_every':0}
    if any(cfg.get(k)!=v for k,v in expected.items()):
        raise ValueError('Exact configured batch/accum1/schedule/save recipe required')
    if cfg['data'].get('val') is not None:
        raise ValueError('The inherited recipe has no separate validation split')
    m=cfg['model'];d=cfg['data']['train']
    if jepa:
        validate_jepa_config(cfg)
        d = d['base_config']
    horizon = resolve_scheduler_total_steps(steps, cfg.get('scheduler_total_steps'))
    if horizon != steps and (not noaction or horizon != 50000):
        raise ValueError('Only the no-reference model may preserve the original 50000-step scheduler')
    if cfg.get('scheduler_total_steps') is not None:
        recipe = m.get('training_recipe', {})
        expected_recipe = dict(batch_per_rank=batch, max_steps=steps, scheduler_total_steps=horizon)
        if any(recipe.get(key) != value for key, value in expected_recipe.items()):
            raise ValueError('Model training_recipe must bind batch, stop max_steps and scheduler_total_steps')
    else:
        recipe_horizon = m.get('training_recipe', {}).get('scheduler_total_steps')
        if recipe_horizon is not None and recipe_horizon != horizon:
            raise ValueError('Model training_recipe scheduler horizon differs from the trainer')
    if noaction:
        required = dict(action_hidden=768, action_ffn=3072, force_hidden=768, force_ffn=2048,
                        force_heads=16, force_shift=1., num_inference_steps=16,
                        action_seed=42, force_seed=43, memory_seed=44, action_space='joint8')
        if (any(m.get(k) != v for k, v in required.items())
                or any(k in m for k in ('control_seed', 'initial_checkpoint', 'action_condition'))
                or d['_target_'] != 'fastwam.cosmos.cache.CosmosCachedDataset'
                or len(d.get('expected_identity_sha256', '')) != 64
                or len(m.get('expected_text_identity', '')) != 64 or not d.get('require_production')
                or m.get('expected_data_identity') != d['expected_identity_sha256']):
            raise ValueError('Bound no-reference Cosmos/AE768/Force768 diffusion configuration required')
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


def validate_jepa_config(cfg):
    """Bind the two from-scratch observation-side experiments to one recipe."""
    m, d = cfg['model'], cfg['data']['train']
    if (cfg.get('batch_size') != 64 or cfg.get('max_steps') != 30000
            or cfg.get('scheduler_total_steps') != 50000):
        raise ValueError('JEPA runs require per-rank B64, 30000 updates and 50000-step scheduler')
    required = dict(reader_width=256, query_count=8, reader_depth=2, reader_heads=4,
                    latent_loss_weight=1., wrench_loss_weight=.1,
                    latent_warmup_steps=2500, jepa_seed=47)
    qw = m.get('query_redundancy_weight', 0.)
    qm = m.get('query_redundancy_margin', .95)
    qs = m.get('query_redundancy_warmup_steps', 2500)
    if (isinstance(qw, bool) or not isinstance(qw, (int, float)) or not math.isfinite(qw) or qw < 0
            or isinstance(qm, bool) or not isinstance(qm, (int, float)) or not math.isfinite(qm)
            or not 0 <= qm < 1 or isinstance(qs, bool) or not isinstance(qs, int) or qs < 0):
        raise ValueError('Invalid query redundancy weight/margin/warmup configuration')
    if (any(m.get(k) != v for k, v in required.items())
            or tuple(m.get('injection_layers', ())) != (7, 14, 21, 28)
            or m.get('allow_preencoded_patches', False)
            or any(k in m for k in ('teacher_checkpoint', 'teacher_identity', 'initial_checkpoint',
                                    'force_latent_cache_root', 'expected_force_latent_identity'))):
        raise ValueError('Exact online-current-Force JEPA reader/loss recipe required; no teacher checkpoint')
    observation = m.get('observation_encoder')
    if not isinstance(observation, dict):
        raise ValueError('Bound online JEPA observation encoder configuration required')
    allowed = {'variant', 'checkpoint', 'expected_sha256', 'checkpoint_suffix'}
    digest = observation.get('expected_sha256', '')
    if (set(observation) != allowed or observation.get('variant') not in ('vitb', 'vitl')
            or not isinstance(observation.get('checkpoint'), str) or not observation['checkpoint']
            or not isinstance(digest, str) or len(digest) != 64
            or any(c not in '0123456789abcdef' for c in digest)
            or observation.get('checkpoint_suffix') is not True):
        raise ValueError('Pinned ViT-B/L EMA checkpoint with final-four Q/V rank8 LoRA required')
    if (d.get('_target_') != JEPA_DATASET or d.get('_recursive_') is not False
            or not isinstance(d.get('base_config'), dict)
            or not isinstance(d.get('observation_cache_root'), str) or not d['observation_cache_root']
            or len(d.get('expected_observation_identity', '')) != 64
            or any(k in d for k in ('force_latent_cache_root', 'expected_force_latent_identity', 'teacher_checkpoint'))):
        raise ValueError('Verified observed-only JEPA RGB wrapper over baseline cache required')
    base = d['base_config']
    if base.get('expected_text_identity') != m.get('expected_text_identity'):
        raise ValueError('JEPA baseline cache and model text identities differ')


class JepaGradientProbe:
    """Collect real backward evidence before ZeRO clears per-step gradients."""
    def __init__(self, model):
        import torch
        from .jepa_observation import variant_spec
        self.encoder = model.dit['jepa_observation']
        spec = variant_spec(self.encoder.variant)
        self.expected = {f'encoder.blocks.{i}.attn.qkv.lora_{part}_{factor}'
                         for i in range(spec['depth'] - 4, spec['depth'])
                         for part in ('q', 'v') for factor in ('a', 'b')}
        self.parameters = dict(self.encoder.named_parameters())
        actual = {name for name, p in self.parameters.items() if p.requires_grad}
        expected_count = 4 * 4 * 8 * spec['width']
        if (actual != self.expected or sum(self.parameters[n].numel() for n in actual) != expected_count
                or any(module.training for module in self.encoder.encoder.modules())):
            raise RuntimeError('Actual JEPA must keep base frozen/eval and only final-four Q/V rank8 trainable')
        self.gradients = {name: [] for name in sorted(self.expected)}
        self.handles = []
        for name in sorted(self.expected):
            def record(gradient, name=name):
                self.gradients[name].append(dict(finite=bool(torch.isfinite(gradient).all()),
                                                 nonzero=bool(gradient.detach().abs().sum() > 0)))
                return gradient
            self.handles.append(self.parameters[name].register_hook(record))
        self.expected_count = expected_count

    def finish(self):
        for handle in self.handles:
            handle.remove()
        base = [p for name, p in self.parameters.items() if name not in self.expected]
        frozen = all(not p.requires_grad for p in base)
        no_base_gradients = all(p.grad is None for p in base)
        seen = all(len(records) >= 2 for records in self.gradients.values())
        finite = all(records and all(r['finite'] for r in records) for records in self.gradients.values())
        # LoRA B is initialized at zero: its A gradient is initially zero and
        # must become nonzero after the first Adam update changes B.
        nonzero = all(any(r['nonzero'] for r in records) for records in self.gradients.values())
        result = dict(variant=self.encoder.variant, base_parameters_frozen=frozen,
                      base_parameter_count=sum(p.numel() for p in base),
                      frozen_base_gradients_absent=no_base_gradients,
                      base_eval=not any(m.training for m in self.encoder.encoder.modules()),
                      lora_trainable_count=sum(self.parameters[n].numel() for n in self.expected),
                      expected_lora_trainable_count=self.expected_count,
                      lora_gradient_records=self.gradients,
                      all_lora_seen_across_updates=seen, all_lora_gradients_finite=finite,
                      all_lora_received_nonzero_gradient=nonzero)
        result['accepted'] = bool(frozen and no_base_gradients and result['base_eval'] and seen and finite and nonzero)
        return result



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
    if hasattr(train_ds, 'observation_identity_sha256'):
        dataset_receipt = dict(dataset_receipt,
            jepa_observation_identity_sha256=train_ds.observation_identity_sha256,
            jepa_observation=train_ds.obs_manifest)
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
    jepa_probe = JepaGradientProbe(model) if 'jepa_observation' in model.dit else None
    layer_calls=[0]*28;handles=[];query_gradients=[]
    def record_query_gradient(gradient):
        query_gradients.append({'finite':bool(torch.isfinite(gradient).all()),
                                'all_queries_nonzero':bool(gradient.float().abs().sum(-1).gt(0).all())})
        return gradient
    force_probe = (model.dit['force'].force_encoder.weight if hasattr(model.dit['force'], 'force_encoder')
                   else model.dit['force'].queries)
    handles.append(force_probe.register_hook(record_query_gradient))
    for i,block in enumerate(model.dit['force'].blocks):
        def count(module,args,out,i=i):layer_calls[i]+=1
        handles.append(block.register_forward_hook(count))
    torch.cuda.synchronize();startup=time.monotonic()-start
    from .probe_updates import run_two_updates
    losses,updates=run_two_updates(trainer,synchronize=torch.cuda.synchronize)
    for handle in handles:handle.remove()
    jepa_gradients = jepa_probe.finish() if jepa_probe is not None else None
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
        'jepa_observation_gradient_contract':jepa_gradients,
        'memory_rng_state':model.dit['runtime'].get_state().tolist(),
        'formal_training_started':False,'checkpoint_saved':False,'distributed_resume_verified':False}
    accepted=(report['actual_engine_updates']==2 and report['adam_state_steps']
        and all(x==2 for x in report['adam_state_steps']) and report['all_parameters_finite']
        and report['adam_master_and_moments_finite'] and min(layer_calls)>=2
        and len(query_gradients)==2 and all(g['finite'] and g['all_queries_nonzero'] for g in query_gradients)
        and (jepa_gradients is None or jepa_gradients['accepted'])
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
