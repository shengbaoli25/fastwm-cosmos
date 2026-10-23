"""Reproducible step10000 expert-grid evaluation; recorded observations only."""
import argparse, gc, itertools, json, platform, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'deploy'), str(Path(__file__).parent)]
import numpy as np
import torch
from fastwam.cache.latent import file_sha256
from fastwam.cosmos.expert_schedule_inference import ExpertScheduleRunner, infer_expert_reference
from fastwam.cosmos.noaction_inference import NoActionInferenceRunner
from fastwam.cosmos.realman_joint import RealManJointSource
from fastwam.cosmos.offline_metrics import action_errors, force_errors, error_stats
from realman_deploy.model_adapter import CosmosJointPolicy
from evaluate_joint8_steps import model_input, recorded_targets, prediction_metrics, aggregate
from validate_joint8_local import preflight, replay_input, comparison, require_equal
SHA = '2e8409d495d11961614b72d81b25bec7cc71bc2e194acb15ea220a72edace1d8'
BASE = 'v16_f16_a16'

def emit(stage, **kw):
    print(json.dumps(dict(stage=stage, **kw), ensure_ascii=False), flush=True)

def save_json(path, obj):
    tmp=path.with_suffix('.tmp'); tmp.write_text(json.dumps(obj,indent=2,ensure_ascii=False,allow_nan=False)+'\n'); tmp.replace(path)

def mode_counts(mode):
    return tuple(int(x[1:]) for x in mode.split('_'))

def to_device(sample, device='cuda'):
    return {k:v.to(device) for k,v in sample.items()}

def prepare(args, policy):
    dest=args.prepared
    receipt=json.loads((args.bundle/'receipt.json').read_text())
    manifest=json.loads((args.bundle/'prepared/features/manifest.json').read_text())
    source=RealManJointSource(args.dataset,manifest['identity']['realman_source']['source_inventory_sha256'])
    normalizer,checks,inputs=preflight(args.bundle,source,receipt)
    if policy.model.manifest['dataset_identity'] != manifest['identity_sha256']:
        raise ValueError('Checkpoint and dataset differ')
    selected={int(i):['midpoint'] for i in receipt['sample_indices']}
    # Selection uses GT only to define evaluation strata, never model inputs.
    # Two additional full-horizon windows per episode, separated from midpoint.
    for ep in range(len(source.episodes)):
        rows=[]
        for i,(e,f,start,n) in enumerate(source.rows):
            lo,hi=source.bounds[i]
            if e!=ep or f-lo<31 or hi-f<33: continue
            a=source.raw['action'][start+f:start+f+32].numpy()
            force=source.raw['force'][start+f+1:start+f+33,:3].numpy()
            rows.append((i,f,float(np.ptp(a[:,7])),float(np.linalg.norm(force-force[0],axis=1).max())))
        occupied=[source.rows[i][1] for i in selected if source.rows[i][0]==ep]
        for col,tag in ((2,'gripper_change'),(3,'force_change')):
            eligible=[r for r in rows if all(abs(r[1]-f)>=40 for f in occupied)]
            if not eligible: continue
            choice=max(eligible,key=lambda r:(r[col],-r[1]))
            selected.setdefault(choice[0],[]).append(tag); occupied.append(choice[1])
    plan=[dict(sample=i,episode=source.rows[i][0],frame=source.rows[i][1],strata=tags) for i,tags in sorted(selected.items())]
    save_json(dest.with_suffix('.protocol.json'),dict(schema='expert-sweep-input-protocol-v1',selection=plan,
        baseline=BASE,grid=[4,8,16],seed_offsets=[0,1,2],dataset_identity=manifest['identity_sha256'],
        source_inventory=source.source_sha,checkpoint_sha256=SHA,
        scope='All episodes were training data; review subset is not training-heldout.',
        selection_episodes=list(range(0,50,2)),review_episodes=list(range(1,50,2)),
        stratum_rule='50 original midpoint windows; max within-window gripper range and force-vector change, full 32-step horizons, >=40-frame separation',
        checks=checks))
    cached_inputs={int(c['sample_index']):(c,o,h) for c,o,h in inputs}
    prepared=[]
    for row in plan:
        i=row['sample']
        if i in cached_inputs: cached,obs,hist=cached_inputs[i]
        else:
            cached=source[i]; obs,hist=replay_input(source,normalizer,i)
            for key,value in zip(('history','history_is_pad','history_channel_valid'),hist):
                require_equal(torch.tensor(value),cached[key],key)
            require_equal(normalizer.encode('state',obs.state)+[0.],cached['proprio'][0],'proprio')
        torch.cuda.synchronize(); t=time.perf_counter()
        with torch.inference_mode(): first=policy.tokenizer.encode(policy._encode_image(obs)[:,:,:1])
        torch.cuda.synchronize(); enc_ms=(time.perf_counter()-t)*1000
        sample=model_input(first,hist,normalizer.encode('state',obs.state),policy.text_index,policy.device)
        target=recorded_targets(source,cached,normalizer)
        gt_pixels=source.window(i).permute(1,2,3,0).add(1).mul(127.5).clamp(0,255).to(torch.uint8)
        prepared.append(dict(**row,sample_input={k:v.cpu() for k,v in sample.items()},target=target,
            encoding_ms=enc_ms,raw_frames=obs.frames,gt_pixels=gt_pixels,history=hist,
            state=obs.state,force=obs.force))
        emit('prepared',done=len(prepared),total=len(plan),sample=i)
    torch.save(prepared,dest)
    save_json(dest.with_suffix('.identity.json'),dict(sha256=file_sha256(dest),windows=len(prepared),checkpoint_sha256=SHA,
        normalization_sha256=file_sha256(Path(policy.config['normalization_file'])),
        dataset_identity=manifest['identity_sha256'],source_inventory=source.source_sha))
    return prepared

def extra_metrics(action,force,target):
    metrics=prediction_metrics(action,force,target)
    valid=target['action_valid']; triple=valid[:-2]&valid[1:-1]&valid[2:]
    metrics['action_dynamics']=dict(acceleration_deg_s2=error_stats(
        np.rad2deg(np.diff(action[:,:7],n=2,axis=0))*900,
        np.rad2deg(np.diff(target['action'][:,:7],n=2,axis=0))*900,triple),
        first_jump_deg=error_stats(np.rad2deg(action[:1,:7]-target['measured_state'][:7]),
        np.rad2deg(target['action'][:1,:7]-target['measured_state'][:7]),valid[:1]))
    metrics['hold_state_vs_recorded']=action_errors(np.tile(target['measured_state'],(32,1)),target['action'],valid)
    return metrics

def timed(call):
    torch.cuda.synchronize(); t=time.perf_counter(); value=call(); torch.cuda.synchronize()
    return value,(time.perf_counter()-t)*1000

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('checkpoint','runtime','prepared','output'): p.add_argument('--'+k,type=Path,required=True)
    for k in ('bundle','dataset'): p.add_argument('--'+k,type=Path)
    p.add_argument('--stage',choices=['prepare','sweep','events','latency'],default='sweep')
    p.add_argument('--modes',nargs='+')
    p.add_argument('--limit',type=int)
    p.add_argument('--repeats',type=int,default=30)
    args=p.parse_args(); args.output.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4)
    if file_sha256(args.checkpoint)!=SHA: raise ValueError('Wrong checkpoint bytes')
    payload=torch.load(args.checkpoint,map_location='cpu',mmap=True,weights_only=True)
    if payload['step']!=10000: raise ValueError('Wrong checkpoint step')
    del payload
    config=json.loads(args.runtime.read_text()); config['inference_backend']='joint_static_cache'
    emit('loading_policy',gpu=torch.cuda.get_device_name())
    policy=CosmosJointPolicy(args.checkpoint,config,config_path=args.runtime)
    if not args.prepared.exists():
        args.prepared.parent.mkdir(parents=True,exist_ok=True); prepared=prepare(args,policy)
    else:
        identity=json.loads(args.prepared.with_suffix('.identity.json').read_text())
        if file_sha256(args.prepared)!=identity['sha256']: raise ValueError('Prepared inputs changed')
        prepared=torch.load(args.prepared,weights_only=False)
    if args.stage=='prepare': policy.close(); return
    selected=[r for r in prepared if ('midpoint' in r['strata'])==(args.stage!='events')]
    if args.stage=='latency': selected=[r for r in prepared if 'midpoint' in r['strata']]
    if args.limit: selected=selected[:args.limit]
    modes=args.modes or [BASE]+[f'v{v}_f{f}_a{a}' for v,f,a in itertools.product((4,8,16),repeat=3) if (v,f,a)!=(16,16,16)]
    report_path=args.output/'report.json'
    if report_path.exists(): report=json.loads(report_path.read_text())
    else: report=dict(status='running',stage=args.stage,checkpoint_sha256=SHA,step=10000,
        input_sha256=file_sha256(args.prepared),command=sys.argv,gpu=torch.cuda.get_device_name(),
        torch=torch.__version__,cuda=torch.version.cuda,python=sys.version,architecture=platform.machine(),
        baseline=BASE,modes=modes,seed_offsets=[0,1,2],parity={},setup_ms={},runner_stats={},
        encoding_ms=[r['encoding_ms'] for r in selected],
        scope='Recorded training observations, not robot success or held-out accuracy',
        latency_scope='Synchronized wall-clock joint core, incl prefill/noise/copy, excluding VAE/RGB/network/robot',
        ordering='Baseline first then grid order; final candidates separately remeasured with alternating blocks',
        source_hashes={str(f.relative_to(ROOT)):file_sha256(f) for f in [Path(__file__),ROOT/'src/fastwam/cosmos/expert_schedule_inference.py',ROOT/'src/fastwam/cosmos/noaction_inference.py']})
    records_path=args.output/'records.jsonl'
    records=[json.loads(x) for x in records_path.read_text().splitlines()] if records_path.exists() else []
    done={(r['mode'],r['sample'],r['seed_offset']) for r in records}
    runner=None
    try:
        for mode in modes:
            if args.stage!='latency' and all((mode,r['sample'],seed) in done for r in selected for seed in range(3)):
                emit('mode_already_complete',mode=mode); continue
            v,f,a=mode_counts(mode); sample=to_device(selected[0]['sample_input'])
            runner=ExpertScheduleRunner(policy.model,video_steps=v,force_steps=f,action_steps=a)
            emit('compile_capture',mode=mode)
            with torch.inference_mode(): warm,ms=timed(lambda:runner.infer_joint(sample))
            report['setup_ms'][mode]=ms
            with torch.inference_mode():
                ref=infer_expert_reference(policy.model,sample,video_steps=v,force_steps=f,action_steps=a)
            parity={k:comparison(warm[k].cpu(),ref[k].cpu()) for k in warm}
            report['parity'][mode]=parity; save_json(report_path,report)
            if any(p['max_abs']>.05 or p['mean_abs']>.01 for p in parity.values()):
                raise ValueError('Full-model parity failed '+mode+': '+str(parity))
            del warm,ref
            emit('parity_passed',mode=mode,parity=parity)
            torch.cuda.reset_peak_memory_stats()
            for ri,row in enumerate(selected):
                sample=to_device(row['sample_input']); target=row['target']
                for seed in range(3):
                    if (mode,row['sample'],seed) in done: continue
                    seeds=dict(seed=42+seed,video_seed=44+seed,force_seed=43+seed)
                    with torch.inference_mode(): pred,ms=timed(lambda:runner.infer_joint(sample,**seeds))
                    an=pred['action'][0].float().cpu().numpy(); fn=pred['force'][0].float().cpu().numpy()
                    action=np.asarray(policy.normalizer.decode('action',an.tolist())); force=np.asarray(policy.normalizer.decode('force',fn.tolist()))
                    metrics=extra_metrics(action,force,target)
                    name=f'sample_{row["sample"]:06d}_seed{seed}_{mode}.npz'
                    np.savez_compressed(args.output/name,action=action,force=force,normalized_action=an,normalized_force=fn,
                        video_latents=pred['video_latents'].float().cpu().numpy().astype(np.float16),
                        recorded_action=target['action'],recorded_force=target['force'],
                        action_valid=target['action_valid'],force_valid=target['force_valid'])
                    rec=dict(mode=mode,steps=mode,sample=row['sample'],episode=row['episode'],frame=row['frame'],
                        strata=row['strata'],seed_offset=seed,joint_core_ms=ms,metrics=metrics,artifact=name)
                    with records_path.open('a') as stream: stream.write(json.dumps(rec,allow_nan=False)+'\n')
                    records.append(rec); done.add((mode,row['sample'],seed)); del pred
                if (ri+1)%5==0: emit('mode_progress',mode=mode,windows=ri+1,total=len(selected),predictions=len(records))
            report['runner_stats'][mode]=dict(runner.stats,peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30)
            report['aggregate']=aggregate(records,modes); save_json(report_path,report)
            emit('mode_complete',mode=mode,summary={k:v for k,v in report['aggregate'][mode].items() if k!='metrics'})
            runner.close(); runner=None; gc.collect(); torch.cuda.empty_cache()
        report['status']='complete'; report['aggregate']=aggregate(records,modes)
    except Exception as exc:
        report.update(status='failed',error=repr(exc)); raise
    finally:
        report['finished']=time.strftime('%Y-%m-%d %H:%M:%S %z'); save_json(report_path,report)
        if runner: runner.close()
        policy.close()
    emit('complete',output=str(args.output))

if __name__=='__main__': main()
