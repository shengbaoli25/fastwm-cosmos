"""Matched batch-one core and decoded-RGB-to-action latency; no hardware I/O."""
import argparse,gc,json,platform,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent))
from evaluate_joint8_experts import (SHA,BASE,CosmosJointPolicy,ExpertScheduleRunner,
    NoActionInferenceRunner,file_sha256,torch,np,to_device,timed,save_json,emit,mode_counts,model_input)
from realman_deploy.contracts import Observation

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('checkpoint','runtime','prepared','output'):p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--modes',nargs='+',required=True)
    p.add_argument('--repeats',type=int,default=30)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True);torch.set_num_threads(4)
    if file_sha256(args.checkpoint)!=SHA:raise ValueError('Wrong checkpoint')
    identity=json.loads(args.prepared.with_suffix('.identity.json').read_text())
    if file_sha256(args.prepared)!=identity['sha256']:raise ValueError('Prepared data changed')
    rows=torch.load(args.prepared,weights_only=False);rows=[r for r in rows if 'midpoint' in r['strata']]
    config=json.loads(args.runtime.read_text());config['inference_backend']='joint_static_cache'
    policy=CosmosJointPolicy(args.checkpoint,config,config_path=args.runtime)
    report=dict(status='running',gpu=torch.cuda.get_device_name(),torch=torch.__version__,cuda=torch.version.cuda,
        architecture=platform.machine(),checkpoint_sha256=SHA,input_sha256=identity['sha256'],
        modes=args.modes,repeats=args.repeats,records=[],setup_ms={},stats={},
        timing='Synchronized wall time. Core uses identical frozen latent/numeric inputs. RGB path starts from decoded RGB bytes, includes resize, CPU-to-GPU, VAE, prefill, joint sampling, D2H and action/force normalization; excludes camera capture, image compression/network and robot execution.',
        ordering='Pairs of configurations plus baseline captured together, round-robin order rotated each repetition. Setup excluded.',
        runtime_sha256=file_sha256(args.runtime),
        source_hashes={str(f.name):file_sha256(f) for f in [Path(__file__),Path(__file__).with_name('evaluate_joint8_experts.py'),
            Path(__file__).resolve().parents[2]/'src/fastwam/cosmos/expert_schedule_inference.py',
            Path(__file__).resolve().parents[2]/'src/fastwam/cosmos/noaction_inference.py']})
    path=args.output/'latency.json';runners={}
    try:
        modes=[m for m in args.modes if m!=BASE]
        groups=[modes[i:i+2] for i in range(0,len(modes),2)] or [[]]
        for group_index,group in enumerate(groups):
            names=[BASE]+group
            for mode in names:
                v,f,a=mode_counts(mode);runner=ExpertScheduleRunner(policy.model,video_steps=v,force_steps=f,action_steps=a)
                with torch.inference_mode():warm,ms=timed(lambda:runner.infer_joint(to_device(rows[0]['sample_input'])))
                report['setup_ms'][f'group{group_index}_{mode}']=ms;del warm;runners[mode]=runner
                emit('latency_warm',group=group_index,mode=mode)
            torch.cuda.reset_peak_memory_stats()
            for rep in range(args.repeats):
                row=rows[(rep*7)%len(rows)]
                for mode in names[rep%len(names):]+names[:rep%len(names)]:
                    runner=runners[mode];sample=to_device(row['sample_input'])
                    seeds=dict(seed=42+rep%3,video_seed=44+rep%3,force_seed=43+rep%3)
                    with torch.inference_mode():pred,core_ms=timed(lambda:runner.infer_joint(sample,**seeds))
                    # Keep only a small parity specimen to avoid contaminating timing.
                    if rep<3:
                        np.savez_compressed(args.output/f'core_sample{row["sample"]}_{mode}_rep{rep}.npz',
                            action=pred['action'].float().cpu().numpy(),force=pred['force'].float().cpu().numpy())
                    del pred
                    obs=Observation.now(row['sample'],row['state'],row['force'],row['raw_frames'],action_space='joint8')
                    torch.cuda.synchronize();start=time.perf_counter()
                    with torch.inference_mode():
                        first=policy.tokenizer.encode(policy._encode_image(obs)[:,:,:1])
                        torch.cuda.synchronize();encoding_ms=(time.perf_counter()-start)*1000
                        fresh=model_input(first,row['history'],policy.normalizer.encode('state',row['state']),policy.text_index,policy.device)
                        pred=runner.infer_joint(fresh,**seeds)
                        action=policy.normalizer.decode('action',pred['action'][0].float().cpu().tolist())
                        force=policy.normalizer.decode('force',pred['force'][0].float().cpu().tolist())
                    torch.cuda.synchronize();rgb_ms=(time.perf_counter()-start)*1000
                    if rep<3:
                        np.savez_compressed(args.output/f'rgb_sample{row["sample"]}_{mode}_rep{rep}.npz',
                            action=np.asarray(action),force=np.asarray(force),
                            first_frame_latents=first.float().cpu().numpy(),
                            normalized_action=pred['action'].float().cpu().numpy(),
                            normalized_force=pred['force'].float().cpu().numpy())
                    del pred
                    report['records'].append(dict(group=group_index,mode=mode,rep=rep,sample=row['sample'],core_ms=core_ms,
                        rgb_to_action_force_ms=rgb_ms,encoding_ms=encoding_ms))
                if (rep+1)%5==0:save_json(path,report);emit('latency_progress',group=group_index,repetitions=rep+1,total=args.repeats)
            report['stats'][f'group{group_index}']=dict(peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
                peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30,runners={k:r.stats for k,r in runners.items()})
            for runner in runners.values():runner.close()
            runners.clear();runner=None;gc.collect();torch.cuda.empty_cache()
        # Audit the prior equal-step optimized engine for a sampler-implementation bias.
        audit=NoActionInferenceRunner(policy.model)
        sample=to_device(rows[0]['sample_input'])
        with torch.inference_mode():pred,_=timed(lambda:audit.infer_joint(sample,num_inference_steps=16))
        del pred;report['prior_equal16_engine_ms']=[]
        for rep in range(10):
            with torch.inference_mode():pred,ms=timed(lambda:audit.infer_joint(sample,num_inference_steps=16))
            report['prior_equal16_engine_ms'].append(ms);del pred
        audit.close()
        report['aggregate']={}
        for mode in args.modes:
            selected=[r for r in report['records'] if r['mode']==mode]
            report['aggregate'][mode]={key:dict(n=len(selected),median=float(np.median([r[key] for r in selected])),
                p95=float(np.percentile([r[key] for r in selected],95)),minimum=float(min(r[key] for r in selected)),
                maximum=float(max(r[key] for r in selected))) for key in ('core_ms','rgb_to_action_force_ms','encoding_ms')}
        report['status']='complete'
    except Exception as exc:report.update(status='failed',error=repr(exc));raise
    finally:
        save_json(path,report)
        for r in runners.values():r.close()
        policy.close()
    emit('latency_complete',aggregate=report['aggregate'])
if __name__=='__main__':main()
