"""Full merged50 training-fit evaluation, 16-step joint sampling, no robot I/O."""
import argparse
import gc
import json
from pathlib import Path
import sys
import time
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'src'))
from attention import AttentionProbe
from protocol import anchors, channel_metrics, choose_episode


def dump(path, obj):
    path = Path(path); temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False)+'\n'); temp.replace(path)


def cpu(x):
    return x.detach().float().cpu().numpy()


def decode_numeric(source, key, x):
    from fastwam.datasets.lerobot.utils.normalizer import SingleFieldLinearNormalizer
    stats = {k:torch.tensor(v) for k,v in source.stats[key].items()}
    norm = SingleFieldLinearNormalizer(stats, mode='z-score')
    return norm.backward(torch.as_tensor(x).float()).numpy()


def fit_summary(records):
    result = {}
    for field in ('action', 'force'):
        p = np.stack([r[field] for r in records]); y = np.stack([r[field+'_target'] for r in records])
        m = np.stack([r[field+'_valid'] for r in records])
        result[field] = channel_metrics(p, y, m)
        result[field+'_hold'] = channel_metrics(np.stack([r[field+'_hold'] for r in records]), y, m)
        result[field+'_per_horizon'] = [channel_metrics(p[:,i], y[:,i], m[:,i]) for i in range(32)]
    result['joint_mean_mae_deg'] = float(np.mean(result['action']['mae'][:7])*180/np.pi)
    result['hold_joint_mean_mae_deg'] = float(np.mean(result['action_hold']['mae'][:7])*180/np.pi)
    return result


def plot_episode(out, ep, rows):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for key, labels in [('action', ['J'+str(j) for j in range(1,8)]+['Gripper']),
                        ('force', ['Fx','Fy','Fz','Tx','Ty','Tz'])]:
        fig, axes=plt.subplots(len(labels)//2, 2, figsize=(13, 2.1*len(labels)), squeeze=False)
        for col, ax in enumerate(axes.flat):
            for r in rows:
                offset = 0 if key == 'action' else 1
                x=(np.arange(32)+int(r['frame'])+offset)/30
                mask=r[key+'_valid'][:,col]
                scale=180/np.pi if key=='action' and col<7 else 1
                ax.plot(x,np.where(mask,r[key+'_target'][:,col]*scale,np.nan),color='black',lw=1.3)
                ax.plot(x,np.where(mask,r[key][:,col]*scale,np.nan),color='#d55e00',lw=1)
            unit=('deg' if col<7 else 'openness') if key=='action' else ('N' if col<3 else 'Nm')
            ax.set(title=labels[col],xlabel='Recorded episode time (s)',ylabel=unit); ax.grid(alpha=.2)
        fig.suptitle(f'Episode {ep:03d}: black recorded target / orange prediction; gaps remain blank')
        fig.tight_layout(); fig.savefig(out/f'episode_{ep:03d}_{key}.png',dpi=140);plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('config','dataset','source-sha','checkpoint','tokenizer','output'):
        p.add_argument('--'+name,required=True)
    p.add_argument('--expected-step',type=int,required=True)
    p.add_argument('--limit',type=int,default=0,help='Smoke only; a limited run is never marked complete')
    p.add_argument('--skip-plots',action='store_true',help='Export all predictions; render paired plots later on CPU')
    a=p.parse_args(); out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
    # Fail before loading a GPU model if later rendering cannot run.
    import PIL, av
    if not a.skip_plots:import matplotlib
    from hydra.utils import instantiate
    from omegaconf import OmegaConf
    from fastwam.cache.latent import file_sha256
    from fastwam.cosmos.cache import CosmosCachedDataset
    from fastwam.cosmos.realman_joint import RealManJointSource
    from fastwam.cosmos.assets import load_tokenizer
    from fastwam.cosmos.noaction import SCHEMA, QUERY_POSTTRAIN_SCHEMA
    torch.set_num_threads(4)
    if torch.cuda.device_count()!=1:
        raise RuntimeError('Isolate exactly one allocated GPU')
    cfg=OmegaConf.load(a.config)
    checkpoint=Path(a.checkpoint)
    before=(checkpoint.stat().st_size,checkpoint.stat().st_mtime_ns)
    digest=file_sha256(checkpoint)
    payload=torch.load(checkpoint,map_location='cpu',weights_only=True,mmap=True)
    checkpoint_schema = payload.get('schema')
    if payload.get('step') != a.expected_step or checkpoint_schema not in (SCHEMA, QUERY_POSTTRAIN_SCHEMA):
        raise ValueError('Incorrect checkpoint step/schema')
    if not isinstance(payload.get('manifest'), dict) or payload['manifest'].get('schema') != checkpoint_schema:
        raise ValueError('Checkpoint schema is not bound by its manifest')
    if checkpoint_schema == QUERY_POSTTRAIN_SCHEMA:
        warm_start = payload['manifest'].get('warm_start')
        expected_warm_start = dict(source_schema=SCHEMA, source_step=20000,
            source_sha256='34efeb39d0079848c09ca4120655ee94ae4fb1950055a73ecf096f4dff560da6',
            optimizer='fresh AdamW',
            scheduler='fresh 5000-step cosine with 250-step warmup')
        if not isinstance(warm_start, dict) or any(
                warm_start.get(key) != value for key, value in expected_warm_start.items()):
            raise ValueError('Post-training checkpoint is not bound to the verified step-20000 warm start')
    if before!=(checkpoint.stat().st_size,checkpoint.stat().st_mtime_ns):
        raise ValueError('Checkpoint changed during verification')
    manifest=payload['manifest']; del payload
    source=RealManJointSource(a.dataset,a.source_sha)
    if source.identity_sha256!=json.loads(Path(cfg.data.train.cache_root,'manifest.json').read_text())['identity']['numeric_identity']:
        raise ValueError('Dataset identity differs from trained cache')
    ds=CosmosCachedDataset(**{k:v for k,v in OmegaConf.to_container(cfg.data.train,resolve=True).items() if k!='_target_'})
    if manifest['dataset_identity']!=ds.identity_sha256:
        raise ValueError('Checkpoint and source cache differ')
    lookup={(ep,f):j for j,(ep,f,_,_) in enumerate(source.rows)}
    selected=anchors(source.segments,lookup)
    video_episode=choose_episode(source)
    if a.limit: selected=selected[:a.limit]
    ep_indices=[j for j in selected if source.rows[j][0]==video_episode]
    # Predetermined time quantiles and force-change peak, fixed before either model.
    probe_indices=set(ep_indices[::max(1,len(ep_indices)//6)])
    if ep_indices:
        peak=max(ep_indices,key=lambda j:float((source[j]['raw_wrench_future64'][1:,:3]-source[j]['raw_wrench_future64'][:-1,:3]).norm(dim=-1).max()))
        probe_indices.add(peak)
        quiet=min(ep_indices,key=lambda j:float((source[j]['raw_wrench_future64'][1:,:3]-source[j]['raw_wrench_future64'][:-1,:3]).norm(dim=-1).mean()))
        probe_indices.add(quiet)
    protocol=dict(status='running',checkpoint=str(checkpoint),checkpoint_sha256=digest,step=a.expected_step,
        schema=SCHEMA,source_sha256=source.source_sha,source_identity=source.identity_sha256,
        dataset_identity=ds.identity_sha256,training_fit_only=True,episodes=len(source.episodes),
        original_frames=sum(source.lengths),valid_training_rows=len(source),windows=len(selected),
        anchor_stride=32,video_episode=video_episode,num_inference_steps=16,seeds=dict(action=42,force=43,video=44),
        video_protocol='recorded current observation/history per chunk, future offsets4..32; no predicted state feedback',
        attention_layers=[0,6,13,20,27],attention_steps=[0,7,15],probe_indices=sorted(probe_indices),
        action_offsets=list(range(32)),force_offsets=list(range(1,33)),image_offsets=list(range(0,33,4)),
        gpu=torch.cuda.get_device_name(),limited_smoke=bool(a.limit))
    dump(out/'protocol.json',protocol)
    model=instantiate(cfg.model,device='cuda'); model.load_checkpoint(checkpoint);model.eval();model.gradient_checkpointing=False
    probe=AttentionProbe(model); records=[]; timings=[]; videos=[]; attention_errors=[]
    for pos,j in enumerate(selected):
        cached=ds[j]; numeric=source[j]
        for k,v in numeric.items():
            if not torch.equal(v,cached[k]): raise ValueError('Source/cache mismatch '+k)
        # Only current proprio, observed frame and recorded past enter inference.
        sample={k:cached[k].unsqueeze(0) for k in ('first_frame_latents','history','history_is_pad','history_channel_valid','cosmos_text_index','action_is_pad')}
        sample['proprio']=cached['proprio'][:1].unsqueeze(0)
        torch.cuda.synchronize(); t=time.perf_counter()
        with torch.inference_mode():
            result=model.infer_joint(sample,num_inference_steps=16,seed=42,force_seed=43,video_seed=44,use_cache=True)
        torch.cuda.synchronize(); timings.append(time.perf_counter()-t)
        ep,f,start,n=source.rows[j]; lo,hi=source.bounds[j]
        act=decode_numeric(source,'action',cpu(result['action'][0]));force=decode_numeric(source,'force',cpu(result['force'][0]))
        ia=start+np.minimum(f+np.arange(32),hi-1); iw=start+np.minimum(f+np.arange(1,33),hi-1)
        record=dict(sample=j,episode=ep,frame=f,action=act,force=force,
            action_target=source.raw['action'][ia].numpy(),force_target=source.raw['force'][iw].numpy(),
            action_valid=np.repeat((~cached['action_is_pad'].numpy())[:,None],8,axis=1),
            force_valid=cached['wrench_valid'].numpy() & ~cached['force_is_pad'].numpy()[:,None],
            action_hold=np.repeat(source.raw['state'][start+f].numpy()[None],32,axis=0),
            force_hold=np.repeat(source.raw['force'][start+f].numpy()[None],32,axis=0))
        records.append(record)
        if ep==video_episode:
            videos.append(dict(sample=j,frame=f,hi=hi,latents=result['video_latents'].cpu()))
        if j in probe_indices:
            with torch.inference_mode(),probe.recording():
                checked=model.infer_joint(sample,num_inference_steps=16,seed=42,force_seed=43,video_seed=44,use_cache=True)
            differences={k:float((checked[k]-result[k]).abs().max()) for k in result}
            if any(v!=0 for v in differences.values()): raise ValueError('Attention instrumentation changed output')
            probe.save(out/f'attention_{j:06d}.npz'); attention_errors.append(dict(sample=j,max_abs=differences))
        del result
        if pos%20==0:
            dump(out/'progress.json',dict(done=pos+1,total=len(selected),sample=j,episode=ep))
            print(json.dumps(dict(stage='inference',done=pos+1,total=len(selected),episode=ep)),flush=True)
    summary=fit_summary(records)
    summary.update(by_episode={str(ep):fit_summary([r for r in records if r['episode']==ep]) for ep in sorted(set(r['episode'] for r in records))},
        inference_s=dict(median=float(np.median(timings[1:] or timings)),p95=float(np.percentile(timings[1:] or timings,95))),
        peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
        attention_output_parity=attention_errors)
    np.savez_compressed(out/'predictions.npz',**{k:np.stack([r[k] for r in records]) for k in records[0]})
    dump(out/'metrics.json',summary)
    # Decode on the same allocated card after releasing the three-expert model.
    upstream=cfg.model.upstream;del probe,model;gc.collect();torch.cuda.empty_cache()
    if videos:
        vae=load_tokenizer(upstream,a.tokenizer,device='cuda')
        source.decode_episode(video_episode)
        raw=((source._frames+1)/2).clamp(0,1).permute(0,2,3,1).numpy()
        np.savez_compressed(out/f'episode_{video_episode:03d}_gt.npz',rgb=(raw*255).round().astype(np.uint8))
        sparse={}; errors=[]
        for entry in videos:
            with torch.inference_mode(): x=cpu(vae.decode(entry['latents'].to('cuda',torch.bfloat16)))[0]
            if x.shape!=(3,9,224,448):raise ValueError('Unexpected native decoder shape '+str(x.shape))
            x=np.clip((x.transpose(1,2,3,0)+1)/2,0,1)
            for offset in range(1,9):
                frame=entry['frame']+4*offset
                if frame>=entry['hi']:continue
                sparse[frame]=np.round(x[offset]*255).astype(np.uint8)
                for view in range(2):
                    sl=slice(view*224,(view+1)*224)
                    mse=float(np.mean((x[offset,:,sl]-raw[frame,:,sl])**2))
                    errors.append(dict(frame=frame,view=view,mse=mse,psnr=-10*np.log10(max(mse,1e-12))))
        np.savez_compressed(out/'video_predictions.npz',frames=np.array(sorted(sparse)),rgb=np.stack([sparse[k] for k in sorted(sparse)]))
        dump(out/'video_metrics.json',dict(per_frame=errors,mean_psnr_by_view={str(v):float(np.mean([r['psnr'] for r in errors if r['view']==v])) for v in range(2)}))
        del vae;gc.collect();torch.cuda.empty_cache()
    if not a.skip_plots:
        for ep in sorted(set(r['episode'] for r in records)):
            plot_episode(out,ep,[r for r in records if r['episode']==ep])
    protocol.update(status='smoke_complete' if a.limit else 'complete',completed_windows=len(records),ended_unix=time.time(),plots_deferred=a.skip_plots)
    dump(out/'protocol.json',protocol)
    print('JOINT8_EVALUATION_COMPLETE',flush=True)

if __name__=='__main__':main()
