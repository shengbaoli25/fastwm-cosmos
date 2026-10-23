"""Physical-unit summaries, paired episode CIs, worst cases and sampling plots."""
import argparse,csv,itertools,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
BASE='v16_f16_a16'
METRICS={'joint_deg':('action_vs_recorded','joint_deg'),'gripper':('action_vs_recorded','gripper_openness'),
 'velocity_deg_s':('action_vs_recorded','joint_velocity_deg_s'),'force_N':('force_vs_recorded','force_N'),
 'torque_Nm':('force_vs_recorded','torque_Nm'),'accel_deg_s2':('action_dynamics','acceleration_deg_s2'),
 'first_jump_error_deg':('action_dynamics','first_jump_deg')}

def event_info(pred,gt):
    if len(gt)<2 or np.ptp(gt)<.1:return None
    k=int(np.argmax(np.abs(np.diff(gt))));direction=np.sign(gt[k+1]-gt[k]);threshold=(gt.min()+gt.max())/2
    def crossing(values,anchor):
        ok=values>=threshold if direction>0 else values<=threshold
        ids=np.flatnonzero(ok[1:]&~ok[:-1])+1
        # Include an already-crossed first frame as an early predicted switch.
        if ok[0]:ids=np.concatenate(([0],ids))
        return int(ids[np.argmin(abs(ids-anchor))]) if len(ids) else None
    truth=crossing(gt,k+1);prediction=crossing(pred,truth)
    return dict(gt_frame=truth,predicted_frame=prediction,missed=prediction is None,direction=int(direction),
        threshold=float(threshold),
        delta_ms=None if prediction is None else (prediction-truth)*1000/30)

def main():
    p=argparse.ArgumentParser();p.add_argument('results',type=Path);p.add_argument('--output',type=Path)
    args=p.parse_args();out=args.output or args.results;out.mkdir(parents=True,exist_ok=True)
    report=json.loads((args.results/'report.json').read_text())
    records=[json.loads(x) for x in (args.results/'records.jsonl').read_text().splitlines()]
    grouped={m:[r for r in records if r['mode']==m] for m in report['modes']}
    base={(r['sample'],r['seed_offset']):r for r in grouped[BASE]}
    rng=np.random.default_rng(20261021);summary={};events=[]
    for mode,rows in grouped.items():
        if not rows:continue
        result=dict(records=len(rows),core_median_ms=float(np.median([r['joint_core_ms'] for r in rows])),
            core_p95_ms=float(np.percentile([r['joint_core_ms'] for r in rows],95)),metrics={},worst={},paired={},by_seed={})
        for name,(section,group) in METRICS.items():
            stats=[r['metrics'][section][group] for r in rows]
            value=sum(sum(s['absolute_error_sum']) for s in stats)/sum(sum(s['valid_per_channel']) for s in stats)
            per=np.asarray([s['pooled_mae'] for s in stats]); result['metrics'][name]=value
            result['worst'][name]=[dict(sample=r['sample'],episode=r['episode'],seed=r['seed_offset'],value=r['metrics'][section][group]['pooled_mae'])
                for r in sorted(rows,key=lambda r:r['metrics'][section][group]['pooled_mae'],reverse=True)[:10]]
            paired=[]
            for ep in sorted({r['episode'] for r in rows}):
                prs=[r for r in rows if r['episode']==ep and (r['sample'],r['seed_offset']) in base]
                if prs:paired.append(np.mean([r['metrics'][section][group]['pooled_mae']-base[(r['sample'],r['seed_offset'])]['metrics'][section][group]['pooled_mae'] for r in prs]))
            boot=np.mean(rng.choice(paired,size=(5000,len(paired)),replace=True),axis=1) if paired else None
            result['paired'][name]=dict(mean_episode_mae_difference=float(np.mean(paired)) if paired else None,
                ci95=np.percentile(boot,[2.5,97.5]).tolist() if paired else None,
                episodes=len(paired),improved_episodes=int(sum(x<0 for x in paired)),worse_episodes=int(sum(x>0 for x in paired)),
                window_seed_p95=float(np.percentile(per,95)),max_window_seed=float(per.max()))
        for seed in range(3):
            chosen=[r for r in rows if r['seed_offset']==seed]
            result['by_seed'][str(seed)]={name:float(np.mean([r['metrics'][s][g]['pooled_mae'] for r in chosen])) for name,(s,g) in METRICS.items()}
        event_rows=[];invalid=0
        for row in rows:
            with np.load(args.results/row['artifact']) as saved:
                pred=saved['action'];gt=saved['recorded_action'];valid=saved['action_valid']
                invalid+=int(((pred[:,7]<0)|(pred[:,7]>1)).sum())
                event=event_info(pred[valid,7],gt[valid,7])
                if event is not None:
                    event.update(mode=mode,sample=row['sample'],episode=row['episode'],seed_offset=row['seed_offset']);event_rows.append(event)
        result['gripper_out_of_range_values']=invalid;events.extend(event_rows)
        deltas=[e['delta_ms'] for e in event_rows if not e['missed']]
        result['gripper_events']=dict(n=len(event_rows),missed=sum(e['missed'] for e in event_rows),
            median_abs_delta_ms=float(np.median(np.abs(deltas))) if deltas else None,
            p95_abs_delta_ms=float(np.percentile(np.abs(deltas),95)) if deltas else None,
            max_abs_delta_ms=float(np.max(np.abs(deltas))) if deltas else None)
        summary[mode]=result
    cost=[]; times=[]
    for mode,item in summary.items():
        stats=report.get('runner_stats',{}).get(mode)
        if stats:
            cost.append([1,stats['video_forward_evaluations'],stats['force_forward_evaluations'],stats['action_forward_evaluations']])
            times.append(item['core_median_ms'])
    if len(cost)>=4:
        matrix=np.asarray(cost); target=np.asarray(times);coef=np.linalg.lstsq(matrix,target,rcond=None)[0]
        fit=matrix@coef
        (out/'cost-model.json').write_text(json.dumps(dict(
            terms=['intercept','video_forward','force_forward','action_forward'],milliseconds=coef.tolist(),
            r_squared=float(1-np.sum((target-fit)**2)/np.sum((target-target.mean())**2)),
            mean_absolute_residual_ms=float(np.mean(np.abs(target-fit))),
            scope='Descriptive linear fit to measured medians; branch cost attribution is approximate, not a kernel profiler'),indent=2)+'\n')
    (out/'analysis.json').write_text(json.dumps(dict(summary=summary,gripper_events=events,
        ci_method='5000 paired bootstrap resamples of episode means; seed offsets stay paired; train episodes, no generalization claim'),indent=2)+'\n')
    with (out/'summary.csv').open('w',newline='') as stream:
        writer=csv.writer(stream);writer.writerow(['mode','records','core_median_ms','core_p95_ms',*METRICS])
        for mode,s in summary.items():writer.writerow([mode,s['records'],s['core_median_ms'],s['core_p95_ms'],*[s['metrics'][m] for m in METRICS]])
    available=[m for m in summary if m!=BASE]
    fig,axes=plt.subplots(1,3,figsize=(15,5))
    colors={4:'#156c9c',8:'#e7982c',16:'#9040aa'}
    for mode in available+[BASE]:
        s=summary[mode];v,f,a=[int(x[1:]) for x in mode.split('_')]
        for ax,metric in zip(axes,['joint_deg','velocity_deg_s','force_N']):
            ax.scatter(s['core_median_ms'],s['metrics'][metric],c=colors[a],s=40+f*2,marker='s' if mode==BASE else 'o')
            ax.annotate(f'{v}/{f}/{a}',(s['core_median_ms'],s['metrics'][metric]),fontsize=6,xytext=(3,2),textcoords='offset points')
            ax.set(xlabel='Joint core median (ms)',ylabel=metric);ax.grid(alpha=.2)
    windows=len({r['sample'] for r in records});seeds=len({r['seed_offset'] for r in records})
    fig.suptitle(f'Step10000 | V/F/A grids | {windows} training replay windows x {seeds} seeds');fig.tight_layout();fig.savefig(out/'speed-quality.png',dpi=170);plt.close(fig)
    fig,axes=plt.subplots(2,3,figsize=(12,7))
    for row,metric in enumerate(['core_median_ms','velocity_deg_s']):
        for col,a in enumerate([4,8,16]):
            ax=axes[row,col];z=np.full((3,3),np.nan)
            for i,v in enumerate([4,8,16]):
                for j,f in enumerate([4,8,16]):
                    s=summary.get(f'v{v}_f{f}_a{a}')
                    if s:z[i,j]=s[metric] if row==0 else s['metrics'][metric]
            im=ax.imshow(z,cmap='viridis');ax.set(xticks=range(3),xticklabels=[4,8,16],yticks=range(3),yticklabels=[4,8,16],xlabel='Force steps',ylabel='Video steps',title=f'Action {a} | {metric}')
            for (i,j),value in np.ndenumerate(z):
                if np.isfinite(value):ax.text(j,i,f'{value:.0f}' if row==0 else f'{value:.2f}',ha='center',va='center',color='white')
            fig.colorbar(im,ax=ax,fraction=.046)
    fig.tight_layout();fig.savefig(out/'grid-heatmap.png',dpi=170);plt.close(fig)
    print(json.dumps({'modes':len(summary),'records':len(records),'output':str(out)}))
if __name__=='__main__':main()
