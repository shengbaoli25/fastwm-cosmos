"""CPU rendering: complete episode paired cameras and shared-scale attention maps."""
import argparse,json
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from roi import region_statistics


def write_video(path, images, fps=7.5):
    import av
    from fractions import Fraction
    with av.open(str(path),'w') as c:
        stream=c.add_stream('libx264',rate=Fraction(str(fps)))
        first=next(images);stream.width=first.shape[1];stream.height=first.shape[0];stream.pix_fmt='yuv420p'
        import itertools
        for image in itertools.chain([first],images):
            for packet in stream.encode(av.VideoFrame.from_ndarray(image,format='rgb24')):c.mux(packet)
        for packet in stream.encode():c.mux(packet)


def panel(rgb,label):
    im=Image.new('RGB',(rgb.shape[1],rgb.shape[0]+32),'#161616'); im.paste(Image.fromarray(rgb),(0,32))
    ImageDraw.Draw(im).text((5,8),label,fill='white');return np.asarray(im)


def main():
    p=argparse.ArgumentParser();p.add_argument('--runs',nargs='+',required=True);p.add_argument('--output',required=True)
    p.add_argument('--all-attention-images',action='store_true',help='Render every map; all raw maps/statistics are always retained')
    a=p.parse_args();runs=[Path(r) for r in a.runs];out=Path(a.output);out.mkdir(exist_ok=True,parents=True)
    protocols=[json.loads((r/'protocol.json').read_text()) for r in runs]
    ids=('source_sha256','source_identity','dataset_identity','video_episode','windows','seeds','probe_indices')
    if any(any(p[k]!=protocols[0][k] for k in ids) for p in protocols):raise ValueError('Evaluation protocols differ')
    if any(p['status']!='complete' for p in protocols):raise ValueError('Only full completed runs can be compared')
    ep=protocols[0]['video_episode'];gt=np.load(runs[0]/f'episode_{ep:03d}_gt.npz')['rgb']
    predictions=[np.load(r/'video_predictions.npz') for r in runs]
    if any(not np.array_equal(v['frames'],predictions[0]['frames']) for v in predictions):raise ValueError('Video timestamps differ')
    # One common timeline includes all recorded frames, including invalid prefixes/gaps.
    # Display prediction only at its exact target timestamp; sample-and-hold is not scored.
    frame_list=sorted(set(range(0,len(gt),4))|set(predictions[0]['frames'].tolist()))
    # Use 30 fps so irregular per-segment offsets retain exact recorded timing.
    for view,name in enumerate(('global','wrist')):
        sl=slice(view*224,(view+1)*224)
        def images():
            lookup={int(f):i for i,f in enumerate(predictions[0]['frames'])}
            latest=None
            for f in range(len(gt)):
                if f in lookup:latest=f
                # At most three duplicated frames, never carry across a capture gap.
                held=latest is not None and 0<=f-latest<4
                row=[panel(gt[f,:,sl],f'GT t={f/30:.2f}s')]
                for v,protocol in zip(predictions,protocols):
                    image=v['rgb'][lookup[latest],:,sl] if held else np.full_like(gt[f,:,sl],64)
                    label=f"step{protocol['step']} target={latest/30:.2f}s" if held else f"step{protocol['step']} no valid prediction"
                    row.append(panel(image,label))
                yield np.concatenate(row,axis=1)
        write_video(out/f'episode_{ep:03d}_{name}_comparison.mp4',images(),fps=30)
    metrics=[json.loads((r/'metrics.json').read_text()) for r in runs]
    (out/'comparison-metrics.json').write_text(json.dumps({str(p['step']):m for p,m in zip(protocols,metrics)},indent=2))
    data=[np.load(r/'predictions.npz') for r in runs]
    if any(not np.array_equal(d['sample'],data[0]['sample']) for d in data):raise ValueError('Numeric anchor mismatch')
    for episode in np.unique(data[0]['episode']):
        indices=np.flatnonzero(data[0]['episode']==episode)
        for field,labels in [('action',[f'J{i}' for i in range(1,8)]+['Gripper']),('force',['Fx','Fy','Fz','Tx','Ty','Tz'])]:
            fig,axes=plt.subplots(len(labels)//2,2,figsize=(13,2*len(labels)),squeeze=False)
            for ch,ax in enumerate(axes.flat):
                scale=180/np.pi if field=='action' and ch<7 else 1
                for ix in indices:
                    valid=data[0][field+'_valid'][ix,:,ch]
                    x=(data[0]['frame'][ix]+np.arange(32)+(field=='force'))/30
                    ax.plot(x,np.where(valid,data[0][field+'_target'][ix,:,ch]*scale,np.nan),'k-',lw=1.4)
                    for d,proto,color in zip(data,protocols,('#0072b2','#d55e00')):
                        ax.plot(x,np.where(valid,d[field][ix,:,ch]*scale,np.nan),color=color,lw=1)
                ax.set(title=labels[ch],xlabel='Recorded time (s)',ylabel=('deg' if ch<7 else 'openness') if field=='action' else ('N' if ch<3 else 'Nm'));ax.grid(alpha=.2)
            legend='; '.join(['Black GT']+[f'{color} {p["step"]}' for color,p in zip(('blue','orange'),protocols)])
            fig.suptitle(legend+'. Recorded-observation chunk predictions.')
            fig.tight_layout();fig.savefig(out/f'episode_{episode:03d}_{field}_comparison.png',dpi=140);plt.close(fig)
    attention_summary=[];roi_summary=[]
    roi=json.loads(Path(__file__).with_name('contact-roi.json').read_text())
    if roi['episode']!=ep:raise ValueError('ROI annotations belong to another episode')
    attention_arrays={index:[np.load(r/f'attention_{index:06d}.npz') for r in runs]
                      for index in protocols[0]['probe_indices']}
    attention_vmax={}
    for arrays in attention_arrays.values():
        for key in arrays[0].files:
            visual_maps=[]
            for array in arrays:
                prob=array[key].mean((0,1)); visual_maps.append(prob[:1176].reshape(3,14,28))
            for temporal in range(3):
                scale_key=(key,temporal)
                attention_vmax[scale_key]=max(attention_vmax.get(scale_key,0.0),
                    *(float(m[temporal].max()) for m in visual_maps))
    (out/'attention-scales.json').write_text(json.dumps({f'{key}:{temporal}':value
        for (key,temporal),value in attention_vmax.items()},indent=2))
    # O/V share the actual 14x28 token grid: global left, wrist right.
    for index in protocols[0]['probe_indices']:
        arrays=attention_arrays[index]
        ix=int(np.flatnonzero(data[0]['sample']==index)[0]);frame=int(data[0]['frame'][ix]);bg=gt[frame]
        force=data[0]['force_target'][ix,:,:3]
        valid=data[0]['force_valid'][ix,:,:3].all(-1)
        change=np.linalg.norm(np.diff(force,axis=0),axis=-1)[valid[1:] & valid[:-1]]
        peak_force_change=float(change.max()) if len(change) else None
        for key in arrays[0].files:
            maps=[];masses=[]
            for array in arrays:
                prob=array[key].mean((0,1)); visual=prob[:1176].reshape(3,14,28)
                masses.append(visual.sum(axis=(1,2)).tolist())
                # Keep observed/future maps separate: do not project future frames onto current RGB.
                maps.append(visual)
            for m,proto in zip(maps,protocols):
                for view,name in enumerate(('global','wrist')):
                    stats=region_statistics(m[0],roi['frames'][str(frame)][name],view)
                    roi_summary.append(dict(sample=index,frame=frame,key=key,step=proto['step'],view=name,
                                            max_force_increment_N=peak_force_change,**stats))
            sampling,layer,_=key.split('_');sampling=int(sampling[1:]);layer=int(layer[1:])
            for temporal,typename in enumerate(('observed','future_latent1','future_latent2')):
                render=(sampling in (0,15) and layer in (0,13,27)) if temporal==0 else (sampling==15 and layer in (13,27))
                if not a.all_attention_images and not render:continue
                vmax=attention_vmax[(key,temporal)]
                fig,axes=plt.subplots(2,len(runs),figsize=(5*len(runs),7),squeeze=False)
                for col,(m,proto) in enumerate(zip(maps,protocols)):
                    heat=m[temporal];axes[0,col].imshow(bg)
                    if temporal==0:
                        for view,name in enumerate(('global','wrist')):
                            for x0,y0,x1,y1 in roi['frames'][str(frame)][name]:
                                axes[0,col].add_patch(Rectangle((x0+view*224,y0),x1-x0,y1-y0,fill=False,edgecolor='#00ff9f',linewidth=1))
                        axes[0,col].imshow(heat,extent=(0,448,224,0),cmap='inferno',vmin=0,vmax=vmax,alpha=.45,interpolation='nearest')
                    axes[0,col].set_title(f"step{proto['step']}: "+('observed attention overlay' if temporal==0 else 'recorded anchor (context)'))
                    im=axes[1,col].imshow(heat,cmap='inferno',vmin=0,vmax=vmax,interpolation='nearest')
                    axes[1,col].set_title(f"{typename} attention, mass={heat.sum():.3f}")
                    for ax in axes[:,col]:ax.axis('off')
                fig.colorbar(im,ax=list(axes[1]),fraction=.03,label='Mean softmax probability / key')
                fig.suptitle(f'anchor{frame} {key}; shared scale across checkpoints and all anchors')
                fig.savefig(out/f'attention_{index:06d}_{key}_{typename}.png',dpi=130);plt.close(fig)
            attention_summary.append(dict(sample=index,key=key,visual_frame_mass={str(p['step']):m for p,m in zip(protocols,masses)}))
    (out/'attention-mass.json').write_text(json.dumps(attention_summary,indent=2))
    (out/'attention-roi.json').write_text(json.dumps(dict(annotation=roi,statistics=roi_summary),indent=2))
    grouped={}
    for row in roi_summary:
        sampling,layer,branch=row['key'].split('_')
        group=(row['step'],sampling,branch,row['view'])
        grouped.setdefault(group,[]).append(row)
    summaries=[]
    for (step,sampling,branch,view),rows in sorted(grouped.items()):
        entry=dict(step=step,sampling=sampling,branch=branch,view=view,layer_anchor_count=len(rows))
        for field in ('view_attention_mass','region_attention_mass','within_view_fraction','enrichment'):
            values=[r[field] for r in rows if r[field] is not None]
            entry[field+'_mean']=float(np.mean(values)) if values else None
        summaries.append(entry)
    (out/'attention-roi-summary.json').write_text(json.dumps(summaries,indent=2))
    print('COMPARISON_RENDER_COMPLETE',flush=True)

if __name__=='__main__':main()
