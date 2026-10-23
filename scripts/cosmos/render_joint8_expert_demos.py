"""Render recorded-video / predicted-video comparisons and physical-unit curves."""
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'deploy')]
import numpy as np
import torch
from fastwam.cosmos.assets import load_tokenizer
from PIL import Image,ImageDraw,ImageFont
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import av

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('prepared','results','runtime','output'):p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--modes',nargs='+',required=True)
    p.add_argument('--samples',nargs='+',type=int,required=True)
    args=p.parse_args();args.output.mkdir(exist_ok=True,parents=True);torch.set_num_threads(4)
    rows={r['sample']:r for r in torch.load(args.prepared,weights_only=False)}
    cfg=json.loads(args.runtime.read_text());vae=load_tokenizer(cfg['model_kwargs']['upstream'],cfg['tokenizer'])
    for index in args.samples:
        row=rows[index];target=row['target'];predictions={};videos={'Recorded cameras':row['gt_pixels'].numpy()}
        for mode in args.modes:
            path=args.results/f'sample_{index:06d}_seed0_{mode}.npz'
            if not path.exists():raise FileNotFoundError(path)
            with np.load(path) as saved:
                predictions[mode]=(saved['action'].copy(),saved['force'].copy())
                latent=torch.tensor(saved['video_latents'],dtype=torch.bfloat16,device='cuda')
            with torch.inference_mode():v=vae.decode(latent).float().cpu()[0].permute(1,2,3,0)
            videos[mode]=v.add(1).mul(127.5).clamp(0,255).byte().numpy()
            del latent,v;torch.cuda.empty_cache()
        fig,axes=plt.subplots(4,2,figsize=(10,10))
        time=np.arange(32)/30
        for j,ax in enumerate(axes.flat):
            truth=target['action'][:,j].copy();truth[~target['action_valid']]=np.nan
            if j<7:truth=np.rad2deg(truth)
            ax.plot(time,truth,'k',lw=2,label='Recorded target')
            for mode,(action,force) in predictions.items():ax.plot(time,np.rad2deg(action[:,j]) if j<7 else action[:,j],label=mode,lw=1)
            ax.set_title(f'Joint {j+1} (deg)' if j<7 else 'Gripper openness');ax.grid(alpha=.25)
        axes.flat[0].legend(fontsize=7);fig.suptitle(f'Episode {row["episode"]}, frame {row["frame"]}, seed 0 | offline replay')
        fig.tight_layout();fig.savefig(args.output/f'sample{index}_actions.png',dpi=140);plt.close(fig)
        fig,axes=plt.subplots(3,2,figsize=(10,7))
        for j,ax in enumerate(axes.flat):
            truth=target['force'][:,j].copy();truth[~target['force_valid'][:,j]]=np.nan
            ax.plot((np.arange(32)+1)/30,truth,'k',lw=2,label='Recorded GT')
            for mode,(action,force) in predictions.items():ax.plot((np.arange(32)+1)/30,force[:,j],label=mode,lw=1)
            ax.set_title(['Fx (N)','Fy (N)','Fz (N)','Tx (Nm)','Ty (Nm)','Tz (Nm)'][j]);ax.grid(alpha=.25)
        axes.flat[0].legend(fontsize=7);fig.tight_layout();fig.savefig(args.output/f'sample{index}_forces.png',dpi=140);plt.close(fig)
        # 2x2 panels, includes original observation followed by 8 future video frames.
        names=list(videos);out=av.open(str(args.output/f'sample{index}_video.mp4'),'w')
        stream=out.add_stream('libx264',rate=15);stream.width=960;stream.height=600;stream.pix_fmt='yuv420p';stream.options={'crf':'20'}
        for loop in range(2):
            for fi in range(72):
                frame_id=min(8,fi//8)
                canvas=Image.new('RGB',(960,600),'#141922');draw=ImageDraw.Draw(canvas)
                draw.text((18,8),f'Step10000 | ep {row["episode"]} frame {row["frame"]} | offline replay, not robot success',fill='white')
                for j,name in enumerate(names[:4]):
                    video=videos[name];x=16+(j%2)*472;y=40+(j//2)*270
                    draw.text((x,y),f'{name} | t={frame_id/7.5:.2f}s | slow motion',fill='#ddd')
                    canvas.paste(Image.fromarray(video[min(frame_id,len(video)-1)]).resize((448,224)),(x,y+22))
                draw.text((18,580),'Predicted video does not verify action execution. Curves are saved alongside.',fill='#bbb')
                for packet in stream.encode(av.VideoFrame.from_image(canvas)):out.mux(packet)
        for packet in stream.encode():out.mux(packet)
        out.close();print(json.dumps(dict(sample=index,output=str(args.output))),flush=True)
    (args.output/'README.md').write_text('离线教师观测回放：录制视频与模型预测视频对照；曲线分别显示动作和力与 GT 的差值。不是闭环机械臂执行，也不是成功率验证。视频解码不计入推理时延。所有演示使用固定 seed 0。\n')
if __name__=='__main__':main()
