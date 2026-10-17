"""Offline videos and a standalone interactive viewer for actual query traces."""
import base64
from io import BytesIO
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

RGB_OFFSETS = [0, 4, 8, 12, 16, 20, 24, 28, 32]
LATENT_GROUPS = [-1, 0, 0, 0, 0, 1, 1, 1, 1]


def validate_trace(frames, attention):
    if frames.shape != (9, 224, 448, 3) or frames.dtype != np.uint8:
        raise ValueError('Nine uint8 224x448 RGB frames required')
    if (attention.ndim != 6 or attention.shape[:2] != (2, 2)
            or attention.shape[3] != 2 or min(attention.shape[-2:]) < 1
            or not np.isfinite(attention).all() or (attention < 0).any()
            or not np.allclose(attention.sum((-1,-2)), 1, atol=2e-3)):
        raise ValueError('Expected normalized finite maps [future=2,view=2,pair,endpoint=2,H,W]')


def map_center(values):
    h, w = values.shape
    x = ((np.arange(w)+.5)/w * values.sum(0)).sum() / values.sum()
    y = ((np.arange(h)+.5)/h * values.sum(1)).sum() / values.sum()
    return float(x), float(y)


def overlay_frame(frame, attention, frame_index, pair=0, all_pairs=False):
    """No interpolation across groups; the initial frame has no future map."""
    if not 0 <= frame_index < 9 or not 0 <= pair < attention.shape[2]:
        raise ValueError('Frame or pair outside trace')
    group = LATENT_GROUPS[frame_index]
    image = Image.fromarray(frame.copy())
    if group >= 0:
        pixels = np.asarray(image).astype(np.float32).copy()
        if not all_pairs:
            for view in range(2):
                for endpoint, color in enumerate(([255,65,50],[20,210,255])):
                    values = attention[group,view,pair,endpoint]
                    heat = Image.fromarray((values/max(float(values.max()),1e-12)*255).astype(np.uint8))
                    alpha = np.asarray(heat.resize((224,224),Image.Resampling.BILINEAR)).astype(np.float32)/255*.32
                    crop = pixels[:,view*224:(view+1)*224]
                    crop[:] = crop*(1-alpha[...,None])+np.array(color)*alpha[...,None]
        image = Image.fromarray(np.clip(pixels,0,255).astype(np.uint8))
        draw = ImageDraw.Draw(image)
        pairs = range(attention.shape[2]) if all_pairs else [pair]
        for view in range(2):
            for identity in pairs:
                pts = []
                for endpoint, color in enumerate(((255,65,50),(20,210,255))):
                    x,y = map_center(attention[group,view,identity,endpoint])
                    point = (view*224+x*224,y*224)
                    pts.append(point)
                draw.line(pts, fill=(255,240,80), width=1)
                for point,color in zip(pts,((255,65,50),(20,210,255))):
                    draw.ellipse((point[0]-3,point[1]-3,point[0]+3,point[1]+3),fill=color)
                draw.text(pts[1],str(identity),fill=(255,255,255),stroke_width=1,stroke_fill=(0,0,0))
    canvas = Image.new('RGB',(448,272),(16,20,27));canvas.paste(image,(0,48))
    draw = ImageDraw.Draw(canvas)
    label = ('no future map' if group < 0 else f'group {group+1} (map held)')
    draw.text((7,3),f'Generated | offset {RGB_OFFSETS[frame_index]} | Main left / Wrist right',fill='white')
    draw.text((7,17),label+' | '+('all candidate pairs' if all_pairs else f'pair {pair}: A red / B cyan'),fill='white')
    draw.text((7,31),'Attention centroid; not contact / per-frame tracking',fill=(210,210,210))
    return np.asarray(canvas)


def save_viewer(path, frames, attention, metadata):
    images=[]
    for frame in frames:
        buf=BytesIO();Image.fromarray(frame).save(buf,format='JPEG',quality=90)
        images.append('data:image/jpeg;base64,'+base64.b64encode(buf.getvalue()).decode())
    data=dict(images=images,maps=attention.tolist(),metadata=metadata,offsets=RGB_OFFSETS,groups=LATENT_GROUPS)
    encoded=json.dumps(data,ensure_ascii=False,allow_nan=False).replace('</','<\\/')
    html='''<!doctype html><meta charset="utf-8"><title>Paired query video viewer</title>
<style>body{max-width:1050px;margin:28px auto;background:#111923;color:#edf2f7;font:16px system-ui}canvas{width:896px;max-width:100%;image-rendering:auto}select,button,input{margin:8px}pre{white-space:pre-wrap;color:#b7c8d9}p{line-height:1.6}</style>
<h2>成对视觉 query：离线检查</h2>
<p>红色 A／青色 B 为两个端点，黄线连接同一候选对。位置是 patch 注意力质心；亮度是相对热度，不是接触概率。每个视角有独立候选，编号相同不代表已标定的同一个三维位置。</p>
<label>候选对 <select id="pair"></select></label><button id="play">播放 / 暂停</button>
<label><input type="checkbox" id="heat" checked>热图</label><input id="frame" type="range" min="0" max="8" value="0"><span id="status"></span>
<canvas id="canvas" width="448" height="224"></canvas><pre id="detail"></pre>
<p>第 0 帧没有未来读取图。其后每 4 个 RGB 采样帧共享一个 latent 分组图；这不是逐 RGB 帧重新定位。注意力来自生成完成后的中间层刷新，不能当成全部去噪步骤的注意力。先检查两端落点和热图分散程度，再用遮挡对照检验它们是否影响预测。</p>
<pre id="metadata"></pre><script>const D=__DATA__;
const sel=document.querySelector('#pair'), slider=document.querySelector('#frame'), cv=document.querySelector('#canvas'), ctx=cv.getContext('2d');
const images=D.images.map(url=>{const im=new Image();im.src=url;im.onload=draw;return im;});
for(let p=0;p<D.maps[0][0].length;p++)sel.add(new Option(String(p),p));
const center=a=>{let sx=0,sy=0,tot=0;for(let y=0;y<a.length;y++)for(let x=0;x<a[0].length;x++){let w=a[y][x];sx+=(x+.5)/a[0].length*w;sy+=(y+.5)/a.length*w;tot+=w;}return[sx/tot,sy/tot];};
function draw(){let n=+slider.value,p=+sel.value,g=D.groups[n];ctx.clearRect(0,0,448,224);if(images[n].complete)ctx.drawImage(images[n],0,0,448,224);let notes=[];
 if(g>=0)for(let v=0;v<2;v++){let pts=[];for(let e=0;e<2;e++){let a=D.maps[g][v][p][e],flat=a.flat(),max=Math.max(...flat),c=e?'#14d2ff':'#ff4132';
 if(document.querySelector('#heat').checked){ctx.fillStyle=c;for(let y=0;y<a.length;y++)for(let x=0;x<a[0].length;x++){ctx.globalAlpha=.30*a[y][x]/max;ctx.fillRect(v*224+x*224/a[0].length,y*224/a.length,224/a[0].length+.2,224/a.length+.2);}ctx.globalAlpha=1;}
 let [x,y]=center(a);pts.push([v*224+x*224,y*224]);ctx.fillStyle=c;ctx.beginPath();ctx.arc(pts[e][0],pts[e][1],4,0,7);ctx.fill();let entropy=-flat.reduce((s,z)=>s+(z>0?z*Math.log(z):0),0)/Math.log(flat.length);notes.push((v?'Wrist':'Main')+' '+(e?'B':'A')+': x='+x.toFixed(3)+' y='+y.toFixed(3)+' normalized entropy='+entropy.toFixed(3));}
 ctx.strokeStyle='#fff050';ctx.beginPath();ctx.moveTo(...pts[0]);ctx.lineTo(...pts[1]);ctx.stroke();}
 document.querySelector('#status').textContent='offset '+D.offsets[n]+' | '+(g<0?'初始帧：无未来图':'latent group '+(g+1)+'（组内保持）');document.querySelector('#detail').textContent=notes.join('\\n');}
sel.onchange=slider.oninput=document.querySelector('#heat').onchange=draw;
let timer=null;document.querySelector('#play').onclick=()=>{if(timer){clearInterval(timer);timer=null;}else timer=setInterval(()=>{slider.value=(+slider.value+1)%9;draw();},1000/7.5);};
document.querySelector('#metadata').textContent=JSON.stringify(D.metadata,null,2);draw();</script>'''
    Path(path).write_text(html.replace('__DATA__',encoded),encoding='utf-8')


def write_bundle(directory, frames, attention, metadata, *, pair=0):
    import imageio.v2 as imageio
    validate_trace(frames,attention)
    root=Path(directory);root.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(root/'region_maps.npz',attention=attention,rgb_offsets=RGB_OFFSETS,
                        latent_groups=LATENT_GROUPS)
    np.save(root/'generated_rgb.npy',frames,allow_pickle=False)
    (root/'metadata.json').write_text(json.dumps(metadata,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    save_viewer(root/'viewer.html',frames,attention,metadata)
    for name, overview in [('regions_overview.mp4',True),(f'pair_{pair:02d}.mp4',False)]:
        with imageio.get_writer(root/name,fps=7.5,codec='libx264',macro_block_size=1) as writer:
            for n,frame in enumerate(frames):writer.append_data(overlay_frame(frame,attention,n,pair,overview))
        with imageio.get_reader(root/name) as reader:
            if reader.count_frames()!=9:raise RuntimeError('Encoded query video lost frames')
    # All pairs, both views, both future groups remain inspectable without choosing highlights.
    tiles=[]
    for identity in range(attention.shape[2]):
        tile=Image.fromarray(overlay_frame(frames[4],attention,4,identity)).resize((224,136))
        tiles.append(tile)
    rows=(len(tiles)+3)//4;sheet=Image.new('RGB',(4*224,rows*136))
    for n,tile in enumerate(tiles):sheet.paste(tile,((n%4)*224,(n//4)*136))
    sheet.save(root/'all_pairs.png')
