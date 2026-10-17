"""Inspectable, unpaired query maps on generated RGB and optional references."""
import base64
from io import BytesIO
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from .region_visualization import LATENT_GROUPS, RGB_OFFSETS


def validate_query_trace(frames, attention, reference=None):
    if frames.shape != (9, 224, 448, 3) or frames.dtype != np.uint8:
        raise ValueError('Nine uint8 224x448 generated frames required')
    if (attention.ndim != 5 or attention.shape[0] != 2 or attention.shape[2] != 2
            or attention.shape[1] < 1 or min(attention.shape[-2:]) < 1
            or not np.isfinite(attention).all() or (attention < 0).any()
            or not np.allclose(attention.sum((-1, -2, -3)), 1, atol=2e-3)):
        raise ValueError('Expected joint-camera probabilities [future=2,query,view=2,H,W]')
    if reference is not None and (reference.shape != frames.shape or reference.dtype != np.uint8):
        raise ValueError('Reference must match generated RGB shape/dtype')


def peak_xy(values):
    y, x = np.unravel_index(values.argmax(), values.shape)
    return (x + .5) / values.shape[1], (y + .5) / values.shape[0]


def _draw_map(frame, maps, query, all_queries):
    image = Image.fromarray(frame.copy())
    selected = range(len(maps)) if all_queries else [query]
    if not all_queries:
        pixels = np.asarray(image).astype(np.float32).copy()
        scale = max(float(maps[query].max()), 1e-12)
        for view in range(2):
            values = (maps[query, view] / scale * 255).round().astype(np.uint8)
            heat = np.asarray(Image.fromarray(values).resize((224, 224), Image.Resampling.BILINEAR)) / 255.
            alpha = heat * .38
            crop = pixels[:, view * 224:(view + 1) * 224]
            crop[:] = crop * (1 - alpha[..., None]) + np.array([255, 100, 30]) * alpha[..., None]
        image = Image.fromarray(pixels.clip(0, 255).astype(np.uint8))
    draw = ImageDraw.Draw(image)
    for q in selected:
        for view in range(2):
            mass = float(maps[q, view].sum())
            if mass < .10:  # Do not draw a confident-looking point for an unused view.
                continue
            x, y = peak_xy(maps[q, view]); x = view * 224 + x * 224; y *= 224
            draw.ellipse((x - 3, y - 3, x + 3, y + 3), fill=(255, 90, 30))
            draw.text((x + 4, y), str(q), fill='white', stroke_width=1, stroke_fill='black')
    return image


def overlay_query_frame(frames, attention, frame_index, *, query=0, all_queries=False, reference=None):
    if not 0 <= query < attention.shape[1] or not 0 <= frame_index < 9:
        raise ValueError('Query/frame outside trace')
    group = LATENT_GROUPS[frame_index]
    generated = Image.fromarray(frames[frame_index]) if group < 0 else _draw_map(frames[frame_index], attention[group], query, all_queries)
    height = 272 if reference is None else 520
    canvas = Image.new('RGB', (448, height), (16, 20, 27)); canvas.paste(generated, (0, 48))
    draw = ImageDraw.Draw(canvas)
    draw.text((6, 3), f'Generated | offset {RGB_OFFSETS[frame_index]} | Main left / Wrist right', fill='white')
    label = 'no future map' if group < 0 else f'latent group {group + 1}; map held within group'
    draw.text((6, 17), label, fill='white')
    draw.text((6, 31), 'All query peaks' if all_queries else f'Query {query}: attention read weights, not contact probability', fill='white')
    if reference is not None:
        draw.text((6, 277), 'GT VAE reconstruction reference; NOT raw recorded RGB', fill='white')
        # Coordinates are shared for comparison, not independently measured on GT.
        draw.text((6, 291), 'Same generated-query map projected to reference coordinates', fill=(210, 210, 210))
        ref = Image.fromarray(reference[frame_index]) if group < 0 else _draw_map(reference[frame_index], attention[group], query, all_queries)
        canvas.paste(ref, (0, 304))
    return np.asarray(canvas)


def _images(frames):
    if frames is None: return None
    result = []
    for frame in frames:
        stream = BytesIO(); Image.fromarray(frame).save(stream, format='JPEG', quality=92)
        result.append('data:image/jpeg;base64,' + base64.b64encode(stream.getvalue()).decode())
    return result


def save_query_viewer(path, frames, attention, metadata, reference=None):
    data = dict(images=_images(frames), reference=_images(reference), maps=attention.tolist(),
                groups=LATENT_GROUPS, offsets=RGB_OFFSETS, metadata=metadata)
    encoded = json.dumps(data, ensure_ascii=False, allow_nan=False).replace('</', '<\\/')
    html = '''<!doctype html><meta charset="utf-8"><title>32 query inspection</title>
<style>body{max-width:1050px;margin:24px auto;background:#111923;color:#edf2f7;font:16px system-ui}canvas{width:896px;max-width:100%}select,button,input{margin:8px}pre{white-space:pre-wrap}p{line-height:1.6}</style>
<h2>独立视觉 query：离线检查</h2>
<p>选择一个 query 查看它在双视角中的读取概率。两个视角共用一个概率分布，不强制成对。橙点是热图峰值；某视角概率不足 10% 时不标点。</p>
<label>Query <select id="query"></select></label><button id="play">播放 / 暂停</button>
<label><input id="heat" type="checkbox" checked>热图</label>
<label><input id="all" type="checkbox">全部 query 峰值</label>
<input id="frame" type="range" min="0" max="8" value="0"><span id="status"></span>
<p>上方：生成视频。下方若显示参考，则为 GT latent 的 VAE 解码，不是原始录像；下方叠加同一张生成热图，仅比较位置。</p>
<canvas id="canvas" width="448" height="224"></canvas><pre id="detail"></pre>
<p>初始帧没有未来读取图。其后每 4 个采样帧共享一个 latent 时间组的热图；这不是逐帧追踪或接触检测结果。同一编号不保证持续跟随同一物体。</p>
<pre id="metadata"></pre><script>
const D=__DATA__;
const sel=document.querySelector('#query'),slider=document.querySelector('#frame'),cv=document.querySelector('#canvas'),ctx=cv.getContext('2d');
const load=urls=>(urls||[]).map(url=>{let im=new Image();im.src=url;im.onload=()=>draw();return im;});
const images=load(D.images),refs=load(D.reference);
cv.height=refs.length?472:224;
for(let q=0;q<D.maps[0].length;q++)sel.add(new Option(String(q),q));
const sum=a=>a.flat().reduce((s,v)=>s+v,0);
function paint(n,y0,g,q,notes){if(g<0)return;let ids=document.querySelector('#all').checked?D.maps[g].map((_,i)=>i):[q];
 for(let id of ids){let both=D.maps[g][id],scale=Math.max(...both.flat(2),1e-12);
  for(let v=0;v<2;v++){let a=both[v],mass=sum(a),h=a.length,w=a[0].length;
   if(ids.length===1&&document.querySelector('#heat').checked){ctx.fillStyle='#ff641e';for(let y=0;y<h;y++)for(let x=0;x<w;x++){ctx.globalAlpha=.38*a[y][x]/scale;ctx.fillRect(v*224+x*224/w,y0+y*224/h,224/w+.1,224/h+.1);}ctx.globalAlpha=1;}
   let flat=a.flat(),p=flat.indexOf(Math.max(...flat)),x=(p%w+.5)/w,y=(Math.floor(p/w)+.5)/h;
   if(mass>=.10){ctx.fillStyle='#ff5a1e';ctx.beginPath();ctx.arc(v*224+x*224,y0+y*224,3,0,7);ctx.fill();ctx.fillStyle='white';ctx.fillText(String(id),v*224+x*224+4,y0+y*224);}
   if(ids.length===1)notes.push((v?'Wrist':'Main')+' mass='+mass.toFixed(3)+' peak=('+x.toFixed(3)+', '+y.toFixed(3)+')');
  }
 }
}
function draw(){let n=+slider.value,q=+sel.value,g=D.groups[n],notes=[];ctx.clearRect(0,0,cv.width,cv.height);if(images[n]&&images[n].complete)ctx.drawImage(images[n],0,0,448,224);paint(n,0,g,q,notes);
 if(refs.length){ctx.fillStyle='white';ctx.fillText('GT VAE reconstruction; same generated map',5,240);if(refs[n].complete)ctx.drawImage(refs[n],0,248,448,224);paint(n,248,g,q,[]);}
 if(g>=0){let flat=D.maps[g][q].flat(2);notes.push('Joint normalized entropy='+(-flat.reduce((s,p)=>s+(p>0?p*Math.log(p):0),0)/Math.log(flat.length)).toFixed(3));}
 document.querySelector('#status').textContent='offset '+D.offsets[n]+' | '+(g<0?'无未来图':'latent group '+(g+1)+'（组内保持）');document.querySelector('#detail').textContent=notes.join('\\n');}
sel.onchange=slider.oninput=document.querySelector('#heat').onchange=document.querySelector('#all').onchange=draw;
let timer=null;document.querySelector('#play').onclick=()=>{if(timer){clearInterval(timer);timer=null;}else timer=setInterval(()=>{slider.value=(+slider.value+1)%9;draw();},1000/7.5);};
document.querySelector('#metadata').textContent=JSON.stringify(D.metadata,null,2);draw();
</script>'''
    Path(path).write_text(html.replace('__DATA__', encoded), encoding='utf-8')


def write_query_bundle(directory, frames, attention, metadata, *, query=0, reference=None):
    import imageio.v2 as imageio
    validate_query_trace(frames, attention, reference)
    if not 0 <= query < attention.shape[1]: raise ValueError('Query outside trace')
    root = Path(directory); root.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(root / 'region_maps.npz', attention=attention, rgb_offsets=RGB_OFFSETS, latent_groups=LATENT_GROUPS)
    np.save(root / 'generated_rgb.npy', frames, allow_pickle=False)
    if reference is not None: np.save(root / 'gt_vae_reference_rgb.npy', reference, allow_pickle=False)
    (root / 'metadata.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    save_query_viewer(root / 'viewer.html', frames, attention, metadata, reference)
    for name, all_queries in [('regions_overview.mp4', True), (f'query_{query:02d}.mp4', False)]:
        with imageio.get_writer(root / name, fps=7.5, codec='libx264', macro_block_size=1) as writer:
            for n in range(9): writer.append_data(overlay_query_frame(frames, attention, n, query=query, all_queries=all_queries, reference=reference))
        with imageio.get_reader(root / name) as reader:
            if reader.count_frames() != 9: raise RuntimeError('Encoded query video frame count differs')
    for group, frame in enumerate((4, 8), 1):
        tiles = []
        for q in range(attention.shape[1]):
            tile = Image.new('RGB', (224, 136), (16, 20, 27))
            tile.paste(_draw_map(frames[frame], attention[group - 1], q, False).resize((224, 112)), (0, 24))
            ImageDraw.Draw(tile).text((6, 6), f'Query {q} | latent group {group}', fill='white')
            tiles.append(tile)
        sheet = Image.new('RGB', (896, ((len(tiles) + 3) // 4) * 136))
        for q, tile in enumerate(tiles): sheet.paste(tile, ((q % 4) * 224, (q // 4) * 136))
        sheet.save(root / f'all_queries_group{group}.png')
