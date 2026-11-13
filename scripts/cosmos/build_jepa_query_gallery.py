"""Build a local-only, sharded query-attention gallery from completed census shards.

No model is run and no annotations are generated. The optional legacy importer
turns the earlier 36-sample diagnostic into the same gallery for UI preview.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import tarfile

import numpy as np


SHAPE = (2, 4, 8, 392)
ROW_BYTES = int(np.prod(SHAPE)) * 2


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, separators=(',', ':'), allow_nan=False) + '\n', encoding='utf-8')
    temp.replace(path)


def jsonlines(path):
    return [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines() if line.strip()]


def safe_members(archive):
    """Reject traversal, links, device nodes, duplicate paths, and absolute names."""
    seen = set()
    for member in archive.getmembers():
        path = PurePosixPath(member.name)
        if (path.is_absolute() or '..' in path.parts or '\\' in member.name or ':' in member.name
                or member.name in seen or not (member.isfile() or member.isdir())):
            raise ValueError('Unsafe or duplicate archive member: ' + member.name)
        seen.add(member.name)
        yield member


def extract_shard(path, directory, run_identity=None):
    digest = sha(path)
    receipt = directory / 'gallery-extraction.json'
    if directory.exists():
        saved = json.loads(receipt.read_text(encoding='utf-8')) if receipt.is_file() else {}
        if saved.get('archive_sha256') != digest or saved.get('run_identity') != run_identity:
            raise ValueError('Existing extraction is incomplete or refers to changed shard: ' + str(directory))
        return
    final_directory = directory
    import uuid
    directory = directory.with_name(directory.name + '.building-' + uuid.uuid4().hex[:8])
    directory.mkdir(parents=True)
    with tarfile.open(path, 'r:*') as archive:
        members = list(safe_members(archive))
        for member in members:
            target = directory.joinpath(*PurePosixPath(member.name).parts)
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                with archive.extractfile(member) as source, target.open('xb') as destination:
                    shutil.copyfileobj(source, destination)
    manifest_file = directory / 'shard-manifest.json'
    if not manifest_file.exists():
        raise ValueError('Missing shard manifest')
    if manifest_file.exists():
        manifest = json.loads(manifest_file.read_text(encoding='utf-8'))
        if (manifest.get('schema') != 'jepa-query-census-shard-v1'
                or (run_identity is not None and manifest.get('run_identity') != run_identity)):
            raise ValueError('Unknown census shard schema')
        # Receipt shape is accepted from either a path->digest mapping or the
        # runner's list of member objects. The array is additionally checked below.
        members = manifest.get('members', manifest.get('files', {}))
        if isinstance(members, dict):
            records = [dict(path=k, sha256=v if isinstance(v, str) else v.get('sha256')) for k, v in members.items()]
        else:
            records = members
        for record in records:
            relative = record.get('path', record.get('name'))
            digest_expected = record.get('sha256')
            if relative and digest_expected:
                target = directory / relative
                if directory.resolve() not in target.resolve().parents or sha(target) != digest_expected:
                    raise ValueError('Shard member checksum mismatch: ' + str(relative))
    declared_ids = manifest.get('sample_ids')
    if declared_ids != [int(r['sample_index']) for r in jsonlines(directory / 'samples.jsonl')]:
        raise ValueError('Shard sample IDs differ from manifest')
    write_json(directory / 'gallery-extraction.json', dict(archive_sha256=digest,
        archive=str(path.resolve()), run_identity=run_identity))
    directory.rename(final_directory)


def descriptors(attention):
    """Same descriptors as census: means of individual layer/head/query values."""
    p = np.asarray(attention, dtype=np.float32)
    p = p / p.sum(-1, keepdims=True)
    entropy = -(p * np.log(np.maximum(p, 1e-30))).sum(-1) / np.log(392)
    unit = p / np.maximum(np.linalg.norm(p, axis=-1, keepdims=True), 1e-30)
    gram = np.einsum('lhqd,lhkd->lhqk', unit, unit)
    ii, jj = np.triu_indices(8, 1)
    return dict(entropy_normalized=float(entropy.mean()), peak_probability=float(p.max(-1).mean()),
                query_cosine_similarity=float(gram[:, :, ii, jj].mean()),
                camera_main_mass=float(p[..., :196].sum(-1).mean()))


def import_legacy(source, target):
    """Import real previous outputs, without rerunning or inventing samples."""
    from PIL import Image
    samples = json.loads((source / 'samples.json').read_text(encoding='utf-8'))['samples']
    directory = target / 'shards' / 'legacy36'
    directory.mkdir(parents=True, exist_ok=True)
    (directory / 'thumbnails').mkdir(exist_ok=True)
    values, metrics = [], []
    for row in samples:
        index = row['sample_index']
        full = np.load(source / 'attention' / f'{index:06d}.npz', allow_pickle=False)['attention']
        if full.shape != (16, *SHAPE):
            raise ValueError('Legacy attention shape differs')
        mean = full.mean(0)
        values.append(mean.astype('<f2'))
        metrics.append(dict(sample_index=index, **descriptors(mean)))
        raw = Image.open(source / 'raw' / f'{index:06d}.png')
        for camera, name in enumerate(('main', 'wrist')):
            raw.crop((camera * 224, 264, camera * 224 + 224, 488)).save(
                directory / 'thumbnails' / f'{index:06d}-{name}.jpg', quality=88, subsampling=0)
    np.save(directory / 'attention.npy', np.stack(values), allow_pickle=False)
    for filename, rows in [('samples.jsonl', samples), ('metrics.jsonl', metrics)]:
        (directory / filename).write_text(''.join(json.dumps(row) + '\n' for row in rows), encoding='utf-8')
    return [directory]


def build_index(directories, output):
    compact, shards, ids = [], [], set()
    for number, directory in enumerate(directories):
        data = np.load(directory / 'attention.npy', mmap_mode='r', allow_pickle=False)
        samples = jsonlines(directory / 'samples.jsonl')
        metrics = {int(m['sample_index']): m for m in jsonlines(directory / 'metrics.jsonl')}
        if data.shape != (len(samples), *SHAPE) or data.dtype != np.dtype('<f2') or not data.flags.c_contiguous:
            raise ValueError('Unexpected attention dtype/layout/shape: ' + str(directory))
        if (not np.isfinite(data).all() or (data < 0).any()
                or np.max(np.abs(data.sum(-1, dtype=np.float32) - 1)) > .002):
            raise ValueError('Attention probabilities are invalid')
        relative = directory.relative_to(output).as_posix()
        shard_rows = []
        for local, sample in enumerate(samples):
            index = int(sample['sample_index'])
            if index in ids:
                raise ValueError('Duplicate census sample index: ' + str(index))
            ids.add(index)
            metric = metrics[index]
            fields = [float(metric[k]) for k in ('entropy_normalized', 'peak_probability', 'query_cosine_similarity', 'camera_main_mass')]
            if not np.isfinite(fields).all():
                raise ValueError('Nonfinite overview descriptor')
            for camera in ('main', 'wrist'):
                if not (directory / 'thumbnails' / f'{index:06d}-{camera}.jpg').is_file():
                    raise ValueError('Missing thumbnail')
            compact.append([index, int(sample['episode_index']), int(sample['frame_index']), number, local, *fields])
            shard_rows.append(dict(sample, metrics={key: metric[key] for key in metric if key != 'sample_index'}))
        write_json(directory / 'index.json', shard_rows)
        shards.append(dict(path=relative, count=len(samples), array_offset=int(data.offset), row_bytes=ROW_BYTES,
                           shape=list(data.shape), dtype='<f2', array_bytes=(directory / 'attention.npy').stat().st_size))
        del data
    compact.sort(key=lambda x: x[0])
    return compact, shards


SERVER = r'''"""Serve this gallery on localhost with bounded binary Range reads."""
import argparse
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler
import os
from pathlib import Path
import re

class Handler(SimpleHTTPRequestHandler):
    def send_head(self):
        value = self.headers.get('Range')
        if not value:
            return super().send_head()
        path = Path(self.translate_path(self.path))
        if not path.is_file():
            return super().send_head()
        match = re.fullmatch(r'bytes=(\d+)-(\d*)', value)
        size = path.stat().st_size
        if not match:
            self.send_error(416, 'Only a single explicit byte range is supported'); return None
        start = int(match[1]); end = min(int(match[2]) if match[2] else size - 1, size - 1)
        if start >= size or end < start:
            self.send_response(416); self.send_header('Content-Range', f'bytes */{size}'); self.end_headers(); return None
        stream = path.open('rb'); stream.seek(start)
        self.send_response(206)
        self.send_header('Content-Type', self.guess_type(str(path)))
        self.send_header('Content-Range', f'bytes {start}-{end}/{size}')
        self.send_header('Content-Length', str(end - start + 1))
        self.send_header('Accept-Ranges', 'bytes')
        self.end_headers(); self.range_remaining = end - start + 1
        return stream

    def copyfile(self, source, target):
        remaining = getattr(self, 'range_remaining', None)
        if remaining is None:
            return super().copyfile(source, target)
        try:
            while remaining:
                block = source.read(min(1 << 20, remaining))
                if not block: break
                target.write(block); remaining -= len(block)
        finally:
            del self.range_remaining

if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--port', type=int, default=8765)
    args = parser.parse_args(); os.chdir(Path(__file__).resolve().parent)
    server = ThreadingHTTPServer(('127.0.0.1', args.port), Handler)
    print(f'Local gallery: http://127.0.0.1:{server.server_port}/', flush=True)
    print('This serves only the gallery directory on localhost. Ctrl+C stops it.', flush=True)
    try: server.serve_forever()
    except KeyboardInterrupt: pass
    finally: server.server_close()
'''


HTML = r'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>JEPA 全窗口 Query 图库</title>
<style>:root{font:14px system-ui,"Microsoft YaHei",sans-serif;color:#19313e;background:#f3f6f8}body{max-width:1480px;margin:24px auto;padding:0 22px}h1{font-size:28px;margin:0 0 8px}p{line-height:1.6}.notice{border-left:4px solid #bd871c;background:#fff4d7;padding:12px}.controls{display:flex;flex-wrap:wrap;gap:10px;align-items:end;padding:16px;background:white;border:1px solid #d7e1e7;border-radius:10px}label{color:#486575;font-size:12px}select,input,button{padding:7px;border:1px solid #abc0cb;border-radius:5px;background:white;color:#19313e;font-size:14px}label select,label input{display:block}button{cursor:pointer}.views{display:grid;grid-template-columns:1fr 1fr;gap:16px;margin:16px 0}.view{background:white;border:1px solid #d7e1e7;border-radius:10px;padding:14px}.pair{display:grid;grid-template-columns:1fr 1fr;gap:10px}canvas{display:block;width:100%;height:auto;image-rendering:pixelated}h2{font-size:18px;margin:0 0 10px}h3{font-size:12px;margin:0 0 6px;color:#486575}.stats{min-height:42px;white-space:pre-line}#timeline{width:100%;box-sizing:border-box}table{border-collapse:collapse;width:100%;font-size:12px}th,td{text-align:left;padding:7px;border-bottom:1px solid #dbe4e9}tr.active{background:#d7e9f5}tbody tr{cursor:pointer}tbody tr:hover{background:#e8f0f5}.gridtable{max-height:360px;overflow:auto;border:1px solid #dbe4e9;background:#fff}thead{position:sticky;top:0;background:#e9f0f4}.error{color:#aa281a}#hover{min-height:26px;font:12px ui-monospace,monospace}.muted{color:#4f6877;font-size:12px}.legend{display:flex;gap:12px;align-items:center;margin:12px 0;font-size:12px}.colorbar{width:220px;height:14px;background:linear-gradient(90deg,#00224e,#414d6b,#7d7c78,#bcae6c,#fee838)}@media(max-width:900px){.views{grid-template-columns:1fr}}</style>
<h1>JEPA · 全窗口视觉 Query 图库</h1><p id="coverage">正在读取本地清单…</p>
<p class="notice">这是现有 checkpoint 的推理测评，不是新增 loss 后重新训练的模型。排序中的“尖锐 / 集中 / 相似”只是注意力形状描述，不等于“好的接触定位”。大部分窗口没有人工接触标签；已有 36 样本的盲标也只是可见接口代理区域，不能当作真实力接触点。所有数据属于训练分布。</p>
<div class="controls"><label>Episode<select id="episode"><option value="all">全部</option></select></label><label>描述性排序<select id="sort"><option value="index">原始窗口顺序</option><option value="entropy-up">低平均熵（更尖锐）</option><option value="entropy-down">高平均熵（更分散）</option><option value="peak-down">高平均峰值</option><option value="similarity-up">低 Query 相似度</option><option value="similarity-down">高 Query 相似度</option></select></label><label>样本 ID<input id="jump" type="number" style="width:100px"></label><button id="go">跳转</button><label>当前读取层<select id="layer"><option value="0">Layer 0</option><option value="1">Layer 1</option></select></label><label>Head<select id="head"><option value="-1">均值 4 heads</option></select></label><label>Query<select id="query"><option value="-1">均值 8 queries</option></select></label><label>色标<select id="mode"><option value="relative">视角内相对均匀倍数</option><option value="global">全局 patch 概率</option></select></label><label>共享上限<input id="scale" type="number" value="4" min="0.00001" step="any" style="width:75px"></label><label>叠加浓度<input id="alpha" type="range" min="0" max="100" value="65" style="width:90px"></label></div>
<p id="selection"></p><input id="timeline" type="range" min="0" max="0" value="0"><div class="legend"><button id="prev">←</button><button id="next">→</button><span>0</span><span class="colorbar"></span><span id="upper">4×</span><span>固定共享色标；超过上限截色。悬停可看未截断数值。</span></div>
<div class="views"><section class="view"><h2>主视角 · main</h2><div class="pair"><div><h3>当前输入 RGB（JPEG 预览）</h3><canvas id="raw0" width="448" height="448"></canvas></div><div><h3>真实 14×14 patch 注意力</h3><canvas id="map0" width="448" height="448"></canvas></div></div><p id="stats0" class="stats"></p></section><section class="view"><h2>腕视角 · wrist</h2><div class="pair"><div><h3>当前输入 RGB（JPEG 预览）</h3><canvas id="raw1" width="448" height="448"></canvas></div><div><h3>真实 14×14 patch 注意力</h3><canvas id="map1" width="448" height="448"></canvas></div></div><p id="stats1" class="stats"></p></section></div><p id="hover">悬停热图查看数值。</p>
<p><button id="pageprev">上一页</button> <button id="pagenext">下一页</button> <span id="paging"></span></p><div class="gridtable"><table><thead><tr><th>样本</th><th>episode</th><th>frame</th><th>平均归一化熵</th><th>平均峰值概率</th><th>Query 余弦相似</th><th>主视角质量</th></tr></thead><tbody id="rows"></tbody></table></div>
<p class="muted">熵和峰值是每个 layer/head/query 的指标再平均；相似度是每个 layer/head 内 8 query 的 28 对余弦相似再平均。其高低均没有“好接触”的监督含义。面板默认先平均选择的 head/query，再在当前视角内归一化，因此面板数值与表中“单 query 指标的均值”不是同一种统计量。这里保存的是全部 16 个去噪步的均值，不能恢复去噪阶段变化；每个 token 又融合过去/当前两帧，所以当前 RGB 只是叠加背景。橙框仅显示已有盲标的可见接口代理，其余样本保持未标注。实际接触、任务成功、未见场景泛化仍需要独立验证。</p><p id="lineage" class="muted"></p>
<script>
const $=id=>document.getElementById(id);let catalog,overview,filtered=[],position=0,page=0,generation=0,currentMaps=[],roi={},detailsCache=new Map(),dataCache=new Map();const f=x=>Number(x).toFixed(4);for(let i=0;i<4;i++)$('head').add(new Option('Head '+i,i));for(let i=0;i<8;i++)$('query').add(new Option('Query '+i,i));
function half(x){const sign=(x&32768)?-1:1,exp=(x>>10)&31,mant=x&1023;return sign*(exp===0?Math.pow(2,-14)*(mant/1024):exp===31?(mant?NaN:Infinity):Math.pow(2,exp-15)*(1+mant/1024))}
const stops=[[0,34,78],[65,77,107],[125,124,120],[188,174,108],[254,232,56]];function color(t){t=Math.max(0,Math.min(1,t))*4;let a=Math.min(3,Math.floor(t)),b=t-a;return stops[a].map((x,j)=>Math.round(x*(1-b)+stops[a+1][j]*b))}
async function getJSON(url){const r=await fetch(url);if(!r.ok)throw Error(url+': '+r.status);return r.json()}
async function details(shard){if(!detailsCache.has(shard)){detailsCache.set(shard,await getJSON(catalog.shards[shard].path+'/index.json'));if(detailsCache.size>4)detailsCache.delete(detailsCache.keys().next().value)}return detailsCache.get(shard)}
async function attention(row){const key=row[0];if(dataCache.has(key))return dataCache.get(key);const s=catalog.shards[row[3]],start=s.array_offset+row[4]*s.row_bytes,end=start+s.row_bytes-1,r=await fetch(s.path+'/attention.npy',{headers:{Range:`bytes=${start}-${end}`}});if(!r.ok)throw Error('Attention fetch failed '+r.status);let b=await r.arrayBuffer();if(r.status===200)b=b.slice(start,end+1);if(b.byteLength!==50176)throw Error('Attention row byte count differs');const v=new DataView(b),a=new Float32Array(25088);for(let i=0;i<a.length;i++){a[i]=half(v.getUint16(i*2,true));if(!Number.isFinite(a[i])||a[i]<0)throw Error('Nonfinite/negative attention')}dataCache.set(key,a);if(dataCache.size>8)dataCache.delete(dataCache.keys().next().value);return a}
async function img(url){const im=new Image();im.src=url;await im.decode();return im}
function coverage(box,r,c){return Math.max(0,Math.min(box[2],(c+1)/14)-Math.max(box[0],c/14))*Math.max(0,Math.min(box[3],(r+1)/14)-Math.max(box[1],r/14))*196}
function renderTable(){const first=page*200,slice=filtered.slice(first,first+200);$('rows').innerHTML=slice.map((r,j)=>`<tr data-pos="${first+j}" class="${first+j===position?'active':''}"><td>${r[0]}</td><td>${r[1]}</td><td>${r[2]}</td><td>${f(r[5])}</td><td>${f(r[6])}</td><td>${f(r[7])}</td><td>${f(r[8])}</td></tr>`).join('');$('paging').textContent=`${first+1}–${first+slice.length} / ${filtered.length} 个已完成窗口`;for(const tr of $('rows').querySelectorAll('tr'))tr.onclick=()=>{position=+tr.dataset.pos;$('timeline').value=position;draw();renderTable()}}
function filter(){const ep=$('episode').value;filtered=overview.filter(r=>ep==='all'||r[1]===+ep);const sort=$('sort').value;let col=sort.startsWith('entropy')?5:sort.startsWith('peak')?6:7;filtered.sort(sort==='index'?((a,b)=>a[0]-b[0]):((a,b)=>(a[col]-b[col])*(sort.endsWith('down')?-1:1)||a[0]-b[0]));position=0;page=0;$('timeline').max=Math.max(0,filtered.length-1);$('timeline').value=0;renderTable();draw()}
async function draw(){if(!filtered.length)return;const g=++generation,row=filtered[position],s=catalog.shards[row[3]];$('selection').textContent=`读取样本 ${row[0]}…`;try{const [a,records,im0,im1]=await Promise.all([attention(row),details(row[3]),img(s.path+'/thumbnails/'+String(row[0]).padStart(6,'0')+'-main.jpg'),img(s.path+'/thumbnails/'+String(row[0]).padStart(6,'0')+'-wrist.jpg')]);if(g!==generation)return;const rec=records[row[4]],l=+$('layer').value,h=+$('head').value,q=+$('query').value,heads=h<0?[0,1,2,3]:[h],queries=q<0?[0,1,2,3,4,5,6,7]:[q],p=new Float64Array(392);for(let hh of heads)for(let qq of queries){const offset=((l*4+hh)*8+qq)*392;for(let j=0;j<392;j++)p[j]+=a[offset+j]/(heads.length*queries.length)}let total=p.reduce((x,y)=>x+y,0);for(let j=0;j<392;j++)p[j]/=total;const max=+$('scale').value;if(!(max>0))throw Error('色标上限必须为正');const relative=$('mode').value==='relative',opacity=+$('alpha').value/100;$('upper').textContent=max+(relative?'×':'');currentMaps=[];for(let v=0;v<2;v++){const view=v?'cam_wrist':'cam_main',im=v?im1:im0,m=Array.from(p.slice(v*196,(v+1)*196)),mass=m.reduce((x,y)=>x+y,0),rr=roi.samples?.[row[0]]?.[view],boxes=rr?.status==='visible'?rr.boxes:[],scaled=m.map(x=>relative?x/mass*196:x);currentMaps[v]={m,mass};for(let kind of ['raw','map']){const ctx=$(kind+v).getContext('2d');ctx.imageSmoothingEnabled=false;ctx.drawImage(im,0,0,448,448);if(kind==='map'){for(let j=0;j<196;j++){ctx.fillStyle=`rgba(${color(scaled[j]/max).join(',')},${opacity})`;ctx.fillRect((j%14)*32,Math.floor(j/14)*32,32,32)}ctx.strokeStyle='rgba(255,255,255,.15)';ctx.lineWidth=.5;for(let j=0;j<=14;j++){ctx.beginPath();ctx.moveTo(j*32,0);ctx.lineTo(j*32,448);ctx.stroke();ctx.beginPath();ctx.moveTo(0,j*32);ctx.lineTo(448,j*32);ctx.stroke()}}ctx.strokeStyle='#ff5b20';ctx.lineWidth=2;for(const b of boxes)ctx.strokeRect(b[0]*448,b[1]*448,(b[2]-b[0])*448,(b[3]-b[1])*448)}let area=0,rm=0;for(let j=0;j<196;j++){let c=Math.min(1,boxes.reduce((sum,b)=>sum+coverage(b,Math.floor(j/14),j%14),0));area+=c/196;rm+=c*m[j]}$('stats'+v).textContent=`当前选择的视角概率 ${(mass*100).toFixed(2)}% · 超色标 ${scaled.filter(x=>x>max).length}/196`+(area?`\n盲标接口代理 ROI 富集 ${(rm/mass/area).toFixed(3)}×；非真实力接触标签。`:`\nROI：${rr?.status||'未标注'}；不生成接触准确率。`)}$('selection').textContent=`样本 ${row[0]} · episode ${row[1]} · frame ${row[2]} · 观测帧 [${rec.observed_frame_indices.join(', ')}] · 排序序号 ${position+1}/${filtered.length} · FP16原始概率和 ${total.toFixed(6)}（仅作显示归一化）`;}catch(e){$('selection').className='error';$('selection').textContent=e.message}}
for(const id of ['layer','head','query','scale','alpha'])$(id).addEventListener('input',draw);$('mode').onchange=()=>{$('scale').value=$('mode').value==='relative'?4:10/392;draw()};$('episode').onchange=filter;$('sort').onchange=filter;$('timeline').oninput=()=>{position=+$('timeline').value;page=Math.floor(position/200);renderTable();draw()};$('prev').onclick=()=>{position=Math.max(0,position-1);$('timeline').value=position;page=Math.floor(position/200);renderTable();draw()};$('next').onclick=()=>{position=Math.min(filtered.length-1,position+1);$('timeline').value=position;page=Math.floor(position/200);renderTable();draw()};$('pageprev').onclick=()=>{page=Math.max(0,page-1);renderTable()};$('pagenext').onclick=()=>{page=Math.min(Math.ceil(filtered.length/200)-1,page+1);renderTable()};$('go').onclick=()=>{const id=+$('jump').value;let p=filtered.findIndex(r=>r[0]===id);if(p<0){$('episode').value='all';filter();p=filtered.findIndex(r=>r[0]===id)}if(p<0){$('selection').textContent='该样本尚未完成或不在此图库中';return}position=p;page=Math.floor(p/200);$('timeline').value=p;renderTable();draw()};for(let v=0;v<2;v++)$('map'+v).onmousemove=e=>{if(!currentMaps[v])return;const b=e.currentTarget.getBoundingClientRect(),c=Math.min(13,Math.floor((e.clientX-b.left)/b.width*14)),r=Math.min(13,Math.floor((e.clientY-b.top)/b.height*14)),m=currentMaps[v],x=m.m[r*14+c];$('hover').textContent=`${v?'wrist':'main'} row ${r}, col ${c}: global P=${x.toFixed(7)}; within-view P=${(x/m.mass).toFixed(7)}; relative=${(x/m.mass*196).toFixed(3)}×`};
(async()=>{try{if(location.protocol==='file:')throw Error('请先运行 serve-gallery.cmd 或 python serve_gallery.py，然后打开 http://127.0.0.1:8765/ 。本图库通过本地分片读取，无外网请求。');catalog=await getJSON('catalog.json');overview=await getJSON('overview.json');if(catalog.roi_file)roi=await getJSON(catalog.roi_file);const episodes=[...new Set(overview.map(x=>x[1]))].sort((a,b)=>a-b);for(const ep of episodes)$('episode').add(new Option('Episode '+ep,ep));$('coverage').textContent=`已完成 ${overview.length.toLocaleString()} / ${catalog.expected_samples.toLocaleString()} 窗口 · ${episodes.length} 个 episode · ${catalog.complete?'完整清单':'部分结果，可持续增量更新'} · ${catalog.legacy_preview?'36样本旧测评预览':'全窗口测评'}`;$('lineage').textContent=`Checkpoint step ${catalog.checkpoint_step??'未知'} · SHA256 ${catalog.checkpoint_sha256??'未提供'} · 16步均值；原始FP16分片，描述性指标由FP32计算。数据只由此localhost服务读取，无上传。`;filter()}catch(e){$('coverage').className='error';$('coverage').textContent=e.message}})();
</script></html>'''


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--roi', type=Path, help='Existing frozen annotations only; never inferred from heatmaps')
    parser.add_argument('--legacy36', action='store_true', help='Import completed old 36-sample diagnostic as preview')
    parser.add_argument('--expected-samples', type=int, default=84349)
    args = parser.parse_args(argv)
    source, output = args.source.resolve(), args.output.resolve()
    if output == source:
        raise ValueError('Gallery output must be separate from immutable inference output')
    output.mkdir(parents=True, exist_ok=True)
    run = json.loads((source / 'run.json').read_text(encoding='utf-8')) if (source / 'run.json').exists() else {}
    lineage = run.get('lineage', run)
    if args.legacy36:
        directories = import_legacy(source, output)
    else:
        paths = sorted(p for p in source.rglob('shard*.tar*') if p.name.endswith(('.tar', '.tar.gz')))
        directories = []
        for path in paths:
            name = path.name.removesuffix('.gz').removesuffix('.tar')
            directory = output / 'shards' / name
            extract_shard(path, directory, run_identity=run.get('run_identity'))
            directories.append(directory)
    if not directories:
        raise ValueError('No completed census shard archives found')
    overview, shards = build_index(directories, output)
    roi_file = None
    if args.roi:
        frozen = json.loads(args.roi.read_text(encoding='utf-8'))
        if frozen.get('schema') != 'jepa-contact-roi-v1' or frozen.get('annotation_stage') != 'raw_rgb_before_attention':
            raise ValueError('Use the existing frozen RGB-only interface annotations')
        roi_file = 'frozen-interface-roi.json'
        shutil.copyfile(args.roi, output / roi_file)
    catalog = dict(schema='jepa-query-gallery-v1', columns=['sample_index', 'episode_index', 'frame_index',
        'shard_index', 'row_in_shard', 'entropy_normalized', 'peak_probability', 'query_cosine_similarity', 'camera_main_mass'],
        completed_samples=len(overview), expected_samples=args.expected_samples,
        complete=(len(overview) == args.expected_samples and [r[0] for r in overview] == list(range(args.expected_samples))),
        legacy_preview=args.legacy36, episode_counts=dict(Counter(str(r[1]) for r in overview)), shards=shards,
        checkpoint=lineage.get('checkpoint'), checkpoint_step=lineage.get('checkpoint_step'), checkpoint_sha256=lineage.get('checkpoint_sha256'),
        aggregation='mean_over_16_denoising_steps', inference_training=False, train_distribution_only=True,
        metric_quality_labels=False, roi_file=roi_file, roi_sha256=sha(args.roi) if args.roi else None,
        source_run_sha256=sha(source / 'run.json') if (source / 'run.json').exists() else None,
        descriptor_definitions=dict(entropy='mean entropy over layer/head/query, divided by log(392)',
            peak='mean max probability over layer/head/query',
            redundancy='mean 28 pairwise query cosine similarities within each layer/head',
            contact='unknown except pre-existing visual interface proxy annotations; no exact force contact ground truth'))
    write_json(output / 'overview.json', overview)
    write_json(output / 'catalog.json', catalog)
    (output / 'index.html').write_text(HTML, encoding='utf-8')
    (output / 'serve_gallery.py').write_text(SERVER, encoding='utf-8')
    import sys
    executable = str(Path(sys.executable).resolve()) if os.name == 'nt' else 'python'
    (output / 'serve-gallery.cmd').write_text(f'@"{executable}" "%~dp0serve_gallery.py" --port 8765\r\n@pause\r\n', encoding='utf-8')
    (output / 'serve-gallery.sh').write_text('#!/usr/bin/env bash\nset -euo pipefail\nexec python3 "$(dirname -- "$0")/serve_gallery.py" --port 8765\n', encoding='utf-8')
    print(json.dumps(dict(output=str(output), samples=len(overview), shards=len(shards), complete=catalog['complete'],
                         local_url='http://127.0.0.1:8765/', row_bytes=ROW_BYTES)))


if __name__ == '__main__':
    main()
