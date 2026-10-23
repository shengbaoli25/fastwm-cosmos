"""Add verified 4090-vs-GB10 benchmark and transport measurements to the report."""
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

import argparse
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("experiment", type=Path)
ROOT = parser.parse_args().experiment.resolve()
REPORT=ROOT/'评测报告.md'
HTML=ROOT/'演示与结果.html'
g=json.loads((ROOT/'results/gb10-latency/latency.json').read_text())
rpath=ROOT/'results/4090-latency/latency.json'
r=json.loads(rpath.read_text())
net=json.loads((ROOT/'results/4090-network/transport.json').read_text())
assert r['status']=='complete' and len(r['records'])==330
assert (ROOT/'results/4090-latency/exit-code.txt').read_text().strip()=='0'
assert g['checkpoint_sha256']==r['checkpoint_sha256']
assert g['input_sha256']==r['input_sha256']
assert g['source_hashes']==r['source_hashes']

rows=[]
for mode in r['modes']:
    row={'mode':mode}
    med={}
    for label,data in [('GB10',g),('4090',r)]:
        xs=[x for x in data['records'] if x['mode']==mode]
        assert len(xs)==(120 if mode=='v16_f16_a16' else 30)
        med[label]=float(np.median([x['rgb_to_action_force_ms'] for x in xs]))
        row[label]=dict(n=len(xs),core_p50=float(np.median([x['core_ms'] for x in xs])),
            core_p95=float(np.percentile([x['core_ms'] for x in xs],95)),
            rgb_p50=med[label],rgb_p95=float(np.percentile([x['rgb_to_action_force_ms'] for x in xs],95)))
        if mode=='v16_f16_a16':row[label]['paired_speedup']=1.0
        else:
            base={(x['group'],x['rep']):x for x in data['records'] if x['mode']=='v16_f16_a16'}
            row[label]['paired_speedup']=float(np.median([
                base[(x['group'],x['rep'])]['rgb_to_action_force_ms']/x['rgb_to_action_force_ms'] for x in xs]))
    row['gb10_over_4090']=med['GB10']/med['4090']
    rows.append(row)

# Compare representative outputs with identical selected samples/seeds. These are parity probes, not quality metrics.
parity={}
for kind in ('core','rgb'):
    per_key={}
    for mode in r['modes']:
        for rep in range(3):
            record=next(x for x in r['records'] if x['mode']==mode and x['rep']==rep)
            name=f'{kind}_sample{record["sample"]}_{mode}_rep{rep}.npz'
            a_path=ROOT/'results/gb10-latency'/name
            b_path=ROOT/'results/4090-latency'/name
            assert a_path.exists() and b_path.exists(),name
            with np.load(a_path) as aa,np.load(b_path) as bb:
                for key in aa.files:
                    if key not in bb or not np.issubdtype(aa[key].dtype,np.number):continue
                    diff=np.abs(aa[key].astype(np.float64)-bb[key].astype(np.float64))
                    z=per_key.setdefault(key,{'max_abs':0.0,'sum_abs':0.0,'count':0,'arrays':0})
                    z['max_abs']=max(z['max_abs'],float(diff.max(initial=0)))
                    z['sum_abs']+=float(diff.sum());z['count']+=diff.size;z['arrays']+=1
    parity[kind]={k:{'max_abs':v['max_abs'],'mean_abs':v['sum_abs']/max(v['count'],1),'arrays':v['arrays']} for k,v in per_key.items()}

fig,ax=plt.subplots(figsize=(10.5,5.2))
labels=[x['mode'].replace('v','').replace('f','').replace('a','') for x in rows]
y=np.arange(len(rows));h=.37
ax.barh(y-h/2,[x['GB10']['rgb_p50'] for x in rows],height=h,label='GB10 (measured)')
ax.barh(y+h/2,[x['4090']['rgb_p50'] for x in rows],height=h,label='RTX 4090 (measured)')
ax.set_yticks(y,labels);ax.invert_yaxis();ax.set_xlabel('RGB bytes → action + force (ms, P50)')
ax.set_title('Joint8 step10000: matched input and checkpoint')
ax.grid(axis='x',alpha=.25);ax.legend();fig.tight_layout()
chart=ROOT/'results/4090-latency/hardware-comparison.png';fig.savefig(chart,dpi=160);plt.close(fig)

lines=[]
lines.append('## GB10 与 RTX 4090 实测对照')
lines.append('')
lines.append('4090 实例已完成实测。表中每个候选为 30 次同输入重复，16/16/16 基线在四个交替组共 120 次；同组每次轮换执行顺序。核心延时和 RGB→输出延时分别给出 P50/P95。两个设备使用完全相同的权重 SHA、输入 SHA、评测源码 SHA、batch=1、BF16 和采样配置；4090 为 Torch '+r['torch']+' / CUDA '+r['cuda']+' / x86_64，GB10 为 Torch '+g['torch']+' / CUDA 13 / ARM64。')
lines.append('')
lines.append('| Video/Force/Action | 核心 GB10 P50/P95 | 核心 4090 P50/P95 | RGB→输出 GB10 P50/P95 | RGB→输出 4090 P50/P95 | GB10/4090 P50 比 | 同设备相对16步：GB10 / 4090 |')
lines.append('|---|---:|---:|---:|---:|---:|---:|')
for x in rows:
 m=x['mode'].replace('v','').replace('_f','/').replace('_a','/')
 a,b=x['GB10'],x['4090']
 lines.append(f"| {m} | {a['core_p50']:.0f}/{a['core_p95']:.0f} ms | {b['core_p50']:.0f}/{b['core_p95']:.0f} ms | {a['rgb_p50']:.0f}/{a['rgb_p95']:.0f} ms | {b['rgb_p50']:.0f}/{b['rgb_p95']:.0f} ms | {x['gb10_over_4090']:.2f}× | {a['paired_speedup']:.2f}× / {b['paired_speedup']:.2f}× |")
lines += ['',f"![GB10 与 RTX 4090 延时对比]({chart.relative_to(ROOT)})",'']
recommended = next(x for x in rows if x['mode'] == 'v4_f4_a16')
fastest = next(x for x in rows if x['mode'] == 'v4_f4_a4')
lines.append(f"4090 对 16/16/16 RGB→输出为 **P50 {rows[0]['4090']['rgb_p50']:.0f} ms / P95 {rows[0]['4090']['rgb_p95']:.0f} ms**，GB10 为 **{rows[0]['GB10']['rgb_p50']:.0f}/{rows[0]['GB10']['rgb_p95']:.0f} ms**，中位数约快 **{rows[0]['gb10_over_4090']:.2f}×**。推荐候选 4/4/16 在 4090 为 **{recommended['4090']['rgb_p50']:.0f}/{recommended['4090']['rgb_p95']:.0f} ms**，GB10 为 **{recommended['GB10']['rgb_p50']:.0f}/{recommended['GB10']['rgb_p95']:.0f} ms**；同一配置的跨设备中位数约快 **{recommended['gb10_over_4090']:.2f}×**。")
lines.append('')
lines.append(f"4/4/4 在 4090 最快，RGB→输出 P50/P95={fastest['4090']['rgb_p50']:.0f}/{fastest['4090']['rgb_p95']:.0f} ms；但离线事件窗口显示动作速度差分误差增加约 23%，所以速度数据不改变 4/4/16 作为首个真机候选的判断。4090 将相同动作/力专家设置的 16/16/16 计算压到约 0.76 s，4/4/16 约 0.34 s；异步控制仍应结合实际网络、摄像头周期和动作块衔接一起测。")
alloc=max(x['peak_allocated_gib'] for x in r['stats'].values());reserved=max(x['peak_reserved_gib']for x in r['stats'].values())
lines.append(f"PyTorch 在 4090 的峰值 allocated/reserved 为 **{alloc:.2f}/{reserved:.2f} GiB**，测试时 `nvidia-smi` 曾观测到约 12.7 GB 板载显存占用；GB10 分别为 **8.35/11.05 GiB**，其统一内存不能用 `nvidia-smi` memory 字段直接比较。")
lines.append('')
lines.append('### 到服务器的传输开销实测')
lines.append('')
lines.append('为了估计数据流上云的成本，本机通过当前启智 Jupyter HTTPS 代理向运行中的 4090 实例发 50 次轮换探测，每次服务器返回固定 16 KiB 确认数据。该数值包含一次 HTTP 请求往返和上传，不含推理，也不代表未来生产服务 API 的网络 SLA。')
lines.append('')
lines.append('| 请求内容 | 上传大小 P50 | 请求往返 P50/P95 | 本机 JPEG 编码 P50/P95 |')
lines.append('|---|---:|---:|---:|')
raw=net['summary']['rgb_2x640x480'];jpg=net['summary']['jpeg90_2x640x480'];ping=net['summary']['ping_1KiB']
lines.append(f"| 1 KiB 探测 | 1 KiB | {ping['transport_ms']['p50']:.1f}/{ping['transport_ms']['p95']:.1f} ms | — |")
lines.append(f"| 两路未压缩 640×480 RGB | {raw['request_bytes']['p50']/1024:.0f} KiB | {raw['transport_ms']['p50']:.1f}/{raw['transport_ms']['p95']:.1f} ms | — |")
lines.append(f"| 两路 JPEG quality=90 | {jpg['request_bytes']['p50']/1024:.0f} KiB | {jpg['transport_ms']['p50']:.1f}/{jpg['transport_ms']['p95']:.1f} ms | {jpg['jpeg_encode_ms']['p50']:.1f}/{jpg['jpeg_encode_ms']['p95']:.1f} ms |")
lines.append('')
lines.append('')
raw_p50 = recommended['4090']['rgb_p50'] + raw['transport_ms']['p50']
raw_p95 = recommended['4090']['rgb_p95'] + raw['transport_ms']['p95']
jpg_p50 = recommended['4090']['rgb_p50'] + jpg['transport_ms']['p50'] + jpg['jpeg_encode_ms']['p50']
jpg_p95 = recommended['4090']['rgb_p95'] + jpg['transport_ms']['p95'] + jpg['jpeg_encode_ms']['p95']
lines.append(f"把传输探测与 4090 的 4/4/16 RGB 完整处理相加，未压缩输入的粗略 P50/P95 约 **{raw_p50:.0f}/{raw_p95:.0f} ms**；JPEG quality=90 加上本机编码后约 **{jpg_p50:.0f}/{jpg_p95:.0f} ms**。这是把两组独立测量相加的粗略估算，尚未含远端 JPEG 解码、真实服务响应体、机器人接口及网络高峰排队；应把 GB10 的本地推理作为稳定性基准。")

parity_path=ROOT/'provenance/device-parity.json'
parity_path.write_text(json.dumps({'checkpoint_sha256':r['checkpoint_sha256'],'input_sha256':r['input_sha256'],'source_hashes_match':True,'paired_samples_per_mode':3,'parity':parity},indent=2,ensure_ascii=False)+'\n')
summary_path=ROOT/'results/device-comparison.json'
summary_path.write_text(json.dumps({'devices':{'GB10':{'torch':g['torch'],'cuda':g['cuda'],'architecture':g['architecture']},'RTX4090':{'name':r['gpu'],'torch':r['torch'],'cuda':r['cuda'],'architecture':r['architecture']}},'checkpoint_sha256':r['checkpoint_sha256'],'input_sha256':r['input_sha256'],'source_hashes_match':True,'rows':rows,'memory':{'GB10':{'allocated_gib':max(x['peak_allocated_gib']for x in g['stats'].values()),'reserved_gib':max(x['peak_reserved_gib']for x in g['stats'].values())},'RTX4090':{'allocated_gib':alloc,'reserved_gib':reserved}},'network_summary':net['summary'],'parity':parity},indent=2,ensure_ascii=False)+'\n')

old=REPORT.read_text()
start_header='## 4090 对照状态' if '## 4090 对照状态' in old else '## GB10 与 RTX 4090 实测对照'
start=old.index(start_header)
end=old.index('## 复现',start)
new=old[:start]+'\n'.join(lines)+'\n\n'+old[end:]
new=new.replace('本机采样、事件窗口、RGB 视频质量、交替时延和离线演示已完成；4090 尚未完成实测。下面硬件比较仅列已测本机和远端待测项目，不提供推测数值冒充实测。','本机 step10000 采样评测与 4090 同输入延时对照均已完成；结论只适用于本报告列出的离线推理，不代表机械臂闭环成功率。')
async_base='32 个 30Hz 动作点约覆盖 1.067 s。4/4/16 的本机处理 P95≈0.779 s，若完整执行 32 点，留给传感器、传输、排队、接续的理论余量约 0.288 s；因此值得实现并测试异步。但若每次仅执行前 8 点（267 ms），当前时延仍不足以持续更新。8/8/16 的 P95≈1.146 s 已超过该动作块覆盖时间。'
async_note='4090 上 4/4/16 RGB→输出 P95 为 0.351 s；若执行完整 32 点，理论余量约 0.716 s。通过本次启智代理的 JPEG 探测加本机编码，粗略 P95 约 0.383 s，仍高于 8 点动作块的 0.267 s；代理探测不含生产服务排队和远端解码，所以不能据此承诺闭环异步周期。'
if async_note not in new:
 new=new.replace(async_base,async_base+'\n\n'+async_note)
REPORT.write_text(new)

html=HTML.read_text()
section=f'''<section><h2>GB10 与桌面 RTX 4090 实测</h2><p>同一权重、输入和源码；16/16/16 基线 P50：GB10 {rows[0]['GB10']['rgb_p50']:.0f} ms，4090 {rows[0]['4090']['rgb_p50']:.0f} ms。推荐候选 4/4/16：GB10 {next(x for x in rows if x['mode']=='v4_f4_a16')['GB10']['rgb_p50']:.0f} ms，4090 {next(x for x in rows if x['mode']=='v4_f4_a16')['4090']['rgb_p50']:.0f} ms。服务器数据上传另计；网络测试不含推理服务排队或机械臂通信。</p><img src="results/4090-latency/hardware-comparison.png"><p><a href="评测报告.md#gb10-与-rtx-4090-实测对照">阅读设备、网络、显存和数值一致性完整报告</a> · <a href="results/device-comparison.json">机器可读汇总</a></p></section>'''
if 'GB10 与桌面 RTX 4090 实测' not in html:
 html=html.replace('</html>',section+'</html>')
HTML.write_text(html)
print('report updated',REPORT,'chart',chart,'parity',parity_path,'comparison',summary_path)
