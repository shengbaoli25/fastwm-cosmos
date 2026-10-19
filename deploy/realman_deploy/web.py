"""Loopback-only web dashboard and checkpoint download API."""
from __future__ import annotations

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import threading
from pathlib import Path
from urllib.parse import urlparse

from .checkpoints import CheckpointCatalog, CheckpointStore
from .model_adapter import CosmosJointPolicy
from .runtime import DeploymentRuntime


PAGE = r'''<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>RealMan 模型部署台</title><style>
:root{color-scheme:dark;--bg:#0b1020;--panel:#141c2e;--line:#26344d;--fg:#e9eef8;--muted:#94a3b8;--cyan:#5eead4;--red:#fb7185;--amber:#fbbf24}
*{box-sizing:border-box}body{margin:0;background:radial-gradient(ellipse at top,#17243b,var(--bg) 58%);color:var(--fg);font:15px/1.5 system-ui,sans-serif}
main{max-width:1100px;margin:40px auto;padding:0 22px}.head{display:flex;justify-content:space-between;align-items:flex-end;margin-bottom:22px}.eyebrow{color:var(--cyan);font-size:12px;letter-spacing:.14em;text-transform:uppercase}.head h1{font-size:29px;margin:5px 0}.head p{margin:0;color:var(--muted)}.grid{display:grid;grid-template-columns:1.2fr .8fr;gap:16px}.card{background:linear-gradient(145deg,#182338,#121a2a);border:1px solid var(--line);border-radius:16px;padding:20px;box-shadow:0 20px 50px #0003}.wide{grid-column:1/-1}.card h2{font-size:16px;margin:0 0 16px}.row{display:flex;gap:10px;align-items:center}.row>*{flex:1}select,button{font:inherit;border:1px solid #34445f;border-radius:10px;background:#0e1727;color:var(--fg);padding:11px 13px}button{cursor:pointer;font-weight:650;transition:.15s}button:hover{transform:translateY(-1px);border-color:var(--cyan)}button:disabled{opacity:.4;cursor:not-allowed;transform:none}.primary{background:#0d776d;border-color:#16877d}.danger{background:#6b2431;border-color:#98384a}.secondary{background:#1b2a40}.muted{color:var(--muted)}.metrics{display:grid;grid-template-columns:repeat(4,1fr);gap:10px}.metric{padding:13px;background:#0c1524;border:1px solid #223049;border-radius:12px}.metric small{display:block;color:var(--muted);font-size:11px}.metric strong{display:block;margin-top:5px;font-size:17px}.badge{display:inline-block;border-radius:99px;background:#26344d;padding:5px 10px;font-size:12px}.badge.good{color:#82f2d4;background:#12453e}.badge.warn{color:#ffd775;background:#4e3a15}.badge.bad{color:#ff9baa;background:#4b1d29}.note{padding:12px 14px;border-left:3px solid var(--amber);background:#2b2414;color:#f8dda0;border-radius:6px;margin-top:14px}.log{font:12px ui-monospace,monospace;color:var(--muted);white-space:pre-wrap;max-height:120px;overflow:auto}.foot{color:var(--muted);font-size:12px;margin-top:18px}@media(max-width:760px){.grid{grid-template-columns:1fr}.wide{grid-column:auto}.metrics{grid-template-columns:repeat(2,1fr)}.head{display:block}}
</style></head><body><main>
<div class="head"><div><div class="eyebrow">FastWAM · RealMan RM75</div><h1>真机推理部署台</h1><p id="modelMode">双视角观测 · 30 Hz 动作接口</p></div><span id="badge" class="badge warn">正在检查</span></div>
<div class="grid">
<section class="card"><h2>模型 checkpoint</h2><div class="row"><select id="checkpoint"></select><button id="refresh" class="secondary">刷新列表</button></div><div class="row" style="margin-top:10px"><button id="download" class="secondary">下载到桌面 checkpoints</button><button id="load" class="primary">加载模型</button></div><p id="modelHint" class="muted">目录：<code>~/桌面/checkpoints</code></p><div id="downloadProgress" class="log"></div></section>
<section class="card"><h2>真机状态</h2><div class="metrics"><div class="metric"><small>机械臂</small><strong id="robot">未连接</strong></div><div class="metric"><small>双相机</small><strong id="cameras">未连接</strong></div><div class="metric"><small>六轴力</small><strong id="force">未连接</strong></div><div class="metric"><small>夹爪</small><strong id="gripper">未连接</strong></div><div class="metric"><small>CUDA</small><strong id="cuda">未连接</strong></div><div class="metric"><small>模型延迟</small><strong id="latency">—</strong></div></div><div class="row" style="margin-top:13px"><button id="connect" class="secondary">连接硬件</button><button id="reload" class="secondary">重载状态</button></div></section>
<section class="card wide"><h2>闭环控制</h2><div class="row"><button id="start" class="primary">开始推理与控制</button><button id="pause" class="secondary">暂停 / 缓停</button><button id="estop" class="danger">急停</button></div><p id="modeHint" class="muted">正在读取调度模式…</p><div id="warning" class="note">启动前请确认工作区无障碍、机械臂处于可控状态。</div></section>
<section class="card wide"><h2>运行指标</h2><div class="metrics"><div class="metric"><small>状态 / epoch</small><strong id="state">idle / 0</strong></div><div class="metric"><small>观测序号</small><strong id="sequence">—</strong></div><div class="metric"><small>动作缓冲</small><strong id="queue">0</strong></div><div class="metric"><small>丢弃结果</small><strong id="discarded">0</strong></div><div class="metric"><small>执行模式</small><strong id="executionMode">—</strong></div><div class="metric"><small>执行阶段</small><strong id="executionPhase">—</strong></div><div class="metric"><small>模型计算</small><strong id="inferencePhase">空闲</strong></div><div class="metric"><small>输入观测至结果返回</small><strong id="inputAge">—</strong></div></div><p id="error" class="log"></p></section>
</div><div class="foot">服务默认只监听本机。急停按钮调用硬件适配器的独立停止通道；无法确认时应立即使用控制柜物理急停。</div></main>
<script>
const $=id=>document.getElementById(id);let busy=false;
async function api(path,body){const r=await fetch(path,{method:body===undefined?'GET':'POST',headers:{'Content-Type':'application/json'},body:body===undefined?undefined:JSON.stringify(body)});const x=await r.json();if(!r.ok)throw Error(x.error||r.statusText);return x}
function lock(v){busy=v;for(const id of ['download','load','connect','start'])$(id).disabled=v}
function selected(){return $('checkpoint').value}
async function refreshCheckpoints(){try{const x=await api('/api/checkpoints');const s=$('checkpoint'),old=s.value;s.replaceChildren();for(const c of x.checkpoints){const o=document.createElement('option');o.value=c.id;o.textContent=`${c.label}${c.installed?' · 本地文件':' · 待下载'}`;s.append(o)}if([...s.options].some(o=>o.value===old))s.value=old;$('modelHint').textContent=`本地目录：${x.directory} · ${x.checkpoints.length} 个模型`;if(!x.checkpoints.length)$('modelHint').textContent='目录中暂无 checkpoint；远端列表需要在 deploy/checkpoint_catalog.json 配置带 SHA256 的来源。'}catch(e){$('error').textContent=e.message}}
async function refreshStatus(){
  try{
    const s=await api('/api/status');
    $('modelMode').textContent=`双视角观测 · ${s.model_ready?(s.reference_mode==='none'?'无参考轨迹':'滚动 AE 参考'):'模型未加载'} · 30 Hz 动作接口`;
    $('warning').textContent='启动前请确认工作区无障碍、机械臂处于可控状态。'+(s.model_ready?(s.reference_mode==='none'?'当前模型不使用参考轨迹，Video、Force、AE 联合采样。':'首轮 WM 使用当前状态保持作为参考，后续 WM 接收上一轮 AE 预测动作。'):'');
    $('state').textContent=`${s.state} / ${s.epoch}`;
    $('sequence').textContent=s.observation_sequence<0?'—':s.observation_sequence;
    $('queue').textContent=s.action_queue;
    $('discarded').textContent=s.discarded_results;
    const relaxed=s.execution_mode==='async_relaxed';
    const synchronous=s.execution_mode==='synchronous';
    const phases={inferring:'停驻推理中',waiting_for_observation:'停驻取新观测',executing_and_inferring:'执行并推理',executing:'执行中',waiting:'等待新动作',idle:'空闲',ready:'就绪',paused:'已暂停',loading:'加载中',stopping:'停止中',stopped:'已停止',error:'错误',estop:'急停',running:'运行中'};
    $('executionMode').textContent=synchronous?'同步分段':(relaxed?'异步宽松时序':'异步严格时序');
    $('executionPhase').textContent=phases[s.execution_phase]||s.state;
    $('inferencePhase').textContent=s.inference_busy?'推理中':'空闲';
    $('inputAge').textContent=s.input_age_at_result_ms==null?'—':`${s.input_age_at_result_ms} ms`;
    $('modeHint').textContent=synchronous?'同步分段：完整执行一块动作后停驻，获取最新观测再推理；推理期间不执行动作，结果返回后执行下一块。':relaxed?'推理和控制异步：结果返回后执行，动作耗尽时缓停等待，下一块返回后自动继续。模型仍使用推理开始时的观测，运动可能有等待间隔。':'推理和控制异步：按原观测时间丢弃过期动作，动作缓冲耗尽时停止。';
    const warm=s.model_diagnostics?.warmup?.slice(-1)[0];
    $('latency').textContent=s.latency_ms==null?(warm?`预热 ${warm.total_ms} ms`:'—'):`${s.latency_ms} ms`;
    $('robot').textContent=s.hardware?.robot_ready?'就绪':'未就绪';
    $('cameras').textContent=s.hardware?.cameras_ready?'双路就绪':'未就绪';
    $('force').textContent=s.force_ready?'就绪':'未就绪';
    $('gripper').textContent=s.gripper_ready?'就绪':'未就绪';
    $('cuda').textContent=s.cuda?.available?`${s.cuda.device} · ${(s.cuda.total_bytes/1e9).toFixed(1)} GB`:'未连接';
    $('error').textContent=s.error||s.hardware?.error||'';
    const b=$('badge');b.textContent=s.state;
    b.className='badge '+(s.state==='running'?'good':(['error','estop'].includes(s.state)?'bad':'warn'));
    $('start').disabled=busy||!s.model_ready||!s.robot_ready||!s.cameras_ready||!s.force_ready||!s.gripper_ready||!s.cuda?.available||s.inference_busy||!['ready','paused','idle'].includes(s.state);
    $('pause').disabled=!['running','loading','error'].includes(s.state);
    $('pause').textContent=s.state==='error'?'复位错误 / 保持停止':'暂停 / 缓停';
    $('estop').disabled=s.state==='estop';
  }catch(e){$('error').textContent=e.message}
}
async function act(id,path,body={}){lock(true);try{await api(path,body);await refreshStatus()}catch(e){$('error').textContent=e.message}finally{lock(false);refreshStatus()}}
$('refresh').onclick=refreshCheckpoints;$('reload').onclick=refreshStatus;
$('download').onclick=async()=>{lock(true);$('downloadProgress').textContent='开始校验并下载…';try{const x=await api('/api/download',{id:selected()});$('downloadProgress').textContent=`下载完成：${x.path}`;await refreshCheckpoints()}catch(e){$('downloadProgress').textContent=e.message}finally{lock(false)}};
$('load').onclick=()=>act('load','/api/select',{id:selected()});$('connect').onclick=()=>act('connect','/api/connect');$('start').onclick=()=>act('start','/api/start');$('pause').onclick=()=>act('pause','/api/pause');$('estop').onclick=()=>act('estop','/api/estop');
refreshCheckpoints();refreshStatus();setInterval(refreshStatus,1000);
</script></body></html>'''


class DeploymentWeb:
    def __init__(self, runtime: DeploymentRuntime, *, catalog_path: str | Path,
                 config_path: str | Path, hardware_factory: str | None = None,
                 host: str = "127.0.0.1", port: int = 8765):
        self.runtime = runtime
        self.catalog = CheckpointCatalog(catalog_path)
        self.config_path = Path(config_path).resolve()
        self.config = json.loads(self.config_path.read_text())
        checkpoint_root = self.config.get("checkpoint_directory")
        if checkpoint_root:
            checkpoint_root = Path(checkpoint_root).expanduser()
            if not checkpoint_root.is_absolute():
                checkpoint_root = self.config_path.parent / checkpoint_root
        self.store = CheckpointStore(self.catalog, checkpoint_root)
        self.hardware_factory = hardware_factory
        self.server = ThreadingHTTPServer((host, port), self._handler())
        self.server.daemon_threads = True
        self.download_lock = threading.Lock()
        self.connect_lock = threading.Lock()
        self.serving = threading.Event()

    def _handler(self):
        web = self

        class Handler(BaseHTTPRequestHandler):
            server_version = "FastWAMDeploy/1"

            def log_message(self, *_args):
                pass

            def _json(self, value, code=200):
                data = json.dumps(value, ensure_ascii=False, allow_nan=False).encode()
                self.send_response(code); self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(data))); self.send_header("Cache-Control", "no-store")
                self.send_header("X-Content-Type-Options", "nosniff"); self.end_headers(); self.wfile.write(data)

            def _body(self):
                length = int(self.headers.get("Content-Length", "0"))
                if length > 1024 * 1024:
                    raise ValueError("request body too large")
                return json.loads(self.rfile.read(length) or b"{}")

            def _local_origin(self):
                origin = self.headers.get("Origin")
                if origin and urlparse(origin).netloc != self.headers.get("Host"):
                    raise PermissionError("cross-origin control request rejected")

            def do_GET(self):
                if self.path == "/":
                    body = PAGE.encode(); self.send_response(200)
                    self.send_header("Content-Type", "text/html; charset=utf-8")
                    self.send_header("Content-Length", str(len(body)))
                    self.send_header("Content-Security-Policy", "default-src 'self'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; connect-src 'self'; frame-ancestors 'none'")
                    self.send_header("X-Content-Type-Options", "nosniff"); self.end_headers(); self.wfile.write(body)
                elif self.path == "/api/status":
                    self._json(web.runtime.status())
                elif self.path == "/api/checkpoints":
                    self._json({"directory": str(web.store.root), "checkpoints": web.store.list()})
                else:
                    self._json({"error": "not found"}, 404)

            def do_POST(self):
                try:
                    self._local_origin(); body = self._body()
                    if self.path == "/api/download":
                        with web.download_lock:
                            path = web.store.download(str(body.get("id", "")))
                        self._json({"path": str(path)})
                    elif self.path == "/api/select":
                        ident = str(body.get("id", "")); path = web.store.path_for(ident)
                        if not path.is_file():
                            raise FileNotFoundError("checkpoint is not downloaded yet")
                        from .checkpoints import sha256_file
                        spec = web.catalog.specs.get(ident)
                        def factory(checkpoint):
                            if spec and spec.sha256 and sha256_file(Path(checkpoint)) != spec.sha256:
                                raise ValueError("local checkpoint SHA256 differs from catalog")
                            return CosmosJointPolicy(checkpoint, web.config, config_path=web.config_path)
                        web.runtime.load_policy_async(factory, str(path))
                        self._json({"state": "loading", "checkpoint": path.name}, 202)
                    elif self.path == "/api/connect":
                        factory = web.hardware_factory
                        if not factory:
                            raise RuntimeError("未配置硬件插件。设置 FASTWAM_HARDWARE_FACTORY=module:factory 后重启服务。")
                        from .adapters import load_factory
                        hardware_config = Path(web.config["hardware_config"]).expanduser()
                        if not hardware_config.is_absolute():
                            hardware_config = web.config_path.parent / hardware_config
                        if not web.connect_lock.acquire(blocking=False):
                            raise RuntimeError("硬件连接正在进行")
                        try:
                            with web.runtime.lock:
                                web.runtime._require_inactive()
                                existing = web.runtime.hardware
                            if existing is not None:
                                self._json({"state": "connected", "hardware": existing.status()})
                                return
                            adapter = load_factory(factory, config_path=str(hardware_config))
                            try:
                                web.runtime.attach(hardware=adapter)
                            except Exception:
                                adapter.close()
                                raise
                            self._json({"state": "connected", "hardware": adapter.status()})
                        finally:
                            web.connect_lock.release()
                    elif self.path == "/api/start":
                        web.runtime.start(); self._json({"state": "running"})
                    elif self.path == "/api/pause":
                        web.runtime.pause(); self._json({"state": "paused"})
                    elif self.path == "/api/estop":
                        web.runtime.emergency_stop(); self._json({"state": "estop"})
                    else:
                        self._json({"error": "not found"}, 404)
                except FileNotFoundError as exc:
                    self._json({"error": str(exc)}, 404)
                except PermissionError as exc:
                    self._json({"error": str(exc)}, 403)
                except (ValueError, KeyError, RuntimeError, OSError) as exc:
                    self._json({"error": str(exc)}, 400)

        return Handler

    def serve_forever(self):
        self.serving.set()
        try:
            self.server.serve_forever(poll_interval=0.25)
        finally:
            self.serving.clear()
            self.server.server_close()

    def close(self):
        if self.serving.is_set():
            self.server.shutdown()
        self.server.server_close()
        self.runtime.close()
