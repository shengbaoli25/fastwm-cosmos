# 本地硬件 + 启智 H200 远端推理

本机负责相机、机械臂、夹爪、力传感器与动作执行；H200 只加载模型、接收观测并返回 action。原控制台可以选择 **本机 NVIDIA 显卡** 或 **启智 H200 · 远端服务**。本指南所有命令从仓库根目录执行。

```text
本机相机 + joint8 state + force6
       │ 最新观测、32步 state/force 历史、mask、可选动作参考
       ▼
本地 client ── 启智端口转发 HTTPS ── H200 server
       ▲                              │
       └──────── 32×8 action ──────────┘
       │ 本地反归一化 → 关节平滑 → 原有限幅
       ▼
本机机械臂与夹爪，30 Hz 执行
```

每次推理传输两路 RGB、state、force 及训练所需历史。server 返回归一化动作；本机将其转换为关节弧度与夹爪开度。异步模式保持一个请求在途，观测只保留最新一份；server 忙时返回错误，不积压推理请求。

## 1. 准备代码、模型与环境

两端使用同一版本仓库。H200 按[迁移指南](../docs/deploy/MIGRATION_zh.md)准备 NVIDIA 驱动、匹配的 Torch/CUDA 环境以及模型资产；本机可沿用现有部署环境。仅运行远端 client 与合成输入测试不需要本机 CUDA，接入硬件仍需要原有 RealMan、RealSense 等驱动和配置。

本轮 `remote-deploy-20261025.tar.gz` 是覆盖到**已有完整仓库**的 `deploy/` 更新包；它本身不含 `src/` 模型实现、固定上游源码或权重。迁移到全新设备时应先取得完整仓库，再使用这份更新包与模型资产；不能只解压这个包就启动 H200 模型。

| 内容 | H200 server | 本地 client |
|---|---|---|
| 当前训练权重及 `<checkpoint>.force.json` | 必需 | 可省略大权重，使用其真实 SHA256 校验 |
| 同一训练导出的 `normalization.json` | 必需 | 必需，直接复制同一文件 |
| tokenizer、基座权重、完整文本 cache、固定上游源码 | 必需，沿用训练导出的模型身份与配置 | 远端模式不需要；切回本地推理时需要 |
| 本机 hardware JSON、力预处理与标定文件 | 不连接硬件 | 真实执行时必需 |
| `REALMAN_REMOTE_TOKEN` | 启动环境中设置 | 启动环境中设置相同值 |

server 使用完整训练导出的 `runtime.server.json`，修改其中的资产路径与 `device: "cuda:0"` 以适配该实例。相对路径按配置文件所在目录解析。不要用下文的最小 client 配置替代 server 配置。权重若是软链接，loader 会解析其最终路径；`.force.json` 必须放在最终权重文件旁边，避免只迁移软链接而遗漏绑定文件。

在 H200 上先验证所选 CUDA 设备和资产：

```bash
nvidia-smi
python deploy/doctor.py --config /data/wam/runtime.server.json \
  --checkpoint /data/wam/assets/weights/pluginsertion71_step005000.pt --check-cuda
sha256sum /data/wam/assets/weights/pluginsertion71_step005000.pt
```

`cuda:N` 是进程可见 GPU 的逻辑序号。不要把其他 GPU 的编译缓存当作已验证的本机缓存；目标实例自行编译和预热。当前 server 默认要求 `compile_cuda_graph` 生效、整段 sampler 已捕获且没有优化回退，完成后才报告 READY。冷启动时间不计入稳定推理时延。`--allow-optimization-fallback` 仅用于明确记录回退的诊断，不宜用于声称与优化基准相同的测试。

## 2. 在 H200 启动 server

在两端各自启动服务的终端中输入**同一个至少 16 位的随机 token**：

```bash
read -r -s -p 'Remote token: ' REALMAN_REMOTE_TOKEN
export REALMAN_REMOTE_TOKEN
printf '\n'
```

token 只保存在进程环境中，不写入 JSON、页面输入框、命令行参数或 Git。重开终端或重新启动服务时，需要重新提供。

H200 端启动：

```bash
mkdir -p /data/wam/results
nohup python deploy/serve_remote_policy.py \
  --config /data/wam/runtime.server.json \
  --checkpoint /data/wam/assets/weights/pluginsertion71_step005000.pt \
  --sampling-profile 4416 --host 0.0.0.0 --port 8000 \
  --ready-file /data/wam/results/server-ready.json \
  > /data/wam/results/server.log 2>&1 &
echo $! > /data/wam/results/server.pid
```

等待日志出现 `READY: WAM 4416`；`server-ready.json` 记录 GPU、权重 SHA、normalization 身份、采样方案和优化状态，不含 token。上面的启智命令按平台指南监听实例内 `0.0.0.0:8000`，认证 token 仍必需；server 不打开相机、机械臂或夹爪。程序自身默认监听 `127.0.0.1`，只在实例内测试时可以保留该默认值。监听所有实例接口本身不会建立可访问的外部通道，仍须验证下一节的平台转发。

一个 server 固定一份权重和采样方案。切换到 `161616` 时，应停止该 server，并用 `--sampling-profile 161616` 重启，再让 client 选择相同方案。切换同步/异步仅改变本地调度，不需要重启同一采样方案的 server。

## 3. 使用启智官方端口通道

依据[飞书：启智相关文档，第 4 部分「访问」](https://sii-czxy.feishu.cn/wiki/LM9dwnGzjivZ9GkCLR3caL7bnUe)：

1. 在该 H200 实例的 VSCode IDE 底部打开 **Ports / 端口**。
2. 转发 server 正在监听的 **8000** 端口。
3. 复制平台显示的真实访问 URL。指南示例为 `https://nat2-notebook-inspire.sii.edu.cn/.../proxy/8000`，不同实例生成的域名可能不同；必须原样复制平台给出的完整地址，不能仅替换域名或自行拼接中间标识。
4. 将完整 URL 填入本机控制台的「服务地址 / 本机转发地址」，或下文 `remote_inference.url`。

文档说明该地址可供内网设备访问；关闭 VSCode 页面不影响仍在运行的实例服务。这里通过 `nohup` 保持 server 进程，实例停止或 server 退出后，通道自然不可用。此流程依赖官方端口代理，不假设实例存在可用的 SSH 入口。

在本机验证真正的 Python client 能访问 server；仅能在已登录浏览器中打开页面还不足以证明这一步成功：

```bash
export WAM_REMOTE_URL='粘贴平台给出的完整 URL'
python - <<'PY'
import json, os, urllib.request
url = os.environ['WAM_REMOTE_URL'].rstrip('/') + '/health'
req = urllib.request.Request(url, headers={
    'Authorization': 'Bearer ' + os.environ['REALMAN_REMOTE_TOKEN']})
with urllib.request.urlopen(req, timeout=10) as response:
    data = json.load(response)
print(json.dumps(data, ensure_ascii=False, indent=2))
PY
```

应得到 `ready: true` 的 JSON。若返回登录页、403、404 或无法连接，先检查本机所在内网、转发 URL、8000 监听与平台访问设置；不要把浏览器会话视为已打通程序访问。`401 authentication required` 表示两端 token 不一致或本机进程没有继承环境变量。

**2026-10-25 本轮状态：** H200 的真实 step5000 / 4416 服务已通过实例内 HTTP 测试；平台生成的 `ai-notebook-inspire.sii.edu.cn/.../proxy/8000/` 在本机程序和浏览器中仍返回 nginx 403，监听改为 `0.0.0.0` 后也未解决。因此当前没有已验证的本机→启智推理通道，也未证明外部额外开销低于 100 ms。不要用 H200 实例内回环成绩替代跨机测试，需先取得平台可供程序访问的正式转发入口。

## 4. 本地配置与原控制台

通常直接复制现有完整 runtime JSON，在其中加入以下远端设置即可，这样切回本机推理时仍保留全部模型资产路径。只做远端推理也可以使用下面的最小 `runtime.client.json`；按实际位置修改路径、URL 和 SHA，并确保分辨率/相机顺序与 server 一致：

```json
{
  "normalization_file": "/data/wam/normalization.json",
  "checkpoint_directory": "/data/wam/checkpoints",
  "hardware_config": "/data/wam/realman_hardware.json",
  "hardware_factory": "realman_deploy.hardware:make_hardware",
  "camera_order": ["global", "wrist"],
  "camera_width": 640,
  "camera_height": 480,
  "sampling_profile": "4416",
  "execution_mode": "synchronous",
  "inference_device": "qizhi_h200",
  "action_smoothing_alpha": 0.5,
  "max_result_age_s": 0.6,
  "remote_inference": {
    "url": "https://nat2-notebook-inspire.sii.edu.cn/替换为平台真实路径/proxy/8000",
    "checkpoint_sha256": "替换为服务端权重的64位小写SHA256",
    "token_env": "REALMAN_REMOTE_TOKEN",
    "image_encoding": "zlib_rgb",
    "timeout_s": 5.0,
    "max_latency_s": 1.0
  }
}
```

JSON 中的 URL 和 SHA 是待替换占位符。若已有匹配的 checkpoint catalog，可选择该条目使用 catalog 的 SHA；不必为远端模式下载同一份大权重。client 握手会核对权重 SHA、normalization、动作空间、相机顺序与尺寸、采样方案和模型就绪信息，防止模型与本地动作解码不匹配。力数据沿用与训练相同的预处理，并附带身份校验。

启动原控制台：

```bash
python deploy/serve_realman.py --config /data/wam/runtime.client.json
```

打开 `http://127.0.0.1:8765/`。在「推理设置」选择 **启智 H200 · 远端服务 → 4 / 4 / 16 → 同步或异步**，填写地址，应用设置后重新加载模型。远端身份区显示校验后的权重与采样信息；运行指标显示客户端总延迟、服务端模型时延、传输与额外开销、编码耗时。更改设置前需要暂停并等当前请求结束。

加载远端模型只完成连接与身份检查。启动控制台、加载模型和运行下文 benchmark 都不会自动连接或驱动硬件。真实执行继续使用原页面的「连接硬件」与「开始」流程。切换回「本机 NVIDIA 显卡」需要本机权重、模型资产和 CUDA 就绪；最小 client 配置需补齐本地模型配置后才能本地推理。

## 5. 4416 的同步、异步与平滑

| 设置 | 行为 |
|---|---|
| `4416 + synchronous` | 当前 32 条动作及末条控制周期结束，确认停止后取新观测，等待远端结果，再执行下一段。推理期间保持停止。 |
| `4416 + async_relaxed` | 当前动作执行期间，在段末依据最近请求耗时预取；及时返回的下一段在边界衔接。若未及时返回，保持等待，收到有效结果后继续。 |
| `161616` | 基准采样，目前控制台仅开放同步执行；server 必须启动同一方案。 |

异步让计算和动作执行重叠，**不会消除首次请求、观测陈旧度、网络抖动或动作跟随延迟**。它也不是严格时间对齐或预测补偿；任务能否成功仍需实机验证。32 条动作在 30 Hz 下约执行 1.067 s。

本地对已反归一化的前七个关节目标使用 `y[t] = α·x[t] + (1−α)·y[t−1]`，初值为当前实测关节。夹爪开度不做 EMA。`α=1` 关闭平滑；`α=0.5` 对缓慢变化目标的低频滞后约一个控制周期，即约 **33 ms**，不是对任意运动恒定延迟的保证。平滑后的目标仍通过原有硬件限幅；暂停、急停和重启运行时清除平滑历史。

`max_result_age_s: 0.6` 是**本地观测时间到结果返回，以及该段首次动作下发前**的年龄上限；包含采集后的等待、请求与排队等待。600 ms 不是网络传输预算，也不是允许每次请求额外耗时 600 ms。开始执行后不会因该阈值在 32 条动作中途截断；实时传感器新鲜度与原有动作检查仍持续生效。

`remote_inference.max_latency_s` 则限制一次 client 编码、请求与结果解析的总耗时，`timeout_s` 是网络操作超时上限。总耗时在结果解析后核对；网络连接/读写还使用 socket 超时，它不是能精确抢占每段 I/O 的硬实时定时器。任何返回或首动作年龄超过 600 ms 的结果仍不能用于执行，即使请求在 1 s 总限时内成功。连接失败、错误响应或过期结果进入原有错误停机流程；暂停和急停通过 epoch 使旧结果失效，失败请求不自动重试。

## 6. 不接硬件，实测上传到 action 返回

server 报告 READY 后，在**本机**执行下列命令，经过实际通道发送两路合成图像、state、force、历史与 mask，并读取真实模型返回的 `32×8` action：

```bash
mkdir -p /data/wam/results
set -o pipefail
python deploy/benchmark_remote.py \
  --config /data/wam/runtime.client.json \
  --warmup 5 --runs 50 --patterns blank random \
  --output /data/wam/results/remote-zlib.json \
  2>&1 | tee /data/wam/results/remote-zlib.log
```

输入在计时前构造；state/force 取对应 normalization 的均值，历史按现有 `OnlineHistory` 生成并带 padding mask。结果是通道和推理性能测试，不评估动作准确度。加载、编译和预热不纳入稳态样本；每种图像模式另丢弃指定次数 warmup。

| 输入/编码 | 用途 |
|---|---|
| `blank + zlib_rgb` | 空白 RGB 无损压缩后很小，验证连通性与理想可压缩输入的时延下界。不能据此推断真实相机带宽。 |
| `random + zlib_rgb` | 固定种子随机 RGB 很难压缩，检查较大负载与压缩 CPU 开销。不是自然相机画面的统计模型。 |
| `blank/random + raw_rgb` | 将复制的测试配置中 `image_encoding` 改为 `raw_rgb`，测无压缩传输。两路 640×480 RGB 本体共 1,843,200 字节，另加状态/历史元数据。 |

对 raw/random 等探索测试，若超过默认 1 s 请求上限，可另存一份**仅用于 benchmark**的配置，把 `max_latency_s` 和 `timeout_s` 设为 `10.0` 记录慢请求。不要把这份配置用于放宽真实执行的 600 ms 首动作时效。

benchmark 对每次请求独立记录 `success`、`timeout`、错误类型/原因，并在每次完成后原子更新 JSON。某次失败后继续下一个编号的新观测，不重试原请求。正式样本在 `samples`，预热请求另存 `warmup_samples`，两者分别汇总；握手成功后即创建 `completed: false` 的结果文件，全部计划请求尝试完才改为 `true`。中断时保留已完成记录；若初始握手失败，尚无结果文件，应检查终端日志。

`completed: true` 或进程正常退出仅表示测试流程完成，不表示请求全部成功。报告需同时给出 `attempts / successes / failures / timeouts`、成功率、失败率与超时率；P50/P95/P99 只统计成功的正式请求。全部失败时分位数为 `null`。成功样本还包含上传/下载字节数、服务端身份与首条归一化 action。计时口径如下：

| 字段 | 含义 |
|---|---|
| `total_client_ms` | 本机开始编码到解析出 action，含图像编码、上传、服务端解包与推理、回复传输及结果解析。 |
| `server_timing_ms.inference` | server 调用模型 adapter 到得到 CPU action 的时间。 |
| `server_timing_ms.processing` | server 解包 + 模型推理。 |
| `transport_overhead_ms` | client 耗时减 server processing，含客户端编码、HTTP/代理往返、服务端回复序列化与本机解析。它不是纯链路 RTT，也不包含 server 解包。 |
| `non_model_overhead_ms` | `total_client_ms − server_timing_ms.inference`，模型之外的全部额外开销；判断「上传和下发等开销不超过 100 ms」时优先看此值和其 P95。 |

汇总中的 `non_model_under_100ms_fraction` 以**全部正式请求**为分母，失败请求不算达标；`non_model_under_100ms_fraction_among_successes` 才是成功样本内的达标比例。旧字段 `overhead_under_100ms_fraction` 只对 `transport_overhead_ms` 计数，两种开销口径不能混用。不要跨主机直接相减单调时钟来声称单向上传耗时。下面从样本重新计算完整额外开销，也兼容尚未包含新字段的结果文件：

```bash
python - /data/wam/results/remote-zlib.json <<'PY'
import json, sys
sys.path.insert(0, 'deploy')
from benchmark_remote import summarize
data = json.load(open(sys.argv[1]))
for pattern in sorted({r['pattern'] for r in data['samples']}):
    rows = [r for r in data['samples'] if r['pattern'] == pattern]
    successes = [r for r in rows if r.get('success', True)]
    extra = [r['total_client_ms'] - r['server_timing_ms']['inference'] for r in successes]
    passes = sum(v < 100 for v in extra)
    print(pattern, json.dumps({
        'completed': data.get('completed'),
        'attempts': len(rows), 'successes': len(successes),
        'failures': len(rows) - len(successes),
        'model_external_ms': summarize(extra),
        'under_100ms_fraction': passes / len(rows),
        'under_100ms_fraction_among_successes': passes / len(extra) if extra else None
    }, ensure_ascii=False))
PY
```

仅当 H200 模型实测约 200 ms 且同通道额外开销实测约 100 ms 时，才能预期约 300 ms 的 client 请求耗时；真实首动作下发还包含观测取得、控制线程调度与硬件通信。当前迁移指南不预填远端网络测试结论，结果应以本轮实际生成的 JSON 和报告为准。

## 7. 收尾与常见问题

- 本机 CUDA 不可用不阻止远端推理；server 必须确认其实际 NVIDIA 设备和编译/图捕获状态。仅选中「H200」标签不代表远端真实 GPU 已核验。
- normalization / SHA / 采样 / 相机尺寸不匹配：统一两端资产和配置，重新加载；不要绕过身份校验。
- `429 model is busy`：该服务已有请求正在推理，确认没有第二个 benchmark/client 同时使用。
- 网络错误、deadline 或首动作过期：检查通道的完整分位数和负载，暂停后人工确认再恢复；不要自动重复下发同一段 action。
- 结束真机实验先在本机暂停。结束 server 可向自己启动并记录在 `server.pid` 的进程发送正常终止信号；确认停止后，关闭自己本次使用的启智实例以释放 GPU。
