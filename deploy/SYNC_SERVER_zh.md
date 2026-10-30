# 同步推理 Server 与独立机器人 Client

本入口只加载模型、接收观测、返回动作。模型常驻内存；每次请求只做一次推理，
不连接机械臂、串口或相机。现有训练脚本、模型源码和训练配置保持原样。

机器人端位于独立分支 `codex/realman-sync-client`，无需安装 Torch、Cosmos、
DINO、V-JEPA 或下载模型权重。部署时分别检出服务端分支和客户端分支。

## CPU 联调

服务端仅需 Python 3.10+ 标准库：

```sh
python deploy/serve_policy.py --backend mock --host 127.0.0.1 --port 8766
```

在独立客户端分支安装后运行：

```sh
realman-sync-client --server http://127.0.0.1:8766 --cycles 2
```

客户端分支提供 mock 硬件模式与 `--cycles` 参数，安装与真实硬件命令见该分支 README。
mock Server 返回当前关节位置的保持动作，不代表模型推理。

## 加载真实模型

Cosmos 分支继续使用原部署流程导出的 **Joint8 no-reference** runtime JSON、
训练 checkpoint 和匹配的 normalization/force sidecar。原来的
`realman_runtime.example.json` 是旧 EEF 配置，不能直接用于 Joint8 服务。

```sh
python deploy/serve_policy.py --backend cosmos \
  --config /models/runtime-joint8.json \
  --checkpoint /models/checkpoint.pt --host 0.0.0.0 --port 8766
```

DINO/V-JEPA 分支使用新增的 `realman_sync.dense_policy.DensePolicy`，
启动参数为 `--backend dense`；详细训练资产和配置见各分支新增的 dense 说明。
Server 不改采样器、损失或训练逻辑，沿用对应分支的推理 API。

可通过两端同名环境变量 `REALMAN_SERVER_TOKEN` 启用 Bearer 校验；无需将 token 写进配置。
本 HTTP 服务用于可信局域网；跨公网部署应使用 SSH 隧道或 HTTPS 反向代理。

## 接口和时间语义

- `GET /health`：返回 `protocol=realman-sync-v1`、`ready` 和模型输入 metadata。
- `POST /infer`：一个 `session_id` / `request_id` 对应一次 observation→action。
- `POST /session/release`：完成停止后释放会话；推理进行中不能释放。

请求字段为 `protocol`, `session_id`, `request_id`, `force_preprocessing`,
`observation`, `history` 和可选 `video_context`。当前观测包含：

```text
sequence: 客户端递增整数
stamp: 客户端 monotonic 秒
action_space: joint8
state: [joint1_rad, ..., joint7_rad, gripper_openness_0_1]
force: [Fx_N, Fy_N, Fz_N, Tx_Nm, Ty_Nm, Tz_Nm]
force_meta: 原硬件滤波流水线的 preprocessing_id / epoch / source_time 等
frames: {global: {width, height, encoding: rgb8-base64, data}, wrist: {...}}
frame_meta: 原硬件相机元数据
```

原始连续 RGB8 字节使用 base64 无损传输，不使用 pickle/JPEG。请求上限 64 MiB。
`history` 包含截至本轮观测的最近 32 个连续 30 Hz 的 state/force 样本。
客户端在等待网络与执行动作期间仍持续采集，不能把过去 32 次 HTTP 请求
当作 32 个历史时间步。短历史在 Server 侧按原规则左补零并打 padding mask。
历史不能跨力滤波 epoch 或缺失时间段。

Cosmos 只使用当前图像。DINO/V-JEPA 使用当前与前 4 个 30 Hz 步长的图像
（offsets `[-4,0]`，约 133 ms）；metadata 的 `video_context_frames=2`、
`video_context_stride=4` 告知客户端保留并选择对应帧。
所有输入时间戳属于客户端；Server 不与自身 monotonic 时钟做差。

响应包含原 `session_id`、`request_id`、`source_sequence`、`action_space`、
`control_hz`、`inference_ms` 与 `actions`。actions 是已经反归一化的
绝对关节弧度和夹爪开度，客户端不能再次反归一化。夹爪按原 runtime 限制到 [0,1]。

第一版同一模型只接受一个客户端会话，最多一个进行中的推理请求。
重复/倒序请求或并发推理返回 HTTP 409，不自动重试。原会话释放或闲置
`--session-timeout`（默认 300 秒）后，其他客户端可建立会话。
客户端按完整 chunk 执行，等最后一个控制周期结束、slow_stop 确认后，
取更新的观测再请求；推理与动作执行不重叠。

## 验证

```sh
python -m unittest discover -s deploy/tests -p 'test_sync_server.py' -v
python -m unittest discover -s deploy/tests -p 'test_sync_review.py' -v
```

测试覆盖真实本机 HTTP、协议校验、请求去重、会话隔离、30 Hz 历史、
双帧间隔、力预处理一致性，以及 Cosmos 单次归一化/反归一化。
CPU 联调不等于 GPU checkpoint 或真实机器人测试；真实部署需要匹配的
训练资产、CUDA 环境及硬件参数。
