# RealMan 同步推理 Client

独立机器人客户端分支 `codex/realman-sync-client`。只采集机械臂关节、六轴力、夹爪和两路 RGB 图像，向远程 Server 请求动作，再在本机执行。无需训练仓库、checkpoint、PyTorch、CUDA 或模型依赖。

支持相同 `realman-sync-v1` 接口的三个推理分支：

- `codex/realman-cup-joint-30k`
- `codex/realman-cup-joint-30k-dinov3-fwm`
- `codex/realman-cup-joint-30k-vjepa21-fwm`

## 安装和无硬件联调

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .

# Server 端先按其 README 启动 --backend mock；地址也可指向远程机器。
realman-sync-client --server http://127.0.0.1:8766 --cycles 2

# 使用源码运行，无需安装：
PYTHONPATH=src python -m realman_sync_client --server http://127.0.0.1:8766 --cycles 2

# 完整 CPU 回归测试，不访问相机或机械臂：
PYTHONPATH=src python -m unittest discover -s tests -v
```

默认 `--hardware mock --cycles 1`，只产生模拟观测并记录模拟动作。图像尺寸、相机顺序、动作维度和视频上下文由 `/health` 的 metadata 协商；模拟器自动采用 Server 要求的尺寸。Client 不提供模型替身 Server，使用对应 Server 分支的 mock 后端即可验证真正的 HTTP 闭环。

如 Server 设置了鉴权，两端使用相同环境变量；不要把令牌写进命令行参数或配置仓库：

```bash
export REALMAN_SERVER_TOKEN='your-local-token'
realman-sync-client --server http://SERVER_IP:8766 --cycles 2
```

传输支持 HTTP 和 HTTPS。明文 HTTP 适用于受信任机器人局域网；跨网部署使用 HTTPS 入口。客户端不跟随重定向，不使用环境代理，不自动重试推理。

## 连接真实 RM75

真实硬件额外安装：

```bash
python -m pip install -e '.[hardware]'
```

另需安装与现有控制器匹配的 RealMan Python/native SDK，保证 `from Robotic_Arm.rm_robot_interface import RoboticArm` 可导入。SDK 由机器人厂商提供，未改版或打包进本项目。两路 Intel RealSense 使用 `pyrealsense2`。CTAG2F120 继续通过 RealMan 控制器的工具串口 / Modbus RTU 读写，保留原有寄存器映射；不是另写一个 PC USB 串口协议。

复制并核对 `examples/realman_hardware.joint8.example.json`：机械臂 IP/端口、型号、力传感器类型、控制器版本、相机序列号、UDP 端口及软限位参数。JSON 中相对路径基于该 JSON 所在目录。随附的 `force_filter.json` 和 `force_calibration.json` 是源分支的实际配置快照，标定仅适用于原机器人、工具与装配；训练 checkpoint、归一化统计和硬件标定必须使用相同预处理身份。Client 不做自动重标定、校零或修改滤波参数。

```bash
realman-sync-client \
  --server http://SERVER_IP:8766 \
  --hardware realman \
  --hardware-config examples/realman_hardware.joint8.example.json \
  --execute \
  --cycles 0
```

只有同时指定 `--hardware realman --hardware-config ... --execute` 才会连接并执行真实设备。夹爪若尚未使能，可显式增加 `--enable-gripper`；默认不改变使能状态。`--cycles 0` 持续运行，Ctrl+C / SIGTERM 请求停止，停止后关闭相机、还原 UDP 推送配置并释放机器人连接。

所有模型动作都是绝对 `joint8`：`[joint1_rad, ..., joint7_rad, gripper_open]`，夹爪范围 `[0,1]`。只在 RealMan SDK 边界把弧度转换为角度。继续沿用 SDK 软限位、关节跳变检查、关节跟随参数、夹爪限速和 hold/slow-stop 逻辑。模型归一化、图像 resize 和 embedding 均在 Server。Client 侧的力标定/因果滤波属于传感器处理，保持与训练数据一致。

## 同步时序

```text
30 Hz 传感器采样持续运行，保存最近 32 个真实状态/力观测
                  │
完成缓停确认 → 获取缓停之后的新 obs + 连续历史
                  ↓
             POST /infer（仅一个在途请求）
                  ↓
        检查整个动作块、会话、请求编号、源观测和有效期
                  ↓
        按 30 Hz 执行指定步数 → 等待原生 slow_stop 确认
                  └──────────── 下一轮
```

传感器历史在网络等待和动作执行期间持续更新。不是每次推理才增加一个历史点；32 个点始终对应训练时的 30 Hz 测量网格。首次请求等待完整历史；滤波 epoch 改变时清空历史，并拒绝基于先前 epoch 返回的动作。基础分支使用当前图像；DINO 和 VJEPA 的 `video_context_frames=2, video_context_stride=4` 使用 `[-4,0]` 两帧，即间隔 `4/30` 秒，最后一帧与当前 obs 完全相同。

默认执行完整 32 行，`--execute-steps N` 可选择前 `1..32` 行，仍先验证整个返回块。每行周期包括 SDK 调用耗时；超过一个控制周期立即停机，不追赶补发。最后一行执行完一个周期后才缓停，然后才允许下一次请求。请求只携带原始单位，动作只接收已反归一化的原始单位。

常用运行参数：

| 参数 | 默认值 | 含义 |
| --- | --- | --- |
| `--cycles` | `1` | 完成多少轮；`0` 持续运行 |
| `--execute-steps` | `32` | 每次返回块实际执行的前 N 行 |
| `--timeout` | `120` | 单次 HTTP 等待秒数，无自动重试 |
| `--max-result-age` | `120` | 返回动作对应 obs 的最大本机年龄 |
| `--sensor-timeout` | `10` | 相机预热/完整连续历史等待秒数 |

`--max-result-age` 为慢速同步模型预留计算时间，应按部署测得延迟设置。有效期仅用机器人端单调时钟计算，绝不与 Server 的单调时钟相减。网络中断、无效响应、采样失败、过期反馈、控制超时都会停止新动作并要求缓停确认；不会把旧动作重放。退出时尽力调用 `/session/release`，释放失败不覆盖硬件停机异常。

## 协议

- `GET /health`：`{protocol, ready, metadata}`，metadata 明确 `camera_names,width,height,action_space,action_horizon,control_hz,history_length,video_context_frames,video_context_stride,force_preprocessing`。
- `POST /infer`：`{protocol,session_id,request_id,force_preprocessing,observation,history,video_context?}`。obs 含 `sequence,stamp,state[8],force[6],action_space,frames,frame_meta,force_meta`；历史不重复发送图像。
- 图像：`frames[name]={width,height,encoding:"rgb8-base64",data}`，原始 RGB8 无损 base64；相机固定 `global,wrist`。最大 JSON 体积 64 MiB。
- 返回：`{protocol,session_id,request_id,source_sequence,action_space,control_hz,actions[32][8]}`。
- `POST /session/release`：`{protocol,session_id}`，只在同步请求结束后发出。

`src/realman_sync_client/wire.py` 与三个 Server 的 `deploy/realman_sync/wire.py` 保持逐字节相同。无需把训练代码复制到客户端即可执行契约校验。

## 源码与验证范围

`hardware.py,camera.py,gripper.py,sdk.py,contracts.py` 和 `realman_force` 来自基础分支提交 `9a317a1e9d7d37e953102e3ee37f4bbf70ca5906`，逐字节保留；文件哈希记录在 `HARDWARE_SOURCES.json`。本分支是独立 orphan 分支，工作树只包含采集、通信、控制和必要的传感器处理代码。

单元/集成测试覆盖连续历史、同步轮次、VJEPA 间隔、响应身份、过期动作、后续行 NaN、网络不重试、执行周期超时、显式硬件启动、鉴权、真实 HTTP mock CLI 和会话释放。测试不连接物理设备；真实 SDK/相机/机器人和 GPU checkpoint 的端到端效果仍需在对应部署机器验证。
