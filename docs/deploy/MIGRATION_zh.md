# 在新服务器迁移 RealMan 推理服务

本指南适用于将此仓库迁移到另一台带 NVIDIA GPU 的 Linux 设备运行。推理服务与待测模型运行在这台设备上；需要真机测试时，机械臂、相机和夹爪也直接连接到运行服务的同一设备。检查和离线推理不会连接硬件，也不会发送运动指令。

## 1. 准备服务器与 CUDA 环境

先在目标机安装与该 GPU 匹配的 NVIDIA 驱动。驱动必须新到足以支持所选 PyTorch wheel；**无需安装系统 CUDA Toolkit**。检查 `nvidia-smi`，再从仓库安装轻量推理依赖：

```bash
bash deploy/bootstrap.sh --help
```

有适用的官方 Linux wheel 时，按当前驱动和 GPU 选择 `cu126`、`cu128` 或 `cu130`。安装器用独立 venv、只选二进制 wheel，并在改动 Torch 前先检查 wheel 是否可用：

```bash
bash deploy/bootstrap.sh --cuda cu128 --venv "$HOME/.venvs/fastwam-realman"
source "$HOME/.venvs/fastwam-realman/bin/activate"
```

不要把某个 CUDA wheel 当作对所有显卡都通用。新架构（特别是 Blackwell / GB10）、ARM64、Jetson/厂商 SDK 镜像优先保留设备厂商已验证的 Python/Torch/torchvision 环境：

```bash
bash deploy/bootstrap.sh --cuda keep --in-place \
  --python /path/to/vendor-environment/bin/python
```

`keep` 模式不会升级 pip 或替换已有包；若 pinned runtime 依赖与厂商包冲突则停止并报告冲突。缺少的推理包只以 `--no-deps` 补齐。RealSense 和 RealMan SDK 如有 ABI/架构专用包，按厂商提供的 wheel 安装到同一 Python 环境，不要拿 x86 wheel 填入 ARM64 环境。常规 x86 主机可在确认 PyPI wheels 支持后添加 `--hardware`；该选项只装 Python bindings，不装内核驱动或设备固件。

可在 CPU-only 或 GPU 服务器上先做**只读预检**：

```bash
PYTHON=/path/to/active/python
"$PYTHON" deploy/doctor.py --config /path/to/runtime.json
"$PYTHON" deploy/doctor.py --config /path/to/runtime.json --checkpoint /path/to/trained.pt --check-cuda
```

不加 `--check-cuda` 时，doctor 不导入 Torch。加上后，它会在隔离的 Python 子进程中检查 `device` 选定的 `cuda:N`、GPU 型号/架构、空闲显存、wheel 架构列表，并运行实际 BF16 矩阵乘法。检查超时不会卡住 Web 进程。doctor 只读模型资产、配置、源码 pin 与 checkpoint SHA，不加载模型；只验证文件存在和绑定，不判断 checkpoint 能否装入显存。

## 2. 搬运模型与资产

Git 不包含训练权重、官方 tokenizer、Cosmos 基座权重或 Reason1 文本缓存。按受控方式搬运这些文件，并为 checkpoint、tokenizer、text cache 记录来源和校验值。必须包括：

- 训练产生的 no-reference 联合模型 checkpoint；
- 与这个训练 run 精确匹配的 `normalization.json` 及 `<checkpoint>.force.json` 绑定 sidecar；
- 官方 Cosmos tokenizer `tokenizer.pth` 和模型所用的官方 Cosmos 基座：`create_noaction_joint_cosmos` 使用 `base/post-trained`（非 action-conditioned），旧 `create_joint_cosmos_forcewm` 使用 action-conditioned；必须与训练所用版本一致；
- 含完整 `manifest.json` 和对应 SafeTensors payload 的 Cosmos Reason1 text cache；
- 本仓库固定版本的 `third_party/cosmos-predict2.5` 源码（已随仓库提供）；
- 一份经过核实的 hardware JSON，包含真实 RM75 IP、两路相机序列号、力传感器类型/坐标系、CTAG 参数和匹配训练数据的力标定/滤波文件。

用训练导出脚本从实际训练配置建立服务器运行配置，不要把本机 runtime JSON 里的绝对路径直接复制过去：

```bash
python scripts/cosmos/export_noaction_deployment.py \
  --training-config /data/run/resolved.yaml \
  --checkpoint /models/step001000.pt \
  --normalization /data/run/normalization.json \
  --tokenizer /models/cosmos/tokenizer.pth \
  --hardware-config /etc/fastwam/realman_hardware.json \
  --upstream "$PWD/third_party/cosmos-predict2.5" \
  --base-checkpoint /models/cosmos/base-post-trained.pt \
  --text-cache /models/cosmos/reason1-cache \
  --sampling-profile 4416 --execution-mode synchronous \
  --device cuda:0 --optimization auto \
  --output /etc/fastwam/runtime.json
```

导出器的可选参数以 `python ... --help` 为准。`4416` 表示 Video/Force/Action = **4/4/16**，即目前已测得较好的延迟/精度折中采样 schedule。它是推理采样设置，不需要按 4/4/16 重新训练；checkpoint 中的训练身份和架构仍原样校验。`161616` 为标准对照。只有 `4416` 可选择 `synchronous` 或 `async_relaxed`；异步让动作块执行与下一次推理重叠，不代表严格实时，也不提高模型本身的计算速度。首次启动需先在同步模式下完成静态检查和离线验证。

把上述导出示例中的设备序号设为该进程可见 GPU 的**逻辑序号**。若 `CUDA_VISIBLE_DEVICES=3,5`，逻辑 `cuda:1` 指物理序号 5。配置、doctor、预热、计时、模型张量都用同一个序号；不要默认使用 GPU 0。编译缓存可能因驱动、Torch 版本、仓库代码或 GPU 架构不同而失效。迁移时不要复制或挂载其他机器的 TorchInductor、Triton、CUDA 编译缓存；让目标机器在自己的缓存目录编译并预热。

## 3. 离线检查和延迟测试

先检查配置与所有资产，再确认指定设备 BF16 核有效：

```bash
python deploy/doctor.py --config /etc/fastwam/runtime.json \
  --checkpoint /models/step001000.pt --check-cuda
```

离线测模型时延，无须连接 RealMan/相机/夹爪；输入为固定合成 RGB、状态和力，加载时两次 synthetic warmup 独立报告：

```bash
python deploy/benchmark.py \
  --config /etc/fastwam/runtime.json \
  --checkpoint /models/step001000.pt \
  --profiles 161616 4416 --runs 10 --optimization auto \
  --output /data/results/inference-latency.json
```

这个测量包括本地 resize、VAE 编码、联合去噪和 action CPU 回传；不包括相机采集、图片网络上传、机械臂下发/执行、传感器等待或闭环任务质量。报告保留 checkpoint SHA、设备和每次调用计时。实际时延取决于 GPU、驱动、Torch wheel、优化器、温度/功耗和输入；只有同一 checkpoint/输入/设备可见性/优化设置下对比才可归因于采样配置。

## 4. 先启动服务，再决定是否接硬件

新服务器上的可执行服务与硬件必须选一个明确拓扑：

- **只测模型 / 不接硬件**：服务绑定 `127.0.0.1`，使用页面的模型加载和延迟信息，硬件状态保持断开。
- **GPU 服务器也直接连硬件**：RM75 网线和两台 RealSense USB3 相机都接在该服务器上，服务器 GPU 同时本地推理。将服务部署在受信任内网并检查 UI 的双相机和机器人状态。只有使用可信、已隔离的内网时才将 Web 绑定到 `0.0.0.0`；页面不是认证过的公网服务。

先在目标机前台启动并看日志：

```bash
python deploy/serve_realman.py --config /etc/fastwam/runtime.json \
  --catalog /etc/fastwam/checkpoint_catalog.json --host 127.0.0.1 --port 8765
```

本机浏览器打开 `http://127.0.0.1:8765`。跨主机访问要由操作者先配置网络访问控制和可信访问边界。服务启动和模型预热不会连接设备；须在 UI 中显式选择 **连接硬件**，再检查状态。不要在无硬件的推理服务器上按硬件连接或运动相关控件。

## 5. 硬件上电、布线和首次连接

以下要点针对采集时的 RealMan RM75、两台 RealSense RGB 相机和 CTAG2F120。型号、电压、端子、网络参数以各设备铭牌、接线图和厂商手册为准，不要照抄示例 JSON 接其他配置：

1. 机械臂置于无障碍安全区，急停可触达；清空工作空间，先断电布线。RM75 控制柜以以太网直连或接入隔离交换机，并使用静态 IP/同一子网；在硬件配置中填写该控制器 IP 与 SDK 控制端口。确认部署主机的网卡到此 IP 有路由，再从部署主机 `ping` 或检查 TCP 可达性。不要在公网暴露控制端口。
2. 两台 RealSense 分别插入部署主机 USB3 口，尽量用独立 USB3 控制器或有供电的 USB3 hub；不要将两路接到低带宽 USB2 hub。通过 `rs-enumerate-devices` 或厂商工具读取序列号，将全局相机填到 `global`、腕部相机填到 `wrist`。相机序列号必须唯一且与配置一致；先确认两路都能持续出图再启服务。
3. 安装与 OS/Python ABI/CPU 架构匹配的 Intel RealSense SDK、`pyrealsense2`，和 RealMan 官方 Python `Robotic-Arm` SDK；只验证导入及设备枚举，不从不匹配架构的 wheel 安装。
4. 按 CTAG2F120 和 RM75 工具端说明连接夹爪电源和 RS485，核实 A/B、地线、从站地址（本配置默认 1）及通讯参数。部署 worker 会查询报警、行程、使能和反馈，并可能把末端设为 24V、重置 Modbus RTU 模式及写入速度/力等参数；连接检查不下发夹爪目标。未使能时允许连接并显示状态，在页面点击“夹爪使能”会通过官方 RealMan SDK 写单个使能寄存器并回读，不写开合目标。张开/闭合按钮会产生运动，只在推理暂停且硬件就绪时可用；“取消待发目标”不会中止已经发出的运动。不要用软件的初始开合值替代实机确认。
5. 力预处理必须与训练数据完全一致。`force_calibration.json` 是特定机器人、传感器和安装姿态的标定快照，不能跨机器人复制；将匹配训练的 calibration、filter、normalization 与 checkpoint 一起搬运。预检会比对预处理身份。不要通过关闭校验或复制示例标定让不同硬件“先跑起来”。
6. 先保持机械臂急停/伺服安全状态，运行 `doctor.py`，服务启动，页面确认模型、GPU、相机、RM75反馈、力坐标和夹爪反馈后，才按现场流程解除急停、使能和执行。首次联调使用低速、空载、可立即触达急停的受控环境；所有硬件命令仍应遵守工作空间、速度、关节步长限制。

**配置文件是设备参数而不是硬件初始化命令。** doctor 只解析配置；只有 UI 中主动连接硬件会加载硬件 SDK、启动相机与 UDP 状态回调。安全停机用页面暂停/停止后确认 RM75 已停，再断开硬件、退出服务，最后依厂商流程断电。

## 6. CUDA Graph 与夹爪排查

两路 RealSense 使用独立的 `spawn` 子进程。启用 CUDA 的 librealsense 在 RGB 转换线程内申请/释放显存，会干扰同进程的全局 CUDA Graph 捕获；进程隔离保留官方 RGB8 转换及原始主机采集时间戳，并避免 fork 已初始化的 CUDA。模型端维持默认 global 捕获检查，不通过放宽捕获模式消除报错。相机 IPC 队列和历史均有界，过期、身份不符和不递增的帧会被拒绝。

夹爪页面分别显示实际开度、待发目标和最近一次成功下发目标。服务日志中的 `Gripper policy chunk` 是反归一化后的完整预测，可能随后被运行状态丢弃；`Gripper target` 才代表实际 SDK 写入成功，包含请求值、限幅后目标、方向、来源与时间。硬件配置设置 `log_gripper_feedback: true` 可记录每次既有位置轮询结果，不增加 RTU 读取。`log_motion_trace: true` 可进一步记录机械臂每次下发动作。

夹杯后的实际开度不一定等于闭合目标。若目标开度上升后发生松开，应检查模型输出和动作块衔接；若目标保持闭合而实际开度上升，应检查夹持、电源、机械结构及控制器。报警为 0 不能排除滑移。同步等待只取消尚未发送的目标，不主动失能或张开；取消限幅过程可能影响闭合充分程度，须根据日志判断。详细现场证据见 [GB10 相机冲突排查记录](CUDA_CAMERA_zh.md)。
