# RealMan 无参考轨迹联合扩散版本

日期：2026-10-20。分支：`codex/realman-cup-joint-30k`。
本次更新搭建代码，等待新采集的力优化数据集；未启动训练、未连接或控制机械臂。

## 架构与输入

| 分支 | 实现 | 隐藏维度 / FFN | 层数 | 参数量 |
|---|---|---|---|---|
| Video | 官方 Cosmos-Predict2.5-2B **base/post-trained Video2World** | 2048 / 原生配置 | 28 | 2,059,174,912 |
| AE | ActionDiT，联合注意力＋独立文本/当前状态 cross-attention | 768 / 3072 | 28 | 491,060,488 |
| AE 源噪声时间编码 | `tV,tF` → AE 调制量 | 768 | — | 3,937,536 |
| Force | 未来六维力/力矩 flow matching diffusion | 768 / 2048 | 28 | 270,040,838 |
| 当前状态投影 | state8＋占位维 → context | 1024 | — | 10,240 |

总计 2,824,224,014 参数；已在 CPU 严格加载完整官方 Video 权重并构造联合模型，参数计数不代表预测质量。
AE 合计约 4.95 亿参数，Force 约 2.70 亿。三个专家联合注意力的 Q/K/V
均投影到 **16 heads × 128 = 2048**，因此隐藏维度 768 不等于注意力投影宽度 768。
Video 保留原生位置编码、三路 AdaLN 与需要 FP32 的模块。

无 `R` token、无 `action_condition`、无上一轮 AE 参考回填。Video 不接收
未来动作；Force 只接收已观测历史与自己的带噪未来状态，并读取 Video。
历史 H 中已发生的机器人状态/动作与受力仍然保留，当前 proprio 使用 anchor 的第 0 行。
旧 5000 步模型的结构不兼容；新训练由官方非 action-cond Video 权重及新初始化 AE/Force 开始。

| Query / KV | O 已观测视频 | H 已观测历史 | V 带噪未来视频 | F 带噪未来力 | A 带噪动作 |
|---|---|---|---|---|---|
| O | ✓ | — | — | — | — |
| H | ✓ | ✓ | — | — | — |
| V | ✓ | — | ✓ | — | — |
| F | ✓ | ✓ | ✓ | ✓ | — |
| A | ✓ | ✓ | ✓ | ✓ | ✓ |

矩阵描述跨模态 token 的读取关系；固定任务文本仍是条件，AE/Force 还使用当前状态。
仅做前向隔离，Force/AE loss 可以沿允许的读取路径反向更新上游专家。
每层从该层的输入同时形成三个分支的 Q/K/V，再分别做注意力、残差与 FFN。
这版本预测的是观测条件下的未来；**没有定义“给定最终 AE 动作后会发生什么”的动作条件 WM**。

## 训练与推理共用核心

代码入口：`src/fastwam/cosmos/noaction.py::NoActionJointModel`。

- 训练将真实 V/F/A 分别加噪，输入 `V_tV,F_tF,A_tA`；三条噪声时刻独立采样。
- 一次 `joint_denoise_core` 同时预测三条去噪速度，目标均为 `noise - clean`。
- 损失：`loss_video_fm + loss_force_fm + loss_action_fm`，权重 1/1/1。
  padding 与力通道有效性掩码只用于相应监督；无训练内 16 步视频 rollout。
- Force 未来 token 接收 `tF,tV`；H 不加噪、不接收动态时间。
  AE 接收 `tA` 并增加 `tV,tF` 条件，以区分上游预测的噪声水平。
- 推理从三个独立噪声开始，16 轮中每轮先求三个速度，再同时更新 V/F/A。
  Video shift=5，Force/AE shift=1。每轮恢复独立编码的观测首帧。
- 每个推理请求缓存逐层 O/H 和固定文本 K/V；V/F/A 每轮重算。
  缓存在请求内有效，新的观测请求重新构建；训练不使用推理缓存。
- `infer_joint` 和 `infer_forcewm` 返回 `action,force,video_latents`。
  `action` 是归一化的 `[B,32,8]`，`force` 是归一化的 `[B,32,6]`。
  有参考参数的调用会报错，避免静默使用错误输入。

新 Force FM loss 与旧直接回归 Force MSE 的含义不同，不能直接以数值大小比较预测质量。
评估时需要相同窗口、反归一化后的六维力误差和动作误差。

## 数据与配置

双视角、30 Hz 的数据接口保持现有约定：H[-31..0]、A[0..31]、F[1..32]，
图像偏移 0,4,…,32；主相机在左、腕部相机在右，各 224²，拼接为 224×448。
视频缓存是 `[16,3,28,56]`，观测帧独立编码为 `[16,1,28,56]`。
动作/state 使用 `[q1,q2,q3,q4,q5,q6,q7,gripper]`：七轴角度为弧度，夹爪为 0–1 开度。
action 是采集时**实际下发的绝对关节目标**；state 从原始 20 维观测的 `[0..6,19]`
列提取实测关节角和夹爪。proprio 为 8 维状态＋一个历史兼容占位维，history 为 state8＋force6。

新数据 profile 为 `realman-leader-joint8-filtered-v1`，原始 schema 为
`realman_leader_force_gripper_v2`。读取 `observation.force_filtered` 已校准滤波的六维力，
不重复滤波。当前采集与部署配置均采用选定的 Q×48、200 Hz 实际 dt 的因果滤波，
预热 0.5 秒、最大包间隔 30 ms；训练验证每条 episode 的校准与处理身份。
采样仅保留四路 freshness 均有效、动作实际执行、滤波已预热且力反馈不陈旧的行；
按真实单调时钟的间断、滤波重置、跟随会话及锚点变化分段，窗口不跨段。
保留原始视频/episode 行号，在段边界重复边界帧并屏蔽相应监督，不能删除行后重新编号。
不新增人工标注，不生成参考轨迹。新数据需要正常导出视频/text/numeric 缓存与统计；
只有数据来源、官方 tokenizer 和文本身份均一致的现有完整缓存才可复用。

配置入口：

- `configs/model/cosmos_noaction_joint.yaml`
- `configs/task/cosmos_noaction_joint_4gpu_30k.yaml`
- `scripts/cosmos/train_realman.py`：新数据，含缓存导出。
- `scripts/cosmos/train_realman_cached.py`：已验证的完整缓存。

配方：4 GPU、每卡 batch16、global64、30000 步、每5000步保存，
AdamW 1e-4、warmup1500、cosine、BF16/ZeRO1；等新数据就绪后申请 4×H200。
保留既有无独立验证集的训练配方；独立测试 episode 应在收集/导出训练集前预留。
配置中的 `???` 必须由真实缓存生成，不填旧数据的身份摘要。
RealMan resolver 校验统计与缓存中的源数据一致，然后把 normalization.json 的 SHA256
写入模型 manifest；部署必须使用完全相同的文件。

准备好新数据并获得训练资源后执行，以下路径均为待替换的示例，本次未执行：

```bash
PYTHONPATH=src python scripts/cosmos/train_realman.py \
  --dataset /data/new-realman-force \
  --source-sha <新数据文件清单SHA256> \
  --work /runs/new-cup-prepared \
  --run-root /runs/new-cup-noaction-30k \
  --reason1-checkpoint /models/Cosmos-Reason1-7B \
  --checkpoint /models/Cosmos-Predict2.5-2B/base/post-trained/81edfebe-bd6a-4039-8c1d-737df1a790bf_ema_bf16.pt \
  --tokenizer /models/Cosmos-Predict2.5-2B/tokenizer.pth
```

该入口要求平台 `TRAIN_JOB_ID`。正式训练前沿用完整模型四卡两步 Adam、保存、独立进程恢复检查。
旧配置和旧实验记录仍保留，新的 RealMan 入口明确选择无参考版本。

官方 Video 资产固定在 `nvidia/Cosmos-Predict2.5-2B` 的 revision
`15a82a2ec231bc318692aa0456a36537c806e7d4`，文件为上述 `base/post-trained/…pt`，
SHA256=`565bbb2c9645737327983f4461e4d32627bba465b0a8dc26447edea144e1ff47`。
不能用原来的 `robot/action-cond` 权重替代。tokenizer、Reason1 沿用原有官方资产身份。

## 部署导出

新模型使用新 schema `cosmos-noaction-vfa-diffusion28-joint8-v2`。部署加载器自动识别，
恢复训练的结构参数并停用滚动参考。旧模型继续使用旧加载路径。

训练得到新检查点后，在部署主机上执行：

```bash
python scripts/cosmos/export_noaction_deployment.py \
  --training-config /runs/new-cup-noaction-30k/resolved.yaml \
  --checkpoint /models/new-cup/step_005000.pt \
  --normalization /runs/new-cup-prepared/normalization.json \
  --tokenizer /models/cosmos/tokenizer.pth \
  --hardware-config ./deploy/realman_hardware.joint8.example.json \
  --upstream ./third_party/cosmos-predict2.5 \
  --base-checkpoint /models/cosmos/base-post-trained.pt \
  --text-cache /models/new-cup/text \
  --output /models/new-cup/runtime.noaction.json

python deploy/serve_realman.py --config /models/new-cup/runtime.noaction.json
```

导出器校验配置/检查点/normalization，生成 runtime JSON 与过滤力检查点的 `.force.json`。
`--upstream/--base-checkpoint/--text-cache` 用于迁移到部署主机后的路径替换。
默认沿用同步分段控制；可以显式指定 `--execution-mode async_relaxed`。
控制循环时序与模型内部的联合采样是两回事。该脚本只生成配置，不启动网页或控制。

关节部署链路将实测 SDK 角度转成弧度后归一化；预测反归一化后仅在 SDK 边界转回度，
直接下发七轴目标，不做末端 IK。normalization、模型及硬件 action_space 必须一致。
示例控制为 30 Hz 低跟随；采集端原有 200 Hz 高跟随与此控制节奏不同，真机需另行验证。

新采样器后端为 `joint_static_cache`；`reference` 是不使用缓存的数值对照后端，
这里的英文 backend 名称不代表输入参考轨迹。旧 `video_isolated_reuse` 与其
CUDA Graph/串行编译逻辑不适用于新模型，加载时明确拒绝。
本版先保证计算对齐与静态缓存正确，完整模型速度还需在实际权重上测量。

## 验证范围

测试包含官方非 action-cond 原生前向/梯度对齐、分支隔离、下游 loss 的梯度传播、
推理对未来标签不敏感、缓存对照、单次训练核心、activation checkpoint 梯度对齐、
恢复后下一步更新完全一致、部署不读写滚动参考、训练资产错配拒绝。
本地还使用官方结构的缩小模型检查 CPU BF16 和 CUDA BF16 前向/反向。
它们验证实现，不能代表新数据拟合、完整模型显存、延迟或真机性能。

本次执行结果：**159 passed**，命令为：

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=src:deploy python -m pytest -q \
  tests/cosmos deploy/tests \
  --ignore=tests/cosmos/test_reason1_native.py
```

使用 `realman_deploy` 环境（Torch 2.7.1+cu128、PyArrow 24）。本地缺少 Transformers，
Reason1 原生测试在远端环境补验，**3 passed in 4.63s**；其余选中测试全部通过。新增覆盖关节 8 维训练/部署、
关节弧度与 SDK 度转换、不经 IK 下发、采集断点/无效行、原始帧索引与缓存 mask 验证。
CUDA 检查使用 RTX 3080、2 层缩小模型和 3 轮采样；缓存开关的 action/force/video
最大绝对差在这一次 BF16 检查中均为 0。CPU FP32 缓存对照使用数值容差检查。

本机已下载官方非 action-cond 权重，通过固定 SHA256、完整 strict state_dict 加载及有限值检查。
远端环境、资产和准备凭据见 `joint8-environment-preparation-20261020.md`。
50 条新数据已定版，1000 步和 20000 步两组四卡 H200 任务已提交排队；缓存、四卡预检、
正式训练和新模型离线拟合评估仍待 GPU 启动后执行。详见
[本轮训练记录](experiments/realman-joint8-merged50-20261020.md)。
