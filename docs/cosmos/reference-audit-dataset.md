# 完整参考轨迹、观测审计网络与 WAM 联合训练

## 1. 当前完成到哪里

本分支基于 `codex/cosmos-rm71d78-videoisolated-delta30k` 的
`931ae1b6048944f44696d2677695fa53a1690a78`。本分支已经实现以下代码路径：

1. 从完整 episode 的真实测量轨迹生成稀疏、可追溯的 reference bank。
2. 在取样时生成变长、非均匀、带几何扰动和缺失的参考输入，并自动产生对齐监督。
3. 检查 reference bank 与原 Cosmos Joint8 cache 的来源、连续段和 episode/frame 对应关系。
4. `ReferenceCachedDataset` 与变长 collate，把完整参考和监督分开连接到原缓存样本。
5. `ObservationReferenceAuditor` 仅根据当前/因果观测推断参考进度，并产生 32 个可微局部条件 token。
6. `ReferenceJointModel` 把条件只加入 Action Expert，使用原 FM 损失与自动对齐 CE 联合训练。
7. 启智四卡训练入口、资产核验和更新/保存/恢复检查入口，默认单卡 batch 64、30,000 步。

原 `NoActionJointModel` 保持 `reference_mode='none'`，基线文件不改；新模型通过独立 factory
`fastwam.cosmos.reference_model.create_reference_joint_cosmos` 启用参考分支。
不能只把新字段塞给旧 factory 就认为已经使用了参考。

代码实现与训练证据分开记录：本文不声称远端 PyTorch 测试、四卡检查或正式训练已经通过/开始。
已有 CPU bank 与来源绑定结果见第 7 节；运行后的配置、日志与 checkpoint 才是训练状态的依据。
原始数据与既有缓存不被改写；参考 bank 可在本地 CPU 制作，模型训练在启智执行。

## 2. 算法目标与职责边界

已实现的数据流为：

```text
完整参考 R（训练：GT 派生；部署：VLM 提供）
               + 当前/因果历史观测 O_t
                          ↓
                observation-only auditor
             预测进度分布 + 学习前瞻范围
                          ↓
              完整变长 M → 32 个软读取 token
                          ↓
          条件 adapter → 只加到 AE 输入 token
                          ↓
                  原格式的动作块 A_t
```

审计网络的输入不包括预测动作、未来 GT 动作、GT 进度或 episode 时钟。它从当前图像、当前真实
state/TCP、因果力历史和参考内容中推断应该读取哪一段参考。预测动作参与失败检测、回退和上层重规划，
属于另一个外部模块，不并入这里的输入合同。

WAM 仍预测现有动作块；不把输出改成“进度＋动作”，也不要求动作是参考的数值残差。
参考只提供任务路径和顺序信息，执行动作仍由观测条件下的 flow matching 学习。
GT 派生参考与 GT 动作监督分开保存，增强只作用于前者。

这是一种训练接口设计，尚不是“已证明消除参考依赖”的结论。加噪也不能保证参考无法直接执行：
尤其默认保留段端点时，简单任务仍可能存在复制捷径，需要后续无参考、错误参考、不同扰动强度和真实 VLM
参考实验来验证模型是否使用视觉与力。

## 3. 离线 reference bank

### 3.1 几何来自真实测量，不来自 Joint8 动作标签

原始 `observation.state` 为 `[N,20]`：

| 列 | 含义 | 在此数据层的用途 |
| --- | --- | --- |
| `0:7` | 真实关节角 | 原 WAM state 合同保持不变 |
| `7:10` | 真实 TCP xyz，米 | 参考位置与当前 TCP |
| `10:13` | 真实 TCP Euler xyz，弧度 | 转为参考与当前 TCP 四元数 |
| `13:19` | 滤波后的六维力/力矩 | 校验与原力输入一致 |
| `19` | 真实夹爪开度 `[0,1]` | 参考与当前夹爪状态 |

参考的每个点为 **8 维**：

```text
[x_m, y_m, z_m, qx, qy, qz, qw, gripper_open_0_1]
```

姿态约定为原始记录的 extrinsic xyz，即 `Rz(rz) Ry(ry) Rx(rx)`；输出为归一化 quaternion xyzw，
处理 Euler 跨 ±π 和四元数符号连续性。默认坐标系名字为 `recorded_controller_tcp_frame`，
只是记录坐标约定，不能把它自动解释成相机坐标或未经验证的机器人基座坐标。
部署时必须把 VLM 参考转换到同一已标定坐标系和单位。

现有 Joint8 cache 的 state 取原始列 `[0,1,2,3,4,5,6,19]`，即关节角＋夹爪，**不是 TCP pose**。
新 bank 另存每个有效观测时刻的 `observation_tcp [N_valid,8]`。
`ReferenceBank.sample()` 只返回与该样本时刻匹配的一行 `current_tcp [8]`，不把整个未来测量序列暴露给模型。
审计网络由它构造相对当前 TCP 的参考表征，无需假定旧 state 已含 TCP，也不用未验证的 FK 替代测量值。

动作标签继续使用既有 `delta_current_joint7` 加绝对夹爪开度；参考的 8 维与动作的 8 维含义不同，不可直接相减。

### 3.2 连续段、简化与文件结构

CPU 分段遵循原 `realman_joint.continuous_segments` 的有效性与断点语义，包括采集有效性、
动作已应用、力时间/滤波预热条件，以及时间缺口、滤波 epoch、跟随 session、leader/follower anchor 变化。
不跨 gap、reset 或 session 边界进行平滑或插值；单行段不作为有效连续段。

每个连续段单独做误差有界的分段简化，默认 `max_stride=12`、位置容差 `0.005 m`、
旋转容差 `0.05 rad`、夹爪阈值 `0.05`。保留段首尾和重要夹爪变化的相邻点。
这组离线容差只约束基础稀疏表示；之后的在线删除/扰动不继续保证同样的误差和最大间隔。
“完整 episode 参考”指覆盖该 episode 的全部有效连续段，不是填补无效行。

```text
reference-bank/<version>/
  manifest.json
  episodes/
    episode_000000.npz
    ...
    episode_000077.npz
```

| NPZ 字段 | 形状 | 用途 |
| --- | --- | --- |
| `reference_pose` | `[M,8]` | 基础稀疏几何参考 |
| `source_frame` | `[M]` | 每点对应的原始 episode 行号，仅监督/追溯 |
| `segment_id` | `[M]` | 每点所属连续段，仅监督/追溯 |
| `segment_bounds` | `[S,2]` | 原始帧区间 `[lo,hi)`，仅监督/追溯 |
| `source_timestamp_s`、`source_capture_time_s` | `[M]` | 来源时间记录，仅追溯 |
| `valid_anchor_frame` | `[N_valid]` | 可与旧训练样本连接的原始帧号 |
| `observation_tcp` | `[N_valid,8]` | 对应有效观测行的真实 TCP＋夹爪 |

Manifest 记录原始文件 SHA256、生成代码 SHA256、坐标/动作约定、简化参数、分段、split 和产物校验和。
新 bank 必须写到原始数据目录之外，输出目录必须不存在；未完成或校验失败的 bank 不可读取。

## 4. 在线增强：变化的是参考，不是动作真值

`augment_reference()` 在 CPU 上执行；`ReferenceBank.sample()` 将它与 episode/frame 查找组合。
不离线固定一套 noisy reference：离线保存可追溯的基础 bank，在线按 epoch 生成不同版本。

默认增强包括：

1. 各连续段独立随机保留内部点，先从 `0.45–1.0` 采样保留概率；边界与重要夹爪事件强制保留，实际保留比例可高于该概率。
2. 以概率 `0.15` 对内部点增加最多 2 个重复槽位，模拟参考中的 hold。
3. 加平滑位置扰动，默认尺度 `0.005 m`；以合法四元数旋转组合施加默认尺度 `0.03 rad` 的姿态扰动。
4. 点 mask 概率 `0.05`；span mask 触发概率 `0.25`，最大 span 比例 `0.2`；整参考 dropout 概率 `0.1`。

因此参考长度 `M` 可变，source-frame 间隔非均匀，可包含重复点；`M` 不等于完整 GT 帧数，也不被固定为动作块长度 32。
当前实现是**非均匀删点＋重复/hold**，未实现独立的连续时间重参数化或插值 time warp。
不要把随机长度描述成模型已学会任意节奏对齐。

默认不随机更改夹爪数值，几何扰动和局部 mask 都保留段端点；局部 mask 仍可隐藏内部事件，整参考 dropout 会隐藏所有端点。
整参考缺失时，任务文本/任务 ID 或其他目标条件应由原观测侧独立提供，不能依靠被丢弃的参考恢复任务意图。
本阶段没有采集真实 GPT/VLM 轨迹，也没有生成实际失败恢复、回退或重规划示范。

### 4.1 输入与监督必须分离

`sample()` 返回 `{'inputs': ..., 'supervision': ...}`：

| 模型可见 `inputs` | 形状 | 语义 |
| --- | --- | --- |
| `reference_pose` | `[M,8]` | 增强后的参考；不可见位置有占位值 |
| `reference_valid` | `[M]` bool | 该槽位几何是否可读 |
| `reference_channel_valid` | `[M,3]` bool | xyz/rotation/gripper 三组可见性；当前实现同步 mask |
| `reference_order` | `[M]` | 增强后槽位的 `linspace(0,1,M)`，不是 GT 时间 |
| `reference_break_before` | `[M]` bool | 段首标记，禁止把两段误接成连续路径 |
| `reference_present` | scalar bool | 是否还有任何有效参考几何 |
| `current_tcp` | `[8]` | 当前一行真实测量值，由 bank reader 补入 |

`source_frame`、原始时间戳、`segment_id`、`segment_bounds`、对齐目标、扰动误差和 anchor/frame ID 都只用于
Dataset 内部连接或 loss/审计，不得拼进 auditor/WAM 输入。不要提供 `frame_index/episode_length` 这样的 GT 时钟。
允许参考自身的顺序和段边界，是因为部署时参考本来也包含路径顺序；不意味着模型可以读取来源帧号。

### 4.2 对齐监督随 mask 重分配

`alignment_target()` 根据生成参考时保存的 source provenance 自动构造软标签，不根据加噪后的空间最近点重新标注。
对观测 anchor 所属的连续段，在仍可见的参考 source coordinate 之间做两邻居线性分配；
重复 source coordinate 的多个可见槽位平分该位置概率。

- 被 mask 或 padding 的节点不能得到目标概率。
- 只剩单侧可见节点时，目标截到该侧最近的可见来源位置；这是“应该读取哪个可见参考”的标签，不声称真实执行进度发生跳跃。
- anchor 所属段完全不可见时，目标全零、`alignment_enabled=False`，即使其他段仍有可见节点也不跨段分配。
- 整参考 dropout 时同样关闭 alignment loss。
- `alignment_target_unmasked` 只用于诊断原拓扑，不参与 masked-key attention loss，也不作为模型输入。

### 4.3 种子、epoch 与持久 worker

当前随机数种子为 `SeedSequence([seed, epoch, episode_index])`，没有 `frame_index`。
同一 episode、同一 epoch、同一配置中的不同 anchor 会看到完全相同的参考计划；新 epoch 才换一个版本。
这避免同一执行窗口中的参考突然跳变，也保证给定种子的可复现性。

`ReferenceCachedDataset.accepts_epoch_indices=True`，sampler 将 `(sample_index, epoch)` 随每个取样请求
显式送到 worker；wrapper 再把 epoch 传给 bank。这样在 `persistent_workers=True` 和预取时，也不会依赖
主进程中未同步的普通 `dataset.epoch`。参考 seed 与增强参数保存在 resolved config；原 sampler 的恢复状态
继续确定 epoch 和取样位置。种子不包含 worker ID/rank，同一 episode/epoch 在各 worker 中使用同一参考计划。

## 5. 与既有 cache 连接及联合训练实现

### 5.1 连接和 batch 合同

使用 **`(episode_index, frame_index)`** 连接 reference bank 与旧 cache 的当前观测时刻；
旧 `sample_index` 是保留样本序号，不是原始帧号。gap、reset 与无效行会使两者不同。
超出有效 anchor 或不匹配的来源应报错，不能用最近帧静默补齐。

当前绑定核对的旧时间合同为：动作 offsets `0..31`，力 offsets `1..32`，历史 offsets `-31..0`，
图像 offsets `0,4,...,32`。这里未来视频/力属于原 WAM 的预测目标，**不因此成为 auditor 的当前观测输入**。

`reference_collate()` 只对参考的 M 维做 padding，不截成动作块长度。实际输出分成两个字典：

```text
reference_inputs:
  reference_pose            [B, M_max, 8]
  reference_valid           [B, M_max]    # 真实槽位且未被增强 mask
  reference_channel_valid   [B, M_max, 3]
  reference_order           [B, M_max]    # 各参考独立的归一化顺序
  reference_break_before    [B, M_max]
  reference_present         [B]
  current_tcp               [B, 8]
reference_supervision:                  # 只送 loss
  alignment_target          [B, M_max]    # padding 与不可见槽位均为 0
  alignment_enabled         [B]
```

当前接口把 padding 与增强 mask 都表示成 `reference_valid=False`，attention 不读取其几何。
不可见点的占位值在模型投影前再次清零，避免 padding 或 NaN 污染有效点。全参考不可见时启用显式 NULL key，
并将 AE 的参考条件置零，连 adapter bias 也一起消除。alignment CE 只在启用样本上计算，避开 `0 * -inf`。
若当前连续段全部隐藏而其他段仍可见，本实现关闭该样本 CE，但不会使用 GT 段号帮助推断，也没有额外监督
“当前段不可见”的 abstain 分类器；这是仍需评估的情况。

### 5.2 当前网络如何读取完整参考

审计器默认宽度 128、4 个 attention head、2 层 Transformer，使用独立于未来扩散状态的当前观测分支：

1. **观测查询。** 独立编码的 `first_frame_latents [B,16,1,H,W]` 空间池化到 4×4；与当前 proprio、
   单行 TCP、32 步因果 state/force 历史及其有效性 mask 一起编码为 observation query。
   不读取未来 `video_latents`、未来力、带噪动作、预测动作或 GT 进度。
2. **完整参考编码。** 每点使用原始 xyz/quaternion/gripper、相对当前 TCP 的位置/旋转、参考自身顺序、
   段边界和 mask 特征。参考 M 保持不变；两层 Transformer 编码后，与观测 query 计算对齐分布 `p_i`。
3. **预测软地址。** 设参考自身顺序为 `u_i`，预测进度为 `s = Σ p_i u_i`；观测分支同时预测
   `h = 0.01 + 0.24 sigmoid(span_head(obs))`。第 k 个局部地址为
   `c_k = clamp(s + h·k/31, 0, 1)`，k=0…31。h 是参考归一化顺序上的前瞻范围，不是 GT 未来 32 帧的索引跨度。
4. **32 个可微局部读数。** 每个地址用 Gaussian 权重读取参考 latent，带宽 `max(h/4, 0.01)`；
   另用对齐分布在各连续段上的概率质量作为软先验。读取的是 latent，不直接平均四元数坐标。
   未来读取不再次叠加尖锐的当前 anchor logits，避免对齐收敛后 32 个槽位全部挤在当前点上。
5. **只条件化 AE。** 32 个局部 latent 与地址/前瞻/槽位顺序编码后，通过 MLP adapter 投影到 AE 宽度，
   加入 AE 的 32 个初始 action token。Video/Force 的 context、attention 可见性和输入准备保持不变。

完整参考没有预先重采样到 32；32 只固定下层条件宽度和既有动作块长度。GT 对齐标签从未参与硬切窗口、
地址计算或教师强制。前向一次计算 auditor，然后执行原 joint denoising，不先预测动作再审计。

### 5.3 联合损失、训练与推理边界

当前损失为：

```text
L = L_video_FM + L_force_FM + L_action_FM + 0.1 L_alignment_CE
```

原三路 FM 的时钟、噪声、目标、loss mask 和权重保持不变。自动 provenance 软标签只用于 CE；
原未来视频、力和动作真值继续用于原 FM。`reference_supervision` 不进入 auditor，推理时不需要提供。
动作 FM 可以通过 AE 输入 adapter → 软读取 → 进度 attention 与 span head 反传，因而能够联合优化参考选择。
审计器与 adapter 都注册在 `model.dit`，包含在原 trainer 的 optimizer 和完整 checkpoint 中。
静态推理缓存仅含原 O/H/text；训练时不复用 detach 的 inference cache。

新增测试覆盖变长到固定 32、mask/NULL、标签隔离、仅 AE 受参考影响、action-only 梯度、checkpointing、
缓存推理与恢复一致性。测试代码存在不等于远端检查已经通过；执行结果必须看对应测试日志。
后层 WM 特征监控、失败回退与重新调用上层 VLM 的模块不在本实现内。

原 V/F/A 专家之间的 attention 可见性保持不变，行是 Query，列是 Key/Value：

```text
        V  F  A
V       1  0  0
F       1  1  0
A       1  1  1
```

新增参考条件是 Action Expert 的独立入口，不把未执行的参考冒充真实已执行 action，也不直接改变 Video/Force WM 的
输入语义。几何增强与 flow matching 扩散状态加噪是两条独立随机过程；动作标签、loss mask 和归一化保持既有合同。

## 6. CPU 准备与验证命令

在仓库根目录执行。使用独立 CPU 环境，仅需 NumPy、PyArrow 等该脚本使用的 CPU 依赖；
不要为了生成 reference bank 安装整个 GPU 训练项目。

```powershell
python -m venv .venv-reference-cpu
.\.venv-reference-cpu\Scripts\python.exe -m pip install -r requirements-reference-cpu.txt
```

以下 `$rawDataset` 为本机已处理的完整 78-episode 原始 LeRobot 数据目录；其他机器按实际位置调整。`rm71d78-v1` 已有产物时，
直接运行 verify；如需重建，选择新的输出目录，不覆盖旧 bank。

```powershell
$cpuPython = '.\.venv-reference-cpu\Scripts\python.exe'
$rawDataset = 'C:\Users\ASUS\Desktop\dataset\rm71d_plug_transfer_20261029'
$bankPath = '.\outputs\reference-bank\rm71d78-v1'

& $cpuPython scripts/cosmos/prepare_reference_bank.py prepare --dataset $rawDataset --output $bankPath --expected-episodes 78 --expected-source-sha 37c1667729162a71b4f5ea5d6d047b0da7c54a01b08a9b140eff889f9a520dfc
& $cpuPython scripts/cosmos/prepare_reference_bank.py verify --root $bankPath
& $cpuPython scripts/cosmos/audit_reference_bank.py --bank $bankPath --output outputs/reference-audit-rerun.json --epochs 8 --anchors-per-episode 3 --seed 42
& $cpuPython scripts/cosmos/verify_reference_cache_binding.py --bank $bankPath --cache-manifest outputs/source-identity/delta78-cache-manifest.json --output outputs/reference-cache-binding-rerun.json
```

`--expected-source-sha` 是此机器已审计的完整原始文件清单 hash，只适用于同一内容集合，不是可随意复制的期望值。
不传此参数时，prepare 只记录 metadata 与 numeric parquet 的来源范围；应在 manifest 中区分它与全量 raw identity。
绑定命令要求新 receipt 文件名，重复运行时另取名称。若 bank 仅含 numeric/metadata 来源，
可向绑定命令增加 `--local-raw-inventory <path-to-sha256-json>`，用于完整文件差异诊断。
输出 audit/receipt 必须放在 bank 和 cache 目录之外。

核心测试可独立在 CPU 运行：

```powershell
$env:PYTHONPATH = (Join-Path (Get-Location) 'src')
& $cpuPython -m unittest discover -s tests/cosmos -p 'test_reference*cpu.py' -v
```

网络测试需要已有 PyTorch/native Cosmos 训练环境，不能用上面的轻量 NumPy 环境冒充执行：

```bash
PYTHONPATH=src python -m pytest tests/cosmos/test_reference_model.py -q
```

### 6.1 启智训练入口与配方

以下命令在新分支仓库根目录执行。launcher 默认使用启智原训练 Python；可通过 `NATIVE_PYTHON`
指定同一已配置环境。`--plan` 只显示路径和配方，不读取资产、不申请 GPU：

```bash
bash scripts/cosmos/launch_reference_audit.sh --plan
```

先在启智 CPU 环境核验现有 78-episode cache、text、normalization、官方权重与 reference bank：

```bash
bash scripts/cosmos/launch_reference_audit.sh --resolve-only \
  --work /inspire/ssd/project/long-range-embodied-collaborative-intelligence/ky26340/rm71d-reference-audit-20261105/resolved/cpu-review-01 \
  --run-root /inspire/ssd/project/long-range-embodied-collaborative-intelligence/ky26340/rm71d-reference-audit-20261105/training/cpu-review-01
```

work/run-root 必须是尚不存在且相互独立的目录；重复验证时换新名字。此命令生成 resolved config 和核验收据，
不运行 GPU forward。随后在启智已分配的**单节点 4×H200 正式训练任务**中执行：

```bash
bash scripts/cosmos/launch_reference_audit.sh --execute --batch-size 64 --max-steps 30000
```

`--execute` 要求平台提供 `TRAIN_JOB_ID`，用该 ID 创建独立 work/run 目录；先执行四 rank 两次 Adam 更新、
保存和独立恢复检查，再启动正式训练。这是入口行为，不代表检查结果已得到。默认复用指定 delta30k prepared
目录中的数值/视频 latent/text cache，不重新 VAE 编码；模型使用固定 SHA 的官方 non-action 视频基础权重，
AE、Force 和 auditor 新建，不从旧 30k policy/optimizer 续训。

| 配方 | 值 |
| --- | --- |
| GPU / ranks | 4×H200 / 4 |
| batch per rank / accumulation / global batch | 64 / 1 / 256 |
| 正式 optimizer updates | 30,000 |
| 学习率调度总长度 / warmup | 50,000 / 2,500，沿用原调度定义 |
| 每次动作 / 局部参考条件 | 32×Joint8 / 32 个 latent token |
| 完整参考长度 | 每 episode/epoch 随增强变化，batch 内仅 padding |
| 对齐 CE 权重 / 整参考 dropout | 0.1 / 0.1 |
| 数据划分 | all 78 episodes train，无 held-out |

默认 reference bank 路径是
`/inspire/ssd/project/long-range-embodied-collaborative-intelligence/ky26340/rm71d-reference-audit-20261105/reference-bank/rm71d78-v1`；
同步位置不同时用 `--reference-bank` 指定，内容 identity 仍须匹配。`--prepared`、`--checkpoint`、
`--upstream` 可指定资产路径；入口核验 pinned 身份，不只检查文件存在。

## 7. 本次实际结果与证据边界

完整数值以生成产物为准：

- `outputs/reference-audit.json`：状态 `passed`，只验证抽样数据和接口不变量。
- `outputs/reference-cache-binding.json`：状态 `bound`，确认 numeric/metadata 来源及 episode/frame 语义绑定。
- `outputs/source-identity/source-identity-report.json`：原始清单差异解释。

| 项目 | 本次结果 |
| --- | --- |
| episode / 原始行数 | 78 / 86,608 |
| 有效观测 anchor / 连续段 | 84,349 / 552 |
| 基础参考总点数 | 12,290 |
| 每 episode 基础长度 min / median / max | 109 / 158 / 194 |
| 抽查增强计划 / anchor 样本 | 624 / 1,872 |
| 抽查增强长度 min / median / max | 89 / 153.5 / 218 |
| 出现变长的 episode | 78 / 78 |
| 出现段内非均匀间隔的计划 | 624 / 624 |
| 整参考 dropout | 68 / 624 |
| 启用 / 禁用 alignment 的样本 | 1,668 / 204 |

Reference bank identity 为 `7d0a0a108d8d338633439358548cd7e8a32b275c74420444c0ce652cf8ddd2b9`。
旧 cache 与本地 raw 的 **396 个 `data/`＋`meta/` 文件全部一致**，无遗漏、无改变。
完整 raw inventory SHA 不同，原因是本地额外保留了 78 个 `force_streams/episode_*.jsonl`；
绑定报告中其余文件无修改或远端独有项。无需删除这些日志或改写原始文件来制造全 SHA 相等。

上述 CPU `bound` 收据不等于已训练可用：其报告明确为
`cache_tensor_payloads_verified=False`、`model_integration_verified=False`、
`training_readiness_evaluated=False`。没有据此验证缓存张量本体、GPU forward/backward、机器人 rollout 或成功率。

## 8. 数据划分与仍需验证的假设

当前 bank 与旧 cache 都是 **all 78 episodes train，val/test 为空**。这使它能复用旧缓存身份，
但不能提供 held-out 泛化结论。`--split-json` 可记录新 train/val/test membership，
不会自动重算旧 normalization，也不能消除已有 all78 checkpoint 看过验证 episode 的事实。
本次绑定器要求原 all78 membership；未来做正式独立评测，需重建相应缓存/统计并明确模型预训练暴露范围。

本地 CPU 负责原始身份审计、reference bank、增强测试与 provenance 绑定；模型接口、joint loss 与训练入口已实现。
启智实际运行仍需验证完整缓存张量、BF16 forward/backward、梯度到达、四卡保存/恢复、显存与吞吐。
30,000 步是否完成只能由训练日志和 checkpoint 证明。

本设计还有明确的研究假设：

- 进度使用 `E[reference_order]`。当后验在重复动作或多个连续段上呈多峰时，均值可能落在两峰之间，
  软段先验不等于已经解决歧义，也不是硬性保证不跨段。需要针对重复路径、遮挡和局部全缺失评测。
- 只用 GT 派生的合成参考训练，没有真实 GPT/VLM 参考分布，迁移效果尚不能由数据增强保证。
- 加噪、稀疏化与 dropout 提供抗复制训练条件，不能单独证明模型不依赖参考；需有无参考、错误参考、
  参考偏差分级，以及视觉/力消融来检查细化是否来自观测。
- 没有恢复、回退、失败监控或上层重规划的真实数据和实机验证；该分支不声称完整闭环恢复框架已经验证。
