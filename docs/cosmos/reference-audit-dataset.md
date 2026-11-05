# 参考轨迹数据集与审计网络接入合同

## 1. 当前完成到哪里

本分支基于 `codex/cosmos-rm71d78-videoisolated-delta30k` 的
`931ae1b6048944f44696d2677695fa53a1690a78`。当前增加的是 **CPU 数据准备与验证层**：

1. 从完整 episode 的真实测量轨迹生成稀疏、可追溯的 reference bank。
2. 在取样时生成变长、非均匀、带几何扰动和缺失的参考输入，并自动产生对齐监督。
3. 检查 reference bank 与原 Cosmos Joint8 cache 的来源、连续段和 episode/frame 对应关系。

**本提交没有实现审计网络、reference-aware WAM、联合训练 Dataset wrapper/collate 或新的训练入口。**
原 `NoActionJointModel` 的 `reference_mode='none'`，会拒绝 `reference`、
`action_reference` 等未来动作/参考条件。把新数据字段直接塞给旧训练命令不会启用此方案。
因此本文只提供已经可执行的 CPU 命令，不提供冒充已接入的训练命令。

本次处理不修改原始数据、既有缓存或基线模型文件，不调用 GPU。启智上的模型接入和训练属于后续工作。

## 2. 算法目标与职责边界

拟接入的数据流为：

```text
完整参考 R（训练：GT 派生；部署：VLM 提供）
               + 当前/因果历史观测 O_t
                          ↓
                observation-only auditor
                判断当前应读取哪些参考点
                          ↓
              可微参考读取 → 局部参考 token
                          ↓
           原 WAM 的 Action Expert 条件分支
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
后续可由它构造相对当前 TCP 的参考表征，无需假定旧 state 已含 TCP，也不在本阶段用不明约定的 FK 替代测量值。

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

未来 Dataset wrapper 必须把 epoch 显式传给 `sample()`，并把 seed、epoch、增强配置写入 checkpoint/run metadata。
`DataLoader(persistent_workers=True)` 时，在主进程修改普通 `dataset.epoch` 不会自动更新 worker 内的副本；
应通过共享 epoch 状态、sampler 传递的 epoch 信息或重建 worker 实现同步。当前 reader 不提供自动传播机制。
多卡下不应无意把 worker ID/rank 加进此种子，导致同一 episode/epoch 在不同 worker 中变成不同计划。

## 5. 与既有 cache 连接及后续联合训练

### 5.1 连接和 batch 合同

使用 **`(episode_index, frame_index)`** 连接 reference bank 与旧 cache 的当前观测时刻；
旧 `sample_index` 是保留样本序号，不是原始帧号。gap、reset 与无效行会使两者不同。
超出有效 anchor 或不匹配的来源应报错，不能用最近帧静默补齐。

当前绑定核对的旧时间合同为：动作 offsets `0..31`，力 offsets `1..32`，历史 offsets `-31..0`，
图像 offsets `0,4,...,32`。这里未来视频/力属于原 WAM 的预测目标，**不因此成为 auditor 的当前观测输入**。

后续 collate 必须独立对参考长度做 padding，而不是裁为动作块长度。建议输出：

```text
reference_pose            [B, M_max, 8]
reference_padding_valid   [B, M_max]    # 真实槽位 / padding
reference_valid           [B, M_max]    # 真实槽位且未被增强 mask
alignment_target          [B, M_max]    # padding 与不可见槽位均为 0
alignment_enabled         [B]
```

两种 mask 分开保存便于诊断；最终参考几何 attention 只读有效槽位。被隐藏点的占位值是零位置、单位四元数、零夹爪，
不能把它当成真实 pose。全参考不可见时必须使用显式 NULL token 或绕过参考分支，
不能对全 `-inf` 的 logits 执行 softmax。alignment CE 只在启用样本和合法支持集计算，避免 `0 * -inf` 产生 NaN。
若当前段完全隐藏、其他段仍可见，模型不能读取监督段号；如何学习 abstain/NULL 仍需接入时明确设计和验证。

### 5.2 拟实施的可微联合训练路径

以下是接入计划，不是本提交已经运行的网络：

1. 编码干净的当前/因果历史观测和增强参考，输出对可见参考点的读取分布 `p_t`。
2. 根据 **预测** `p_t` 进行可微局部读取，例如对参考 latent 做软窗口加权并生成少量有序局部 token。
   不在训练 forward 中用 GT 对齐标签硬切窗口；直接 argmax/top-k 切片会切断动作损失到 auditor 的梯度，需另外处理。
   姿态先编码为 latent 再融合，不直接无约束平均四元数坐标。
3. 局部 token 通过新的条件 adapter/cross-attention 只送入 Action Expert；保持既有动作块和 flow matching target。
4. 联合目标以 `L = L_action_FM + λv L_video_FM + λf L_force_FM + λalign L_alignment` 为起点。
   `L_alignment` 用自动 provenance 软标签训练读取；`L_action_FM` 必须能经参考读取、adapter 反传到 auditor，
   这样学习的是有助于准确动作的参考选择，而不是只有独立进度分类器。权重、冻结策略和稳定性需要后续实验决定。

审计编码必须只取观测信息：不能把含未来目标、带噪未来视频/力 token 或预测动作的 joint hidden state 直接称为
“obs-only”特征。可另建观测编码器或使用明确隔离的 observed context。贡献 3 的后层 WM 特征监控器另行处理。

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

`bound` 不等于已训练可用：当前报告明确为
`cache_tensor_payloads_verified=False`、`model_integration_verified=False`、
`training_readiness_evaluated=False`。没有据此验证缓存张量本体、GPU forward/backward、机器人 rollout 或成功率。

## 8. 数据划分与启智后续工作

当前 bank 与旧 cache 都是 **all 78 episodes train，val/test 为空**。这使它能复用旧缓存身份，
但不能提供 held-out 泛化结论。`--split-json` 可记录新 train/val/test membership，
不会自动重算旧 normalization，也不能消除已有 all78 checkpoint 看过验证 episode 的事实。
本次绑定器要求原 all78 membership；未来做正式独立评测，需重建相应缓存/统计并明确模型预训练暴露范围。

本地 CPU 负责原始身份审计、reference bank、增强测试与 provenance 绑定；不占用本机 GPU。
启智可先用 CPU 校验同步后的 bank/receipt，随后在原训练环境中完成 Dataset wrapper、变长 collate、obs-only auditor、
Action Expert 条件入口与联合 loss。先检查 mask、NULL fallback、无标签泄漏和梯度可达，再做训练。
此后还需要真实 VLM 参考分布和真实接触任务评估，才能判断训练时的合成参考能否迁移、模型是否确实根据感知细化动作。
