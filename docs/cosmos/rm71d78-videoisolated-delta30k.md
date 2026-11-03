# 全78条、单向V/F/A、关节增量：30000步训练

分支：`codex/cosmos-rm71d78-videoisolated-delta30k`。

本实验保留无未来动作条件的三expert joint attention。每层各自生成Q/K/V，按下表决定读取范围；不是把三个expert改成完全独立的模型。

| Query | Video KV | Force KV | Action KV |
|---|---|---|---|
| Video | 是 | 否 | 否 |
| Force | 是 | 是 | 否 |
| Action | 是 | 是 | 是 |

当前图像和实测历史的细粒度mask继续保留。未来动作是Action expert的扩散目标，不作为Video或Force条件；前向隔离不等于梯度冻结。

## 数据与动作

原始库存78条，SHA256：`b24b701239f54ad38d16c819b9ea9f6fad5f9374a683a84a68d2cd7ef27067a8`。
全78条用于训练，不建立验证集或测试集。原有连续段切分、有效传感器过滤和padding mask保留；无效采集行仍不用于监督。

动作表示为 `delta_current_joint7`。对窗口起点t及k=0…31：

```text
target[k, :7] = recorded_absolute_action[t+k, :7] - measured_state[t, :7]
target[k, 7]  = recorded_gripper_openness[t+k]
```

全部32步使用同一实测anchor，不是相邻帧差分。前7维单位rad，第8维夹爪保持绝对0–1。这与当前π0.5、ForceVLA消融的表示一致；当前FastWAM消融仍是绝对关节目标。

在物理量上转换后，对全78条的有效窗口、非padding动作位置重新计算z-score统计；保留原`[-5,5]`裁剪。state和force保留原物理定义，也在全78条上计算统计。新source、normalization、feature-cache和checkpoint均记录表示、anchor、mask和单位，拒绝与旧绝对动作混用。

最近完成的Cosmos任务实际用了60/9/9划分，不能直接复用其feature cache作为全78数据。新任务重建全部图像和数值缓存；复用官方基座、tokenizer、匹配prompt的文本缓存及已有Python环境。

## 配方

| 参数 | 值 |
|---|---|
| GPU | 4×H200，单节点 |
| Batch | 每卡64，全局256，累积1 |
| AdamW | lr1e-4，betas(0.9,0.95)，weight decay0.01 |
| 精度与随机种子 | BF16；42；Action42/Force43/Memory44 |
| 停止步数 | 30000次optimizer update |
| 调度 | 保留最近任务的50000步horizon：warmup2500 + cosine47500，eta_min1e-6 |
| Loss | Video:Force:Action=1:1:1；各项先按有效元素取平均及扩散时间加权 |
| 保存与评测 | 每5000步保存完整状态；eval_every=0 |
| 梯度裁剪 | 沿用原DeepSpeed实际值0.0；配置max_grad_norm1.0不代表引擎启用了裁剪 |
| 初始化 | 官方无动作条件Cosmos基座，独立初始化Force/Action；不续训旧absolute模型 |

停止上限与调度horizon分别记录为`max_steps=30000`、`scheduler_total_steps=50000`，同时绑定到模型training_recipe。缺省scheduler_total_steps的历史配置保持原行为。

## 启动与证据

先在已准备好的新代码目录运行只读计划：

```bash
PY=/inspire/hdd/project/long-range-embodied-collaborative-intelligence/ky26340/realman-fastwam/.venv/bin/python
"$PY" scripts/cosmos/train_rm71d78_delta.py --output-root /absolute/fresh/run
```

正式平台作业使用`dualwm:1.0`、CUDA12.8、4×H200、80CPU/900GB、共享内存200GB；前台命令：

```bash
bash /absolute/new-code/scripts/cosmos/launch_rm71d78_delta.sh
```

启动器校验原配方、官方资产和真实四H200身份，审计全78数据，四卡独立编码并严格组装cache。随后执行真实两步Adam更新、完整保存与独立四rank恢复验证，通过后才从官方初始化启动正式训练。

输出位于新实验根目录的`online/$TRAIN_JOB_ID`，保留`experiment.json`、`prepared/source_audit.json`、新normalization、编码日志、resolved配置、四rank验证回执及完整checkpoints。旧训练和消融作业不写入此目录。

## 部署

新同步HTTP服务和原runtime均先反归一化，再加**产生该次请求的观测anchor**，对外仍返回绝对关节目标。部署必须使用本分支源码或相应新wheel，不能仅替换旧部署中的JSON。

历史固定checkpoint的`evaluate_joint8_*`、`validate_joint8_local.py`等离线工具仍针对旧absolute实验；不要用其`decode('action')`路径解释本实验结果。新评测应使用匹配表示的数据源及`decode_action(..., current_state)`。

本地CPU测试验证动作转换、统计、缓存契约、部署解码、checkpoint兼容和调度一致性；实际GPU验证及训练状态以平台回执为准。
