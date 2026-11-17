# Plug insertion 的 JEPA query 尺度对照

从已启用弱去重的 ViT-L 25k 权重出发，检验共享条件是否压低了 8 个 learned slot 之间的差异。A 组使用原 query 加法，B 组重新分配共享条件与 slot 差异的尺度。两组都短训同一个 reader 架构，并在短训前后评估。这个对照用于判断 query 构造的即时影响和短训后的变化。

## Query 的构造

记 $s_i$ 为第 $i$ 个 learned slot，$\bar s$ 为 8 个 slot 的均值。共享条件 $c_b$ 包含当前候选动作的完整去噪状态、动作时钟、测得的力觉历史、当前 proprio 和任务文本。令 $C(x) = x - \operatorname{mean}_{\mathrm{channel}}(x)$ 表示逐向量做通道中心化。A 组沿用 $q_{b,i}=s_i+c_b$。B 组分别构造共享项和 slot 差异项。

$$
u_b = C(c_b + \bar s), \qquad d_i = C(s_i-\bar s), \qquad
E_b = \operatorname{mean}_{k}(u_{b,k}^{2}), \quad
E_d = \operatorname{mean}_{i,k}(d_{i,k}^{2}).
$$

令 $f(E)=\max(E,10^{-6})$，B 组用固定比例 $r=0.5$ 组合两项。

$$
q_{b,i}^{(B)} =
\frac{\sqrt{f(E_b+E_d)}}{\sqrt{1+r^2}}
\left(\frac{u_b}{\sqrt{f(E_b)}} + r\frac{d_i}{\sqrt{f(E_d)}}\right).
$$

差异项跨 slot 的均值为零，因此原 query 通道中心化后，跨 slot 和通道的 RMS 为 $\sqrt{E_b+E_d}$。当两个分量的能量均高于数值下限时，B 组保留该整体 RMS，并将差异项与共享项的 RMS 比固定为 0.5。底限保护接近零能量的分量。第二个 reader 仍用基础 query 加上第一层视觉输出，因此保留基础 query 的整体尺度也控制了它与视觉残差相加时的相对量级。归一化在 FP32 中计算，组合结果转回原 query dtype。

`query_slot_condition_ratio` 默认为 `None`，沿用原加法、参数键和 checkpoint manifest，可严格加载旧权重。显式启用时固定比例进入新模型的 manifest，且无新增可学习参数。本实验先按原 manifest 严格加载两组共同起点，再为 B 组 reader 启用比例，并独立保存 reader 权重和比例元数据。 [Query 实现](../../src/fastwam/cosmos/jepa_force_latent.py)

## 数据和训练

从 plug insertion 的 78 条原始训练轨迹中，固定 12 条用于短训后的验证，每条 3 个窗口，共 36 个。短训从其他 66 条轨迹均衡抽取 512 个窗口。抽样在源训练缓存内按 episode 轮转，每条轨迹内随机且无放回，抽样种子为 812。训练窗口具有完整的 32 步历史、动作和未来力，保留原缓存中的归一化、相机顺序及观察帧偏移。 [数据打包与核验](../../scripts/cosmos/query_scale_ablation.py)

| 项目 | 两组共用设置 |
| --- | --- |
| 起点 | 同一启用弱去重的 25k checkpoint 的 reader 权重 |
| 更新范围 | `jepa_reader`，包括条件投影、视觉读取器、horizon 读取器及输出头。ViT-L（含 LoRA）、Cosmos、Force、Action 和 Action 私有残差适配器均冻结 |
| 优化 | 各 1,000 次更新，batch 64，AdamW，学习率 `1e-4`，weight decay `0.01`，梯度范数上限 `1.0` |
| 随机性 | 批序种子 `42`，训练候选动作噪声与时钟种子 `43`，辅助验证种子 `44`。两组独立创建优化器，用相同批序与随机数流 |
| 监督 | 原辅助目标，包含在线 clean Force 最终隐特征对齐（权重 `1.0`）、wrench Smooth L1（`0.1`）和弱去重（阈值 `0.95`，权重 `0.01`） |

Force 隐特征由冻结的 25k Force 路径使用真实未来 wrench 生成，作为停止梯度的监督标签。它与冻结的 JEPA patches 一次计算后缓存，两组共用。辅助损失的全局计步从 25k 起算，两个 warmup 均已完成。短训仍按原动作 FM 时钟分布加噪得到 reader 的候选动作条件，优化只使用上述辅助目标。Action FM loss 不参与优化。推理仅使用当前或过去的观测及采样过程生成的候选动作。 [训练与评估 runner](../../scripts/cosmos/query_scale_ablation.py)

## 四格评估

| 组别 | 短训前 `step 0` | 短训后 `step 1000` |
| --- | --- | --- |
| A | 25k reader，原 query 加法 | 原 query 加法，短训 1,000 次 |
| B | 25k reader，立即启用比例 `0.5` | 比例 `0.5`，短训 1,000 次 |

四格对同一 36 个窗口运行完整 16 次联合去噪，统一 Action、Force、Video 随机种子 `113`、`114`、`115`。另外在 `0 / 250 / 500 / 1000` 次更新处记录辅助指标，各自使用相同的 8 组候选动作噪声与时钟抽样。比较各层各相机的 query 对余弦相似度、超过 `0.95` 的比例、各相机获得的注意力权重、可见接触 ROI 富集度和归一化动作误差。记录成对 Force 与 Video 输出的最大差异，核验 Action 私有 Force K/V 路径的隔离。

`step 0` 的 B−A 差异衡量立即替换 query 公式的影响。`step 1000` 的 B−A 衡量适配完成后的总差异。两者之差衡量短训期间组间差异的变化。主要误差指标先在 episode 内聚合，再跨 episode 汇总。验证轨迹只与这轮短训隔离，它们参与过原始 25k 训练。结果用于训练分布内的诊断。判断对未见场景的泛化需要独立轨迹，判断接触区域对预测的因果贡献需要针对性干预，判断真实插接效果需要机器人评估。 [评估记录与配对计算](../../scripts/cosmos/query_scale_ablation.py)
