# O/R/V/F/A masked joint attention

本分支仍为 `codex/cosmos-force-joint-self-attention`。v2 在原 joint v1 上修改
三专家的交互方式，使用真实 Action32 作为参考 R；后续 GPT 轨迹使用同一显式条件接口。

| Query / Key | O | R | V | F | A |
|---|---|---|---|---|---|
| O | 是 | 否 | 否 | 否 | 否 |
| R | 是 | 是 | 否 | 否 | 否 |
| V | 是 | 是 | 是 | 是 | 否 |
| F | 是 | 是 | 是 | 是 | 否 |
| A | 是 | 是 | 是 | 是 | 是 |

O 包含当前双视角视频 token 和测量历史 H。R 为干净的参考动作编码。
V 为两个未来 latent 时间组的视频 token。F 为 32 个 learned future-wrench queries，
不输入真实未来受力。A 是 AE 在当前扩散步的动作状态。

物理拼接顺序为 `[observed video, future video, H, R, F, A]`；由组 ID 生成 mask，
不要求同一逻辑组在物理序列中相邻。有效 O 内部双向，R 内部双向，F 内部双向，
按本次五组表执行，覆盖 v1 的 H/U/W causal submask。
无效输入不得作为有效 query 的 key；无效 query 仅保留自身 sentinel 并在 block 后清零。

观测视频使用 clean-zero 时钟，且首帧的原生 action modulation 为零。
因此 O 的初始表示和各层表示都不依赖 R/V/F/A；R 可以读取 O，但不能经多层间接读取预测。
文本与当前 proprio 作为已观测条件保留各 expert 原本的 context 接口。

## AE 与两个 WM

移除 `ActionDualMemory`，不再分别对 future-video / future-force 做附加 cross-attention。
AE 使用自己的原生 Q/K/V、QK norm、RoPE、残差和 FFN，读取全部 WM prefix K/V 与自身 K/V，
这些 key 在同一个 softmax 中归一化。

由于 O/R/V/F 的 query 行禁止读取 A，可以先计算每层的 WM prefix K/V，再计算 A 行。
这是五组 masked joint attention 的精确分块计算，不是新的双 memory 模块。
prefix 来自对应层 attention 前的原生 K/V，而非 block 后 feature 再投影。
推理时同一 reference 下缓存 prefix，复用于所有 AE 去噪步。

训练时 prefix K/V 保留梯度：ActionFM 可以更新 AE 与被读取的 WM 表示。
这与 v1 的 detached AE memories 不同。前向 mask 不等同于梯度冻结；
WM 的前向输入始终没有 AE 状态、AE diffusion clock 或未来受力标签。

## 生成与监督

保留 VideoFM + ActionFM + ForceMSE，权重 1:1:1。
VideoFM 使用加噪目标视频；Force/AE 使用另一条从观测与独立高斯噪声起步的 16 步视频生成，
整个 rollout no-grad，再执行 differentiable clock0 refresh。
Force 与 AE 不读取监督路径的未来视频标签，也不根据目标 Force masks 筛选输入。
移除重复的 observed-only Video pass，因为 O 已包含在 WM prefix。

## 从 v1 step5000 热启动

显式配置 `model.initial_checkpoint` 和 `model.initial_checkpoint_sha256`。
迁移器要求旧 joint v1 schema 与 step=5000，核对完整文件 SHA256，逐项检查所有保留权重的
名称、形状和有限性。仅允许丢弃旧 `dit.memories.*` 与旧 runtime buffers。
全部保留权重原值加载，重建新 manifest 和 seed44 runtime RNG。
普通 checkpoint load 仍严格拒绝 v1/v2 混载；只有显式迁移入口允许模型权重迁移。

优化器和 scheduler 重新建立。本轮 global_step 从 0 开始、`max_steps=30000`，
即在原 5000 步权重基础上训练新的 30000 步；不把它称为旧 run 的完整状态续训。
新 run 和旧 checkpoint 分开保存。

四 H200 / 四 rank / 每卡 B16 / global64 / BF16 / ZeRO1 / lr1e-4 /
warmup1500 / cosine至30000 / save5000 沿用原设置。原生 FP32 模块维持官方策略。

正式运行前必须在实际四卡上通过两次 Adam 更新、完整保存，以及独立四进程精确恢复。
CPU 检查覆盖五组 mask、多层 O/R 隔离、缓存与完整联合 attention 的输出和梯度一致性、
AE loss 回传、未来标签隔离、padding、BF16、gradient checkpointing、严格迁移和精确恢复。
实际检查结果与资源状态记录于部署凭据，不用 CPU 测试替代 GPU 验证。

训练集真实动作条件下的重建用于结构筛选；仍需要后续 GPT 参考轨迹细化、留出数据和真机闭环验证。
