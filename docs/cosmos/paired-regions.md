# RealAction：每视角 32 对候选关系 query

从 `codex/clean-action-force-chunk` 的 `1f1756ee80ab113ea0ae4bea772a4f469bf0ca08`
新建 `codex/realaction-paired-region-queries32`。这里是可训练模型与离线检查工具；
代码和 CPU 测试不构成真实训练效果，也未申请 GPU、改动运行中的 source 或启动训练。

## 数量与空间含义

输入仍是九帧双视角 `224×448`。Cosmos latent `[16,3,28,56]` 经过空间 `2×2`
patch 后为 `[3,14,28,2048]`。每视角每 latent 时间位置有 196 个空间 token，
去掉初始位置后保留完整 784 个未来 token 给 Force 原有的 dense cross-attention。

- 每个视角 32 对，每对两个端点：两个视角共 **128 个基础可学习 query 向量**。
- 同一组 query 共享到两个未来 latent 时间位置：**256 个未来区域特征**。
- 每视角 32 个目标端点以 `4×8` 软参考网格初始化；另一端全图读取。
  目标端点参考点可移动且没有硬裁剪。每个参考区域平均约 196/32=6.125 个 patch。
- 默认在第 14 个 Cosmos block 后读取（配置零基 `region_layer: 13`）。
  主视角、腕部视角、时间组、关系编号、端点编号全保留。
  同编号的两个相机 query 不代表已识别为同一个三维点。
- query 的两个端点没有预先赋予“真实夹爪／真实物体”的语义。
  它们学习潜在关系，允许多个候选无明确物理意义；不能把全部 32 对都称为接触点。

## 计算路径

1. Cosmos 正常执行至第 14 层，Force 同时保留原 H/U/W 路径与 dense Video 读取。
2. Reader 从同层两个视角的未来 patch 读取成对特征，输入同一份干净归一化 Action32。
   每一对还融合两个端点的相对位置。该次读取没有 Force token 输入。
3. 从这一层开始，256 个区域特征与 96 个 Force H/U/W token 做双向 self-attention；
   区域特征保留 query 身份，通过固定的本次读取图映射回原 Video 空间网格。
   写回只覆盖未来位置，随后 Cosmos 原生 block 继续计算。
4. 后续 block 继续更新同组区域和 Force 特征，并按同组读取图写回。
   因此该分支确实影响后续视频生成，不是生成结束后只画一张热图。
5. 最终 Force 仍由 32 个 query 一次预测完整 `[1..32]` chunk；AE 继续读取 detached
   Video/Force memory，仅 AE 的扩散输入含 noisy Action 与 Action clock。

每层保留原 Force dense Video 读取。桥接宽度 512、8 heads；残差 gate 从 0.01 开始，
目的是小幅接入预训练空间，不保证预训练能力不受影响。
直接 Force-to-Force mask 沿用基线，但跨区域特征可间接交流，不能再声称保留跨层严格因果隔离。
所有输入只来自当下观测／历史力、干净候选动作和生成视频；没有未来真实力输入。

## 训练与自监督的准确范围

基础三项 VideoFM、ActionFM、ForceMSE 权重仍为 1:1:1。新增：

```
L = L_video + L_action + L_force
    + 0.05 * L_region_reconstruction
    + 0.001 * L_target_endpoint_diversity
```

区域重建仅用融合前的区域瓶颈和读取图，重建同层 detached 的标准化生成视频特征，
没有直接读取全部 dense 特征的 decoder 捷径。多样性惩罚作用于不同候选的目标端点，
不强迫同一对的两个端点远离；真实接触时两端可以靠近。

这是**无需人工框标注的生成特征自监督 + 现有力监督**，不是 DINO/CoTracker 教师实现，
也不是已验证的接触定位。全局重建可能关注背景，力历史也可能成为捷径。
应检查注意力分散、候选重复、遮挡敏感性和保留 episode 的力误差；不能只凭热图好看判断有效。

VideoFM 的加噪真实视频路径与 Force/AE 的生成路径分开。
Force 的 16 步视频 rollout 仍不展开反传；最终对独立生成 latent 做一次可微 clock0 refresh，
ForceMSE 与区域重建可沿它更新 Cosmos。未来真实 Video/Force 仅进入原监督项。
这改变了基线的梯度耦合，并增加每次生成时的 Force/桥接计算，实际显存和吞吐必须另做 GPU 预检。

配置 `region_writeback: false` 可做“区域↔Force，但不写回 Video”的对照；
基础 RealAction 默认配置仍完全不创建区域模块。此分支不合并分类器实验。

## 配置与检查点

新 task：`cosmos_forcewm_paired_4gpu_30k`，model：`cosmos_forcewm_paired`。
保留四卡 B16、global64、lr1e-4、30000 步、save5000、原数据与归一化配置。
上述两个区域损失及模块宽度是明确新增的超参数。

```bash
PYTHONPATH=src python scripts/cosmos/resolve_config.py \
  --task cosmos_forcewm_paired_4gpu_30k \
  --text-cache /path/to/verified/text-cache \
  --feature-cache /path/to/verified/feature-cache \
  --checkpoint /path/to/official/robot-action-cond.pt \
  --upstream /path/to/this-repo/third_party/cosmos-predict2.5 \
  --output-dir /path/to/new-paired-run/training \
  --config-output /path/to/paired-resolved.yaml
```

沿用现有 distributed_chunk 的计划／执行和 four-rank save/restore 预检，不能套用旧门禁记录。
本轮只提供入口，没有执行服务器操作。新模块和完整 runtime manifest 都纳入模型、优化器和保存。
新 checkpoint schema 为 `fastwam-cosmos-paired-regions28-v1`；严格拒绝混载旧 RealAction v3，
也不把旧 checkpoint 上随机初始化的 query 当作已经学会定位。默认从同一官方 Cosmos 权重初始化。

## 离线视频检查

完成新分支训练后，以它的真实 checkpoint 和对应展开配置运行：

```bash
PYTHONPATH=src python scripts/cosmos/evaluate_regions.py \
  --config /path/to/paired-resolved.yaml \
  --checkpoint /path/to/new-paired-run/training/checkpoints/weights/step_005000.pt \
  --checkpoint-sha256 ACTUAL_SHA256 \
  --tokenizer /path/to/tokenizer.pth \
  --indices 0,100,200 \
  --output /path/to/new-offline-inspection
```

`indices` 是实际 cache 的样本编号，不是 episode 编号；选择时应记录 episode/anchor。
默认显式传同一 sample.action 给两个 WM 与区域 reader，不运行 AE 去生成替代动作。
每个窗口保存：

- `viewer.html`：独立 HTML，可切换全部 32 对、播放、逐帧、开关热图；两个视角并排。
- `regions_overview.mp4`：全部候选端点与连线；`pair_00.mp4`：指定候选的热图视频。
  `--pair` 只改聚焦视频，HTML 和原始数据仍保留全部候选。
- `region_maps.npz`：两时间组／两视角／32 对／双端点的完整 patch 概率图。
- `all_pairs.png`：一张全部候选的预览；完整两个时间组请使用 HTML 或视频。
- `generated_rgb.npy`、`predictions.npz`、`metadata.json` 和汇总，包含真实 checkpoint SHA、
  配置 SHA、数据身份、episode、anchor、原始帧索引与 padding 标记。

这里可视化的是最终生成后 clock0 refresh 的第 14 层读取权重，叠加到对应生成视频上。
它不是逐次去噪的完整 attention trace，也不是有人工验证的物体位置。
红 A、青 B、黄连线；质心之外一定看完整热图，双峰图的质心可能落在没有物体的空隙。

九个 RGB 采样位置 `[0,4,8,12,16,20,24,28,32]` 对应
`[无未来图,组1,组1,组1,组1,组2,组2,组2,组2]`。
只按 VAE 分组保持，不人为插值出 32 个独立定位结果；VAE/Transformer 的感受野也不局限于该帧。
工具只报告归一化 Force MSE 和同窗口 Hold MSE，不把它写成 N，
也没有拿 VAE 重建的 GT 视频冒充原始 RGB PSNR。

推荐离线判断：位置是否落在夹爪／被操作物体附近；端点是否稳定且关注不同有用关系；
操作区域遮挡与同面积背景遮挡是否影响 Force 预测不同；保留 episode 上 MAE/Hold 比是否改善。
离线推理需要可容纳模型的本地或服务器 GPU；导出的 HTML、PNG、MP4 可以在普通电脑离线查看。

## 本次验证范围

`PYTHONPATH=src python -m pytest tests/cosmos -q`：82 项 CPU 测试通过。
包括实际缩小的 Cosmos 网络，干净 Action 输入、未来真实标签不影响推理、
所有 query 的 Force 梯度、空间写回影响 Video、BF16 与原生 FP32 区域、
gradient checkpointing 一致性、两次 Adam 后保存恢复及下一步逐值一致，
还有左右视角坐标与 MP4 编码/解码帧数。另对 Git 中原始 `1f1756e` 模型进行独立比较，
默认配置的 state、loss、全部参数梯度逐值一致。HTML 脚本已做语法和 DOM/canvas 替身执行检查，
未将这种检查描述为真实浏览器交互验收。渲染用合成测试帧检查，未冒充训练模型预测。

按默认宽度实例化模块计数，Reader 为 6,342,146 参数，15 个 bridge 每个 4,463,874，
共新增 73,300,256 参数。此数不是显存或吞吐实测；额外 Force 执行次数使计算增长不能按
参数增幅简单估算。尚无本分支 GPU 预检、训练、训练后 query 热图或模型质量结论。
