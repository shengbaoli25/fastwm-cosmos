# ForceVLA / LIBERO 两视角：离线 VAE 缓存与 30,000 步配置

最新资源方案为单节点 **4 H200**，每卡 batch 16，梯度累积 1，全局 batch 64。8 卡配置作为显式可选配置保留，不并行创建另一容器。原单视角 2176 步实验的代码、数据与运行进程不修改。

## 训练配置

| 参数 | 值 |
| --- | --- |
| Task | `forcevla_libero_2cam_cached_4gpu_30k` |
| 策略初始化 | 完整官方 `libero_uncond_2cam224.pt`，全部 MoT/proprio 张量严格加载 |
| 模型 | 原 Video30 + Action30，原 MoT、first-frame-causal mask、Gaussian flow matching |
| 每卡 batch / 梯度累积 / GPU 数量 | 16 / 1 / 4 |
| 全局 batch / 最大更新数 | 64 / 30000 |
| 保存 | 每 5000 次更新，权重与原 ZeRO1 恢复状态；终点也保存 |
| 精度 / 优化器 | BF16 / 原 AdamW + ZeRO1 |
| LR / weight decay / 梯度裁剪 | 1e-4 / 1e-2 / 1.0 |
| 调度 / warmup | 原 cosine；1500 次更新 warmup，不乘 GPU 数 |
| Seed / workers | 42 / 每个进程 8 |
| 在线评测 | 关闭；训练后固定训练集五条序列评测 |
| 划分 | 全 50 集、34808 帧训练，无验证/测试划分 |

30000 × 64 = 1,920,000 个名义 anchor 抽样，约 55.16 遍。原 sampler、分布式 dataloader 保留；每遍末尾存在 8 个补齐抽样，因此不以名义抽样证明每条样本已训练指定次数。4 卡每遍约 544 次更新，8 卡为 272 次。

所有模型结构、噪声、timestep、loss 权重与 mask 不变。缓存入口只替代冻结 VAE 的输入编码；视频支路依然不读取生成的动作，因此生成视频不能描述成执行该动作的后果。

## 缓存合同

ForceVLA 的主相机在左、腕部相机在右；每路按官方 LIBERO `ToTensor → Resize(224,224)`，横拼为 224×448，并按原处理器归一化。每个 anchor 取 0、4、…、32 共九帧；VAE 输入 `[1,3,9,224,448]` 显式 BF16，输出已经缩放的 posterior mean `[48,3,14,28]` BF16。

**每个 anchor 的完整九帧窗口独立编码一次。** 不能整集编码后切片，也不能逐帧编码后拼接。包含全部 episode 末尾 anchors，保留原最后一帧补齐及 mask。34808 个 latent 约 3.66 GiB，另有标签与元数据。

缓存保存 stock reader 实际返回的 normalized action/proprio FP32，形状 `[32,7]`、`[32,8]`。原 parquet 数值及额外原始 state/wrench 为 FP64；这两个阶段的精度不能混淆。padding 长度为 9/32/33。T5 每个文本只保存一次，保留原 masked 行清零、最终全 true 的 attention mask。

训练 Dataset 仅打开缓存 safetensors 与 JSON/T5 feature；不会构造 RGB reader、解码图像或调用 VAE encode。缓存缺失、损坏、身份不同或未完成均立即报错，不回退到在线编码。完整 VAE/T5/stats 文件、转换 manifest、实际执行代码及处理配置均绑定 SHA。策略模型仍保留 VAE 组件以兼容原推理，但训练路径不调用它。

## 准备命令（不启动训练）

先运行 lossless 双相机 converter，然后显式执行离线 VAE cache。执行的模型仅为冻结 VAE，不含策略/优化器。现有共享原始数据与基础资产只读复用。

```bash
python scripts/forcevla/precompute_vae_cache.py \
  --task forcevla_libero_2cam_cached_4gpu_30k \
  --dataset /path/to/converted/wipe \
  --stats /path/to/shared_stats/dataset_stats.json \
  --text-cache /path/to/text_embeds_cache/forcevla \
  --vae-weights /path/to/Wan2.2_VAE.pth \
  --t5-weights /path/to/models_t5_umt5-xxl-enc-bf16.pth \
  --cache-root /path/to/vae_cache \
  --max-new-samples 256 --max-cache-seconds 600 --execute-cache
```

默认不加 `--execute-cache` 只打印计划。Pilot 完成后测真实速度，再使用相同身份 `--resume` 完成剩余 anchors；有意预算停止的 partial cache 退出码为 2，不能训练。硬中断留下未登记文件会明确拒绝恢复，需要先保存证据并显式处理。不得在 pilot 后修改缓存 encoder/writer 代码再混用旧分片。

CPU-only `prepare_cached_training.py` 导出全部 resolved YAML 和 JSON；`--check-cache` 逐分片校验完成状态与固定 VAE/stats 身份，再把 identity 绑定到训练 Dataset。未加此项的结果明确为 configuration-only、cache 未检查。输出文件不得覆盖旧计划。

## 启动前实际验收与预算

新任务需要全 34808 缓存校验、标签/T5/末尾 padding 对照及真实 VAE 样本比较，再运行 **本任务 4 卡每卡16的真实两次优化更新检查**。已有单卡 micro1×累积16的 96.59 GiB 结果不能证明本任务显存或吞吐可用。

原每分支总计 5 小时资源上限保留；缓存、预检、保存和评测均占预算。30000 步是否能完成尚未以新配置实际测量，不能静默改目标、延长时间或把部分进度称为完成。每 5000 步保存完整权重和 ZeRO1 状态，六份资产可能接近 576 GB，应检查实际可写空间和共享配额。

本次平台测得 HDD 剩余约 579 GiB，SSD 约 10147 GiB，HDD2 约 90864 GiB。新数据/缓存保存在独立 HDD 准备目录；六次训练权重与恢复状态计划写入独立 SSD 运行目录，给原基线 final save 留出 HDD 空间。文件系统余量不等于用户配额保证，启动时还需确认目标目录可写及实际保存占用。

目前 CPU 检查通过并不代表离线缓存、GPU 检查或新训练已经执行。真实运行状态以平台和校验收据为准。
