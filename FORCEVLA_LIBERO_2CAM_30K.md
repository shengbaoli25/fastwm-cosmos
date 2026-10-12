# ForceVLA 擦黑板：原版 FastWAM、LIBERO 两视角、30000 步

此分支落实用户最新更正：使用 **LIBERO 的两视角图像处理**，替代先前要求的 RoboTwin 三视角。只准备代码与配置，本轮没有启动新训练。现有单视角 2176 步作业的冻结源码、配置、数据均未修改。

## 图像到 VAE

公开数据固定为 `qiaojunyu/ForceVLA-real-data`，revision `6070cbef017e95d695fc804e6fb28cc8baabb1cd`，子集 `flexiv_wipe_board_inputForce`。元数据和 episode0 parquet 的内嵌 schema 均只有两路 RGB：

| 原始列 | 新数据列 | 官方拼图位置 |
| --- | --- | --- |
| `observation.image` | `observation.images.cam_main` | 左侧 |
| `observation.wrist_image` | `observation.images.cam_wrist` | 右侧 |

严格复用 `configs/data/libero_2cam.yaml` 的图像流程与官方 `RobotVideoDataset._get`：各路 480×640 RGB → ToTensor → Resize 224×224 → 按主相机、腕部相机顺序横向拼接 → 保持完整 224×448 → 官方最终 resize/crop、Normalize(mean=.5,std=.5) → `[C,T,H,W]`。不压回 224×224，不复制视角，不将相机维当作 batch。

30Hz 原始时间轴不重采样。每个窗口 33 个原始帧，官方 ratio4 抽取索引 `0,4,...,32`，9 个视频帧对应32个动作。microbatch1 的 VAE 输入为 **`[1,3,9,224,448]`**。按 Wan2.2 VAE 的时间4倍、空间16倍编码结构，latent 为 `[1,48,3,14,28]`；VideoDiT patch `[1,2,2]` 得到294个视频tokens。源码/meta形状检查不等于真实权重 VAE 或 GPU 更新验收。

原版 Video30、Action30、MoT、attention mask、Gaussian flow matching 和完整官方 Base 权重严格加载不变。此配置尚未加入 Force WM。原 Video 分支不读 Action，生成视频是观测和文本条件下的预测，不能据此声称是生成动作的后果视频。

只对齐 LIBERO **图像**流程；ForceVLA 的绝对 pose+gripper 7维动作不改成 LIBERO delta 命令，仍使用冻结的 ForceVLA z-score 统计。state 前7维规范化后，官方 merger 补零至 proprio8，兼容完整 LIBERO 发布权重；六维 wrench 原样保存在数据中，供后续 Force30 使用。

## 本次训练配置

入口：`task=forcevla_libero_2cam_30k`。数据和任务完整定义见 `configs/data/forcevla_libero_2cam.yaml` 与 `configs/task/forcevla_libero_2cam_30k.yaml`。

| 项目 | 值 |
| --- | --- |
| 模型 | 原版 FastWAM Base：Video30 + Action30 + 原 MoT/mask |
| 初始化 | 完整 `libero_uncond_2cam224.pt`，step21700；所有 MoT/proprio tensor strict load |
| 权重 SHA256 | `1000437cfcf55c000094f79a2600634c502bcb5b492476b94bf8509883a49579` |
| 数据 | wipe 全部50集、34808帧；无训练/验证/测试划分 |
| 图像 | 两路224×224，横拼224×448，9帧/窗口 |
| 动作 / proprio | 32×7 / 32×8；最后一维 proprio 为 padding |
| max_steps | **30000 次 optimizer 更新**；覆盖 num_epochs=10 的步数推算 |
| microbatch / 累积 / 有效batch | 1 / 16 / 16（单 GPU） |
| AdamW | lr=1e-4，betas=(0.9,0.95)，weight_decay=.01 |
| 调度 | 官方5% warmup=1500步，之后 cosine |
| 精度 / 分布式 | BF16，官方 ZeRO1，无 offload |
| seed | 42 |
| 保存 / 日志 | 每500步完整 weights+optimizer/scheduler/RNG/dataloader state；每10步日志 |
| 在线验证 / resume | eval_every=0 / null，新鲜初始化 |
| 评测 | 训练后固定0/12/24/36/49，119个完整分块，16步推理；7轴 GT/pred/hold 与两视角视频 |

保存间隔从 LIBERO 的2000改为500，使用其原有保存机制，便于此前5小时资源上限内保留进度。完整状态文件较大，正式启动前须核实磁盘预算。本分支保留已审查的 oversized-flat Adam 分组兼容修复，无新增训练算法。

Hydra `max_grad_norm=1` 保留原配置；官方 DeepSpeed JSON 的实际 effective clipping=0，不能把该字段报告为本次已经启用 clipping。

30000×16 是最多480000次名义样本抽取，约13.79轮 frame anchor 暴露；窗口相互重叠、epoch末批和 reader重试均使它不能被称为覆盖证明或过拟合成功。此前单视角实测速率约5.1秒/更新，30000步仅训练粗估约42.5小时；新的两视角速度和显存尚未实测。**不能声称能在此前每分支5小时总上限内完成，也不能静默将30000降回2176。** 此次只是配置准备，没有延长资源授权或启动 GPU。

## 数据准备与配置导出

沿用已下载且哈希校验的原始完整数据，重新转换到独立两视角目录；旧单视角数据不得覆盖。转换器保留两路编码字节、时间戳、所有帧及原始数值字段；按原列拆出 state7/wrench6。生产转换强制固定下载 manifest、50集和34808帧，不接受缺腕部相机。

```bash
python scripts/forcevla/prepare_forcevla.py --profile libero2cam \
  --source-root /path/to/data_lerobot/flexiv_wipe_board_inputForce \
  --output-root ./data/forcevla_fastwam_libero_2cam/wipe \
  --manifest /path/to/forcevla_wipe_download_manifest.json
```

导出完整 resolved YAML 与 JSON计划，**该命令不会开始训练**：

```bash
python scripts/forcevla/prepare_training.py \
  --shared-repo /path/to/read-only/realman-fastwam \
  --dataset /path/to/new/forcevla_fastwam_libero_2cam/wipe \
  --pretrained-checkpoint /path/to/libero_uncond_2cam224.pt \
  --run-output /path/to/new/runs/forcevla_libero_2cam_30k \
  --output-plan /path/to/new/plans/forcevla_libero_2cam_30k.json \
  --check-data
```

`--check-data` 验证两视角转换身份、文件哈希、完整50集和冻结统计。仅导出配置时可省略此参数，计划会明确标记 `data_check:not_checked`，不能视为训练准备全部通过。

正式启动前必须在独立新 H200 上，用新增 `preflight_gpu.py --task forcevla_libero_2cam_30k --target-steps 30000` 与相同绝对路径覆盖完成连续两次真实 optimizer 更新，再评估显存、吞吐和资源预算。单视角2176 probe不能代替两视角30000 probe。正式训练命令由计划的 `train_argv/train_shell` 给出；只有模型权重初始化用 Base checkpoint，不能偷偷恢复旧2176 optimizer或旧30/28/14分支。

`scripts/baseline/evaluate.py` 从新运行的配置读取图像大小，可复用。旧外部 evidence collector/renderer 的 shape guard 写死224×224，**必须升级到明确绑定224×448的新协议**，不能用旧证据检查脚本给两视角结果盖章。本文及配置导出不声称新数据已在平台完成转换、真实VAE已执行或新训练已启动。
