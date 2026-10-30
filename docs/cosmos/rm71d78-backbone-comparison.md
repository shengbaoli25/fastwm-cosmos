# RM71D78：V-JEPA ViT-B 与 DINOv3 ViT-H+ 对照

本次增加两个 Git 分支，固定下游模型规模，用冻结视觉骨干生成监督特征。

| 分支 | 默认配置 | 冻结视觉骨干 |
|---|---|---|
| `codex/rm71d78-vjepa-vitb-28l-50k` | `rm71d78_vjepa_vitb_28l_50k.yaml` | V-JEPA 2.1 ViT-B/16，官方蒸馏版 |
| `codex/rm71d78-dinov3-vithplus-28l-50k` | `rm71d78_dinov3_vithplus_28l_50k.yaml` | DINOv3 ViT-H+/16 |

分支默认值保存在 `configs/backbone_variant.json`。两个分支共享模型实现和两份配置，
`scripts/backbones/run.py` 默认选择当前分支配置；显式 `--variant` 可以选择另一版本。
旧训练任务和旧缓存无需修改。旧 DINO Base 的 12 层模型是历史对照，不能与这里的
28 层模型混作“只换骨干”的实验。旧 H+ 分支曾复制预训练块到可训练视觉主干；这里
按本次约定使用独立冻结 H+ 编码器和随机初始化的 28 层预测器。

## 精确参数量

以下统计实际构造器的唯一 `nn.Parameter` 元素数。4 卡 DDP 不会使参数量乘 4。

| 模块 | V-JEPA ViT-B | DINOv3 ViT-H+ |
|---|---:|---:|
| 可训练视觉预测器（含输入输出适配） | 354,908,416 | 355,696,384 |
| Action | 491,060,488 | 491,060,488 |
| Force | 270,040,838 | 270,040,838 |
| Proprio 投影 | 10,240 | 10,240 |
| Action 时间投影 | 3,937,536 | 3,937,536 |
| 任务嵌入 | 1,024 | 1,024 |
| **可训练总量** | **1,119,958,542** | **1,120,746,510** |
| **冻结视觉编码器** | **86,833,152** | **840,633,600** |
| **完整逻辑模型总量** | **1,206,791,694** | **1,961,380,110** |

“小版”指 V-JEPA 的视觉编码器；下游预测器、Action、Force 仍采用统一的 28 层配置。
两组可训练参数仅相差 787,968，来自不同特征宽度对应的输入输出适配。
完整逻辑模型总量包含在线 RGB 推理需要的冻结编码器；正式缓存训练不加载该编码器。
不包含缓存张量、buffer、优化器状态、RGB 解码器或另行加载的文本模型。
DINO 计数包含加载时补齐并屏蔽的 K-bias 参数槽；并非只按权重文件字节数估计。

可复现统计命令（只构建 meta 张量，无 GPU、权重加载或下载）：

```bash
python scripts/backbones/count_parameters.py
```

输出为 [backbone-parameter-counts.json](backbone-parameter-counts.json)，记录配置、
源文件 SHA256、骨干 revision、各模块参数及优化器恰好覆盖所有可训练参数的检查。
`git_head_at_audit` 是审计时的基线提交；具体实现由记录中的文件 SHA256 绑定。
BF16 权重字节数不等于推理峰值显存。这里没有填入任何未经实测的速度、显存或精度结果。

## 固定配置与仍然存在的骨干差异

- 数据：`rm71d_plug_transfer_20261029`，78 episodes / 86,608 帧 / 30 FPS；
  数据清单 SHA256 为 `b24b701239f54ad38d16c819b9ea9f6fad5f9374a683a84a68d2cd7ef27067a8`。
- 划分：seed 42，`torch.randperm(78)`，训练 60、验证 9、测试 9；归一化仅使用训练集保留行。
- 双相机：每路 224×224；观测时刻 `[-4, 0]`，目标时刻 `[4,8,12,16,20,24,28,32]`。
  有效窗口必须完整处于连续片段内，当前数据共有 66,919 个有效窗口。
- 缓存布局：`[5,392,C]`，首槽是独立编码的两帧观测，后四槽只作为未来监督；
  V-JEPA `C=768`，DINO H+ `C=1280`。相机顺序为主相机、腕相机。
  完整窗口缓存的纯 BF16 特征约为187.63 GiB / 312.71 GiB，另需预留权重和训练输出空间。
- V-JEPA 使用视频 tubelet=2，目标特征由独立的完整 10 帧前向产生；观察特征只运行两帧。
  DINO 是图像编码器，逐帧独立前向后平均相邻两帧。这一时间聚合差异属于骨干实验定义，
  不能宣称两者除参数量外计算完全相同。
- 可训练预测器：28 层、隐藏宽度 768、12 heads；Action/Force 均 28 层、隐藏宽度 768，
  FFN 分别 3072/2048，联合注意力 16×128，context_dim=1024。
- 损失：未来特征 L1 + Action FM + Force FM，权重均为 1；未来标签不进入推理。
  不同骨干的特征 loss 无法直接作为跨模型质量排名。
- 训练：4 H200，单卡 batch64，全局256，累积1，BF16，50,000步；AdamW lr1e-4、
  weight decay0.01，5% warmup 后 cosine，每1,000步保存和固定验证，保留5k里程碑。
- 推理：观测历史32步，预测 Action `[32,8]`、Force `[32,6]`，默认16步采样。

## 离线资源和缓存准备

权重必须提前放好，所有入口均不下载模型。官方源码固定在 `third_party` 中，
权重记录及 SHA256 在 `src/fastwam/cosmos/backbone_experiments.py`。

| 骨干 | 本地文件名 | 权重 SHA256 |
|---|---|---|
| V-JEPA ViT-B | `vjepa2_1_vitb_dist_vitG_384.pt` | `848a77c33cc9e6649ed2119c9bea1e2c569bcdab9539ff3e7c02ccc2959ddf4d` |
| DINO H+ | `model.safetensors` | `3e1d4d18b9bfa9f28fad8e9de6a783f1313532d3460efa4cd0b12521d81d1a4d` |

V-JEPA 源码 revision：`204698b45b3712590f06245fbfba32d3be539812`；
DINOv3：`6876159a11b4df116f30f667f8c9888617df0751`。
依赖沿用仓库 `pyproject.toml` / `requirements-cosmos.txt`，含 PyTorch2.7.1、timm1.0.15。
DINO 操作需要在已接受许可的环境中设置 `DINOV3_LICENSE_ACCEPTED=1`。

以 V-JEPA 小版为例，在已有 CPU 或单卡准备环境执行：

```bash
export PYTHONPATH="$PWD/src:$PWD${PYTHONPATH:+:$PYTHONPATH}"
DATASET=/path/to/rm71d_plug_transfer_20261029
WEIGHTS=/path/to/vjepa2_1_vitb_dist_vitG_384.pt
CACHE_ROOT=/path/to/cache/rm71d78-vjepa-vitb-video-v2
SOURCE_SHA=b24b701239f54ad38d16c819b9ea9f6fad5f9374a683a84a68d2cd7ef27067a8
WEIGHT_SHA=848a77c33cc9e6649ed2119c9bea1e2c569bcdab9539ff3e7c02ccc2959ddf4d

python scripts/backbones/prepare_cache.py \
  --dataset "$DATASET" --source-sha "$SOURCE_SHA" \
  --teacher vjepa21 --teacher-source third_party/vjepa21 \
  --weights "$WEIGHTS" --weight-sha "$WEIGHT_SHA" \
  --output "$CACHE_ROOT" --device cuda --frame-batch 1

python scripts/backbones/prepare_cache.py \
  --dataset "$DATASET" --source-sha "$SOURCE_SHA" \
  --teacher vjepa21 --teacher-source third_party/vjepa21 \
  --weights "$WEIGHTS" --weight-sha "$WEIGHT_SHA" \
  --output "$CACHE_ROOT" --assemble
```

DINO 大版使用 `--teacher dinov3_vithplus --teacher-source third_party/dinov3`，
换成表中对应权重及 SHA，使用独立目录 `rm71d78-dinov3-vithplus-video-v2`。
CPU 准备改为 `--device cpu`。中断续跑加 `--resume`，组装时验证完整分片清单，
不会重新构建教师。组装成功后才能提交正式训练。v2 缓存绑定骨干、时间聚合、
特征维度、权重 SHA、数据和划分，旧 DINO Base 缓存无法冒充 H+。

## 检查与正式训练入口

先在准备环境检查（没有模型分配或训练子进程）：

```bash
python scripts/backbones/run.py --dataset "$DATASET" \
  --cache-root "$CACHE_ROOT" --weights "$WEIGHTS" --check-only
```

之后在正式分布式任务中运行：

```bash
python scripts/backbones/run.py --dataset "$DATASET" \
  --cache-root "$CACHE_ROOT" --weights "$WEIGHTS" \
  --run-root /path/to/runs/rm71d78-vjepa-vitb-28l-50k/NEW_JOB_ID
```

运行目录必须全新，平台须设置 `TRAIN_JOB_ID`，并分配4张H200。入口依次执行两步
前反向/保存检查、独立四卡恢复检查和全新50k训练；正式训练不继承检查阶段的更新。
检查记录绑定模型配置、优化器配置和缓存身份，日志分别位于 `logs/preflight.log`、
`logs/restore.log`、`logs/formal.log`。此代码提交本身不创建容器或训练任务。

## 推理计时与质量评估口径

`scripts/backbones/benchmark.py` 提供 batch1/BF16/16步的离线计时入口。
`rgb` 模式加载真实冻结编码器，计入预载观察图像的传输、编码和下游预测；
`cached` 模式只计入观察特征传输和下游预测。两种模式必须分开报告。
模型加载、磁盘视频解码和相机采集不在计时范围内。报告包括P50/P95、Hz、
CUDA峰值显存、GPU型号以及配置/权重/数据/缓存标识。它不代表机械臂闭环控制频率。

```bash
python scripts/backbones/benchmark.py \
  --config /path/to/run/training/config.yaml \
  --checkpoint /path/to/run/training/checkpoints/weights/step_050000.pt \
  --dataset "$DATASET" --source-sha "$SOURCE_SHA" --cache-root "$CACHE_ROOT" \
  --teacher-source third_party/vjepa21 --weights "$WEIGHTS" \
  --mode rgb --warmup 10 --iterations 50 --sample-index 0 \
  --output /path/to/results/vjepa-vitb-rgb.json
```

checkpoint 路径以训练输出为准；DINO 切换对应配置、checkpoint、源码目录和权重。
该命令重复一个固定测试窗口；切换 `--sample-index` 可测其他窗口。

同一GPU、软件环境、测试样本及采样步数下，先预热再计时；后续再覆盖多个episode。
正式训练保留原验证器的每个验证episode固定4个窗口的物理量MAE及特征误差，
输出 `training/eval/step_*.json`（实际目录由训练器配置确定）。
完整episode的Action/Force曲线、MAE/RMSE和闭环成功率仍需训练权重与对应评测后报告，
不能由参数量或合成测试推断。

## 本地验证

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=src python -m pytest -q \
  tests/dense tests/backbones tests/cosmos/test_dense_bf16.py
```

覆盖编码器尺寸与映射、双相机顺序、未来信息隔离、71/78集划分、训练集归一化、
缓存交叉使用拒绝、分片完整性、BF16梯度、参数去重和优化器覆盖。
CPU/meta验证不等于真实H+权重加载、四卡训练或推理速度已完成测试。
本次本地相关测试共76项通过；仅有上游timm与CPU后端提示。
