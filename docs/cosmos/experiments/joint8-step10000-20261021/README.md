# Joint8 step10000：推理优化、采样评测与设备对照

实验日期：2026-10-21。代码和结果归档日期：2026-10-23。

- [完整中文报告](评测报告.md) · [PDF](joint8-step10000_三专家采样与GB10-4090评测报告.pdf)
- [离线演示页](演示与结果.html)：下载后在浏览器打开，视频和图片保持原目录关系。
- [机器可读设备对比](results/device-comparison.json) · [27 组汇总](results/gb10-grid/summary.csv)
- [事件窗口汇总](results/gb10-events/summary.csv) · [跨设备输出差异](provenance/device-parity.json)

## 结果与适用范围

V/F/A 指视频、力、动作专家的 Euler 积分步数。两台设备使用同一训练检查点、输入和评测源码，batch=1、BF16；基线为 16/16/16。

| V/F/A | GB10 RGB→输出 P50/P95 | RTX 4090 RGB→输出 P50/P95 | 选择理由 |
|---|---:|---:|---|
| 16/16/16 | 1978/2011 ms | 756/765 ms | 完整步数基线 |
| 4/4/16 | 763/779 ms | 344/351 ms | 首个真机候选，保留动作 16 步 |
| 8/8/16 | 1090/1146 ms | 482/487 ms | 轨迹差分误差更低的对照 |
| 4/4/4 | 627/638 ms | 289/294 ms | 最快，但事件窗口动作速度差分误差比基线高约 23% |

共 6,669 次动作/力质量预测、450 次视频质量评分；每个设备另有 330 组核心和 RGB 计时记录。全部 50 个 episode 参与过训练，因此只能用于离线回放诊断。网络探测和模型计时分别测量，二者 P95 相加只是粗略预算。

## 代码入口

路径均相对仓库根目录：

| 文件 | 用途 |
|---|---|
| `src/fastwam/cosmos/noaction_inference.py` | 等步数联合推理；缓存固定条件、编译 Force/Action 块、捕获 CUDA Graph |
| `src/fastwam/cosmos/multirate_inference.py` | 较少 Video 步数、相同 Force/Action 步数的前期实验 |
| `src/fastwam/cosmos/expert_schedule_inference.py` | 三专家独立网格、固定状态 K/V 复用和完整重算参考实现 |
| `src/fastwam/cosmos/offline_metrics.py` | 带有效目标掩码的动作、力和差分误差 |
| `scripts/cosmos/evaluate_joint8_experts.py` | 输入准备、27 配置和事件窗口评测 |
| `scripts/cosmos/benchmark_joint8_experts.py` | 两设备同协议的核心/RGB 稳态时延 |
| `scripts/cosmos/summarize_expert_sweep.py` | 按 episode 配对 bootstrap、误差汇总和图表 |
| `scripts/cosmos/evaluate_joint8_video_quality.py` | 未来帧 PSNR/SSIM |
| `scripts/cosmos/render_joint8_expert_demos.py` | 离线视频和 GT 曲线 |
| `scripts/cosmos/evaluate_joint8_steps.py` | step1000 的等步数历史评测及共用输入/指标函数 |
| `scripts/cosmos/evaluate_joint8_multirate.py` | step1000 的 Video=4、Force=Action=8 历史评测 |
| `scripts/cosmos/reporting/` | 以实验目录为参数重建设备汇总和 PDF |

固定状态缓存只在专家状态和时钟不变时复用特征；Video 更新后会刷新下游 Force 特征。Force 实际前向次数可能高于积分步数，见 `report.json` 中的 `runner_stats`。本实现没有使用跨变化状态的近似方向外推。

部署适配器新增 `inference_backend: "joint_optimized"`，用于等步数优化推理。独立 V/F/A 网格通过离线评测入口或 `ExpertScheduleRunner` 调用；本提交没有把 4/4/16 配置接到机器人执行接口。

## 环境和外部资产

```bash
git clone --branch codex/realman-cup-joint-30k --recurse-submodules \
  https://github.com/shengbaoli25/fastwm-cosmos.git
cd fastwm-cosmos
git submodule update --init --recursive
```

Cosmos 子模块固定于 `a2c298b0a3df3778b973fe65e9e58877b292d8a7`。实测使用 Python 3.12、Torch 2.9.1、torchvision 0.24.1；GB10 为 CUDA 13.0 / ARM64，4090 为 CUDA 12.8 / x86_64。推理测试环境独立于根目录 `pyproject.toml` 的历史训练环境，勿将 Torch 2.7.1 的训练依赖覆盖到已准备的评测环境。

4090 环境示例：

```bash
python3.12 -m venv .venv-inference
source .venv-inference/bin/activate
python -m pip install torch==2.9.1 torchvision==0.24.1 \
  --index-url https://download.pytorch.org/whl/cu128
python -m pip install numpy==2.2.6 einops==0.8.1 omegaconf==2.3.0 \
  hydra-core==1.3.2 safetensors==0.5.3 pillow==12.0.0 pyarrow==24.0.0 \
  av==15.1.0 attrs loguru termcolor packaging tqdm rich imageio imageio-ffmpeg \
  ftfy iopath transformers==4.49.0 accelerate==1.12.0 scipy==1.16.3 \
  matplotlib==3.10.9 pytest==9.1.1
```

完整推理另需训练权重 `step_010000.pt`、官方 Video/VAE 基础权重、与训练一致的 normalization/text/features 资产包和双相机原始录制数据；也可使用已经准备好的观测张量。实验数据和权重保存在本机资产目录，不随 Git 提交。

- 检查点 SHA256：`2e8409d495d11961614b72d81b25bec7cc71bc2e194acb15ea220a72edace1d8`。
- 冻结输入 SHA256：`2ef2f3eb86783562a7a47385785285285e090b9c700afc98eb0b6f5613f85d17`。
- 文件身份和选窗协议见 `assets/*.json`；该目录只归档元数据。
- 从原训练 runtime 配置生成本机配置，将 `model_kwargs` 中 upstream/checkpoint/text_cache 及顶层 tokenizer/normalization/checkpoint_directory 路径指向本机实际资产，保留身份校验字段。
- 评测脚本绑定上述 step10000 检查点哈希；其他检查点应建立独立协议和输出目录。

## 复现命令

从仓库根目录执行，先修改以下路径。`EXPERIMENT_ROOT` 应为可写实验目录；归档结果另保存在 `docs/`。

```bash
export EXPERIMENT_ROOT=/data/joint8-step10000-replay
export TRAINING_BUNDLE=/data/realman-training-assets
export RECORDED_DATASET=/data/realman-recordings
mkdir -p "$EXPERIMENT_ROOT/assets"

python scripts/cosmos/evaluate_joint8_experts.py \
  --checkpoint "$EXPERIMENT_ROOT/assets/weights/step_010000.pt" \
  --runtime "$EXPERIMENT_ROOT/assets/runtime.local.json" \
  --bundle "$TRAINING_BUNDLE" --dataset "$RECORDED_DATASET" \
  --prepared "$EXPERIMENT_ROOT/assets/prepared-observations.pt" \
  --output "$EXPERIMENT_ROOT/results/prepare" --stage prepare

python scripts/cosmos/evaluate_joint8_experts.py \
  --checkpoint "$EXPERIMENT_ROOT/assets/weights/step_010000.pt" \
  --runtime "$EXPERIMENT_ROOT/assets/runtime.local.json" \
  --prepared "$EXPERIMENT_ROOT/assets/prepared-observations.pt" \
  --output "$EXPERIMENT_ROOT/results/gb10-grid" --stage sweep

python scripts/cosmos/evaluate_joint8_experts.py \
  --checkpoint "$EXPERIMENT_ROOT/assets/weights/step_010000.pt" \
  --runtime "$EXPERIMENT_ROOT/assets/runtime.local.json" \
  --prepared "$EXPERIMENT_ROOT/assets/prepared-observations.pt" \
  --output "$EXPERIMENT_ROOT/results/gb10-events" --stage events \
  --modes v16_f16_a16 v4_f4_a4 v4_f4_a8 v4_f4_a16 v4_f8_a8 \
          v4_f8_a16 v4_f16_a16 v8_f8_a8 v8_f8_a16

python scripts/cosmos/summarize_expert_sweep.py "$EXPERIMENT_ROOT/results/gb10-grid"
python scripts/cosmos/summarize_expert_sweep.py "$EXPERIMENT_ROOT/results/gb10-events"

# 在 GB10 和 4090 分别执行；使用相同 prepared-observations.pt 及其 identity 文件。
python scripts/cosmos/benchmark_joint8_experts.py \
  --checkpoint "$EXPERIMENT_ROOT/assets/weights/step_010000.pt" \
  --runtime "$EXPERIMENT_ROOT/assets/runtime.local.json" \
  --prepared "$EXPERIMENT_ROOT/assets/prepared-observations.pt" \
  --output "$EXPERIMENT_ROOT/results/device-latency" --repeats 30 \
  --modes v16_f16_a16 v4_f4_a4 v4_f4_a8 v4_f4_a16 v4_f8_a16 \
          v4_f16_a16 v8_f8_a8 v8_f8_a16
```

其余脚本参数见 `--help`。稳态计时排除加载、编译和图捕获；这些准备开销单列。RGB 路径包含 resize、VAE 和反归一化，排除相机采集、网络和机器人执行。

## 回归验证

```bash
PYTHONPATH=src:deploy python -m pytest -q \
  tests/cosmos/test_noaction.py tests/cosmos/test_noaction_inference.py \
  tests/cosmos/test_multirate_inference.py tests/cosmos/test_expert_schedule_inference.py \
  tests/cosmos/test_offline_metrics.py \
  deploy/tests/test_model_adapter.py deploy/tests/test_joint8_steps.py
```

这些测试使用小模型验证完整重算与缓存路径一致、请求之间缓存刷新、返回结果不被后续请求覆盖、独立步数网格、掩码指标及部署协议。CUDA 可用时包含 CUDA Graph 检查。全尺寸检查点上的逐配置参考误差和实际前向计数另保存在实验 `report.json`。

## 归档边界和报告重建

归档保留 CSV/JSON 汇总、两设备全部计时记录、每配置 3 个配对输出样本、选窗协议、图和 3 个演示视频。`provenance/published-artifacts.sha256.json` 记录这些文件的原始哈希。四个关键源码的 SHA 位于两设备 `latency.json`，与本次提交内容一致。

完整质量预测 NPZ、质量 `records.jsonl`、准备观测张量、基础权重和训练检查点保留在原实验目录；从头生成质量汇总需这些输出或重新运行评测。文档中原工作站路径属于历史执行记录。

```bash
export ARCHIVE=docs/cosmos/experiments/joint8-step10000-20261021
python scripts/cosmos/reporting/update_device_comparison.py "$ARCHIVE"
# PDF 排版依赖：weasyprint、beautifulsoup4、markdown-it-py，以及系统中文字体/Pango。
python scripts/cosmos/reporting/build_pdf.py "$ARCHIVE"
```

GitHub 网页只预览 Markdown/PDF；本地 HTML 演示页需同时下载 `demos/` 与 `results/` 图片。
