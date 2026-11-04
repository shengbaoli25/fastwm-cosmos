# FastWAM：RM71D78 joint8 基线

该适配复用仓内 `FastWAM`、`ActionDiT`、`MoT`、`RobotVideoDataset` 与 `Wan22Trainer`。
官方代码来源固定为 [FastWAM 7faa711](https://github.com/yuantianyuan01/FastWAM/tree/7faa71108368fbb3b6885649f112af607427a2d4)。
本次核对该提交的 `action_dit.py`、`mot.py` 与仓内实现（统一换行后）一致；仓内 `fastwam.py`
已有额外 VAE 缓存输入支持，本配置走原始 RGB 路径，不使用该扩展。

训练保留官方视频与动作联合 flow-matching 损失、专家注意力以及 VAE；部署调用官方
`infer_action`，直接从当前图像预测动作，不执行未来视频生成。FastWAM 的官方基线没有力输入；
数据集保留的 `observation.force` 不拼接到 state，也不增加 force head。

## 数据与初始化

公共转换器输出 canonical LeRobot v2.1。全部 78 条原始示范参与训练，按连续有效段拆分为
LeRobot episodes；数据身份和文件校验由 `manifest.json` 绑定。全部数据的回放只能称 training-fit，
不表示留出集性能。

- state/action 都是 8 维：7 个绝对关节角（rad）和夹爪 opening `[0,1]`。
- 动作 horizon 32，控制频率 30 Hz；训练视频取 `0,4,...,32` 共 9 帧。
- 每帧按 global、wrist 顺序，各自 RGB resize 为 224×224，横拼为 224×448，再缩放到 `[-1,1]`。
- state/action 使用当前 canonical 训练数据统计的 z-score，沿用官方归一化数值 `[-5,5]` 截断规则。
- 单一任务文本为“拔下插头并插入另一插座”，套用官方 prompt 模板并读取 128 token T5 缓存。

初始化来源为 [官方 LIBERO 完整 policy](https://huggingface.co/yuanty/fastwam/blob/main/libero_uncond_2cam224.pt)，
SHA256 `1000437cfcf55c000094f79a2600634c502bcb5b492476b94bf8509883a49579`。
其 action 是 7D，因此不能 strict-load 到 joint8，也不能补零后当作完整 8D policy：

1. 保留所有 video/action Transformer 主干和其他兼容 MoT 参数。
2. 重建 `mixtures.action.action_encoder.{weight,bias}` 与 `mixtures.action.head.{weight,bias}`。
3. 重建完整 `proprio_encoder`。虽然两者输入都是 8 维，LIBERO 的 EEF/gripper 语义与关节空间不同。
4. 这些新层采用固定 seed 的原始 PyTorch 初始化；没有复制假定对应的关节行/列。
5. 除上述明确白名单，任何缺失键、多余键或 shape 不匹配都会失败。joint8 训练产物则必须全量严格加载。

## 环境与显式命令

训练沿用仓库 `pyproject.toml` 中与官方固定提交一致的依赖版本，使用 Linux CUDA 环境；核心版本 Torch 2.7.1+cu128、
torchvision 0.22.1+cu128、Accelerate 1.12.0、DeepSpeed 0.18.7。现有镜像只提供 Python、CUDA 驱动和
系统工具；bootstrap 创建不继承系统 site-packages 的独立 venv，不修改镜像的 Python 环境。
使用 Python 3.10–3.12、CUDA 12.8 兼容驱动，准备足够 RAM、共享内存（容器可用 `--ipc=host`）
和磁盘保存四卡输入、完整模型、Adam 状态及 checkpoint。DeepSpeed 可能需要宿主 C++ 编译器。

```bash
bash scripts/ablation/bootstrap_fastwam.sh /work/envs/fastwam python3 /work/upstream/FastWAM
```

脚本先安装官方 cu128 Torch wheels，再以 `DS_BUILD_OPS=0`、`--no-build-isolation` 安装固定项目依赖，
使 DeepSpeed 构建元数据能读取已安装的 Torch。额外固定 `sentencepiece=0.2.0`、`ftfy=6.3.1` 支持原始
T5 tokenizer/文本清理。脚本检出官方 `7faa711`，核对本仓 ActionDiT/MoT 与官方文件逐字节一致（统一换行），
执行 `pip check` 与模型/trainer 导入，并保存环境版本、源码 hash 和 `pip freeze`。不下载模型或启动训练。
后续 `python` 均指 `/work/envs/fastwam/bin/python`。

网络受限时可设置 `PIP_INDEX_URL=https://mirrors.aliyun.com/pypi/simple`。bootstrap 先用标准库
预缓存器下载固定官方 Torch/torchvision CUDA12.8 wheel，按官方索引 SHA256 和文件大小验证，随后
从本地 wheel 安装；NVIDIA/CUDA 传递依赖和项目依赖均使用 `PIP_INDEX_URL`，不会被 PyTorch
索引覆盖。此实现没有 uv 或 FlashAttention 安装步骤，注意力沿用原版 PyTorch SDPA。
`FASTWAM_WHEELHOUSE` 可指定共享预缓存目录；`FASTWAM_WHEEL_REUSE_ROOT` 可指定已有 wheel/pip
HTTP `.body` 缓存根目录，只有完全匹配的字节才会复制。也可设置 `FASTWAM_TORCH_WHEEL_BASE_URL`
到包含相同文件名的更快 HTTPS 镜像；SHA 不匹配立即拒绝。CPython3.10/3.11/3.12 Linux x86_64
的六个固定 SHA 与大小见 `scripts/ablation/prefetch_fastwam_wheels.py`，不替换 CUDA 或 Torch 版本。
下载每 15 秒报告真实字节数和速率，支持 `.part` 续传；默认总预算 600 秒，连续 90 秒低于
128KiB/s 切换来源，失败保留 partial 供后续重试。可用 `FASTWAM_WHEEL_TIMEOUT` 修改预算。
各 pip 大阶段默认总预算 900 秒（`FASTWAM_PIP_TIMEOUT`），另有每连接 30 秒、最多 2 次重试；
每 30 秒心跳明确标识为运行状态，不冒充下载进度。超时终止本阶段并返回非零，不继续训练。

当前训练配置为 4 张 H200，每卡真实 microbatch 32、累积 1，全局 batch 128。
入口保留显式配置 batch64/global256 的兼容性，只允许这两个 batch 档位；不会自动降低 batch。
入口选择现有 ZeRO1 启动配置，未指定 `--num-processes` 时采用配置中的 4 个进程。
训练前检查实际可见 GPU 为四张 H200；Accelerate 准备完成后，读取 DeepSpeed engine 的实际
microbatch/global batch/累积值并逐项核对当前配置（默认 32/128/1）。记录分别保存为 `training-hardware.json`、
`training-execution.json`；后者还记录恢复起始 step 与本次循环目标。

Wan2.2 视频专家约 5B，加上动作专家与训练状态，全参数训练应准备大显存 CUDA GPU；
目标平台为 4×H200。2026-11-04 的真实 batch64 两步 smoke 在 rank0 遇到 CUDA OOM：
当时进程占用约 139.79 GiB、GPU 仅剩 12 MiB，额外申请 222 MiB 失败；没有通过两步保存、
第三步恢复和 HTTP 验收。用户授权后将新实验默认改为 batch32；这尚不证明其峰值显存、吞吐或
完整训练可用，必须重新完成同 batch32 的全部 smoke 验收。该改动不修改模型、损失、VAE、
CUDA graph 或 optimizer，也不以梯度累积恢复到原全局 batch256。
最大步数默认 30000、每 5000 步保存一次。现有 trainer 在
`max_steps` 非空时采用按 optimizer step 计数的 while 循环，数据耗尽后进入下一 epoch；
因此 `num_epochs=1000` 不会提前截断这 30000 步，仅在移除 `max_steps` 后用于估算训练预算。

所有入口默认不训练：

```bash
python scripts/ablation/fastwam.py
python scripts/ablation/fastwam.py plan
```

先由公共转换器生成数据，再转换统计（统计产物必须放在数据目录之外，以保持数据 inventory 不变）：

```bash
python scripts/ablation/fastwam.py stats \
  --dataset-root /data/rm71d78-canonical \
  --output /data/ablation-stats/fastwam.json
```

手动准备官方完整 policy、VAE，以及 T5 文本缓存，不会因 plan/stats 或 server 启动而下载。
也可显式运行下列准备命令；固定 revision、文件大小和 SHA/Git blob 来自
[fastwam-assets.lock.json](../../configs/ablation/fastwam-assets.lock.json)。VAE/T5 使用官方代码指定的
ModelScope DiffSynth-Studio 来源，policy/tokenizer 使用其官方 Hugging Face 来源。
复用目录只检查已知文件名并要求字节校验通过，不会复用其他任务的文本 embedding。

```bash
python scripts/ablation/prepare_fastwam_assets.py --asset-root /work/weights/fastwam \
  --reuse-root /path/to/existing/fastwam-checkpoints
```

生成 `fastwam-assets-receipt.json`，并将初始化配置指向
`/work/weights/fastwam/yuanty/fastwam/libero_uncond_2cam224.pt`。
`asset_root` 中的 VAE 固定相对路径为
`DiffSynth-Studio/Wan-Series-Converted-Safetensors/Wan2.2_VAE.safetensors`。
若要用本入口生成文本缓存，还需手动放置官方转换后的
`models_t5_umt5-xxl-enc-bf16.safetensors` 在同一模型目录，以及 tokenizer
`Wan-AI/Wan2.1-T2V-1.3B/google/umt5-xxl/`。`text-cache` 显式调用仓内原始预计算函数，
仅编码文本，不进行训练；服务端仅需要缓存和 VAE。

为避免很长的命令，可以复制 `configs/ablation/fastwam.yaml` 到工作目录，填入五个 `???` 路径。
配置的其余部分保持原样：

```bash
python scripts/ablation/fastwam.py text-cache --config /work/fastwam-rm71d78.yaml
bash scripts/ablation/launch_fastwam_h200.sh /work/envs/fastwam/bin/python /work/fastwam-rm71d78.yaml check-hardware
bash scripts/ablation/launch_fastwam_h200.sh /work/envs/fastwam/bin/python /work/fastwam-rm71d78.yaml train \
  --output /work/fastwam-run
bash scripts/ablation/launch_fastwam_h200.sh /work/envs/fastwam/bin/python /work/fastwam-rm71d78.yaml resume \
  --output /work/fastwam-run \
  --resume /work/fastwam-run/checkpoints/state/step_005000
```

正式 30k 前，用独立目录验证相同四卡/每卡 32 的两步保存、第三步恢复；训练目标从开始就设为 3，
仅对第一次调用显式提前停止。下面的命令可直接使用填好路径的正式配置：

```bash
bash scripts/ablation/launch_fastwam_h200.sh /work/envs/fastwam/bin/python /work/fastwam-rm71d78.yaml train \
  --output /work/fastwam-smoke --set max_steps=3 --set save_every=1 --set log_every=1 --stop-after-steps 2 --smoke-audit
bash scripts/ablation/launch_fastwam_h200.sh /work/envs/fastwam/bin/python /work/fastwam-rm71d78.yaml resume \
  --output /work/fastwam-smoke --set max_steps=3 --set save_every=1 --set log_every=1 \
  --resume /work/fastwam-smoke/checkpoints/state/step_000002 --smoke-audit
```

`--stop-after-steps` 只在 trainer 已构造 scheduler/恢复状态后设置当前训练循环的停止点；配置、
scheduler 总长度和 checkpoint recipe 仍为原目标 3。第一次由原 trainer 完整保存 step_000002，
第二次恢复 optimizer/scheduler/RNG/读取位置后完成 step_000003。核验真实有限 loss/梯度、参数变化、
完整 state 文件和恢复起始 step=2，再启动独立正式目录。新的 batch32 GPU 命令尚未实际执行。

`--smoke-audit` 只允许独立 3 步配置上的 `0→2` 与 `2→3`。它观察真正的
`Accelerator.backward`：Accelerate 1.12 在该调用内执行 DeepSpeed optimizer update，随后读取
真实 loss、全局梯度范数、Adam counter 和 FP32 master/m/v 分片样本。非有限/零梯度、跳过更新、
没有参数变化均直接失败。原模型、损失和 optimizer 不修改。四个 rank 分别保存
`smoke-audit/from_000000-to_000002.rank_*.json`、`from_000002-to_000003.rank_*.json`，
每份包含实际 CUDA allocated/reserved/peak、训练耗时和参数指纹。
每个原始完整 state checkpoint 内附加 `smoke-audit-rank-*.json`；重新构造模型并真实
`load_state` 后，严格比较 scheduler、Adam counter、trainer/data位置、Python/NumPy/Torch/CUDA
RNG，以及参数/master/m/v 样本。样本比较不能声称全张量逐位相同，但能拒绝只改 step 的假恢复。

平台准备脚本生成独立的 `fastwam-smoke.yaml` 与 `fastwam-30k.yaml`（都保持 `resume:null`）。
在各阶段后运行下列验收，只有 `resumed3` 成功才启动正式训练：

```bash
python scripts/ablation/verify_fastwam_smoke.py --smoke-config /work/fastwam-smoke.yaml \
  --formal-config /work/fastwam-30k.yaml --phase saved2
python scripts/ablation/verify_fastwam_smoke.py --smoke-config /work/fastwam-smoke.yaml \
  --formal-config /work/fastwam-30k.yaml --phase resumed3
```

验收器检查全部四个 rank，比较保存/恢复前后真实状态，实际反序列化训练权重归档并核对保存的
policy 样本字节；它不以 exit0 或文件存在作为通过条件。最终写入
`smoke-audit/resumed3-verified.json`。平台包装器还用 `nvidia-smi` 每秒记录完整准备/训练过程显存。
验收按当前配置核对 hardware 和每个 rank 的实际 microbatch/global batch/累积值；batch32 与
batch64 的保存或恢复证据不能交叉使用。batch64 的真实运行因 OOM 失败；batch32 尚待真实 GPU 验收。

`--set key=value` 可覆盖训练预算等字段；当前启动器固定要求 4×H200、每卡 32 或 64、累积 1，默认每卡 32。完整 resume 使用 Accelerate 保存的 optimizer、scheduler、
RNG 与数据读取进度；传入单独权重文件不被视为 resume。首次运行可配置步数、batch、seed 等；
随后完整 resume 必须保持原训练语义，包括 GPU 数、batch、累积、seed、精度、优化器参数和步数。
修改这些参数应建立独立实验，而不是声称完全续训。代码在进入训练前核验数据文件、统计、
官方权重 SHA256、VAE 和文本缓存，并防止把不同身份混入同一个输出目录。

## 部署产物与接口

训练目录保留 `config.yaml`、`fastwam_binding.json`、`normalization.json`、
`checkpoints/weights/step_*.pt` 以及完整训练 state。`normalization.json` 是原统计的字节一致副本；
官方 dataset 另外写出的 `dataset_stats.json` 会重新格式化，不用它替换已绑定的 normalization 文件。
每个 weights checkpoint 内嵌相同绑定信息，包括数据身份、统计/VAE/T5 哈希、官方 pin 与初始化策略。
绑定还包含解析后的 model/data/training 语义 recipe 与哈希；部署从实际配置重新计算并核对，
即使模型权重 shape 相同，改动采样 scheduler、预处理、seed 或训练配方也不能静默通过。

公共 server 通过 `fastwam.ablation.fastwam.load_predictor(config)` 加载，配置必需字段为：

```json
{
  "checkpoint_path": "/deploy/step_030000.pt",
  "training_config_path": "/deploy/config.yaml",
  "binding_path": "/deploy/fastwam_binding.json",
  "normalization_path": "/deploy/normalization.json",
  "asset_root": "/models",
  "text_embedding_cache_dir": "/deploy/text-cache",
  "device": "cuda:0",
  "num_inference_steps": 16,
  "seed": 42
}
```

服务端不需要训练数据目录。输入为 `state[8]`、`images.global/wrist uint8[480,640,3]` 和任务
`prompt`，公共接口携带的 `force[6]` 在此基线不使用。返回反归一化的 `float32[32,8]`，
夹爪按物理范围截断 `[0,1]`，关节输出不做隐式限位。协议、身份、有限值与执行安全检查仍由公共
server/client 处理。推理不导入 trainer，也不会初始化 distributed；默认不启用编译。

目前已在 Windows Torch 2.7.1 环境导入官方 FastWAM 核心，并用微型嵌入 PNG 的 LeRobot v2.1
数据实际实例化原版 RobotVideoDataset，验证双相机布局、32步动作、episode 边界 padding、
部署与原版归一化逐位一致。另有配置、8D head 迁移白名单和严格 checkpoint 身份测试。
官方资产已用于 batch64 的真实 GPU 尝试，但该次 smoke 因 OOM 未完成，不能报告成功训练、
checkpoint 恢复、HTTP 推理或机器人闭环。batch32 的完整 GPU 验收仍待执行。
