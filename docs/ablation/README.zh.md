# RM71D78 ForceVLA 独立消融分支

分支：`codex/ablation-rm71d78-forcevla`。本分支从原始提交
`057f78767f52f358b229f4a05b097c0f1455fa3d` 独立建立，新增入口只支持 ForceVLA。
提供固定官方模型、同源数据准备、训练/恢复入口、checkpoint 身份绑定，以及兼容现有 RealMan
客户端的 HTTP server。继承的原训练入口和模型行为保持不变。

初始本地交付的验证记录见 [validation.json](validation.json)，覆盖代码、固定源码适配、
合成 78 集转换、Hugging Face 图像读取和注入测试预测器的 HTTP 调用。
2026-11-03 的真实 GPU smoke 已完成训练保存与恢复，但首次 HTTP 推理因官方 LIMoE 的
batch1 分组整除约束失败，正式训练被 gate 阻止。本次仅修复服务采样边界：内部重复为 batch4，
保留首项返回，对外仍是单请求 joint8；训练方法、路由和 checkpoint 均不变。
修复后必须重新取得真实 checkpoint 的 HTTP 验证及显存、延迟结果，CPU 测试不代替此 gate。
本次修复的本地记录见 [batch4 验证](validation-batch4-20261103.json)。

完整模型说明、固定版本环境安装和配置参数见 [ForceVLA 指南](forcevla.md)。
源码与初始化来源见 [upstream-lock.json](../../configs/ablation/upstream-lock.json)。

## 模型与数据口径

使用官方 pi0 + 六轴力 projection/LIMoE，保留 flow-matching loss、官方优化器和 Orbax 保存恢复。
当前指定 LoRA：4 张 H200、每卡真实 batch 64、global batch 256、30,000 次更新、seed 42，
每 2,000 步保存，保留每 10,000 步的权重；
无 EMA。`training_method=full` 选择全参数方法。训练与推理使用独立的 Linux CUDA JAX 环境，
机器人客户端可以继续运行在 Windows。单个 JAX 进程使用 `fsdp_devices=4`，mesh 为 `(1,4)`；
官方数据同时沿 batch/fsdp 两轴分片，因此每卡实际处理 256/4=64 个样本，不用梯度累积代替。
启动时要求恰好 4 张 H200，显存不足不会自动降低 batch；真实显存峰值须实际测量。
完整 recipe 及官方源码兼容修复见模型指南。

数据源固定为 `rm71d_plug_transfer_20261029`，任务“拔下插头并插入另一插座”，包含 78 条来源轨迹。
原数据为 LeRobot v3；实测状态的 `[0..6,19]` 映射为 7 个关节角（弧度）和夹爪开度；
动作是实际下发的绝对 `action[8]`，夹爪范围 `[0,1]`，频率 30Hz。

转换保留 `global/wrist` 两相机，把解码后的 RGB 无损保存为 PNG，不做 resize；
写入 Hugging Face Image metadata 供官方 LeRobot v2.1 读取。
力使用 `observation.force_filtered`，审核每集标定、滤波和时间身份，不重复滤波。
断点、未执行动作、过期/无效传感器样本、力滤波 epoch 变化都不跨训练窗口。
每个连续有效段单独成为输出 episode，manifest 保留全部 78 条来源与分段映射。
所有 78 条用于训练，离线回放只报告 **training-fit**，不代表留出集泛化或机器人成功率。

## 准备共享格式数据

数据工具需要 Python、NumPy、PyArrow、PyAV 和 Pillow，不加载模型。转换目标必须不存在且位于
原数据目录之外；统计、报告和运行产物也放在数据目录之外。

```bash
python scripts/ablation/prepare_data.py \
  --source /data/rm71d_plug_transfer_20261029 \
  --report /work/reports/rm71d78-audit.json

python scripts/ablation/prepare_data.py \
  --source /data/rm71d_plug_transfer_20261029 \
  --output /work/data/rm71d78_lerobot21 \
  --report /work/reports/rm71d78-export.json
```

冻结源文件 hash、78 集、数值/时序/力身份及全部相机解码均须通过；版本不符会停止。
转换失败保留显式 `.incomplete-*` 诊断目录，不产生可供训练的正式输出。
PNG 导出比原压缩视频占用更多磁盘空间。

## 计划、训练与恢复

先按 [安装说明](forcevla.md#separate-linux-environment) 准备固定环境、官方 pi0_base 本地权重和
[配置](../../configs/ablation/forcevla.example.json)。填入真实路径、导出 manifest 身份及权重 hash。
默认入口只显示计划；以下 stats 操作也不启动训练：

```bash
python scripts/ablation/forcevla.py --config /work/forcevla-run.json
python scripts/ablation/forcevla.py --config /work/forcevla-run.json --operation stats
# 将生成的 normalization_sha256 填回配置。以下两条才明确启动/恢复训练。
python scripts/ablation/forcevla.py --config /work/forcevla-run.json --operation train
python scripts/ablation/forcevla.py --config /work/forcevla-run.json --operation resume
```

指定资源的 Linux 环境准备与启动入口如下；bootstrap 不下载权重或训练，launcher 明确开始训练：

```bash
bash scripts/ablation/bootstrap_forcevla.sh /work/upstream/ForceVLA /work/envs/forcevla
bash scripts/ablation/launch_forcevla_h200.sh /work/envs/forcevla/bin/python /work/forcevla-run.json train
```

由调度器分配同一台机器的 4 张 H200；不要使用 `torchrun` 启动 4 个 JAX 进程。
`forcevla_launch_receipt.json` 记录设备和实际 batch/mesh 口径，是否已产生优化更新仍以训练日志为准。

正式启动前，独立 smoke 配置和目录保持 4×H200、global256、每卡64、FSDP4，设置
`num_train_steps=3, save_interval=1, keep_period=1, log_interval=1`。先执行
`--operation train --stop-after-steps 2`，再以同一个未改动配置执行 `--operation resume`，
验证恢复 optimizer 后完成第三次更新。调用期提前结束不改变冻结配置、学习率或正式30k目标。
检查有限 loss/梯度、真实参数更新、保存和恢复后再启动独立正式运行目录。
GPU 预检的结果以对应运行的独立 receipt 为准，不把 plan 或 CPU 测试当作训练结果。
统计绑定保存为 `norm_stats.binding.json`；冻结训练记录为 `run_root/forcevla_training_config.json`。
每个 checkpoint 的 `assets/ablation_training_binding.json` 随 Orbax 原子保存，恢复与部署都核对
内嵌记录；仅 step 0 的 run 会拒绝恢复，避免上游静默重新训练。完整恢复需要 `train_state`，
只有推理 weights 不算 resume。

## 导出与运行 server

训练后创建模型 runtime JSON，包含真实 `checkpoint_path`、`normalization_path`、
`normalization_sha256`、`training_binding_path`、`dataset_manifest_sha256` 和固定 `upstream_root`。
checkpoint 路径指向含 `params/` 和 `assets/` 的保存步目录。
导出核验数据、统计和训练身份，对模型资产计算 SHA256；导出本身不加载模型，标记
`model_load_verified=false`。服务启动重新校验资产后才调用真实官方加载器。

```bash
python scripts/ablation/export_server.py \
  --backend forcevla --model-config /work/forcevla-model-runtime.json \
  --dataset /work/data/rm71d78_lerobot21 \
  --output /work/deploy/forcevla-deployment.json

python scripts/ablation/serve.py \
  --deployment /work/deploy/forcevla-deployment.json --host 127.0.0.1 --port 8766
```

server 不连接机械臂。客户端跨主机时，将 `--host` 改为服务机可访问地址；可通过
`REALMAN_SERVER_TOKEN` 环境变量共享鉴权令牌。协议为 `realman-sync-v1`：
`GET /health`、`POST /infer`、`POST /session/release`。
模型处理当前双目 RGB、joint8 和六轴力，输出 32×8 已反归一化的绝对 joint8。
关节角不裁剪，夹爪裁剪至 `[0,1]`；客户端继续负责动作执行、软限位和时序检查。
相机 resize、embedding 和模型归一化均在 server。
官方采样器内部使用重复 batch4，以满足固定四专家 LIMoE 的分组约束；不改变外部请求数或
动作格式。额外计算会影响延迟和显存，须报告真实测量值，详见 [兼容说明](forcevla.md#compatibility-changes)。

运行 server 不需要训练数据或 pi0_base 初始权重。部署记录使用绝对路径；跨主机改变资产路径
时，需在同时可访问数据与模型资产的准备环境中重新导出，完成后服务仅需部署资产和固定上游源码。

## 真实权重回放与性能验证

以下命令需要真实训练 checkpoint。它加载模型并启动本机临时 HTTP server，从同一验证数据中
读取 RGB/state/force，实际调用健康检查、推理和会话释放，不操作机器人。

```bash
python scripts/ablation/replay.py \
  --deployment /work/deploy/forcevla-deployment.json \
  --dataset /work/data/rm71d78_lerobot21 \
  --warmup 2 --samples 20 --report /work/reports/forcevla-replay.json
```

报告包含加载时间、HTTP P50/P95、服务内部推理时间、JAX allocator 显存统计、有效动作段关节
MAE（rad）及夹爪 MAE。末端 padding 不计入误差；离线时钟按 30Hz 重置，来源片段由 manifest
校验。真实模型报告尚未生成，不提前填写性能实测值。

## 当前分支验证

```bash
PYTHONPATH=src:deploy FORCEVLA_SOURCE_ROOT=/work/upstream/ForceVLA \
  python -m pytest tests/ablation deploy/tests/test_sync_server.py -q
```

`FORCEVLA_SOURCE_ROOT` 可选，用于对固定官方源码实际执行改动回归。测试包含合成 78 集转换、
HF 图像解码、joint8/力通道和数值契约、统计与 checkpoint 身份拒绝、无数据目录加载路径
（显式模型 stub），以及注入测试预测器的完整真实 HTTP/replay。详见本分支的
[validation.json](validation.json)。这些结果没有启动正式训练，也不证明真实模型/GPU 已通过。
