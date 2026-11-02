# RM71D78 FastWAM 独立分支

分支为 `codex/ablation-rm71d78-fastwam`，从原始训练仓基线
`057f78767f52f358b229f4a05b097c0f1455fa3d` 分出。新增的训练配置、模型适配与 HTTP 服务
只支持 FastWAM；原仓历史训练代码保持原样。

这次交付是可审查的实现与本分支验证结果，没有启动正式训练、下载大权重或运行机器人。
真实 78 条数据尚未在本机完整核验；真实 checkpoint 加载、GPU 显存、吞吐与闭环成功率尚未验证。
代码/CPU/模拟 HTTP 测试证据见 [validation.json](validation.json)。

当前执行目标为 4×H200、每卡真实 batch 64、累积 1、全局 batch 256、30,000 步。
独立 Linux venv bootstrap、真实 GPU/DeepSpeed batch 预检，以及保持相同硬件的两步保存/第三步恢复
命令见 [FastWAM 环境与训练指南](fastwam.md)。本地尚未执行这些 GPU 操作。

## 任务契约

数据集是 `rm71d_plug_transfer_20261029`，78 条原始示范，任务文本“拔下插头并插入另一插座”。
来源 inventory SHA256 固定为
`b24b701239f54ad38d16c819b9ea9f6fad5f9374a683a84a68d2cd7ef27067a8`。
转换保留原数据，依据采样间断/滤波 epoch/跟随会话切分连续段，每段成为独立 LeRobot v2.1
episode，动作窗口不跨间断。global/wrist RGB 均为 640×480、30 Hz；转换保存无损 PNG。

输入 state 与输出 action 都是绝对 joint8：7 个关节角弧度和夹爪 opening `[0,1]`。
训练采用当前数据计算的统计，输出 32 步动作块。六轴力以已标定、因果滤波的原值保留，
FastWAM 官方基线不使用力作为模型输入。全部 78 条示范用于训练，离线回放指标仅为 training-fit。

官方 FastWAM 代码/权重 pin 见 [upstream-lock.json](../../configs/ablation/upstream-lock.json)。
保留视频和动作 Transformer 主干，显式重建官方 7D policy 的动作输入、动作输出与 proprio 投影。
初始化策略、环境与显存边界见 [fastwam.md](fastwam.md)。

## 准备数据与训练

先按仓库安装训练依赖；训练使用 Linux CUDA 与现有 Accelerate/DeepSpeed ZeRO1。
以下路径必须替换为实际资产，统计、报告和输出放在原始/转换数据目录之外。

```bash
python scripts/ablation/fastwam.py plan

# 省略 --output 时仅审计；指定时生成新目录。
python scripts/ablation/prepare_data.py \
  --source /data/rm71d_plug_transfer_20261029 \
  --output /data/rm71d78-canonical --report /work/fastwam/data-audit.json

python scripts/ablation/fastwam.py stats \
  --dataset-root /data/rm71d78-canonical --output /work/fastwam/stats.json
```

复制 [fastwam.yaml](../../configs/ablation/fastwam.yaml) 到 `/work/fastwam/config.yaml`，填写
`dataset_root`、`normalization_path`、`asset_root`、`text_embedding_cache_dir`、
`pretrained_fastwam_checkpoint` 五个路径。手动准备固定官方 policy、Wan VAE、T5/tokenizer
或已生成的 T5 缓存；入口不会自动下载。默认 plan 不加载模型也不训练。

```bash
# 缺少现成缓存时显式编码文本；此命令不做优化训练。
python scripts/ablation/fastwam.py text-cache --config /work/fastwam/config.yaml
python scripts/ablation/fastwam.py train --config /work/fastwam/config.yaml \
  --output /work/fastwam/run --num-processes 4
python scripts/ablation/fastwam.py resume --config /work/fastwam/config.yaml \
  --output /work/fastwam/run --num-processes 4 \
  --resume /work/fastwam/run/checkpoints/state/step_005000
```

当前配置为 4×H200、每卡 batch64、梯度累积 1、全局 batch256，训练至 30000 optimizer steps，
每 5000 步保存。启动入口默认从配置读取 4 个进程，不会把每卡 batch64 转换成累积后的等效 batch。
实际显存与吞吐仍需平台运行验证。完整 resume 恢复 optimizer、
scheduler、RNG 与读取进度，要求已绑定的训练语义保持一致。训练产生 `config.yaml`、
`fastwam_binding.json`、字节一致的 `normalization.json`、`checkpoints/weights/step_*.pt`
及完整 state。部署使用 `normalization.json`，不能以重新格式化的 `dataset_stats.json` 替换。

## 导出 Server 与回放

创建 `/work/fastwam/model.json`，填入实际训练产物：

```json
{
  "checkpoint_path": "/work/fastwam/run/checkpoints/weights/step_030000.pt",
  "normalization_path": "/work/fastwam/run/normalization.json",
  "training_config_path": "/work/fastwam/run/config.yaml",
  "binding_path": "/work/fastwam/run/fastwam_binding.json",
  "asset_root": "/models",
  "text_embedding_cache_dir": "/work/fastwam/text-cache",
  "device": "cuda:0",
  "num_inference_steps": 16,
  "seed": 42
}
```

```bash
python scripts/ablation/export_server.py --backend fastwam \
  --model-config /work/fastwam/model.json --dataset /data/rm71d78-canonical \
  --output /work/fastwam/deployment.json
python scripts/ablation/serve.py --deployment /work/fastwam/deployment.json \
  --host 127.0.0.1 --port 8766

# 单独运行：加载模型并启动临时本机 HTTP server，避免与上述 server 争抢显存。
python scripts/ablation/replay.py --deployment /work/fastwam/deployment.json \
  --dataset /data/rm71d78-canonical --samples 20 --warmup 2 --report /work/fastwam/replay.json
```

服务复用 `GET /health`、`POST /infer`、`POST /session/release`，兼容现有同步 Client。
通过环境变量 `REALMAN_SERVER_TOKEN` 配置与 Client 相同的 token。模型预处理、归一化和反归一化
位于 Server，机器人采集和执行由 Client 完成。Server 无需训练数据目录。

导出只核验资产与身份，报告 `model_load_verified=false`；实际加载和回放成功后才能认定权重可服务。
回放工具记录 HTTP p50/p95、显存及训练数据拟合误差，不连接机器人，不表示任务成功率。
