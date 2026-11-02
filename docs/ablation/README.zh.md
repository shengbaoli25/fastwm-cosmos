# RM71D78：π0.5 独立消融分支

本分支 `codex/ablation-rm71d78-pi05` 提供官方 π0.5 的数据准备、训练与恢复入口、
checkpoint 身份绑定及现有 RealMan 客户端可调用的 HTTP server。
**训练没有启动，大权重没有下载，真实 GPU 前反向、参数更新、恢复和模型推理尚未验证。**

环境和模型配置见 [pi05.md](pi05.md)，源码、初始化来源与数据身份见
[upstream-lock.json](../../configs/ablation/upstream-lock.json)。训练采用独立 Linux CUDA JAX
环境；客户端可继续运行在 Windows。当前分支检查结果单独记录在
[validation.json](validation.json)。

## 数据与训练口径

数据源固定为 `rm71d_plug_transfer_20261029`，任务“拔下插头并插入另一插座”，78 条原始轨迹。
LeRobot v3 的实测状态 `[0..6,19]` 转成 7 个关节角（弧度）和夹爪开度，动作保留实际下发的
绝对 `action[8]`。夹爪范围 `[0,1]`，频率 30 Hz。

保留 `global/wrist` 两相机，视频解码出的 RGB 无损存为 PNG，不在转换阶段 resize；
写入 Hugging Face Image metadata 供官方 LeRobot v2.1 加载器解码。
六轴力仍保留在通用数据协议中，并校验标定和因果滤波身份；π0.5 模型不使用这些力数据。
断点、未执行动作、过期传感器样本和滤波 epoch 变化不跨训练窗口；连续有效段分别成为
输出 episode。manifest 保留全部 78 条来源及分段映射，输出 episode 数可能大于 78。
全部原始轨迹用于训练，离线回放只报告 **training-fit**，不代表留出集泛化或机器人成功率。

默认计划为官方 OpenPI JAX LoRA，50k 更新、global batch 8、每 5k 保存、seed 42。
内部采用 32D padding 和 32 步输出，7 个关节以官方 delta transform 训练并在推理后还原绝对值。
默认资源规划预留一张 80GB GPU；实际显存和速度须后续预检。具体归一化、优化器、全量微调
选项及依赖版本见 [pi05.md](pi05.md)。这些数值是可配置计划，未证明收敛。

## 数据准备

数据工具只需 Python、NumPy、PyArrow、PyAV 和 Pillow，不加载训练模型。
转换目标必须不存在，并位于原数据目录之外。报告与运行产物也放在数据目录外。

```bash
# 审核冻结文件 SHA、78 集、图像、数值、时序与力预处理身份。
python scripts/ablation/prepare_data.py \
  --source /data/rm71d_plug_transfer_20261029 \
  --report /work/reports/rm71d78-audit.json

# 审核并转换；使用与其他实验相同的冻结数据及划分口径。
python scripts/ablation/prepare_data.py \
  --source /data/rm71d_plug_transfer_20261029 \
  --output /work/data/rm71d78_lerobot21 \
  --report /work/reports/rm71d78-export.json
```

源数据 hash 与冻结记录不一致即停止。失败时保留 `.incomplete-*` 诊断目录，不产出正式数据目录。
PNG 导出所需磁盘空间高于原视频；原数据保持不变。

## 计划、统计、训练和恢复

先按模型文档安装固定版本 OpenPI，下载官方基座到本地，然后复制并填写
[pi05.example.json](../../configs/ablation/pi05.example.json)。下列命令均从本仓库根目录运行；
除 plan 外，使用固定 OpenPI 环境中的 Python。

```bash
# 默认只显示计划，不创建运行目录，不加载模型。
python scripts/ablation/pi05.py --config /work/pi05-run.json

/work/upstream/openpi/.venv/bin/python scripts/ablation/pi05.py stats --config /work/pi05-run.json

# 仅显式 train / resume 会开始优化；本次未执行。
/work/upstream/openpi/.venv/bin/python scripts/ablation/pi05.py train --config /work/pi05-run.json
/work/upstream/openpi/.venv/bin/python scripts/ablation/pi05.py resume --config /work/pi05-run.json
```

运行绑定记录数据 manifest、统计文件 SHA、源码版本、实际基座参数 SHA、训练设置及唯一 run_id。
Orbax checkpoint 内嵌 `assets/ablation_training_binding.json`，随 assets 原子保存。
完整恢复需要真实 optimizer/RNG 等状态，并核验嵌入绑定；仅权重文件不算恢复。

正式启动前，应在独立试运行目录用小 batch 和少量更新检查有限 loss/梯度、参数更新、保存及恢复，
再冻结正式配置。当前代码检查没有替代这些运行验证，计划输出也不证明模型能够训练。

## 导出与启动 server

训练后按模型文档准备 `/work/pi05-model-runtime.json`，填写真实 checkpoint、其 normalization
和 training binding 路径。导出校验数据身份和嵌入绑定，为部署资产计算 SHA256；启动服务时再校验。
导出结果本身标记 `model_load_verified=false`，实际模型加载发生在 server 启动时。

```bash
python scripts/ablation/export_server.py \
  --backend pi05 \
  --model-config /work/pi05-model-runtime.json \
  --dataset /work/data/rm71d78_lerobot21 \
  --output /work/deploy/pi05-deployment.json

/work/upstream/openpi/.venv/bin/python scripts/ablation/serve.py \
  --deployment /work/deploy/pi05-deployment.json --host 127.0.0.1 --port 8766
```

服务只推理模型，不连接机械臂。跨机器使用时将 `--host` 设置为服务机可访问地址。
可使用 `REALMAN_SERVER_TOKEN` 环境变量与客户端共享鉴权令牌。
协议沿用 `realman-sync-v1` 的 `/health`、`/infer` 和 `/session/release`，每次返回
32×8 已反归一化的绝对 joint8；关节角不裁剪，夹爪限定到 `[0,1]`。
客户端继续负责动作执行、软限位和时序检查；图像 resize、归一化及 embedding 均在 server。

server 不需要训练数据目录或基座参数目录，但仍需固定版本 OpenPI 源码及所选训练 checkpoint。
复制资产到另一机器后，用实际资产路径重新导出 deployment；导出阶段仍需校验数据集，serve 阶段无需。

## 真实 checkpoint 回放

以下命令加载真实训练模型并运行临时本机 HTTP server，从已验证数据读取 RGB/state/force，
调用健康检查、推理及会话释放接口，不操作机器人。

```bash
/work/upstream/openpi/.venv/bin/python scripts/ablation/replay.py \
  --deployment /work/deploy/pi05-deployment.json \
  --dataset /work/data/rm71d78_lerobot21 \
  --warmup 2 --samples 20 --report /work/reports/pi05-replay.json
```

报告包括模型加载时间、HTTP P50/P95、服务推理时间、JAX 显存统计、有效动作段的关节 MAE（rad）
和夹爪 MAE。padding 不计入误差。离线时间戳按 30 Hz 重置，数据身份保留在 manifest 中。
该真实 checkpoint 回放尚未运行，报告中没有预填实测结果。

## 本分支本地检查

```bash
PYTHONPATH=src:deploy python -m pytest tests/ablation deploy/tests/test_sync_server.py -q
```

检查覆盖合成 78 集数据转换、HF 图像解码、π0.5 输入/输出契约、checkpoint/run 绑定及恢复拒绝、
注入测试预测器的真实 HTTP 调用、文件篡改拒绝。测试使用模型替身验证协议，不代表真实权重推理。
本分支单独执行的命令、结果和未运行项目见 [validation.json](validation.json)。
