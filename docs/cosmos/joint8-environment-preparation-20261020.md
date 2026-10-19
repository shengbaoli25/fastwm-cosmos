# Joint8 训练环境与资产准备

CPU 环境与全部权重已准备完成，通过完整模型加载。随后用户授权合并 50 条数据并启动
两组新训练：4×H200、每卡尝试 batch64，一组 1000 步保存，一组上限 20000 步。
两组使用新关节版；正式训练前仍须做对应配置的四卡更新和保存/恢复检查。

## 环境与持久化位置

- 镜像：`docker.sii.shaipower.online/inspire-studio/dualwm:1.0`。
- 本轮准备实例：`realman-joint8-prep-1020`，ID `fdea7d62-e501-492d-8455-cd1df355f393`；4 CPU / 16 GB；准备完成后已执行停止，资产均位于持久化共享目录。
- 共享根目录：`/inspire/hdd/project/long-range-embodied-collaborative-intelligence/ky26340`，下文路径均相对此目录。
- Python：`realman-fastwam/.venv/bin/python`，3.12.11。
- 主要依赖已实测导入：Torch 2.7.1+cu128、Transformers 4.49.0、Accelerate 1.12.0、DeepSpeed 0.18.7、PyArrow 24.0.0。
- 本轮目录：`realman-noaction-joint8/`；训练固定使用 `code-82cb9d2/`，新底座 `assets/`、日志/凭据 `preparation/`。
- 复用原有环境与完整 Reason1/tokenizer 文件，不改动旧项目、旧训练或采集数据。

## 资产

| 资产 | 远端路径 | 固定身份 |
|---|---|---|
| 官方非 action-cond Video2World | `realman-noaction-joint8/assets/base/base/post-trained/81edfebe-bd6a-4039-8c1d-737df1a790bf_ema_bf16.pt` | 4,118,575,006 bytes；SHA256 `565bbb2c9645737327983f4461e4d32627bba465b0a8dc26447edea144e1ff47` |
| 官方 tokenizer | `realman-dualwm/checkpoints/tokenizer.pth` | 507,609,880 bytes；SHA256 `38071ab59bd94681c686fa51d75a1968f64e470262043be31f7a094e442fd981` |
| Cosmos-Reason1-7B | `cosmos-reason1-upload-20261012/` | 四个完整 safetensors 分片，逐一匹配 `src/fastwam/cosmos/reason1_weights.json`，另含本地 processor/config |

基础权重从公开 ModelScope 分发源下载，最终内容必须匹配仓库固定的 NVIDIA 官方 SHA。
本地及远端均已严格加载完整 Video 官方权重并构造 Joint8 模型。
远端凭据：`preparation/environment_assets_verified.json`；全部权重 SHA 和本地 processor 检查通过。

CPU 审计入口 `scripts/cosmos/prepare_environment.py`：验证依赖、官方资产 SHA、固定上游代码、
Reason1 本地 processor 和 512 token 输入、完整非 action-cond Video 严格加载，并构造 Joint8 模型。
它不创建训练器，不启动训练，也不代表 GPU 算子/显存验证。

## 数据与下一步

- 七个实测关节角和实际下发目标均为弧度；加夹爪开度后 action/state 均 8 维，history 14 维。
- 使用 `observation.force_filtered`，与采集端 Q×48 因果滤波合同一致；不重复滤波。
- `1020_cup` 的 43 条和 `rm71d_20261020_032019` 的 7 条已生成独立合并副本 `1020_cup_merged50`。
- 共 28,992 帧、27,783 个有效窗口起点；源清单 SHA256 为 `e4feb413686fdcc44e1bc046e4510250d187ef0bd6ae1dc6c1c32e1aad02d60d`。
- 七轴动作、滤波力及视频内容保持原样，仅重新编号 episode/global index 并更新元数据；源目录不变。
- 远端数据：`realman-noaction-joint8/data/1020_cup_merged50`；导出一次共享缓存，分别重新初始化两组模型和优化器。
- 新配方：4×H200、每卡尝试 batch64/global256、双视角；短训 1000 步/save1000/warmup50，长训 20000 步/save5000/warmup1000。
- 如实际更新检查 OOM，需根据记录降低 microbatch；不把梯度累积视作物理 batch64 成功。

本地相关完整测试：159 项通过；远端 Reason1 原生测试：3 项通过（4.63 秒）。
两组训练于 2026-10-20 03:48（北京时间）提交，目前排队，batch64 的四卡显存预检尚未执行。
任务链接、冻结版本、日志及保存路径见 [50 条数据训练记录](experiments/realman-joint8-merged50-20261020.md)。
详细架构、输入和训练/部署入口见 [实现说明](noaction-joint-diffusion.md)。
