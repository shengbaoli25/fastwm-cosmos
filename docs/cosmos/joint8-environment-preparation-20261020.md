# Joint8 训练环境与资产准备

范围：只准备代码、CPU 环境和权重；未启动训练，未申请 H200，未连接机械臂。
正式训练等待新采集数据定版，再做缓存/统计和四卡检查。

## 环境与持久化位置

- 镜像：`docker.sii.shaipower.online/inspire-studio/dualwm:1.0`。
- 本轮准备实例：`realman-joint8-prep-1020`，ID `fdea7d62-e501-492d-8455-cd1df355f393`；4 CPU / 16 GB，2 小时自动停止。
- 共享根目录：`/inspire/hdd/project/long-range-embodied-collaborative-intelligence/ky26340`，下文路径均相对此目录。
- Python：`realman-fastwam/.venv/bin/python`，3.12.11。
- 主要依赖已实测导入：Torch 2.7.1+cu128、Transformers 4.49.0、Accelerate 1.12.0、DeepSpeed 0.18.7、PyArrow 24.0.0。
- 本轮目录：`realman-noaction-joint8/`；独立代码目录 `code/`、新底座 `assets/`、日志/凭据 `preparation/`。
- 复用原有环境与完整 Reason1/tokenizer 文件，不改动旧项目、旧训练或采集数据。

## 资产

| 资产 | 远端路径 | 固定身份 |
|---|---|---|
| 官方非 action-cond Video2World | `realman-noaction-joint8/assets/base/base/post-trained/81edfebe-bd6a-4039-8c1d-737df1a790bf_ema_bf16.pt` | 4,118,575,006 bytes；SHA256 `565bbb2c9645737327983f4461e4d32627bba465b0a8dc26447edea144e1ff47` |
| 官方 tokenizer | `realman-dualwm/checkpoints/tokenizer.pth` | 507,609,880 bytes；SHA256 `38071ab59bd94681c686fa51d75a1968f64e470262043be31f7a094e442fd981` |
| Cosmos-Reason1-7B | `cosmos-reason1-upload-20261012/` | 四个完整 safetensors 分片，逐一匹配 `src/fastwam/cosmos/reason1_weights.json`，另含本地 processor/config |

基础权重从公开 ModelScope 分发源下载，最终内容必须匹配仓库固定的 NVIDIA 官方 SHA。
本地已严格加载完整 Video 官方权重并构造 Joint8 模型；远端检查进度以 `preparation/` 凭据为准。

CPU 审计入口 `scripts/cosmos/prepare_environment.py`：验证依赖、官方资产 SHA、固定上游代码、
Reason1 本地 processor 和 512 token 输入、完整非 action-cond Video 严格加载，并构造 Joint8 模型。
它不创建训练器，不启动训练，也不代表 GPU 算子/显存验证。

## 数据与下一步

- 七个实测关节角和实际下发目标均为弧度；加夹爪开度后 action/state 均 8 维，history 14 维。
- 使用 `observation.force_filtered`，与采集端 Q×48 因果滤波合同一致；不重复滤波。
- 当前采集中的 `1020_cup` 只做过只读格式检查，未冻结数据身份、未生成最终归一化统计。
- 新数据完成后，冻结目录并生成清单身份；导出文本/视频/numeric 缓存与 normalization，再做四卡两步 Adam、保存/恢复检查。
- 原定训练为 4×H200、每卡 batch16、双视角、30000 步、每 5000 步保存；新模型显存与可提升 batch 尚未实测。

本地验证：156 项测试通过；Reason1 原生测试需在含 Transformers 的远端环境补验。
详细架构、输入和训练/部署入口见 [实现说明](noaction-joint-diffusion.md)。
