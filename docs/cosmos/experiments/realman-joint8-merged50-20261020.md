# Realman Joint8 merged50：1000 步和 20000 步

2026-10-20 03:48（北京时间）提交。两组均已通过真实四卡 batch64 更新和保存/恢复预检。
短训于05:31:22达到1000/1000步，05:32:22完成最终保存并以returncode=0退出。
末步Action FM为0.0072、Force FM为0.0197、Video FM为0.0407，总loss为0.0676。
这些数值是训练FM loss，不能替代采样后的动作误差。
长训于04:15失败：正式训练前的磁盘容量检查不通过，尚未进入正式step。
共享盘剩余约112GiB，不足以保存每5000步一次、共4份完整状态及10%余量（约188GiB）。
尚未删除历史资产、变更保存策略或重启长训。

## 任务

| 用途 | 平台任务 | 资源与 batch | 训练上限 | 保存 | warmup |
|---|---|---|---:|---:|---:|
| 短训拟合、部署对齐验证 | [realman-joint8-merged50-b64-1k](https://qz.sii.edu.cn/jobs/distributedTrainingDetail/job-713837da-ca6c-4e24-8235-f9e39e0cbbd7?spaceId=ws-9dcc0e1f-80a4-4af2-bc2f-0e352e7b17e6) | 单节点 4×H200 141GB，每卡64，global256 | 1000 | 1000 | 50 |
| 完整训练 | [realman-joint8-merged50-b64-20k](https://qz.sii.edu.cn/jobs/distributedTrainingDetail/job-14e430fd-4bf6-4c15-9e02-6de2c657339d?spaceId=ws-9dcc0e1f-80a4-4af2-bc2f-0e352e7b17e6) | 单节点 4×H200 141GB，每卡64，global256 | 20000 | 每5000 | 1000 |

两组均为普通优先级4、80 CPU、900GB内存、200GB共享内存；镜像 `dualwm:1.0`。
提交时所选 H200 资源池显示 GPU 可用0/160；上述状态来自任务实际启动后的日志。
BF16、AdamW学习率1e-4、ZeRO1、梯度累积1。每组正式训练前执行实际更新和独立保存/恢复检查。

两组固定源代码 `82cb9d2fe4c855d146ab13b0045b175cd833f48a`，目录：
`/inspire/hdd/project/long-range-embodied-collaborative-intelligence/ky26340/realman-noaction-joint8/code-82cb9d2`。
后续文档提交不会改变任务绑定的这份源码。入口 `scripts/cosmos/launch_joint8_merged50.py`，
短训使用 `--mode short`，长训使用 `--mode long`，共同参数：
`--run-tag merged50-82cb9d2-b64 --batch-size 64`。

两组只复用相同数据的文本/视频缓存，分别从官方非 action-cond Cosmos 和随机 AE/Force 开始，
分别创建优化器与对应长度的训练计划。长训不接续1000步权重。
长训等待短训写出完整共享缓存凭据；如缓存准备失败则退出，等待上限4小时。

## 数据身份

- 本地：`/home/jaylen/桌面/dataset/1020_cup_merged50`。
- 源：`1020_cup` 43条、25,119帧；`rm71d_20261020_032019` 7条、3,873帧。
- 合并：50条、28,992帧、27,783个有效窗口起点。原始两份不变。
- 远端：`/inspire/hdd/project/long-range-embodied-collaborative-intelligence/ky26340/realman-noaction-joint8/data/1020_cup_merged50`。
- 源清单SHA256：`e4feb413686fdcc44e1bc046e4510250d187ef0bd6ae1dc6c1c32e1aad02d60d`。
- 上传归档SHA256：`f0db321eb59f4df9500c9e229262192268a6b3654eba59ae82689ba441cb158a`；远端实测匹配。

视频无重编码，100个双视角视频均逐文件校验。动作、状态、力数值保持原内容，
仅重编episode/global索引并更新metadata/statistics。使用Q×48因果滤波后的六维力，训练不重复滤波。
关节action/state均为7个绝对关节弧度加夹爪开度，history为state8+force6。
模型为无参考V/F/A逐层联合扩散，AE768/3072、Force768/2048；训练和部署使用相同Joint8协议。

## 日志与检查点

以下路径均相对共享根：
`/inspire/hdd/project/long-range-embodied-collaborative-intelligence/ky26340/realman-noaction-joint8/runs/merged50-82cb9d2-b64/`。
短训路径已产生实际日志和预检凭据；长训 `formal-watch` 尚不存在，失败日志在 `long/formal-controller-watch/probe.log`。

| 内容 | 相对路径 |
|---|---|
| 共享缓存准备状态 | `prepared/status.json`、`prepared/prepared.json` |
| Reason1文本编码日志 | `prepared/text.log` |
| 四卡视频编码日志 | `prepared/encode_rank0.log` 至 `encode_rank3.log` |
| 归一化 | `prepared/normalization.json` |
| 短训配置/长训配置 | `prepared/resolved.yaml` / `long-inputs/resolved.yaml` |
| 短训/长训分发状态 | `short/dispatch.json` / `long/dispatch.json` |
| 短训/长训预检 | `short/preflight/preflight.json` / `long/preflight/preflight.json` |
| 短训/长训日志 | `short/formal-watch/training.log` / `long/formal-watch/training.log` |
| 1000步权重 | `short/training/checkpoints/weights/step_001000.pt` |
| 长训权重 | `long/training/checkpoints/weights/step_005000.pt`、`step_010000.pt`、`step_015000.pt`、`step_020000.pt` |

预检若OOM，应记录实际峰值和失败阶段后下调microbatch，使用独立新输出目录；
梯度累积达到global256不等于物理每卡batch64成功。

## 已完成验证与待验证项

- 本地模型/数据/部署相关完整suite：159 passed；另有运行配方及resume完成状态相关44 passed。
- 远端Reason1原生兼容测试：3 passed in 4.63s，生产Transformers仍为4.49。
- 本地和远端官方Video完整严格加载、模型构造、Reason1四分片及tokenizer SHA校验通过。
- 远端合并数据重新读取确认50条/28,992帧/27,783窗口；冻结代码HEAD和工作树已核对。
- 已完成：两组物理每卡batch64真实更新/恢复预检，短训正式loss，独立本地推理资产下载。
- 本地：4个真实窗口输入对齐和step0完整部署回放通过，相关44项测试通过。
- 正式1000步权重已下载，大小、SHA256、step和schema通过校验，本地runtime及力预处理绑定已导出。
- 05:54正式回放的4窗口CPU输入对齐通过；GPU推理因本地驱动要求Reset而停止，尚无训练后拟合曲线及部署输出对照。长训因容量门槛失败。
- 本地准备与后续入口见 [Joint8本地验证](../joint8-local-validation-20261020.md)。

1000步保存后应先在同一段记录数据上比较训练读取与部署读取的相机顺序/帧、关节弧度、
夹爪定义、Q48力历史与归一化；固定采样seed和16轮联合采样，比较AE归一化输出及反归一化关节曲线。
部署SDK边界只做一次弧度到度转换、不经IK。验证通过后再由现场用户启动真机。
单凭1000步真机效果差无法区分模型尚未拟合和部署不对齐；必须结合记录数据的目标动作误差及两条路径输出差异。

## 最终保存凭据

远端最终权重为6,038,120,329字节，CPU监视日志两次记录相同稳定SHA256：
`38c402207973b9da74488ccf7475cd9a9f7b9280db1f700c9fda582ae4e2a425`。
最终日志明确包含`max_steps reached step=1000`，随后NCCL正常销毁，进程退出码为0。
Rich移除了原始日志字符串中的`[done]`标记，导致自动下载凭据未生成；该问题不影响训练保存。
本地凭据明确标记为依据已下载的远端日志、两次稳定哈希和退出码重建，未声称由远端脚本自动签发。
凭据脚本已兼容原始和Rich渲染后的日志，并拒绝在失败退出时签发；相关7项测试通过。
