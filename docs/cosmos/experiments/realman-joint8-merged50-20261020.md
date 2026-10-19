# Realman Joint8 merged50：1000 步和 20000 步

2026-10-20 03:48（北京时间）提交。03:49 两个任务均显示排队中；尚无 GPU 预检、正式 step 或 loss。
本次按用户要求每卡先尝试物理 batch64，不能在实际运行前断言 H200 显存足够。

## 任务

| 用途 | 平台任务 | 资源与 batch | 训练上限 | 保存 | warmup |
|---|---|---|---:|---:|---:|
| 短训拟合、部署对齐验证 | [realman-joint8-merged50-b64-1k](https://qz.sii.edu.cn/jobs/distributedTrainingDetail/job-713837da-ca6c-4e24-8235-f9e39e0cbbd7?spaceId=ws-9dcc0e1f-80a4-4af2-bc2f-0e352e7b17e6) | 单节点 4×H200 141GB，每卡64，global256 | 1000 | 1000 | 50 |
| 完整训练 | [realman-joint8-merged50-b64-20k](https://qz.sii.edu.cn/jobs/distributedTrainingDetail/job-14e430fd-4bf6-4c15-9e02-6de2c657339d?spaceId=ws-9dcc0e1f-80a4-4af2-bc2f-0e352e7b17e6) | 单节点 4×H200 141GB，每卡64，global256 | 20000 | 每5000 | 1000 |

两组均为普通优先级4、80 CPU、900GB内存、200GB共享内存；镜像 `dualwm:1.0`。
提交时所选 H200 资源池显示 GPU 可用0/160，因此没有可靠的启动时间。
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
它们是配置的输出位置，排队时尚未产生训练结果。

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
- 尚未完成：batch64 GPU容量、正式训练loss、1000步拟合曲线及真实部署的数值对照。

1000步保存后应先在同一段记录数据上比较训练读取与部署读取的相机顺序/帧、关节弧度、
夹爪定义、Q48力历史与归一化；固定采样seed和16轮联合采样，比较AE归一化输出及反归一化关节曲线。
部署SDK边界只做一次弧度到度转换、不经IK。验证通过后再由现场用户启动真机。
单凭1000步真机效果差无法区分模型尚未拟合和部署不对齐；必须结合记录数据的目标动作误差及两条路径输出差异。
