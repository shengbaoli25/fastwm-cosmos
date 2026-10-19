# Joint8 step1000 本地验证

本地准备于 2026-10-20 完成。正式 step1000 权重尚未保存，不能将下列运行预检解释为训练后拟合结果。

## 已完成

- Python：`/home/jaylen/miniconda3/envs/realman_deploy/bin/python`，Torch 2.7.1+cu128；本轮补齐 PyAV 15.1.0 和 Matplotlib 3.10.7。RealSense 与 RealMan SDK 导入通过，没有连接硬件。
- 官方非 action-cond Cosmos、tokenizer、固定上游源码均通过加载检查。
- 本轮配置、精确 normalization、Reason1 文本和 4 个原始训练 shard 已通过内置浏览器下载，12 个文件逐一 SHA256 校验。归档 SHA256：`aa141de3f8b3d9e2adf44e59634908a2566a6c1878e4be0b19a8e2a87648a990`。
- 新目录：`/home/jaylen/桌面/checkpoints/realman_joint8_merged50_step001000_assets`；不覆盖旧 step5000 配置。硬件配置为 Joint8，绑定本次 Q48 力滤波及标定，保留用户此前设置的运动限制偏好。
- 样本 24、56、24066、24098，来自 episode0 和43，覆盖两份来源数据的片段起点及完整历史位置。在线 history、当前 proprio、所有数值标签和掩码与训练缓存完全一致；原始 RGB 经部署预处理与训练读图结果完全一致。记录力不重复滤波。
- 真实首帧在 H200 与 RTX3080 的 BF16 tokenizer 结果并非逐位一致，平均绝对差 0.000775–0.000803、最大 0.015625；单独报告这项跨设备差异。
- 使用明确标记的 **step0 测试权重** 完成保存、导出、严格加载、两次预热和 4 窗口部署回放；部署适配器与直接使用训练数值输入、相同本地首帧 latent 的动作/力/视频输出完全一致。此项只验证代码链路，不判断训练质量。
- RTX3080 20GB，16 轮 V/F/A 联合采样：4 次部署回放 1.942–1.960 秒，峰值 allocated 6.375 GiB / reserved 6.631 GiB。诊断捕获含输出复制开销；不是闭环控制频率。测试后 GPU 回到约2MiB。
- 未训练分支的缓存/非缓存动作平均绝对差约0.00348、最大0.02344（归一化单位），来自 BF16 计算路径；正式权重必须重新复核。

预检凭据在资产目录的 `preflight-01/report.json`、`gpu-assets-probe.json`、`gpu-real-input-probe.json`。
完整执行预检在 `execution-smoke-TEST-ONLY/report.json`，其 step0 权重和图不得作为 step1000 结果展示。

05:13复查发现本地驱动处于 `GPU Recovery Action: Reset`，CUDA无法重新初始化。
内核在05:05:15报告Xid119（GSP RPC超时），05:05:27报告Xid154（GPU Reset Required）。
完整推理输出在该进程内已完成并保存在上述报告中，但不能据此声称当前显卡仍可使用。
桌面Xorg/WebKit等仍打开设备，且当前无免密管理员权限；本轮没有重置显卡或终止桌面程序。
正式权重仍可下载、校验和导出，GPU回放需要用户恢复驱动/重启后再执行。
相关测试首次44项通过；驱动异常后的复跑为43项通过、1项CUDA测试跳过。

## 权重到达后

只下载 `short/training/checkpoints/weights/step_001000.pt`，不下载约43GiB的完整优化器状态。
训练在 save_every 和 max_steps 处会再次保存同名文件，必须等最终 `[done] max_steps reached step=1000`，
没有 `.pt.tmp` 且最终保存结束，再生成并下载凭据：

```json
{
  "schema": "joint8-step1000-download-v1",
  "step": 1000,
  "training_complete": true,
  "size_bytes": 0,
  "sha256": "填写远端最终权重的真实SHA256"
}
```

上面的0是格式示例，必须替换为真实大小；不能直接作为下载成功凭据。
通过内置浏览器项目文件页面下载权重和凭据。执行资产目录的 `finish_local.sh 权重路径 凭据路径`，
它调用 [finish_joint8_local.py](../../scripts/cosmos/finish_joint8_local.py)：

1. 校验远端 SHA256、大小、checkpoint schema 和 step1000，原子复制到独立 weights 目录。
2. 从真实训练配置和 checkpoint manifest 导出 runtime，保留 training_recipe；重新绑定精确归一化、文本、Joint8 和力预处理。
3. 调用 [validate_joint8_local.py](../../scripts/cosmos/validate_joint8_local.py)，保持16步及 A42/V44/F43 种子。
4. 分开报告部署路径一致性、本地/训练缓存首帧差异、缓存/非缓存差异、训练数据动作拟合和六轴力拟合。

验证输出为每个窗口的 NPZ、8通道动作图、6通道力图和 JSON 指标。动作同时对比原始记录目标、
经过归一化clip后的训练目标，以及保持当前测量状态的基线；仅有效未来位置计入误差。
这4个训练窗口用于排查链路和初步拟合，不代表完整训练集指标或未见数据泛化。
验证不加载机器人SDK连接，不发送动作。完成后释放模型。

正式 runtime 和 catalog 导出后，可用资产目录的 `serve_local.sh` 启动本地部署网页。
脚本只启动服务，模型加载、硬件连接和开始控制仍由用户在网页操作。

本对话已建立每10分钟一次的保存点跟进 `joint8-1000`，仅在保存、失败或需用户处理时通知，
下载验证完成后停止跟进。
