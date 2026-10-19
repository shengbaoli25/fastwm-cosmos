# Joint8 step1000 本地验证

正式step1000权重已于2026-10-20下载并校验，本地runtime、力预处理绑定和模型目录已导出。
重启后06:07–06:09完成正式16步回放，4个窗口输入对齐和部署/直接模型输出一致性通过。
回放退出后06:10再次发生GSP超时及Xid119/154，目前CUDA又不可用；成功的离线结果已经保存，当前网页模型尚未加载。
下列step0运行预检不能解释为训练后拟合结果。

## 正式权重与当前阻塞

- 训练05:31:22达到1000步，05:32:22最终保存完成并以returncode=0退出。
- 末步训练FM loss：Action 0.0072、Force 0.0197、Video 0.0407、总计0.0676。
- 下载文件位于`/home/jaylen/文档/step_001000.pt`，已安装到资产目录`weights/step_001000.pt`。
- 文件6,038,120,329字节，下载及安装后的SHA256均与远端两次稳定记录一致：
  `38c402207973b9da74488ccf7475cd9a9f7b9280db1f700c9fda582ae4e2a425`。
- checkpoint的step1000和schema检查通过；`runtime.local.json`、`checkpoint_catalog.json`及`.force.json`绑定已导出，固定16轮联合采样。
- 正式记录：资产目录`validation-20261020-055411/report.json`，输入对齐通过；推理因驱动要求Reset而失败，模型尚未加载到GPU。
- 最新成功报告：资产目录`validation-20261020-060758/report.json`，状态`inference_passed`；每个窗口的动作/力PNG及NPZ均已保存。
- 4窗口部署与直接模型的action、force、video_latents逐位一致（相同本地首帧latent）；首个窗口缓存/非缓存动作平均差0.000779、最大0.015625，低于既定数值诊断阈值。
- RTX3080 20GB正式16步回放耗时1.929–1.945秒，峰值allocated 6.375GiB、reserved 6.631GiB；含诊断复制开销，不代表闭环频率。
- 4窗口7关节平均MAE为0.428°，保持当前关节基线为0.623°。各窗口模型/基线为0.389/0.409°、0.369/0.902°、0.609/0.088°、0.344/1.091°；静止起始窗口明显劣于保持基线，不能据此声称已稳定拟合或真机成功。
- 夹爪MAE为0.0102–0.0151开度；六轴力曲线已保存，部分窗口未来力仍有漂移。这里只检查4个训练窗口，不代表完整数据或泛化。
- 06:10内核再次记录GSP RPC超时及GPU Reset Required，`kernel-after-reboot.log`已保存。故障发生在离线结果保存之后，根因尚未定位。未擅自重启、重置GPU或修改驱动。
- 网页服务已恢复到`http://127.0.0.1:8765/`，step1000已选中，当前模型未加载、硬件未就绪；不能将离线验证通过等同于当前可以启动真机。先前服务还以139退出并输出RealMan寄存器读取错误，硬件连接问题需单独定位。

自动凭据监视误将Rich渲染日志中已消失的`[done]`作为必需标记。
已下载最终训练日志、退出码和两次远端稳定哈希，保存在资产目录`training-completion/`。
凭据明确标记为本地依据这些远端证据重建，未声称由远端自动生成；脚本修复及7项相关测试通过。

驱动恢复后执行（不会连接或运动机器人）：

```bash
bundle='/home/jaylen/桌面/checkpoints/realman_joint8_merged50_step001000_assets'
"$bundle/finish_local.sh" "$bundle/weights/step_001000.pt" "$bundle/checkpoint-download-receipt.json"
```

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
正式权重已下载、校验和导出，GPU回放需要用户恢复驱动后再执行。本轮没有重启系统。
相关测试首次44项通过；驱动异常后的复跑为43项通过、1项CUDA测试跳过。

## 下载与复跑流程

只下载 `short/training/checkpoints/weights/step_001000.pt`，不下载约43GiB的完整优化器状态。
训练在 save_every 和 max_steps 处会再次保存同名文件，必须等最终`max_steps reached step=1000`（原始日志可能含`[done]`），
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

保存点跟进`joint8-1000`的下载与正式离线验证已完成，应停止该跟进；后续驱动与真机连接问题单独处理。
