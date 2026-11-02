# 采集端与 Windows 推理 Client 数据契约核对

审计日期：2026-11-01。核对范围包括静态代码、配置、纯函数契约比较及主代理保存的本机验收记录。本次文档复核没有连接相机、机械臂或夹爪，也没有执行动作。结论是 **joint8、RGB 输入和当前 Cosmos 力预处理契约有一致证据，但不能据此宣称现场全部一致**。下文明确区分数据格式、运行策略和未完成的实机验证。

## 来源与版本

- 权威采集仓库：`references/realman-data-collection`，Git commit `762e4a2bb4ff29c6a7ec447592f92688990b8237`；审计时 Git 工作区干净，未发现该工作区下的 AGENTS.md。
- 推理 Client：`C:/work/infer/client`，不是 Git checkout；[LOCAL_SOURCE.json](LOCAL_SOURCE.json) 记录来源分支 `codex/realman-sync-client`、归档 commit `78fff14d77b3b446135a5bc9fdf0a3d25765e923`，并明确有 Windows 和网页本地修改。该 commit 不能代表当前全部文件。
- 训练输入交叉证据来自本地 Cosmos [realman_joint.py](C:/work/infer/cosmos/code/src/fastwam/cosmos/realman_joint.py#L14) 与已下载的 [normalization.json](C:/work/infer/cosmos/config/normalization.json:131)。本报告没有逐一验证其它四份实验权重的 normalization。
- 以下“采集路径”相对根为 `C:/work/realman/references/realman-data-collection`。

## 已确认的数据语义

| 项目 | 采集证据 | 当前推理端 | 结论 |
|---|---|---|---|
| 关节 state | `realman_leader_reader/collection_dataset.py:203` 将实测从臂 J1–J7 从 degree 转 rad；原始 state 为 20 维：关节7、TCP6、滤波力6、实测夹爪1 | `hardware.py:447` 发送七轴实测 radians 和实测夹爪开度 | 一致。原始20维与请求8维不是遗漏：训练 `RealManJointSource.STATE_COLUMNS=[0,1,2,3,4,5,6,19]` 明确抽取这8列 |
| action | 采集 `collection_dataset.py:205` 保存真正发送的七轴绝对目标，转成 rad；`:216` 加入最后已确认的夹爪目标 | `hardware.py:534` 按绝对关节 rad 解释，在 SDK 边界转 degree，再调用 `rm_movej_canfd` | 绝对目标语义一致；不是末端增量，也不是关节增量 |
| 夹爪方向 | `leader_gripper.py:901`：`1-(raw-lower)/(upper-lower)`；数据说明为0闭、1开 | `gripper.py:318` 用同公式；`:190` 反变换为 `lower+(1-opening)*(upper-lower)` | 方向一致；两者读取已保存行程寄存器，未硬编码12000为通用终点 |
| force 字段 | state[13:19] 和 `observation.force_filtered` 是同一补偿后因果滤波力 | 单独发送6维 force，server 拼成8维state+6维force历史 | 与训练14维历史布局一致 |
| 双相机角色 | 示例相机配置为 global、wrist；训练 CAMERAS 固定 global、wrist | `client.py:48` 与 `hardware.py:265` 强制相同顺序 | 逻辑角色顺序一致；物理摆放尚未现场确认 |
| RGB 与尺寸 | `camera_worker.py:141` 使用 rgb8；配置640×480、30fps | `camera.py:40` 使用 rgb8，当前 hardware.json 同尺寸/帧率；传输为 RGB 原始字节 | 采集/模型输入颜色格式一致。网页 JPEG 只是预览，不作为模型输入 |
| 采样时间 | 采集30Hz网格；先选网格之前的力/关节同包，再按力时间选择相机和夹爪 | 30Hz collector；按同包反馈时间选择相机和夹爪；禁止未来样本 | 因果方向一致；允许年龄和故障处理有下述差异 |

关键来源：[采集 state/action 打包](https://github.com/shengbaoli25/realman-data-collection/blob/762e4a2bb4ff29c6a7ec447592f92688990b8237/realman_leader_reader/collection_dataset.py#L203)、[采集夹爪映射](https://github.com/shengbaoli25/realman-data-collection/blob/762e4a2bb4ff29c6a7ec447592f92688990b8237/realman_leader_reader/leader_gripper.py#L899)、[推理观测](src/realman_sync_client/hardware.py#L422)、[推理动作](src/realman_sync_client/hardware.py#L534)。

## 力预处理的确定一致证据与边界

三份 `realman_force/contract.py`、`force_filter.py`、`force_calibration.py` 在采集仓库与 Client 中，按文本读取规范化 LF/CRLF 后完全相同。Client 当前 force_filter.json 等于权威 `SELECTED_CONFIG`。

纯配置计算 `input_contract(calibration, make_manifest(filter, calibration))` 与当前 Cosmos normalization 的 `force_preprocessing` **整对象相等**：

- 契约 ID：`41dff495c3ac878829234b43004c450720f8fd45751b0136e39dca5853af5236`
- 校准 ID：`2c89319c6642d4635543cd93e60de65e997b2b0f23dde1ada56ec94bf9bfdfd3`
- 输入为控制器 `zero_force`，坐标为 World/base axes、控制器 wrench origin，单位 `N,N,N,Nm,Nm,Nm`。
- 随机游走 Kalman：测量方差 `[.25,.25,.25,.0004,.0004,.0004]`；每秒过程方差前三项 `.03662109375`、后三项 `.00005859375`。
- 所有原生反馈包参与实际 dt 递推；至少100包、0.5秒热身；超过30ms间断触发重置。
- 网格样本选 `latest_source_at_or_before_grid`，最大源年龄30ms，不插值。采集在30Hz采样时处理尚未处理的原始历史，Client 在每个 UDP 包到达时处理并保存对应结果；连续有效包条件下采用相同因果递推。
- 采集使用本机 `time.monotonic()`，Windows Client 使用统一 `clock.now()/perf_counter()`。两边均用同一台主机的单调钟做本机比较，不应跨机器比较绝对数值。

证据：[共享契约](src/realman_force/contract.py#L18)、[原生包滤波及模型取样](src/realman_sync_client/hardware.py#L107)、[采集取样](https://github.com/shengbaoli25/realman-data-collection/blob/762e4a2bb4ff29c6a7ec447592f92688990b8237/realman_leader_reader/collect.py#L321)。

仓库没有入库现场 `local_setup/force_calibration/active.json`，因此不能从权威 Git commit 验证当前机器实际使用哪份标定。上述哈希一致证明 Client 与当前下载的训练契约一致，不证明当前 TCP、World frame、工具负载或传感器零偏仍符合标定条件。

## 差异处理结果与仍需区分的项目

1. **已关闭：相机自动曝光与内部队列设置。** Client [camera.py](src/realman_sync_client/camera.py#L38) 已按采集端补充 `enable_auto_exposure=1`、`auto_exposure_priority=0`、`frames_queue_size=1`，保存旧值并在退出恢复；两边都预热15帧。主代理报告真实相机选项读回为 **1/0/1**，原始实测 receipt 另存。本审计复读确认实现存在；该选项读回不等于证明所有场景都能稳定30fps。

2. **已关闭：夹爪生产参数不一致。** 权威 `leader_gripper.py:842` 的真实 worker 构造 `Config(speed=100,torque=50)`，并非示例值。Client hardware.json 已改为 **100/50**。主代理已在显式恢复流程中将寄存器参数对齐并真实读回 **[100,50,100,100]**（速度、力矩、加速度、减速度）；随后 `measurement_ready=true`、夹爪仍失能。原始实测 receipt 另存；这些操作不是由本审计代理执行。

3. **已关闭：网页入口 frames/DH 核验缺项。** [WebHardware._verify_frames_and_dh](src/realman_sync_client/web_hardware.py#L317) 已根据绑定力标定 context 核验当前工具/工作坐标系名称与 identity pose、SDK和控制器全部 DH 参数一致、物理 DH d 匹配。主代理报告实机核验通过，receipt 另存。此处仅关闭 frames/DH 缺项；没有把它扩大为采集端另有的“FK与控制器TCP误差比较”也已在新入口实现，更不证明实际工具负载、传感器零偏长期不变。

4. **已关闭：网页反馈入口系统故障门控。** [web_hardware.py](src/realman_sync_client/web_hardware.py#L96) 已检查 `state.err.err[:state.err.err_len]` 及数组长度；发现系统故障会拒绝该包、清理模型历史并重置滤波状态。本审计复读确认逻辑已存在；未在真机人为注入故障。本修正范围是网页入口的 `_MonitoredFeedback`，并非声称未改动的基础 CLI 入口也自动获得了该检查。

5. **夹爪调度和物理响应不同，不能直接删安全限制。** 权威生产 `RetargetingTrigger` 参数为最小间隔10ms、基础死区0.6%、移动死区1%、追赶合并200ms、稳定60ms；`gripper_schedule.py:43` 的反馈软间隔活动120ms/空闲700ms、反馈开始预算300ms、目标核验延迟400ms。Client 是独立 worker、100ms最小命令间隔、每次最大开度步长0.15。新版每轮先读使能和连续位置/报警块，再写目标；100ms轮询期限从读取开始计算，慢读完成后不再额外等待100ms。它接收模型 action chunk，采集调度接收操作者连续输入，不能机械照搬。保留 Client 约束并记录实际已确认目标/时延；未经专项验证，不能声称动作波形、夹持后张开策略或执行频率与采集相同。

6. **夹爪观测年龄已对齐，运动与中断门控仍有区别。** 采集 `collection_telemetry.py:87` 在夹爪年龄超过1秒时拒收，`force_recording.py:70` 的默认最大年龄也是1秒，`lerobot_recording.py:118` 验证保存年龄在0..1秒。当前 Client 的 `gripper_feedback_max_age_s=1.0` 已对齐这一观测上限；网页执行就绪、新动作接收及手动操作仍要求位置反馈不超过0.5秒。状态中的 `measurement_ready` 与运动 `ready` 分开：失能但无报警的真实开度可观测，不可由模型执行。采集相机年龄120ms、两相机差50ms；Client 除此还要求每路相机与力/关节包差不超过50ms。采集错过30Hz槽会跳过并记录 discontinuity，训练切断这些区间；Client 在开始采集历史后网格 overrun 会停止，不伪造补帧。

7. **诊断字段及夹爪时间定义不同。** 采集同时保留未补偿 `sensor.force`、反馈开始/完成时间和已确认 action 时间；Client 模型只需已处理force，不保留完整raw诊断。采集夹爪样本以事务开始时间描述年龄，并要求完成时间不晚于力包；Client 以读回完成时间存入历史，也要求不晚于力包，所以无未来泄漏，但年龄统计不能直接当成完全同口径。新版网页的开度年龄、Hz和间隔均从位置历史计算，使能 ACK 不会刷新位置时效，HTTP轮询也不会产生伪采样。`gripper_poll_s=0.1` 是从读取开始计时的目标周期，并非保证10Hz；界面使用实际 `hz/last_interval_ms/max_interval_ms`。

8. **尚未证明物理相机角色/标定与训练现场一致。** 权威仓库相机配置是序列号占位示例；当前 Client 固定 global=`REPLACE_GLOBAL_CAMERA_SERIAL`、wrist=`REPLACE_WRIST_CAMERA_SERIAL`。逻辑顺序及上述曝光控制选项已核对，真实安装位置、镜头参数、场景和训练时的成像分布没有在本审计中验证。其它四个模型的输入契约仍须各自从对应 normalization/config 核对。

9. **已修复：暂停后旧 collector 错误污染恢复。** 主代理真实诊断曾遇到暂停期间 collector 锁存“夹爪反馈过期”，而后来原生反馈已经恢复健康。`webapp.py` 现在暂停时关闭并释放模型 collector，保持原生传感器监测；恢复时创建新会话和完整32步历史。新增离线回归确认旧 collector 关闭、旧错误不复用、原生硬件对象保持连接、恢复后能完成推理且不发动作。本次仅运行网页后端测试：**33项通过，1.070秒**。

生产配置证据：[相机设置与恢复](https://github.com/shengbaoli25/realman-data-collection/blob/762e4a2bb4ff29c6a7ec447592f92688990b8237/realman_quest_teleop/scripts/camera_worker.py#L131)、[夹爪生产入口](https://github.com/shengbaoli25/realman-data-collection/blob/762e4a2bb4ff29c6a7ec447592f92688990b8237/realman_leader_reader/leader_gripper.py#L842)、[生产总线调度](https://github.com/shengbaoli25/realman-data-collection/blob/762e4a2bb4ff29c6a7ec447592f92688990b8237/realman_leader_reader/gripper_schedule.py#L43)、[frames/DH 校验](https://github.com/shengbaoli25/realman-data-collection/blob/762e4a2bb4ff29c6a7ec447592f92688990b8237/realman_quest_teleop/scripts/quest_rm75_local.py#L235)、[系统故障门控](https://github.com/shengbaoli25/realman-data-collection/blob/762e4a2bb4ff29c6a7ec447592f92688990b8237/realman_quest_teleop/scripts/realtime_feedback.py#L34)。

## 验证状态

本审计代理完成源码阅读、Git版本确认、纯配置契约计算、规范化文本比较及上述离线网页回归；没有操作硬件或改动训练/硬件实现。主代理另行执行并保存真实验证 receipt，本报告将其结果明确列为主代理实测，并以复读代码确认实现范围。相机选项、frames/DH、系统故障门控及夹爪参数四项原缺口已按上文范围关闭；调度差异、物理成像/标定适用性及其它模型契约仍保留明确边界。不能将这些检查解释为已完成动作执行或所有模型真机测试。

本轮文档复读了以下原始记录，没有重新连接硬件：

- [gripper-recovery-confirmed.json](diagnostics/gripper-recovery-confirmed.json)：恢复及参数对齐完成，`1837.06 ms`，实测参数 `[100,100,100,100]→[100,50,100,100]`，夹爪仍失能、观测就绪。报警前后均0且 `reset_written=false`；这不是本轮“非零报警清零”的实机证据。
- [real-observation-inference-confirmed.json](diagnostics/real-observation-inference-confirmed.json)：真实 Cosmos 观测请求累计11次，当前返回32×8 action，记录计算耗时约2.156–2.484秒；state/force199.82Hz、两相机29.97/29.98fps，当前Arm_Tip/World和DH核验通过。该快照 `execution.enabled=false`、夹爪失能，未执行模型动作。

上述实机快照早于本轮1秒观测阈值与手动控制按钮更新。之后用户自行使能和开合，主代理观察到实测开度1→0→1；用户又两次启动模型执行，真实日志包含policy目标写入，随后在18:45:39、18:46:37因夹爪未就绪暂停。位置反馈间隔观察到664.33ms，超出500ms运动限制。实际动作已经发生，但不能因此宣称稳定运动部署已经完成。

本轮为该故障优化了串行总线调度，并增加关节写入前的夹爪500ms检查，保留原后置检查。新 worker 9项、网页硬件适配器38项离线测试通过：覆盖连续目标不饿死反馈、合并寄存器偏移、停止和hold优先、失败读取不写目标、ACK不刷新位置、已取消派发不推进已发送位置。每次事务90ms的虚拟测试反馈间隔≤270ms，仅证明该测试条件下的调度。已经提交到SDK的写入无法撤回，两路硬件写入也不是原子事务；停止后不会继续排队发送已取消目标。

新版后续 [continuous-runtime.json](diagnostics/continuous-runtime.json) 有60.218秒、117次只读状态采样，包含用户执行。常态约8.3Hz/120ms已观察到，但18:58:53仍因真实位置年龄521.52ms触发前置guard，拒绝该次关节写入并暂停。暂停阶段也出现超过1秒的年龄及1316.20ms间隔，故冗余读取优化有效不等于长期运动稳定性已通过。

对采集端的后续只读核对显示：生产 worker 完成SDK初始化后改用独立JSON Modbus socket，且在独立进程中运行；其 `gripper_health_error` 不将单独的位置年龄作为停臂条件。该部署仍保留500ms运动门控，不能直接照搬该策略。源仓库 JSON 桥接基准也有约504ms超时和500–702ms失败事务，切换JSON不能保证消除物理或控制器长延迟。本轮没有新增JSON通路。SDK `rm_set_timeout` 是全局设置，初始化/缓停的不同值需要后续统一管理，但当前证据不足以将其认定为根因。完整说明见 [GRIPPER_TIMING_20261101.zh.md](diagnostics/GRIPPER_TIMING_20261101.zh.md)；本次文档更新没有发出任何动作或控制请求。
