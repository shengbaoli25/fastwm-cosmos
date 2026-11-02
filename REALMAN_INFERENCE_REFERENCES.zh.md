# RealMan 具身推理链路参考与本机适用性

核查日期：2026-11-01。本文基于已完成的官方文档与公开源码审阅，不代表这些项目已在本机或当前夹爪上通过实机验证。调研期间未向硬件发送控制命令。

本机目标链路：Windows 本地 Server 推理；Client 采集相机、机械臂状态与力，按同步的 observation → action chunk → 执行流程工作。模型使用 `joint8`：前七项为绝对关节角度，单位 rad；最后一项为夹爪开度，范围 `[0, 1]`。控制节拍为 30 Hz，训练动作 horizon 为 32；夹爪为接在机械臂末端 RS485 的 CTAG2F120。

## 1. RealMan 官方协议与运动接口

- [第三代控制器 JSON 协议版本说明](https://develop.realman-robotics.com/robot/json/readme/)
- [Python 运动接口：rm_movej_canfd](https://develop.realman-robotics.com/en/robot/apipython/classes/movePlan/)
- [JSON 运动指令](https://develop.realman-robotics.com/en/robot/json/motionConfig/)
- [JSON Modbus 指令](https://develop.realman-robotics.com/robot/json/modbus/)
- [第三代控制器系统与版本查询指令](https://develop.realman-robotics.com/robot/json/systemConfig/)
- [控制器版本说明与配套版本表](https://develop.realman-robotics.com/robot/releaseNotes/releaseNotes/)

### 可以采纳的约束

1. `rm_movej_canfd` 的 SDK 边界使用角度制。模型输入输出继续保留 rad，仅在硬件 SDK 边界转换，不能因为参考项目直接读写 degree 而改变训练契约。
2. 高跟随模式要求较短的连续发送周期，官方接口说明要求周期不超过 10 ms。本机 30 Hz 的周期约 33.3 ms，应继续使用低跟随 `follow=false`，不能把其他示例中的 `follow=true` 当作通用提速选项。
3. JSON 协议版本说明的 **3.7.3，日期 2024-09-27**，记录了低跟随透传改用在线轨迹规划，并解除此前目标角度差不超过 10° 的限制。因此旧运动说明中的 10° 条件不能直接当作所有固件、所有跟随模式的永久限制。
4. 仍应校验本机真实关节软限位、数值有效性、控制器错误、急停与命令有效期。解除应用层 8° 阈值不意味着取消这些检查。

### 版本号不能跨命名空间直接比较

JSON 文档版本 `3.7.3` 与控制器 `ctrl_info.version` / `plan_info.version` 的 `V1.x.x` 不是同一套版本编号。不能使用字符串或数值大小比较，直接得出“本机版本高于 3.7.3”或“已具备该能力”。应保存实际返回的固件版本和编译时间，再结合厂家配套版本表、发布说明确认功能对应关系。本文不声称已完成这一步实机核验，也不建议为此自动升级固件。

官方支持以下只读查询：

```json
{"command":"get_arm_software_info"}
{"command":"get_tool_software_version"}
```

第一项示例响应的 `command` 是 `arm_software_info`，包含 `ctrl_info`、`plan_info` 等字段；第二项第三代控制器示例响应使用 `state: "tool_software_version"`。接收匹配不能假定所有响应都原样回显请求名称。

### 末端 Modbus 对本机的意义

- `port=1` 对应末端接口板 RS485；`set_modbus_mode.timeout` 单位为 100 ms，值 `2` 表示 200 ms，而不是 2 ms。
- 官方提供批量读写寄存器指令，可在设备寄存器映射允许时减少事务次数；不能把不连续、语义不同的 CTAG 寄存器任意拼成批量读取。
- 官方示例通过 `read_state` / `write_state` 表示失败，没有提供可用于任意并发请求的事务 ID。当前单一通信所有者、按请求顺序完成事务的设计应保留。
- 本次未查到厂家明确规定的“JSON 到末端 Modbus 请求最小间隔”或推荐轮询 Hz。通用 Modbus RTU 的字符间隔不能直接作为本机 50–70 ms 事务延迟或掉线的已证实根因。

## 2. FlagOpen / RoboCOIN

该项目是 RoboCOIN 的原始项目仓库，提供包括 RealMan 在内的机器人适配与模型部署代码；它不是 RealMan 厂家的底层驱动保证。

固定审阅 commit：`b3261fe7cf92d18d9f8545c4b8ad9813dd1d2edd`。

- [RealMan 适配器，重点 L58–95](https://github.com/FlagOpen/RoboCOIN/blob/b3261fe7cf92d18d9f8545c4b8ad9813dd1d2edd/src/lerobot/robots/realman/realman.py#L58)
- [OpenPI 同步客户端，重点 L179–226](https://github.com/FlagOpen/RoboCOIN/blob/b3261fe7cf92d18d9f8545c4b8ad9813dd1d2edd/src/lerobot/scripts/server/robot_client_openpi.py#L179)
- [动作队列与异步客户端，重点 L488–737](https://github.com/FlagOpen/RoboCOIN/blob/b3261fe7cf92d18d9f8545c4b8ad9813dd1d2edd/src/lerobot/scripts/server/robot_client.py#L488)

### 值得采纳的逻辑

- 将 observation 和 action 的特征顺序作为显式契约，而不是根据数组长度猜测机械臂型号。
- 将序列化、网络请求、动作获取、实际执行分别计时；用动作 timestep 标识已经执行与尚未执行的部分。
- 动作队列锁只覆盖队列操作，硬件调用放在锁外，避免阻塞生产者或状态读取。
- 未来如果明确改为异步，可参考按 timestep 排除过期动作与替换未来动作的做法。当前同步链路仍应依靠请求序列、会话代际和停止状态，阻止旧请求结果在重启后继续执行。

### 不能照搬的部分

- RealMan 适配器使用规划运动 `rm_movej`，随后调用 `rm_set_gripper_position`；状态获取使用 `rm_get_joint_degree` 和原生 `rm_get_gripper_state`。这不是本机 30 Hz CANFD 透传与 CTAG 末端 Modbus 的执行方式。
- 其原生夹爪位置尺度与本机 `[0, 1]` 开度不同。不能把原生 gripper API 当成 CTAG 协议适配器，也不能让原生夹爪的阻塞等待进入本机每个 33.3 ms 控制周期。
- OpenPI 示例的动作处理硬编码访问 `action[6]` 和最后一项作为夹爪相关量。本机 `action[6]` 是第七个关节，照搬会直接篡改关节目标。
- OpenPI 示例在整块动作循环之后才按 `frequency` sleep，不能据此保证块内 30 Hz；本机应保持逐步扣除硬件调用耗时的节拍。
- 异步客户端的 gRPC、队列聚合和部署依赖不能作为本次同步 HTTP 链路的替换模板。借鉴其计时与有效期逻辑即可。

## 3. QinDeyun / act-plus-plus-realman

这是作者提供的 RealMan ACT / Diffusion Policy 实验代码，属于直接源码参考，不是厂商认证实现。

固定审阅 commit：`9bcb960c4b131e2dc40dcfbcd995accee559d886`。

- [相机与机械臂状态采集线程，L244–274](https://github.com/QinDeyun/act-plus-plus-realman/blob/9bcb960c4b131e2dc40dcfbcd995accee559d886/imitate_episodes_deploy.py#L244)
- [推理预热、动作块和执行循环，L475–576](https://github.com/QinDeyun/act-plus-plus-realman/blob/9bcb960c4b131e2dc40dcfbcd995accee559d886/imitate_episodes_deploy.py#L475)
- [默认 FPS 与任务常量](https://github.com/QinDeyun/act-plus-plus-realman/blob/9bcb960c4b131e2dc40dcfbcd995accee559d886/constants.py)

### 值得采纳的逻辑

- 推理预热应在正式执行前完成，单独记录首次加载与稳定推理延迟。
- 采集与模型计算分离；只消费所需的新数据，避免无限积压图像队列。
- 控制周期等待应扣除本步已经消耗的时间，并记录超期，而不是在耗时操作之后再固定 sleep 一个完整周期。

### 不能照搬的部分

- 其相机帧与机械臂状态放入两个独立的最新值队列，没有提供与本机等价的采集时间戳配对保证。不能用这种做法替换本机相机/关节对齐与历史帧选择。
- 默认 FPS 为 50，但部署中调用 `rm_movej_canfd(..., True, ...)`，且该处仍留有透传验证 TODO。不能把这些参数当作当前 30 Hz 系统的可靠控制模板。
- 该文件还保留了其他机器人/任务的动作切片、常量与初始化姿态。不能复制其中初始位姿、动作维度处理、夹爪映射或固定 IP 后直接运行。
- ACT 的时间聚合会改变执行轨迹；本机当前模型的整块动作语义、动作时间网格与训练归一化必须保持一致，不能仅为“平滑”直接套用。

## 4. 对当前链路的落地建议

### 把跟踪误差与命令跳变分别记录

审阅时，本地执行路径的“关节跳变”计算的是当前目标与最新实测关节角之间的差；chunk 预检计算的则是相邻目标差，两处共用同一阈值。这会把控制器跟随落后与模型轨迹突变混为一谈。解除应用层 8° 拦截后，建议保留两项独立诊断：

- `tracking_error_deg`：当前目标与实测关节角的差，并记录反馈时间。
- `adjacent_command_delta_deg`：当前目标与上一条实际发送目标的差，并记录命令间隔。

第一条命令还应明确标记为新会话/新动作块起点，不能把跨停机阶段的差直接归为连续轨迹跳变。真实关节软限位与控制器状态检查继续生效。

### 按串行事务预算安排夹爪读写

本机日志观察到单次事务约 50–70 ms；据此估算，一个串行通信通道在不计额外等待时也仅能完成约 14–20 次事务/秒。位置 20 Hz、使能 4 Hz，再加动作写入不能被视为同时可满足的独立频率。这是基于本机测量的容量估算，不是厂家发布的性能上限。

适合本机的优化是：单 worker 保持事务顺序；合并尚未发出的重复位置目标；在持续动作写入时仍给位置和使能反馈保留到期执行机会；超时之后区分通信状态、反馈新鲜度与目标是否执行。不能用目标位置覆盖实际读回位置，也不能因为写入结果不确定而盲目重发运动命令。

### 保持同步执行语义，分别显示三种时间

32 个动作以 30 Hz 执行，名义动作块时间约为 `32 / 30 = 1.067 s`。同步模式还需加上采集、编码、请求、推理及停止/刷新等耗时，所以“块内执行 30 Hz”不等于整条链路无间断地执行 30 Hz。

网页与实验日志宜区分：Server 推理耗时、动作块内执行周期、完整 observation → action → 执行周期。还应单独记录相机帧龄/对齐误差、机械臂反馈龄、夹爪事务延迟与最近成功读回时间，以便定位阻挡点。当前已具备的严格时间戳对齐与周期扣时应保留，不因公开示例较简单就退回无时间戳的“取最新值”。

## 5. 本文边界

本文固定了参考源码版本与适用性判断，没有改变训练代码、模型归一化、模型采样步数或硬件控制配置，也没有证明任何示例在当前 RM75、CTAG2F120 与 Windows 组合上可直接运行。后续本机实验结果应独立记录，不能以公开仓库存在作为通信稳定性或任务成功率的证据。
