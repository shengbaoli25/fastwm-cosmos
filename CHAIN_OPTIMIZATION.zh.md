# 真机推理链路调整记录（2026-11-01）

运行目录：`C:\work\infer\client`。本轮只修改 Client，未改训练代码、模型权重和归一化。Server 为本机 V-JEPA ViT-B 20k，当前采样为 dense10。

## 已部署

1. `config/hardware.json` 的 `max_joint_step_deg` 设为 `null`，关闭两个本地跳变判定。原先其中一个判定实际比较“当前实测角度到目标角度”，会把控制器跟踪滞后也计为跳变。关节范围、有限值、急停和反馈有效性检查仍生效。
2. JSON 夹爪通信在目标写入前优先读取使能和位置，依据最近事务耗时为后续反馈读取预留时间。目标始终取最新值，预算不足时推迟写入。当前位置和使能的 500 ms 动作新鲜度要求没有放宽。
3. 修正超时恢复时复用旧接收截止时间的问题：等待旧回复后，新发出的只读同步请求获得自己的事务时间预算，仍受整个 2 秒恢复窗口约束。不重发状态不确定的运动写入。
4. `start-web.ps1` 默认将运行日志写入 `diagnostics/client-runtime.log`，20 MiB 轮转，保留 3 份备份；RTU 明细另存 `diagnostics/gripper-transactions.jsonl`。
5. `Inference cycle` 记录观测等待、请求构建、Server 推理、HTTP 往返、动作执行和整个周期时间。`Motion trace` 分别记录实测跟踪误差与相邻目标差值；`feedback_schedule` 记录反馈预算和目标推迟原因。

## 验证及实际限制

- 全量 254 项测试通过，见 `diagnostics/chain-all-tests-20261101.log`。
- 使用实际 worker 的模拟通信回放：60 ms 读取/70 ms 写入时，6 秒发送 31 次目标，反馈最大年龄约 250 ms；单次 200 ms 慢读时最大约 390 ms。它是软件回放，不等于真实夹爪运动验证。
- 22:12:48 重启 Client 后，连接真实硬件、仅推理的 dense10 检查完成 41 次请求；约第 39.5 秒仍发生“夹爪运动反馈过期”，随后通信恢复。见 `diagnostics/chain-live-dense10-20261101.json`。因此不能宣称通信已完全稳定。
- 这次失败时相机约 30 FPS，机械臂/力反馈约 191 Hz；夹爪位置年龄达到约 1348 ms，transport 报 `Controller response timeout`。这不是相机对齐错误或本地 8° 阈值。
- 用户开始真机实验后，后台验证已经结束；不再自动切换控制状态或重启进程。

## 如何读日志

- `Motion trace.measured_tracking_error_deg`：实测位置与当前指令差值。
- `Motion trace.adjacent_command_delta_deg`：相邻两条实际下发的关节目标差值。
- `Inference cycle.cycle_ms`：一次同步观测→推理→动作执行的总时间。32 个动作按 30 Hz 执行约需 1.07 秒，还需加上推理/通信时间；30 Hz 是动作块内频率。
- RTU 的 `stage=receive` 超时与 `stage=sync_receive`/恢复等待超时要分开统计，同一次失联可能生成多条超时记录。
- 恢复只读通信不能证明之前的运动写入是否成功；对于不确定写入保持取消，不自动重放。

已审查的 RealMan 官方文档和公开推理实现见 `REALMAN_INFERENCE_REFERENCES.zh.md`。本轮吸收了阶段计时、最新目标及有限队列等思路；没有直接照搬其他机器人维度或高跟随参数。
