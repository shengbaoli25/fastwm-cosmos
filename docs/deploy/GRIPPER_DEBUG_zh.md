# CTAG2F120 夹爪调试

推理页面新增「夹爪调试」。本机连接 RM75 末端 RS485 上的 CTAG2F120，沿用 runtime 指定的 hardware_config；不依赖加载模型、CUDA 或双相机。独立调试与完整硬件共用一个夹爪通道，不另起进程争用 Modbus。

## 页面操作

1. 暂停推理，等待正在执行的推理结束，再点击「连接夹爪」。如报警造成旧连接退出，可重新连接。
2. 读取状态，查看报警码、使能状态、开度与用户位置。连接会初始化末端供电/Modbus 通讯，但不会自动清错、使能或发送开合目标。
3. 有报警时排除对应原因，再点击「清除报警」。此操作先确认失能，写入厂商报警复位寄存器 `0x0403`，回读报警状态；成功后仍保持失能，不执行回零。
4. 点击「使能」前会用当前实测位置替换残留目标，防止重新使能后执行上次目标。操作必须得到真实使能回读。
5. 「打开」是开度 100%，「闭合」是 0%；也可设置目标百分比、速度和力度。移动按配置限幅，并等待反馈确认。处于力矩保持状态时，张开操作会先确认失能，再以实测位置重新使能后张开。超时或通信失败会显示失败，不自动重发运动命令。
6. 「失能」解除夹爪驱动使能并验证回读。暂停或软件急停可以打断调试，清除未执行目标；急停锁存后禁止新的调试动作。

开合、使能会影响实机。先确认夹爪附近无手指、工件状态允许操作；失能可能释放夹持。运动命令只在操作者点击后发送，不随页面刷新执行。

推理运行、模型加载、其他硬件操作期间调试按钮禁用；调试期间禁止启动模型闭环。清错只处理夹爪报警，不解除软件急停，也不自动恢复模型控制。若之前推理因夹爪错误退出，恢复夹爪后先点「复位错误 / 保持停止」，确认全部设备就绪再启动。

## 命令行

先启动 `serve_local.sh` 或 `deploy/serve_realman.py`。下列命令调用同一服务端保护逻辑，不直接建立第二条硬件连接。逐条按需执行。

```bash
python deploy/gripper_debug.py status --cached
python deploy/gripper_debug.py connect
python deploy/gripper_debug.py status
python deploy/gripper_debug.py clear_error
python deploy/gripper_debug.py enable
python deploy/gripper_debug.py open --speed 30 --torque 50
python deploy/gripper_debug.py move --opening 50 --speed 30 --torque 50
python deploy/gripper_debug.py close --speed 30 --torque 50
python deploy/gripper_debug.py disable
```

API：`GET /api/gripper` 读缓存；`POST /api/gripper` 接收 `action`，移动可附 `opening`（0..1）、`speed` 和 `torque`（1..100）。原有硬件 readiness、训练力处理身份以及推理保护保持有效。

协议依据：本机采集系统的 `ConfirmedStreamingGripper.clear_alarm / set_enabled / enable_at_measured_position`，及厂商随设备提供的 Changingtek RTU SDK。固件报警位按实际返回显示，未知位原样保留。
