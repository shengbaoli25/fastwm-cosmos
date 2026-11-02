# π0.5 · RM71D78 消融实验

分支：`codex/ablation-rm71d78-pi05`。基于原始训练仓库提交
`057f78767f52f358b229f4a05b097c0f1455fa3d`，独立提供官方 π0.5 的
78 条插头换插座数据适配、训练/恢复入口及 RealMan HTTP server。

**训练尚未启动，真实权重加载和 GPU 训练/恢复尚未验证。** 当前分支的本地检查结果见
[validation.json](docs/ablation/validation.json)。默认命令只显示计划：

```bash
python scripts/ablation/pi05.py --config configs/ablation/pi05.example.json
```

- [准备数据、训练、导出、server 与回放指南](docs/ablation/README.zh.md)
- [π0.5 官方环境、模型适配及配置说明](docs/ablation/pi05.md)
- [官方源码、权重来源和数据身份](configs/ablation/upstream-lock.json)

数据源为 `rm71d_plug_transfer_20261029`，原始 78 条轨迹，任务“拔下插头并插入另一插座”。
输入采用 `global/wrist` 双相机与 joint8 状态，模型不使用力输入；输出为 30 Hz、32×8 的
绝对关节弧度与夹爪开度。HTTP 协议兼容现有 `realman-sync-v1` 客户端。
