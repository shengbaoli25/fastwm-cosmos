# V-JEPA ViT-B 28L 资产与部署

本适配器对应训练源码基线 `9810104d6ffb44f22f485c5f1e9282eabbdebf06`（`codex/rm71d78-vjepa-vitb-28l-50k`），不是旧 V-JEPA 分支。

从训练任务目录下载匹配的 `step_020000.pt`、原始训练 YAML、normalization 和力预处理合同；从官方来源取得训练 YAML 指定 SHA 的 ViT-B 编码器。按 `infer/vjepa-vitb-28l/{checkpoints,config,assets}` 分目录保存，不与 Cosmos 同名文件混放。此模型不使用 Cosmos tokenizer 或 Reason1 文本缓存。

安装此分支 `deploy/windows` 中的 `verify_checkpoint.py`、`prepare_runtime.py` 和测试到 `infer/vjepa-vitb-28l`。它们为这一已固定实验校验身份，不能修改常量来绕过其他 checkpoint 的不匹配。

```powershell
$py = '.\.venv-server\Scripts\python.exe'
& $py -X utf8 .\vjepa-vitb-28l\verify_checkpoint.py
if ($LASTEXITCODE -ne 0) { throw 'checkpoint verification failed' }
& $py -X utf8 .\vjepa-vitb-28l\prepare_runtime.py --write-runtime
if ($LASTEXITCODE -ne 0) { throw 'runtime verification failed' }
& .\runtime\native-server.ps1 -Action Serve -Experiment vjepa-vitb-28l -Config .\vjepa-vitb-28l\config\runtime.step_020000.json -Checkpoint .\vjepa-vitb-28l\checkpoints\step_020000.pt -Venv .\.venv-server -DryRun
```

先通过 ZIP/全 SHA/CPU 严格 state 和有限值检查，再启动真实 Server。生成的 runtime 配置与报告留在本机，模型资产和报告不随源码发布。使用与 Client 一致的 REALMAN_SERVER_TOKEN 环境变量；确认路径后去掉 DryRun。视觉单次前向，力/动作可选 `dense16` 或 `dense10`。下载与推理检查都不等于机器人任务成功率验证。
