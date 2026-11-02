# V-JEPA ViT-B Windows 布局辅助文件

`prepare_runtime.py`、`verify_checkpoint.py`、`test_prepare_runtime.py` 原样来自已验证部署。把三者复制到 `infer/vjepa-vitb-28l/`，当前分支完整代码放到该目录的 `code/`；**不能直接在 deploy/windows 中运行**，脚本按自身目录寻找 code/config/assets/checkpoints。

通用启动器使用 Cosmos 部署分支 `deploy/windows` 的布局包，安装到 `infer/runtime/`。先执行 verify_checkpoint.py，再用 prepare_runtime.py --write-runtime 进行 CPU 严格模型加载、manifest 和张量检查，成功后才用 native-server.ps1 启动 `-Experiment vjepa-vitb-28l`。脚本固定于本实验20k身份；不要更改 hash/身份常量绕过错误。模型、实际训练配置/统计、报告与服务密钥均保留本机。

新增 dense 推理适配器支持当前28层ViT-B合同和 `dense16` / `dense10` 两种等步数力/动作采样。训练源码、模型结构、训练权重都保持原样。
