# Windows 同步推理部署文件

本目录保存已验证的 Windows 启动与资产校验工具。它是 `infer` 布局的安装材料，**不能直接在仓库 `deploy/windows` 中执行启动器**。模型训练源码和训练配置未修改；不包含 checkpoint、tokenizer、编码器、文本缓存、实际下载收据或服务密钥。

## 放置位置

将本目录的脚本、requirements 和 tests 复制到自行选择的 `infer/runtime/`；将 `code-layout.example.json` 放到 `infer/code-layout.json` 并填写本机绝对路径、实际代码版本。保留实际推理提交 ID，示例中的 commit 是补丁的训练源码基线。

```text
infer/
  code-layout.json
  .venv-server/
  runtime/                 本目录脚本安装到这里
    upstream/cosmos-predict2.5/
  cosmos/
    code/                  当前 Cosmos 分支的完整源码
    config/                原训练配置、normalization、经校验的 runtime JSON
    checkpoints/           训练 checkpoint 与 tokenizer
    assets/text/           匹配该任务的 manifest 与文本张量
  vjepa-vitb-28l/
    code/                  V-JEPA ViT-B 分支的完整源码
    config/
    checkpoints/
    assets/
  client/                  独立 Client 分支源码
```

Cosmos 上游源码须为 `a2c298b0a3df3778b973fe65e9e58877b292d8a7`；按仓库原有上游校验检查，不以其他版本替换。未使用的实验可以不列入 code-layout.json。使用已有环境时显式传 `-Venv`，避免依赖本机原工作区的路径回退。`server.py check --verify` 另外要求由实际安装目录生成的 `code-inventory.sha256.json`；本发布不携带旧主机的清单。`check_assets.py` 需要本机独立维护的 assets-manifest.json。

## 启动

使用已安装的 Python 3.12 环境；新环境安装命令为 `native-server.ps1 -Action Install -BootstrapPython <python.exe> -Venv <venv>`。这会安装推理依赖，不安装完整训练环境。服务认证通过进程环境变量 `REALMAN_SERVER_TOKEN` 提供，Client 必须使用同一值；不要写入源码、命令行参数或 Git。

在 `infer` 目录，先做仅打印的启动检查：

```powershell
& .\runtime\native-server.ps1 -Action Serve -Experiment cosmos -Config .\cosmos\config\runtime.step_020000.json -Checkpoint .\cosmos\checkpoints\step_020000.pt -CheckpointBootstrap -Venv .\.venv-server -DryRun
```

确认路径、真实资产和配置后去掉 `-DryRun`。默认仅监听 `127.0.0.1:8766`；一次运行一个 Server，启动前确认端口未被另一模型使用。`-CheckpointBootstrap` 选择新增的完整训练 checkpoint loader；仍需匹配的 tokenizer、任务专属文本、normalization、force 合同及配置身份字段。不会自动连接机器人。

Cosmos 档位：`161616`（16/16/16）、`4416`（4/4/16）、`3310`（3/3/10）。运行配置中的训练模型构造参数 `num_inference_steps` 和 `joint_steps` 仍保持16；只改变部署采样调度，Windows 路径关闭编译。V-JEPA 档位是 `dense16` / `dense10`，不能套用 Cosmos 的三元步数。

资产下载和校验见 [Cosmos](COSMOS_ASSET_DOWNLOAD.zh.md)、[V-JEPA ViT-B](VJEPA_VITB_ASSET_DOWNLOAD.zh.md)。这些说明不提供免登录下载地址，不代表新机器已经完成校验或真机测试。
