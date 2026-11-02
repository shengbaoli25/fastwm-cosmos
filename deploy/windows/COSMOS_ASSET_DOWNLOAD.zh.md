# Cosmos 资产下载与校验

复用已登录且能访问训练平台的外部浏览器，在项目文件输入完整目录并核对面包屑、任务与文件名。训练配置中的文本缓存路径是选择任务专属文本的依据；不要用同名公共缓存替代。文件点击后没有保存框也可能已经落在浏览器 Downloads，先核查，避免反复下载。

下载到对应模型目录：训练 checkpoint 和官方 tokenizer 放 `cosmos/checkpoints`；原训练 YAML、normalization 及 force 合同放 `cosmos/config`；匹配的 manifest 与文本张量放 `cosmos/assets/text`。相同文件名的其他模型必须分目录。保留训练资产原样，不生成替代 normalization。

网页提示成功不是落盘证据。等待对应 `.crdownload` / `.crswap` / `.part` 消失，正式文件非零，再校验精确大小、完整 SHA256、checkpoint ZIP CRC、模型 manifest 和文本身份。不要重命名临时文件或覆盖未知的同名文件。

安装到 `infer/runtime` 后，把 `COSMOS_ASSETS.lock.example.json` 复制为本地 `COSMOS_ASSETS.lock.json`。根据可信训练输出填写实际绝对远端路径、相对目标路径、精确字节数、SHA256 和 text_identity；示例含故意无效的占位值，不能当作已验证清单。不要提交生成的本机锁、校验报告或登录凭据。

```powershell
python -X utf8 .\runtime\cosmos_asset_download.py plan
python -X utf8 .\runtime\cosmos_asset_download.py import --asset training_config --source <已下载的准确本地文件>
python -X utf8 .\runtime\cosmos_asset_download.py check --crc --finite
```

工具只提供计划、已完成本地文件导入、校验；没有提取 cookie/token 或请求私有签名 URL 的功能。import 校验后原子发布，拒绝覆盖不同内容。文本有限值检查专门对应当前 Reason1 BF16 [512,100352] 合同，不能当作其他缓存格式的通用验证器。`open_cosmos_download.ps1` 默认只显示流程；显式 `-LaunchBrowser` 才打开用户已有的专用 Edge 快捷方式，不修改代理设置。
