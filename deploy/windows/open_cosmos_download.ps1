param([switch]$LaunchBrowser)
$ErrorActionPreference = 'Stop'
$shortcut = Join-Path $env:USERPROFILE 'Desktop\启智直连 Edge.lnk'
Write-Output '下载方式：登录平台后，从项目文件列表手动下载；不提取凭据、不生成下载接口 URL。'
Write-Output '清单与本地验证：python -X utf8 cosmos_asset_download.py plan'
Write-Output '来源依据：diagnostics/edge-direct-shortcut.json、edge-direct-runtime-verification.json 和 download-receipts.json。'
if (-not $LaunchBrowser) {
    Write-Output '当前为说明模式；指定 -LaunchBrowser 才打开已有的专用直连 Edge 快捷方式。'
    exit 0
}
if (-not (Test-Path -LiteralPath $shortcut -PathType Leaf)) {
    throw '已验证的启智直连 Edge 快捷方式不存在。请使用当前已登录的外部 Edge；脚本不会创建新代理或浏览器配置。'
}
# Explicit interactive invocation only. No profile/cookie files are inspected.
# This launch opens the existing platform shortcut, not a private download URL.
Start-Process -FilePath $shortcut -WindowStyle Normal
