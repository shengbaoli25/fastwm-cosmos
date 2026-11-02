param(
    [string]$Server = 'http://127.0.0.1:8766',
    [int]$Port = 8767,
    [ValidateSet('realman','mock')][string]$Mode = 'realman',
    [string]$HardwareConfig = (Join-Path $PSScriptRoot 'config\hardware.json'),
    [string]$TokenFile,
    [string]$Python,
    [string]$LogFile = (Join-Path $PSScriptRoot 'diagnostics\client-runtime.log'),
    [switch]$NoAutoInfer
)
$ErrorActionPreference = 'Stop'
if (@(Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue).Count) {
    throw "Client page already has a listener on port $Port. Open http://127.0.0.1:$Port instead."
}
if (-not $Python) {
    $Python = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
    if (-not (Test-Path -LiteralPath $Python)) {
        $Python = (Get-Command python -CommandType Application -ErrorAction Stop).Source
    }
}
if (-not (Test-Path -LiteralPath $Python)) { throw 'Client Python environment was not found. Supply -Python.' }
$PreviousPath = $env:PYTHONPATH
$PreviousUtf8 = $env:PYTHONUTF8
try {
    $env:PYTHONPATH = (Join-Path $PSScriptRoot 'src')
    $env:PYTHONUTF8 = '1'
    $ClientArguments = @('-X','utf8','-m','realman_sync_client.webapp','--server',$Server,
        '--port', $Port, '--mode',$Mode,'--hardware-config',$HardwareConfig)
    if ($TokenFile) { $ClientArguments += @('--token-file', $TokenFile) }
    if ($NoAutoInfer) { $ClientArguments += '--no-auto-infer' }
    if ($LogFile) { $ClientArguments += @('--log-file', $LogFile) }
    & $Python @ClientArguments
    $ClientExit = $LASTEXITCODE
} finally {
    $env:PYTHONPATH = $PreviousPath
    $env:PYTHONUTF8 = $PreviousUtf8
}
exit $ClientExit
