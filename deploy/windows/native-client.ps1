param(
    [string]$Python,
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$ClientArguments
)
$ErrorActionPreference = 'Stop'
if (-not $Python) { $Python = Join-Path (Split-Path -Parent $PSScriptRoot) '.venv-client\Scripts\python.exe' }
if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) { throw 'Supply a client environment Python with -Python; environments were not copied.' }
& $Python -X utf8 (Join-Path $PSScriptRoot 'client.py') @ClientArguments
exit $LASTEXITCODE
