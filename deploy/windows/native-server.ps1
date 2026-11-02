param(
    [ValidateSet('Install', 'Check', 'Serve', 'Imports')]
    [string]$Action = 'Check',
    [string]$Experiment,
    [string]$Config,
    [string]$Checkpoint,
    [string]$BootstrapPython,
    [string]$Venv,
    [string]$ListenHost = '127.0.0.1',
    [ValidateRange(1, 65535)]
    [int]$Port = 8766,
    [string]$TokenEnv = 'REALMAN_SERVER_TOKEN',
    [switch]$CheckpointBootstrap,
    [switch]$Verify,
    [switch]$DryRun
)
$ErrorActionPreference = 'Stop'
$InferRoot = Split-Path -Parent $PSScriptRoot
if (-not $Venv) {
    $Venv = Join-Path $InferRoot '.venv-server'
    if ($Action -ne 'Install') {
        if ($env:REALMAN_SERVER_VENV) {
            $Venv = $env:REALMAN_SERVER_VENV
        } elseif (-not (Test-Path -LiteralPath (Join-Path $Venv 'Scripts\python.exe') -PathType Leaf)) {
            $Layout = Get-Content -LiteralPath (Join-Path $InferRoot 'code-layout.json') -Raw | ConvertFrom-Json
            $CosmosSource = ($Layout.experiments | Where-Object { $_.name -eq 'cosmos' }).source_path
            $WorkspacePath = Split-Path -Parent (Split-Path -Parent $CosmosSource)
            $ExistingVenv = Join-Path $WorkspacePath '.venv-server'
            if (Test-Path -LiteralPath (Join-Path $ExistingVenv 'Scripts\python.exe') -PathType Leaf) { $Venv = $ExistingVenv }
        }
    }
}
$ServerPython = Join-Path $Venv 'Scripts\python.exe'
if ($Action -eq 'Install') {
    if (-not $BootstrapPython) { throw 'Supply the real Python 3.12 executable with -BootstrapPython.' }
    $SelectedPython = $BootstrapPython
    $Arguments = @('-X', 'utf8', (Join-Path $PSScriptRoot 'server.py'), 'install', '--python', $BootstrapPython, '--venv', $Venv)
} else {
    $SelectedPython = $ServerPython
    if ($BootstrapPython) { $SelectedPython = $BootstrapPython }
    $Arguments = @('-X', 'utf8', (Join-Path $PSScriptRoot 'server.py'), $Action.ToLowerInvariant())
    if ($Action -eq 'Check') {
        if ($Verify) { $Arguments += '--verify' }
    } elseif ($Action -eq 'Imports') {
        $Arguments += @('--venv', $Venv)
    } else {
        if (-not $Experiment -or -not $Config -or -not $Checkpoint) { throw 'Serve needs -Experiment, -Config, and -Checkpoint.' }
        $Arguments += @('--experiment', $Experiment, '--config', $Config, '--checkpoint', $Checkpoint, '--venv', $Venv,
                        '--host', $ListenHost, '--port', $Port, '--token-env', $TokenEnv)
        if ($CheckpointBootstrap) { $Arguments += '--checkpoint-bootstrap' }
    }
}
if (-not (Test-Path -LiteralPath $SelectedPython -PathType Leaf)) { throw 'Selected Python does not exist. Environments were intentionally not copied.' }
if ($DryRun -and $Action -ne 'Check') { $Arguments += '--dry-run' }
& $SelectedPython @Arguments
exit $LASTEXITCODE
