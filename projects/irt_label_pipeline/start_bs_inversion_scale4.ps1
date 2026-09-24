param(
    [int]$Workers = 4
)

$ErrorActionPreference = "Stop"
$Python = "C:\Users\pc\miniconda3\envs\sigmap\python.exe"
$ScriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
& $Python (Join-Path $ScriptRoot "run_bs_inversion_pilot.py") `
    --config (Join-Path $ScriptRoot "config_bs_inversion_scale4_v1.json") `
    --workers $Workers
exit $LASTEXITCODE
