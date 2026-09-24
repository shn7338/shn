param(
    [int]$Epochs = 15
)

$ErrorActionPreference = "Stop"
$Python = "C:\Users\pc\miniconda3\envs\sigmap\python.exe"
$ScriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$Config = Join-Path $ScriptRoot "config_stage2b_bs_inversion_pilot_finetune_v1.json"

& $Python (Join-Path $ScriptRoot "finetune_pilot_stage2b.py") `
    --config $Config `
    --epochs $Epochs

exit $LASTEXITCODE
