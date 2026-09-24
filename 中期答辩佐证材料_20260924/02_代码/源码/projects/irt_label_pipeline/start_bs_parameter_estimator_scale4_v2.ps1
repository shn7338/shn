$ErrorActionPreference = "Stop"
$Python = "C:\Users\pc\miniconda3\envs\sigmap\python.exe"
$ScriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
& $Python (Join-Path $ScriptRoot "train_bs_parameter_estimator.py") `
    --config (Join-Path $ScriptRoot "config_bs_parameter_estimator_scale4_v2.json") `
    --init-checkpoint "D:\桌面\dac\models\bs_parameter_estimator_pilot_v1\best.pt"
exit $LASTEXITCODE
