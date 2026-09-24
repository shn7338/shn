$ErrorActionPreference = "Stop"
$Python = "C:\Users\pc\miniconda3\envs\sigmap\python.exe"
$Project = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location -LiteralPath $Project
& $Python (Join-Path $Project "train_joint_estimator_stage2b_v6.py") `
    --config (Join-Path $Project "config_joint_estimator_stage2b_scale4_v6.json")
exit $LASTEXITCODE
