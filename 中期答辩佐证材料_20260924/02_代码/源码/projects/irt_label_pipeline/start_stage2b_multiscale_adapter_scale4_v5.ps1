$ErrorActionPreference = "Stop"
$Python = "C:\Users\pc\miniconda3\envs\sigmap\python.exe"
$Project = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location -LiteralPath $Project
& $Python (Join-Path $Project "train_stage2b_latent_adapter_v5.py") `
    --config (Join-Path $Project "config_stage2b_multiscale_adapter_scale4_v5.json")
exit $LASTEXITCODE
