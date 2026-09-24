$ErrorActionPreference = "Stop"
$Python = "C:\Users\pc\miniconda3\envs\sigmap\python.exe"
$ScriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
& $Python (Join-Path $ScriptRoot "finetune_pilot_stage2b.py") `
    --config (Join-Path $ScriptRoot "config_stage2b_latent_fusion_scale4_v4.json")
exit $LASTEXITCODE

