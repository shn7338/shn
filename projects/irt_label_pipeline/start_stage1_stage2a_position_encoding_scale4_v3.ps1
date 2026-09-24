$ErrorActionPreference = "Stop"
$Python = "C:\Users\pc\miniconda3\envs\sigmap\python.exe"
$ScriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
& $Python (Join-Path $ScriptRoot "finetune_pilot_stage1_stage2a.py") `
    --config (Join-Path $ScriptRoot "config_stage1_stage2a_position_encoding_scale4_v3.json")
exit $LASTEXITCODE

