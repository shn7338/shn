param(
    [int]$Epochs = 15,
    [string]$Resume = ""
)

$ErrorActionPreference = "Stop"
$Python = "C:\Users\pc\miniconda3\envs\sigmap\python.exe"
$ScriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$Arguments = @(
    (Join-Path $ScriptRoot "finetune_pilot_stage1_stage2a.py"),
    "--config",
    (Join-Path $ScriptRoot "config_stage1_stage2a_bs_inversion_pilot_finetune_v1.json"),
    "--epochs",
    $Epochs
)
if ($Resume) {
    $Arguments += @("--resume", $Resume)
}
& $Python @Arguments
exit $LASTEXITCODE
