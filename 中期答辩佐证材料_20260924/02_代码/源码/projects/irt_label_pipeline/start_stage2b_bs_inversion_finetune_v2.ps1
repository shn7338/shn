param(
    [int]$Epochs = 12,
    [string]$Resume = ""
)

$ErrorActionPreference = "Stop"
$Python = "C:\Users\pc\miniconda3\envs\sigmap\python.exe"
$ScriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$Arguments = @(
    (Join-Path $ScriptRoot "finetune_pilot_stage2b.py"),
    "--config",
    (Join-Path $ScriptRoot "config_stage2b_bs_inversion_pilot_finetune_v2.json"),
    "--epochs",
    $Epochs
)
if ($Resume) {
    $Arguments += @("--resume", $Resume)
}
& $Python @Arguments
exit $LASTEXITCODE
