param(
  [ValidateSet('Smoke', 'Full', 'Resume')]
  [string]$Mode = 'Smoke',
  [string]$Checkpoint = ''
)

$ErrorActionPreference = 'Stop'
$python = 'C:\Users\pc\miniconda3\envs\sigmap\python.exe'
$script = Join-Path $PSScriptRoot 'train_bs_parameter_estimator.py'
$config = Join-Path $PSScriptRoot 'config_bs_parameter_estimator_pilot_v1.json'

switch ($Mode) {
  'Smoke' {
    & $python $script --config $config --smoke
  }
  'Full' {
    & $python $script --config $config
  }
  'Resume' {
    if (-not $Checkpoint) {
      throw 'Resume requires -Checkpoint.'
    }
    & $python $script --config $config --resume $Checkpoint
  }
}
