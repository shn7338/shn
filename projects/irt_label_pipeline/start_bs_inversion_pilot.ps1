param(
  [ValidateSet('DryRun', 'Smoke', 'Full', 'Status', 'Finalize', 'Stop')]
  [string]$Mode = 'Status',
  [int]$Workers = 4
)

$ErrorActionPreference = 'Stop'
$repoRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$python = 'C:\Users\pc\miniconda3\envs\sigmap\python.exe'
$script = Join-Path $PSScriptRoot 'run_bs_inversion_pilot.py'
$config = Join-Path $PSScriptRoot 'config_bs_inversion_pilot_v1.json'
$outputRoot = 'D:\桌面\dac\02_inversion\bs_inversion_pilot_v1'

switch ($Mode) {
  'DryRun' {
    & $python $script --config $config --dry-run
  }
  'Smoke' {
    if (Test-Path -LiteralPath (Join-Path $outputRoot 'STOP')) {
      Remove-Item -LiteralPath (Join-Path $outputRoot 'STOP')
    }
    & $python $script --config $config --limit-sites 1 --workers 1
  }
  'Full' {
    if (Test-Path -LiteralPath (Join-Path $outputRoot 'STOP')) {
      Remove-Item -LiteralPath (Join-Path $outputRoot 'STOP')
    }
    & $python $script --config $config --workers $Workers
  }
  'Status' {
    & $python $script --config $config --status
  }
  'Finalize' {
    & $python $script --config $config --finalize-only
  }
  'Stop' {
    New-Item -ItemType Directory -Path $outputRoot -Force | Out-Null
    Set-Content -LiteralPath (Join-Path $outputRoot 'STOP') -Value 'stop requested' -Encoding UTF8
    Write-Output 'Stop requested. Active sites will finish; no new sites will start.'
  }
}
