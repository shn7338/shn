param(
    [string]$DataRoot = $env:DPM_DATA_ROOT,
    [string]$RunsRoot = $env:DPM_RUNS_ROOT,
    [string]$ProjectRoot = $env:DPM_PROJECT_ROOT,
    [string]$Python = 'C:\Users\pc\miniconda3\envs\sigmap\python.exe',
    [int]$TimeoutMinutes = 180
)

$ErrorActionPreference = 'Stop'
$required = @(
    (Join-Path $RunsRoot 'ablation_height\test_metrics.json'),
    (Join-Path $RunsRoot 'ablation_height_tx\test_metrics.json'),
    (Join-Path $RunsRoot 'baseline_v1\test_metrics.json')
)
$deadline = (Get-Date).AddMinutes($TimeoutMinutes)
while ((Get-Date) -lt $deadline) {
    if (($required | Where-Object { -not (Test-Path -LiteralPath $_) }).Count -eq 0) {
        & $Python (Join-Path $ProjectRoot 'generate_stage1_report.py') `
            --data-root $DataRoot `
            --runs-root $RunsRoot `
            --output (Join-Path $RunsRoot 'stage1_results.md')
        exit $LASTEXITCODE
    }
    Start-Sleep -Seconds 30
}
throw "Timed out waiting for ablation result files after $TimeoutMinutes minutes."
