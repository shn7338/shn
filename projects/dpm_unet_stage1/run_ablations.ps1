param(
    [string]$DataRoot = $env:DPM_DATA_ROOT,
    [string]$RunsRoot = $env:DPM_RUNS_ROOT,
    [string]$Python = 'C:\Users\pc\miniconda3\envs\sigmap\python.exe',
    [int]$Epochs = 60,
    [int]$BatchSize = 16,
    [int]$Workers = 4
)

$ErrorActionPreference = 'Stop'
$project = $PSScriptRoot
if ([string]::IsNullOrWhiteSpace($DataRoot) -or [string]::IsNullOrWhiteSpace($RunsRoot)) {
    throw 'Set DPM_DATA_ROOT and DPM_RUNS_ROOT, or pass -DataRoot and -RunsRoot explicitly.'
}
$python = [System.IO.Path]::GetFullPath($Python)
if (-not (Test-Path -LiteralPath $python)) {
    throw "Python executable not found: $python"
}

function Run-Ablation {
    param([string]$Name, [string]$InputMode)

    $output = Join-Path $RunsRoot $Name
    $checkpoint = Join-Path $output 'best.pt'
    Write-Host "=== $Name ($InputMode) ==="
    if (-not (Test-Path -LiteralPath $checkpoint)) {
        & $python (Join-Path $project 'train.py') `
            --data-root $DataRoot `
            --output-dir $output `
            --epochs $Epochs `
            --batch-size $BatchSize `
            --workers $Workers `
            --base-channels 32 `
            --learning-rate 3e-4 `
            --patience 12 `
            --input-mode $InputMode
        if ($LASTEXITCODE -ne 0) { throw "$Name training failed with exit code $LASTEXITCODE" }
    } else {
        Write-Host "Checkpoint already exists; skipping training: $checkpoint"
    }

    & $python (Join-Path $project 'evaluate.py') `
        --data-root $DataRoot `
        --checkpoint $checkpoint `
        --split test `
        --batch-size $BatchSize `
        --workers $Workers `
        --output (Join-Path $output 'test_metrics.json')
    if ($LASTEXITCODE -ne 0) { throw "$Name evaluation failed with exit code $LASTEXITCODE" }
}

Run-Ablation -Name 'ablation_height' -InputMode 'height'
Run-Ablation -Name 'ablation_height_tx' -InputMode 'height_tx'

# Re-evaluate the already trained complete model using the same current code.
$baseline = Join-Path $RunsRoot 'baseline_v1'
& $python (Join-Path $project 'evaluate.py') `
    --data-root $DataRoot `
    --checkpoint (Join-Path $baseline 'best.pt') `
    --split test `
    --batch-size $BatchSize `
    --workers $Workers `
    --output (Join-Path $baseline 'test_metrics.json')
if ($LASTEXITCODE -ne 0) { throw "Baseline evaluation failed with exit code $LASTEXITCODE" }

& $python (Join-Path $project 'generate_stage1_report.py') `
    --data-root $DataRoot `
    --runs-root $RunsRoot `
    --output (Join-Path $RunsRoot 'stage1_results.md')
if ($LASTEXITCODE -ne 0) { throw "Report generation failed with exit code $LASTEXITCODE" }

Write-Host 'All ablation experiments, test evaluations, and the Stage-1 report are complete.'
