param(
    [ValidateSet('Stats', 'Smoke', 'Full', 'Resume')]
    [string]$Mode = 'Smoke',
    [string]$ConfigPath = (Join-Path $PSScriptRoot 'config_stage2a_sionna35_depth8_27360_paper_v1.json'),
    [string]$Checkpoint
)

$ErrorActionPreference = 'Stop'
$pythonExe = 'C:\Users\pc\miniconda3\envs\sigmap\python.exe'
$statsScript = Join-Path $PSScriptRoot 'compute_stage2a_residual_stats.py'
$trainScript = Join-Path $PSScriptRoot 'train_stage2a_iso_refine.py'
$fullStats = 'E:\dac_sionna_35ghz_depth8_27360_v3\stage2a_residual_stats_train.json'
$smokeStats = 'E:\dac_sionna_35ghz_depth8_27360_v3\stage2a_residual_stats_smoke512.json'

foreach ($requiredPath in @($pythonExe, $ConfigPath, $statsScript, $trainScript)) {
    if (-not (Test-Path -LiteralPath $requiredPath)) {
        throw "Required path does not exist: $requiredPath"
    }
}

switch ($Mode) {
    'Stats' {
        & $pythonExe $statsScript --config $ConfigPath
        exit $LASTEXITCODE
    }
    'Smoke' {
        if (-not (Test-Path -LiteralPath $smokeStats)) {
            & $pythonExe $statsScript `
                --config $ConfigPath `
                --output $smokeStats `
                --limit 512
            if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
        }
        $stamp = Get-Date -Format 'yyyyMMdd_HHmmss'
        $smokeOutput = "E:\dac_stage2a_sionna35_depth8_paper_smoke_$stamp"
        & $pythonExe $trainScript `
            --config $ConfigPath `
            --residual-stats $smokeStats `
            --allow-partial-residual-stats `
            --output-dir $smokeOutput `
            --epochs 2 `
            --train-limit 512 `
            --val-limit 128 `
            --test-limit 128
        exit $LASTEXITCODE
    }
    'Full' {
        if (-not (Test-Path -LiteralPath $fullStats)) {
            Write-Host 'Full train residual statistics are missing; computing them first.'
            & $pythonExe $statsScript --config $ConfigPath
            if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
        }
        & $pythonExe $trainScript --config $ConfigPath
        exit $LASTEXITCODE
    }
    'Resume' {
        if ([string]::IsNullOrWhiteSpace($Checkpoint)) {
            throw '-Checkpoint is required in Resume mode.'
        }
        if (-not (Test-Path -LiteralPath $Checkpoint)) {
            throw "Checkpoint does not exist: $Checkpoint"
        }
        & $pythonExe $trainScript `
            --config $ConfigPath `
            --resume $Checkpoint
        exit $LASTEXITCODE
    }
}
