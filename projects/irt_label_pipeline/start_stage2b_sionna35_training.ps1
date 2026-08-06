param(
    [ValidateSet('Smoke', 'Full', 'Resume')]
    [string]$Mode = 'Smoke',
    [string]$ConfigPath = (Join-Path $PSScriptRoot 'config_stage2b_sionna35_depth8_27360_paper_v1.json'),
    [string]$Checkpoint
)

$ErrorActionPreference = 'Stop'
$pythonExe = 'C:\Users\pc\miniconda3\envs\sigmap\python.exe'
$trainScript = Join-Path $PSScriptRoot 'train_stage2b_directional_ss.py'

foreach ($requiredPath in @($pythonExe, $ConfigPath, $trainScript)) {
    if (-not (Test-Path -LiteralPath $requiredPath)) {
        throw "Required path does not exist: $requiredPath"
    }
}

switch ($Mode) {
    'Smoke' {
        $stamp = Get-Date -Format 'yyyyMMdd_HHmmss'
        $smokeOutput = "E:\dac_stage2b_sionna35_depth8_paper_smoke_$stamp"
        & $pythonExe $trainScript `
            --config $ConfigPath `
            --output-dir $smokeOutput `
            --epochs 2 `
            --train-limit-tiles 128 `
            --val-limit-tiles 32 `
            --test-limit-tiles 32
        exit $LASTEXITCODE
    }
    'Full' {
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
