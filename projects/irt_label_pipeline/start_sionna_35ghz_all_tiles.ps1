param(
    [ValidateSet('Pilot32', 'Full', 'Status', 'Stop', 'DryRun', 'Finalize')]
    [string]$Mode = 'Pilot32',
    [ValidateRange(1, 4)]
    [int]$Workers = 4
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$scriptRoot = $PSScriptRoot
$datasetConfig = Join-Path $scriptRoot 'config_sionna201_35ghz_depth8_27360_v3.json'
$runner = Join-Path $scriptRoot 'run_sionna_dataset.py'
$sharder = Join-Path $scriptRoot 'build_irt_shards.py'
$verifier = Join-Path $scriptRoot 'verify_irt_shards.py'
$statsScript = Join-Path $scriptRoot 'compute_irt_shard_stats.py'
$python = 'E:\dac_sionna_rt_env\Scripts\python.exe'
$outputRoot = 'E:\dac_sionna_35ghz_depth8_27360_v3'
$selectionCsv = 'D:\桌面\dac\normalized_4m_v1\splits.csv'
$stopFlag = Join-Path $outputRoot 'STOP'
$exitCode = 0

if (-not (Test-Path -LiteralPath $python -PathType Leaf)) {
    throw "Sionna Python was not found: $python"
}

function Clear-StopFlag {
    if (Test-Path -LiteralPath $stopFlag -PathType Leaf) {
        Remove-Item -LiteralPath $stopFlag -Force
    }
}

switch ($Mode) {
    'Pilot32' {
        Clear-StopFlag
        & $python $runner --config $datasetConfig --pilot --workers $Workers
        $exitCode = $LASTEXITCODE
    }
    'Full' {
        Clear-StopFlag
        & $python $runner --config $datasetConfig --workers $Workers
        $exitCode = $LASTEXITCODE
    }
    'Status' {
        & $python $runner --config $datasetConfig --status
        $exitCode = $LASTEXITCODE
    }
    'Stop' {
        New-Item -ItemType Directory -Path $outputRoot -Force | Out-Null
        New-Item -ItemType File -Path $stopFlag -Force | Out-Null
        Write-Host 'Stop requested. Active tiles will finish safely; no new tile will start.'
        Write-Host $stopFlag
    }
    'DryRun' {
        & $python $runner --config $datasetConfig --dry-run --workers $Workers
        $exitCode = $LASTEXITCODE
    }
    'Finalize' {
        & $python $runner --config $datasetConfig --status
        if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
        & $python $sharder --config $datasetConfig
        if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
        & $python $verifier --metadata (Join-Path $outputRoot 'shard_metadata.json')
        if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
        & $python $statsScript `
            --shard-root (Join-Path $outputRoot 'shards') `
            --shard-manifest (Join-Path $outputRoot 'shard_manifest.csv') `
            --selection-csv $selectionCsv `
            --output (Join-Path $outputRoot 'normalization_irt.json')
        $exitCode = $LASTEXITCODE
    }
}

exit $exitCode
