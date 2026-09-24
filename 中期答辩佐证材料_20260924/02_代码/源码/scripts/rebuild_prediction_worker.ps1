<#
.SYNOPSIS
    Cleans shapefiles and rebuilds a disjoint shard of WinProp ODB tiles.
#>

[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$DatasetRoot,

    [Parameter(Mandatory = $true)]
    [string]$RebuildRoot,

    [Parameter(Mandatory = $true)]
    [string]$TileListPath,

    [ValidateRange(1, 16)]
    [int]$WorkerCount = 8,

    [ValidateRange(0, 15)]
    [int]$WorkerIndex = 0,

    [string]$PythonPath = '',

    [string]$CleanerScript = '',

    [string]$GeneratorScript = '',

    [double]$Gap = 0.02,

    [ValidateRange(0, 10)]
    [double]$SimplifyTolerance = 0,

    [ValidateRange(0, 10)]
    [double]$MinimumClearance = 0,

    [switch]$PreserveOverlaps
)

Set-StrictMode -Version 2.0
$ErrorActionPreference = 'Stop'

$repositoryRoot = Split-Path -Parent $PSScriptRoot
if ([string]::IsNullOrWhiteSpace($PythonPath)) {
    $PythonPath = Join-Path $repositoryRoot 'bjtu_osm_buildings\.venv\Scripts\python.exe'
}
if ($WorkerIndex -ge $WorkerCount) {
    throw "WorkerIndex must be lower than WorkerCount."
}
if ([string]::IsNullOrWhiteSpace($CleanerScript)) {
    $CleanerScript = Join-Path $PSScriptRoot 'clean_winprop_shapefile.py'
}
if ([string]::IsNullOrWhiteSpace($GeneratorScript)) {
    $GeneratorScript = Join-Path $PSScriptRoot 'generate_winprop_odb.ps1'
}
foreach ($required in @($DatasetRoot, $TileListPath, $PythonPath, $CleanerScript, $GeneratorScript)) {
    if (-not (Test-Path -LiteralPath $required)) {
        throw "Required path does not exist: $required"
    }
}
New-Item -ItemType Directory -Path $RebuildRoot -Force | Out-Null

$resolvedDatasetRoot = (Resolve-Path -LiteralPath $DatasetRoot).Path
$resolvedRebuildRoot = (Resolve-Path -LiteralPath $RebuildRoot).Path
$tiles = @(
    Get-Content -LiteralPath $TileListPath -Encoding UTF8 |
        ForEach-Object { $_.Trim() } |
        Where-Object { $_ -match '^tile_\d{6}$' }
)

$assignedTiles = New-Object System.Collections.Generic.List[string]
for ($index = 0; $index -lt $tiles.Count; $index++) {
    if (($index % $WorkerCount) -eq $WorkerIndex) {
        $assignedTiles.Add($tiles[$index])
    }
}

$logPath = Join-Path $resolvedRebuildRoot `
    ('_prediction_rebuild_worker_{0:D2}_of_{1:D2}.csv' -f ($WorkerIndex + 1), $WorkerCount)
$logExists = Test-Path -LiteralPath $logPath
$writer = New-Object System.IO.StreamWriter(
    $logPath,
    $true,
    (New-Object System.Text.UTF8Encoding($true))
)
if (-not $logExists -or (Get-Item -LiteralPath $logPath).Length -eq 0) {
    $writer.WriteLine('"Timestamp","Tile","Status","Message","DurationSeconds","OutputBytes"')
    $writer.Flush()
}

function CsvField {
    param([object]$Value)
    $text = if ($null -eq $Value) { '' } else { [string]$Value }
    return '"' + $text.Replace('"', '""') + '"'
}

function Write-RebuildLog {
    param(
        [string]$Tile,
        [string]$Status,
        [string]$Message,
        [double]$DurationSeconds,
        [long]$OutputBytes
    )
    $values = @(
        (Get-Date).ToString('yyyy-MM-dd HH:mm:ss'),
        $Tile,
        $Status,
        $Message,
        $DurationSeconds.ToString('0.000', [Globalization.CultureInfo]::InvariantCulture),
        $OutputBytes
    )
    $writer.WriteLine((($values | ForEach-Object { CsvField $_ }) -join ','))
    $writer.Flush()
}

try {
    for ($position = 0; $position -lt $assignedTiles.Count; $position++) {
        $tile = $assignedTiles[$position]
        $timer = [Diagnostics.Stopwatch]::StartNew()
        $sourceDir = Join-Path $resolvedDatasetRoot $tile
        $sourceShp = Join-Path $sourceDir "$tile.shp"
        $tileNumber = [int]$tile.Substring(5)
        $destinationDir = Join-Path $resolvedRebuildRoot $tile
        $destinationShp = Join-Path $destinationDir "$tile.shp"
        $destinationOdb = Join-Path $destinationDir "$tile.odb"
        $cleanLog = Join-Path $destinationDir '_cleaner_output.txt'
        $generatorLog = Join-Path $destinationDir '_generator_output.txt'

        Write-Progress `
            -Activity "Prediction ODB rebuild worker $($WorkerIndex + 1)/$WorkerCount" `
            -Status "$($position + 1)/$($assignedTiles.Count): $tile" `
            -PercentComplete ([int](100.0 * ($position + 1) / $assignedTiles.Count))

        if (Test-Path -LiteralPath $destinationOdb -PathType Leaf) {
            $timer.Stop()
            Write-RebuildLog $tile 'SKIPPED_EXISTS' 'Existing rebuilt ODB was kept.' `
                $timer.Elapsed.TotalSeconds (Get-Item -LiteralPath $destinationOdb).Length
            continue
        }

        try {
            if (-not (Test-Path -LiteralPath $sourceShp -PathType Leaf)) {
                throw "Missing source shapefile: $sourceShp"
            }
            New-Item -ItemType Directory -Path $destinationDir -Force | Out-Null

            $previousErrorActionPreference = $ErrorActionPreference
            $ErrorActionPreference = 'Continue'
            try {
                $cleanerArguments = @(
                    $CleanerScript,
                    '--source-shp', $sourceShp,
                    '--output-shp', $destinationShp,
                    '--height-field', 'HEIGHT_M',
                    '--gap', $Gap,
                    '--simplify-tolerance', $SimplifyTolerance,
                    '--minimum-clearance', $MinimumClearance
                )
                if ($PreserveOverlaps) {
                    $cleanerArguments += '--preserve-overlaps'
                }
                $cleanerOutput = & $PythonPath @cleanerArguments 2>&1
                $cleanerExit = $LASTEXITCODE
            }
            finally {
                $ErrorActionPreference = $previousErrorActionPreference
            }
            [IO.File]::WriteAllLines(
                $cleanLog,
                [string[]]@($cleanerOutput | ForEach-Object { [string]$_ }),
                (New-Object Text.UTF8Encoding($true))
            )
            if ($cleanerExit -ne 0) {
                throw "Geometry cleaner exited with code $cleanerExit. See $cleanLog"
            }

            $generatorArguments = @(
                '-NoProfile',
                '-ExecutionPolicy', 'Bypass',
                '-File', $GeneratorScript,
                '-BaseDir', $resolvedRebuildRoot,
                '-StartTile', $tileNumber,
                '-EndTile', $tileNumber,
                '-OverwriteOda',
                '-OverwriteOdb'
            )
            $generatorOutput = & powershell @generatorArguments 2>&1
            $generatorExit = $LASTEXITCODE
            [IO.File]::WriteAllLines(
                $generatorLog,
                [string[]]@($generatorOutput | ForEach-Object { [string]$_ }),
                (New-Object Text.UTF8Encoding($true))
            )
            if ($generatorExit -ne 0 -or
                -not (Test-Path -LiteralPath $destinationOdb -PathType Leaf)) {
                throw "ODB generator failed with code $generatorExit. See $generatorLog"
            }

            $timer.Stop()
            Write-RebuildLog $tile 'OK' '' $timer.Elapsed.TotalSeconds `
                (Get-Item -LiteralPath $destinationOdb).Length
        }
        catch {
            $timer.Stop()
            Write-Warning "[$tile] $($_.Exception.Message)"
            Write-RebuildLog $tile 'FAILED' $_.Exception.Message $timer.Elapsed.TotalSeconds 0
        }
    }
}
finally {
    Write-Progress -Activity 'Prediction ODB rebuild' -Completed
    $writer.Dispose()
}
