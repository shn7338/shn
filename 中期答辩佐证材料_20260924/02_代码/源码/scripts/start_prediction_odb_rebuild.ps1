<#
.SYNOPSIS
    Starts parallel geometry-cleaning and ODB rebuild workers for second-pass warnings.
#>

[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$DatasetRoot,

    [Parameter(Mandatory = $true)]
    [string]$RecheckRoot,

    [ValidateRange(1, 16)]
    [int]$Workers = 8,

    [double]$Gap = 0.02,

    [ValidateRange(0, 10)]
    [double]$SimplifyTolerance = 0,

    [ValidateRange(0, 10)]
    [double]$MinimumClearance = 0,

    [switch]$PreserveOverlaps,

    [string]$RebuildRoot = '',

    [switch]$DryRun
)

Set-StrictMode -Version 2.0
$ErrorActionPreference = 'Stop'

if ([string]::IsNullOrWhiteSpace($RebuildRoot)) {
    $RebuildRoot = Join-Path (Split-Path -Parent $PSScriptRoot) 'prediction_rebuild'
}
foreach ($required in @($DatasetRoot, $RecheckRoot)) {
    if (-not (Test-Path -LiteralPath $required -PathType Container)) {
        throw "Directory does not exist: $required"
    }
}
New-Item -ItemType Directory -Path $RebuildRoot -Force | Out-Null
$resolvedDatasetRoot = (Resolve-Path -LiteralPath $DatasetRoot).Path
$resolvedRecheckRoot = (Resolve-Path -LiteralPath $RecheckRoot).Path
$resolvedRebuildRoot = (Resolve-Path -LiteralPath $RebuildRoot).Path
$workerScript = Join-Path $PSScriptRoot 'rebuild_prediction_worker.ps1'

$runningRechecks = @(
    Get-CimInstance Win32_Process -Filter "Name='powershell.exe'" |
        Where-Object {
            $_.CommandLine -match '(?i)-File\s+.*batch_check_winprop_odb\.ps1'
        }
)
if ($runningRechecks.Count -gt 0) {
    throw "WallMan recheck workers are still running: $($runningRechecks.ProcessId -join ', ')"
}

$logs = @(
    Get-ChildItem -LiteralPath $resolvedRecheckRoot -File `
        -Filter '_wallman_check_database_worker_*_of_08.csv'
)
$rows = @()
foreach ($log in $logs) {
    $rows += Import-Csv -LiteralPath $log.FullName
}
$nonSkipped = @($rows | Where-Object { $_.Status -notlike 'SKIPPED*' })
if ($nonSkipped.Count -lt 501) {
    throw "Second-pass recheck is incomplete: $($nonSkipped.Count)/501 warning inputs have a result."
}

$latestBySource = @{}
foreach ($row in $nonSkipped) {
    $key = $row.Source.ToLowerInvariant()
    $timestamp = [datetime]$row.Timestamp
    if (-not $latestBySource.ContainsKey($key) -or
        $timestamp -gt $latestBySource[$key].Timestamp) {
        $latestBySource[$key] = [pscustomobject]@{
            Timestamp = $timestamp
            Source = $row.Source
            Status = $row.Status
        }
    }
}

$tiles = @(
    $latestBySource.Values |
        Where-Object { $_.Status -in @('OK_DATABASE_WARNING', 'FAILED') } |
        ForEach-Object {
            $name = Split-Path $_.Source -Leaf
            if ($name -notmatch '^(tile_\d{6})_checked\.odb$') {
                throw "Unexpected recheck source name: $name"
            }
            $matches[1]
        } |
        Sort-Object -Unique
)

$tileListPath = Join-Path $resolvedRebuildRoot '_prediction_rebuild_tiles.txt'
[IO.File]::WriteAllLines(
    $tileListPath,
    [string[]]$tiles,
    (New-Object Text.UTF8Encoding($false))
)

Write-Host "Second-pass results : $($latestBySource.Count)"
Write-Host "Tiles to rebuild    : $($tiles.Count)"
Write-Host "Rebuild root        : $resolvedRebuildRoot"
Write-Host "Tile list           : $tileListPath"

if ($DryRun) {
    $tiles | Select-Object -First 30
    exit 0
}
if ($tiles.Count -eq 0) {
    Write-Host 'No tile needs geometry rebuild.'
    exit 0
}

$started = @()
for ($workerIndex = 0; $workerIndex -lt $Workers; $workerIndex++) {
    $arguments = @(
        '-NoProfile',
        '-ExecutionPolicy', 'Bypass',
        ('-File "{0}"' -f $workerScript),
        ('-DatasetRoot "{0}"' -f $resolvedDatasetRoot),
        ('-RebuildRoot "{0}"' -f $resolvedRebuildRoot),
        ('-TileListPath "{0}"' -f $tileListPath),
        ("-WorkerCount $Workers"),
        ("-WorkerIndex $workerIndex"),
        ("-Gap $Gap"),
        ("-SimplifyTolerance $SimplifyTolerance"),
        ("-MinimumClearance $MinimumClearance")
    ) -join ' '
    if ($PreserveOverlaps) {
        $arguments += ' -PreserveOverlaps'
    }
    $process = Start-Process powershell.exe -ArgumentList $arguments `
        -WindowStyle Minimized -PassThru
    $started += [pscustomobject]@{
        Worker = $workerIndex + 1
        ProcessId = $process.Id
    }
}

Write-Host "Started $Workers geometry-cleaning/ODB-rebuild workers."
$started | Format-Table -AutoSize
