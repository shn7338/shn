<#
.SYNOPSIS
    Rechecks every checked ODB whose latest WallMan status is a database warning.

.DESCRIPTION
    Reads all existing WallMan CSV logs, selects sources whose latest non-skip
    status is OK_DATABASE_WARNING, and also includes checked ODBs with no known
    status. The existing checked files are used as inputs. Second-pass outputs
    and logs are written to a separate workspace and never overwrite dataset
    files.
#>

[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$RootPath,

    [ValidateRange(1, 16)]
    [int]$Workers = 8,

    [string]$OutputRoot = '',

    [string]$WorkerScript = '',

    [switch]$DryRun
)

Set-StrictMode -Version 2.0
$ErrorActionPreference = 'Stop'

if ([string]::IsNullOrWhiteSpace($WorkerScript)) {
    $WorkerScript = Join-Path $PSScriptRoot 'batch_check_winprop_odb.ps1'
}
if ([string]::IsNullOrWhiteSpace($OutputRoot)) {
    $OutputRoot = Join-Path (Split-Path -Parent $PSScriptRoot) 'prediction_recheck'
}
if (-not (Test-Path -LiteralPath $RootPath -PathType Container)) {
    throw "RootPath is not a directory: $RootPath"
}
if (-not (Test-Path -LiteralPath $WorkerScript -PathType Leaf)) {
    throw "Worker script was not found: $WorkerScript"
}

$resolvedRoot = (Resolve-Path -LiteralPath $RootPath).Path
$resolvedWorkerScript = (Resolve-Path -LiteralPath $WorkerScript).Path
New-Item -ItemType Directory -Path $OutputRoot -Force | Out-Null
$resolvedOutputRoot = (Resolve-Path -LiteralPath $OutputRoot).Path

$existingWorkers = @(
    Get-CimInstance Win32_Process -Filter "Name='powershell.exe'" |
        Where-Object {
            $_.CommandLine -match '(?i)-File\s+.*batch_check_winprop_odb\.ps1'
        }
)
if ($existingWorkers.Count -gt 0) {
    throw "Existing WallMan batch workers are running: $($existingWorkers.ProcessId -join ', ')"
}

Write-Host 'Collecting WallMan logs and latest source status...'
$logs = @(
    Get-ChildItem -LiteralPath $resolvedRoot -File -Filter '_wallman_check_database*.csv'
)
$logs += @(
    Get-ChildItem -LiteralPath $resolvedRoot -Recurse -File `
        -Filter '_wallman_check_database_log.csv' -ErrorAction SilentlyContinue |
        Where-Object { $_.DirectoryName -ne $resolvedRoot }
)
$logs = @($logs | Sort-Object FullName -Unique)

$latestBySource = @{}
foreach ($log in $logs) {
    foreach ($row in (Import-Csv -LiteralPath $log.FullName)) {
        if ($row.Status -like 'SKIPPED*') {
            continue
        }
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
}

Write-Host 'Finding checked ODBs that require prediction-readiness verification...'
$originalSources = @(
    Get-ChildItem -LiteralPath $resolvedRoot -Recurse -File -Filter '*.odb' |
        Where-Object {
            -not $_.BaseName.EndsWith('_checked', [System.StringComparison]::OrdinalIgnoreCase) -and
            $_.Name -notlike '__wmchk_????????.odb'
        } |
        Sort-Object FullName
)

$recheckSources = New-Object System.Collections.Generic.List[string]
$warningCount = 0
$unknownCount = 0
foreach ($source in $originalSources) {
    $key = $source.FullName.ToLowerInvariant()
    $include = $false
    if (-not $latestBySource.ContainsKey($key)) {
        $include = $true
        $unknownCount++
    }
    elseif ($latestBySource[$key].Status -eq 'OK_DATABASE_WARNING') {
        $include = $true
        $warningCount++
    }

    if ($include) {
        $checkedPath = Join-Path $source.DirectoryName ($source.BaseName + '_checked.odb')
        if (-not (Test-Path -LiteralPath $checkedPath -PathType Leaf)) {
            throw "The checked ODB selected for recheck is missing: $checkedPath"
        }
        $recheckSources.Add((Resolve-Path -LiteralPath $checkedPath).Path)
    }
}

$manifestPath = Join-Path $resolvedOutputRoot '_prediction_recheck_sources.txt'
[System.IO.File]::WriteAllLines(
    $manifestPath,
    [string[]]$recheckSources,
    (New-Object System.Text.UTF8Encoding($false))
)

Write-Host "Latest warning sources : $warningCount"
Write-Host "Unknown-status sources : $unknownCount"
Write-Host "Second-pass inputs     : $($recheckSources.Count)"
Write-Host "Output root            : $resolvedOutputRoot"
Write-Host "Manifest               : $manifestPath"

if ($DryRun) {
    $recheckSources | Select-Object -First 20
    exit 0
}
if ($recheckSources.Count -eq 0) {
    Write-Host 'No checked ODB requires a second pass.'
    exit 0
}

$started = New-Object System.Collections.Generic.List[object]
for ($shardIndex = 0; $shardIndex -lt $Workers; $shardIndex++) {
    $arguments = @(
        '-NoProfile'
        '-ExecutionPolicy Bypass'
        ('-File "{0}"' -f $resolvedWorkerScript)
        ('-RootPath "{0}"' -f $resolvedRoot)
        ('-SourceListPath "{0}"' -f $manifestPath)
        ('-OutputRoot "{0}"' -f $resolvedOutputRoot)
        '-OutputSuffix _rechecked'
        ("-ShardCount $Workers")
        ("-ShardIndex $shardIndex")
        '-SaveConcurrency 1'
        '-MaxAttempts 2'
    ) -join ' '

    $process = Start-Process `
        -FilePath 'powershell.exe' `
        -ArgumentList $arguments `
        -WindowStyle Minimized `
        -PassThru

    $started.Add([pscustomobject]@{
        Worker = $shardIndex + 1
        ProcessId = $process.Id
        Log = Join-Path $resolvedOutputRoot `
            ('_wallman_check_database_worker_{0:D2}_of_{1:D2}.csv' -f ($shardIndex + 1), $Workers)
    })
}

Write-Host "Started $Workers prediction-readiness recheck workers."
$started | Format-Table Worker, ProcessId, Log -AutoSize
