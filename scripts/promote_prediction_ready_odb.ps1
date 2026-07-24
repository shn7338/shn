<#
.SYNOPSIS
    Promotes verified prediction-ready ODBs into the dataset with backups.

.DESCRIPTION
    For each source selected by the prediction recheck, uses the second-pass
    output when its status is OK; otherwise uses an OK checked ODB from the
    geometry-rebuild workspace. Existing dataset checked ODBs are copied to a
    backup tree before an atomic same-volume replacement.
#>

[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$DatasetRoot,

    [Parameter(Mandatory = $true)]
    [string]$RecheckRoot,

    [Parameter(Mandatory = $true)]
    [string]$RebuildRoot,

    [string]$ReplacementManifestPath = '',

    [string]$BackupRoot = '',

    [ValidateRange(1, 1000000)]
    [int]$ExpectedPromotions = 502,

    [switch]$DryRun
)

Set-StrictMode -Version 2.0
$ErrorActionPreference = 'Stop'

if ([string]::IsNullOrWhiteSpace($BackupRoot)) {
    $BackupRoot = Join-Path (Split-Path -Parent $PSScriptRoot) 'prediction_warning_backups'
}
foreach ($directory in @($DatasetRoot, $RecheckRoot, $RebuildRoot)) {
    if (-not (Test-Path -LiteralPath $directory -PathType Container)) {
        throw "Directory does not exist: $directory"
    }
}

$running = @(
    Get-CimInstance Win32_Process |
        Where-Object {
            $_.Name -match '^powershell(\.exe)?$' -and
            ($_.CommandLine -match '(?i)batch_check_winprop_odb\.ps1' -or
             $_.CommandLine -match '(?i)rebuild_prediction_worker\.ps1')
        }
)
if ($running.Count -gt 0) {
    throw "Prediction preparation workers are still running: $($running.ProcessId -join ', ')"
}

$resolvedDatasetRoot = (Resolve-Path -LiteralPath $DatasetRoot).Path
$resolvedRecheckRoot = (Resolve-Path -LiteralPath $RecheckRoot).Path
$resolvedRebuildRoot = (Resolve-Path -LiteralPath $RebuildRoot).Path
New-Item -ItemType Directory -Path $BackupRoot -Force | Out-Null
$resolvedBackupRoot = (Resolve-Path -LiteralPath $BackupRoot).Path

function Get-LatestNonSkipRows {
    param([string]$Root)
    $latest = @{}
    $logs = @(
        Get-ChildItem -LiteralPath $Root -File -Filter '_wallman_check_database*.csv'
    )
    foreach ($log in $logs) {
        foreach ($row in (Import-Csv -LiteralPath $log.FullName)) {
            if ($row.Status -like 'SKIPPED*') {
                continue
            }
            $key = $row.Source.ToLowerInvariant()
            $timestamp = [datetime]$row.Timestamp
            if (-not $latest.ContainsKey($key) -or
                $timestamp -gt $latest[$key].Timestamp) {
                $latest[$key] = [pscustomobject]@{
                    Timestamp = $timestamp
                    Source = $row.Source
                    Output = $row.Output
                    Status = $row.Status
                }
            }
        }
    }
    return $latest
}

$plan = @()
if (-not [string]::IsNullOrWhiteSpace($ReplacementManifestPath)) {
    if (-not (Test-Path -LiteralPath $ReplacementManifestPath -PathType Leaf)) {
        throw "Replacement manifest does not exist: $ReplacementManifestPath"
    }
    foreach ($manifestRow in (Import-Csv -LiteralPath $ReplacementManifestPath)) {
        $tile = [string]$manifestRow.Tile
        $replacement = [string]$manifestRow.Replacement
        $category = [string]$manifestRow.Category
        if ($tile -notmatch '^tile_\d{6}$') {
            throw "Invalid tile in replacement manifest: $tile"
        }
        if ([string]::IsNullOrWhiteSpace($category)) {
            throw "Missing category in replacement manifest for $tile."
        }
        $datasetChecked = Join-Path $resolvedDatasetRoot "$tile\$tile`_checked.odb"
        $backupDirectory = Join-Path $resolvedBackupRoot $tile
        $backup = Join-Path $backupDirectory "$tile`_checked_warning_backup.odb"
        $plan += [pscustomobject]@{
            Tile = $tile
            Category = $category
            Current = $datasetChecked
            Replacement = $replacement
            Backup = $backup
        }
    }
}
else {
    $recheckLatest = Get-LatestNonSkipRows -Root $resolvedRecheckRoot
    $rebuildLatest = Get-LatestNonSkipRows -Root $resolvedRebuildRoot

    foreach ($row in $recheckLatest.Values) {
        $sourceName = Split-Path $row.Source -Leaf
        if ($sourceName -notmatch '^(tile_\d{6})_checked\.odb$') {
            continue
        }
        $tile = $matches[1]
        $datasetChecked = Join-Path $resolvedDatasetRoot "$tile\$tile`_checked.odb"

        if ($row.Status -eq 'OK') {
            $replacement = $row.Output
            $category = 'RECHECKED'
        }
        else {
            $rebuiltSource = Join-Path $resolvedRebuildRoot "$tile\$tile.odb"
            $rebuiltKey = $rebuiltSource.ToLowerInvariant()
            if (-not $rebuildLatest.ContainsKey($rebuiltKey) -or
                $rebuildLatest[$rebuiltKey].Status -ne 'OK') {
                throw "No verified OK rebuilt result is available for $tile."
            }
            $replacement = $rebuildLatest[$rebuiltKey].Output
            $category = 'REBUILT'
        }

        $backupDirectory = Join-Path $resolvedBackupRoot $tile
        $backup = Join-Path $backupDirectory "$tile`_checked_warning_backup.odb"
        $plan += [pscustomobject]@{
            Tile = $tile
            Category = $category
            Current = $datasetChecked
            Replacement = $replacement
            Backup = $backup
        }
    }
}

foreach ($item in $plan) {
    foreach ($requiredFile in @($item.Current, $item.Replacement)) {
        if (-not (Test-Path -LiteralPath $requiredFile -PathType Leaf)) {
            throw "Required promotion file is missing: $requiredFile"
        }
    }
    if ((Get-Item -LiteralPath $item.Replacement).Length -le 0) {
        throw "Replacement ODB is empty: $($item.Replacement)"
    }
}

$plan = @($plan | Sort-Object Tile -Unique)
if ($plan.Count -ne $ExpectedPromotions) {
    throw "Promotion preflight found $($plan.Count) tiles; expected $ExpectedPromotions."
}

Write-Host "Promotion files : $($plan.Count)"
$plan |
    Group-Object Category |
    Sort-Object Name |
    ForEach-Object {
        Write-Host ('{0,-24}: {1}' -f $_.Name, $_.Count)
    }
Write-Host "Backup root     : $resolvedBackupRoot"

if ($DryRun) {
    $plan | Select-Object -First 30 Tile, Category, Current, Replacement
    exit 0
}

$logPath = Join-Path $resolvedBackupRoot '_promotion_log.csv'
$logExists = Test-Path -LiteralPath $logPath
$writer = New-Object IO.StreamWriter(
    $logPath,
    $true,
    (New-Object Text.UTF8Encoding($true))
)
if (-not $logExists -or (Get-Item -LiteralPath $logPath).Length -eq 0) {
    $writer.WriteLine('"Timestamp","Tile","Category","Status","CurrentSHA256","ReplacementSHA256","Backup"')
    $writer.Flush()
}

function CsvField {
    param([object]$Value)
    $text = if ($null -eq $Value) { '' } else { [string]$Value }
    return '"' + $text.Replace('"', '""') + '"'
}

try {
    for ($index = 0; $index -lt $plan.Count; $index++) {
        $item = $plan[$index]
        Write-Progress `
            -Activity 'Promoting prediction-ready ODBs' `
            -Status "$($index + 1)/$($plan.Count): $($item.Tile)" `
            -PercentComplete ([int](100.0 * ($index + 1) / $plan.Count))

        $currentHash = (Get-FileHash -LiteralPath $item.Current -Algorithm SHA256).Hash
        $replacementHash = (Get-FileHash -LiteralPath $item.Replacement -Algorithm SHA256).Hash
        $status = 'PROMOTED'

        if ($currentHash -eq $replacementHash) {
            $status = 'ALREADY_PROMOTED'
        }
        else {
            $backupDirectory = Split-Path -Parent $item.Backup
            New-Item -ItemType Directory -Path $backupDirectory -Force | Out-Null
            if (-not (Test-Path -LiteralPath $item.Backup -PathType Leaf)) {
                Copy-Item -LiteralPath $item.Current -Destination $item.Backup
            }
            $backupHash = (Get-FileHash -LiteralPath $item.Backup -Algorithm SHA256).Hash
            if ($backupHash -ne $currentHash) {
                throw "Backup verification failed for $($item.Tile)."
            }

            $temporary = Join-Path (Split-Path -Parent $item.Current) `
                ('__predready_{0}.odb' -f [Guid]::NewGuid().ToString('N').Substring(0, 8))
            $atomicBackup = Join-Path (Split-Path -Parent $item.Current) `
                ('__predprevious_{0}.odb' -f [Guid]::NewGuid().ToString('N').Substring(0, 8))
            $replacementVerified = $false
            try {
                Copy-Item -LiteralPath $item.Replacement -Destination $temporary
                $temporaryHash = (Get-FileHash -LiteralPath $temporary -Algorithm SHA256).Hash
                if ($temporaryHash -ne $replacementHash) {
                    throw "Temporary replacement verification failed for $($item.Tile)."
                }
                [IO.File]::Replace($temporary, $item.Current, $atomicBackup, $true)
                $promotedHash = (Get-FileHash -LiteralPath $item.Current -Algorithm SHA256).Hash
                if ($promotedHash -ne $replacementHash) {
                    throw "Promoted file verification failed for $($item.Tile)."
                }
                $replacementVerified = $true
            }
            finally {
                if (Test-Path -LiteralPath $temporary) {
                    Remove-Item -LiteralPath $temporary -Force -ErrorAction SilentlyContinue
                }
                # Keep the same-directory rollback file if post-replacement
                # verification fails. The permanent preflight backup is retained
                # independently under BackupRoot.
                if ($replacementVerified -and (Test-Path -LiteralPath $atomicBackup)) {
                    Remove-Item -LiteralPath $atomicBackup -Force -ErrorAction SilentlyContinue
                }
            }
        }

        $values = @(
            (Get-Date).ToString('yyyy-MM-dd HH:mm:ss'),
            $item.Tile,
            $item.Category,
            $status,
            $currentHash,
            $replacementHash,
            $item.Backup
        )
        $writer.WriteLine((($values | ForEach-Object { CsvField $_ }) -join ','))
        $writer.Flush()
    }
}
finally {
    Write-Progress -Activity 'Promoting prediction-ready ODBs' -Completed
    $writer.Dispose()
}

Write-Host "Promotion completed. Log: $logPath"
