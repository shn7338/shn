<#
.SYNOPSIS
    Shows an aggregate status for sharded WallMan Check Database workers.

.EXAMPLE
    powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\show_wallman_check_status.ps1 `
      -RootPath 'D:\桌面\dac\512mdata' -Workers 8 -Watch
#>

[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$RootPath,

    [ValidateRange(1, 16)]
    [int]$Workers = 8,

    [ValidateRange(1, 2147483647)]
    [int]$TotalSources = 27361,

    [ValidateRange(2, 300)]
    [int]$RefreshSeconds = 10,

    [switch]$Watch
)

Set-StrictMode -Version 2.0
$ErrorActionPreference = 'Stop'

if (-not (Test-Path -LiteralPath $RootPath -PathType Container)) {
    throw "RootPath is not a directory: $RootPath"
}
$resolvedRoot = (Resolve-Path -LiteralPath $RootPath).Path

function Get-WallManStatus {
    $workerProcesses = @(
        Get-CimInstance Win32_Process -Filter "Name='powershell.exe'" |
            Where-Object {
                $_.CommandLine -match '(?i)-File\s+.*batch_check_winprop_odb\.ps1' -and
                $_.CommandLine -match ("(?i)-ShardCount\s+{0}(?:\s|$)" -f $Workers)
            }
    )

    $runStart = if ($workerProcesses.Count -gt 0) {
        $workerProcesses |
            ForEach-Object { [datetime]$_.CreationDate } |
            Sort-Object |
            Select-Object -First 1
    }
    else {
        $null
    }

    $baseCount = [math]::Floor($TotalSources / $Workers)
    $extraCount = $TotalSources % $Workers
    $totalRows = 0
    $totalProcessed = 0
    $totalWarnings = 0
    $totalFailed = 0
    $latestUpdate = $null
    $firstProcessedUpdate = $null
    $latestProcessedUpdate = $null

    $workerTable = @(
        for ($shardIndex = 0; $shardIndex -lt $Workers; $shardIndex++) {
            $assignedCount = $baseCount + $(if ($shardIndex -lt $extraCount) { 1 } else { 0 })
            $logPath = Join-Path $resolvedRoot `
                ('_wallman_check_database_worker_{0:D2}_of_{1:D2}.csv' -f ($shardIndex + 1), $Workers)

            $rows = if (Test-Path -LiteralPath $logPath) {
                @(Import-Csv -LiteralPath $logPath)
            }
            else {
                @()
            }

            if ($null -ne $runStart) {
                $rows = @($rows | Where-Object { [datetime]$_.Timestamp -ge $runStart })
            }

            $processedRows = @($rows | Where-Object { $_.Status -notlike 'SKIPPED*' })
            $warningRows = @($processedRows | Where-Object { $_.Status -eq 'OK_DATABASE_WARNING' })
            $failedRows = @($processedRows | Where-Object { $_.Status -eq 'FAILED' })

            $totalRows += $rows.Count
            $totalProcessed += $processedRows.Count
            $totalWarnings += $warningRows.Count
            $totalFailed += $failedRows.Count

            if ($rows.Count -gt 0) {
                $rowTime = [datetime]$rows[-1].Timestamp
                if ($null -eq $latestUpdate -or $rowTime -gt $latestUpdate) {
                    $latestUpdate = $rowTime
                }
            }
            if ($processedRows.Count -gt 0) {
                $firstWorkerProcessed = [datetime]$processedRows[0].Timestamp
                $lastWorkerProcessed = [datetime]$processedRows[-1].Timestamp
                if ($null -eq $firstProcessedUpdate -or $firstWorkerProcessed -lt $firstProcessedUpdate) {
                    $firstProcessedUpdate = $firstWorkerProcessed
                }
                if ($null -eq $latestProcessedUpdate -or $lastWorkerProcessed -gt $latestProcessedUpdate) {
                    $latestProcessedUpdate = $lastWorkerProcessed
                }
            }

            [pscustomobject]@{
                Worker = $shardIndex + 1
                Position = $rows.Count
                Total = $assignedCount
                Percent = [math]::Round(100.0 * $rows.Count / $assignedCount, 1)
                New = $processedRows.Count
                Warnings = $warningRows.Count
                Failed = $failedRows.Count
                LastTile = if ($rows.Count -gt 0) { Split-Path $rows[-1].Source -Leaf } else { '' }
                Updated = if ($rows.Count -gt 0) { $rows[-1].Timestamp } else { '' }
            }
        }
    )

    # SKIPPED_EXISTS rows are emitted very quickly after a restart and must not
    # inflate the real WallMan throughput. Measure only actual processing rows
    # between the first and latest completed item in this run.
    $processingWindowSeconds = if ($null -ne $firstProcessedUpdate -and $null -ne $latestProcessedUpdate) {
        ($latestProcessedUpdate - $firstProcessedUpdate).TotalSeconds
    }
    else {
        0
    }
    $ratePerHour = if ($totalProcessed -gt 1 -and $processingWindowSeconds -gt 0) {
        3600.0 * ($totalProcessed - 1) / $processingWindowSeconds
    }
    else {
        0
    }
    $remaining = [math]::Max(0, $TotalSources - $totalRows)

    [pscustomobject]@{
        WorkerTable = $workerTable
        Summary = [pscustomobject]@{
            RunningWorkers = $workerProcesses.Count
            Position = $totalRows
            Total = $TotalSources
            Percent = [math]::Round(100.0 * $totalRows / $TotalSources, 1)
            NewlyProcessed = $totalProcessed
            Warnings = $totalWarnings
            Failed = $totalFailed
            RatePerHour = [math]::Round($ratePerHour, 0)
            HoursRemaining = if ($ratePerHour -gt 0) { [math]::Round($remaining / $ratePerHour, 2) } else { 0 }
            SecondsSinceUpdate = if ($null -ne $latestUpdate) {
                [math]::Round(((Get-Date) - $latestUpdate).TotalSeconds, 1)
            }
            else {
                -1
            }
        }
    }
}

do {
    if ($Watch) {
        Clear-Host
    }

    $status = Get-WallManStatus
    Write-Host "WallMan Check Database status at $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
    Write-Host ''
    $status.WorkerTable | Format-Table Worker, Position, Total, Percent, New, Warnings, Failed, LastTile, Updated -AutoSize
    Write-Host ''
    $status.Summary | Format-List

    if ($Watch) {
        Write-Host "Refreshing every $RefreshSeconds seconds. Press Ctrl+C to close this monitor only."
        Start-Sleep -Seconds $RefreshSeconds
    }
} while ($Watch)
