<#
.SYNOPSIS
    Starts multiple disjoint WallMan Check Database workers.

.EXAMPLE
    powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\start_wallman_check_parallel.ps1 `
      -RootPath 'D:\桌面\dac\01_data\512mdata' -Workers 8
#>

[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$RootPath,

    [ValidateRange(1, 16)]
    [int]$Workers = 8,

    [ValidateRange(1, 4)]
    [int]$SaveConcurrency = 1,

    [string]$WorkerScript = ''
)

Set-StrictMode -Version 2.0
$ErrorActionPreference = 'Stop'

if ([string]::IsNullOrWhiteSpace($WorkerScript)) {
    $WorkerScript = Join-Path $PSScriptRoot 'batch_check_winprop_odb.ps1'
}

if (-not (Test-Path -LiteralPath $RootPath -PathType Container)) {
    throw "RootPath is not a directory: $RootPath"
}
if (-not (Test-Path -LiteralPath $WorkerScript -PathType Leaf)) {
    throw "Worker script was not found: $WorkerScript"
}

$resolvedRoot = (Resolve-Path -LiteralPath $RootPath).Path
$resolvedWorkerScript = (Resolve-Path -LiteralPath $WorkerScript).Path

$existingWorkers = @(
    Get-CimInstance Win32_Process -Filter "Name='powershell.exe'" |
        Where-Object {
            $_.CommandLine -match '(?i)-File\s+.*batch_check_winprop_odb\.ps1'
        }
)

if ($existingWorkers.Count -gt 0) {
    $details = ($existingWorkers | ForEach-Object {
        $shard = if ($_.CommandLine -match '(?i)-ShardIndex\s+(\d+)') { $matches[1] } else { '?' }
        "PID=$($_.ProcessId), ShardIndex=$shard"
    }) -join '; '
    throw "Existing WallMan batch workers are still running ($details). Stop every old worker with Ctrl+C before changing the shard count."
}

$started = New-Object System.Collections.Generic.List[object]
for ($shardIndex = 0; $shardIndex -lt $Workers; $shardIndex++) {
    $arguments = '-NoProfile -ExecutionPolicy Bypass -File "{0}" -RootPath "{1}" -ShardCount {2} -ShardIndex {3} -SaveConcurrency {4}' -f `
        $resolvedWorkerScript, $resolvedRoot, $Workers, $shardIndex, $SaveConcurrency

    $process = Start-Process `
        -FilePath 'powershell.exe' `
        -ArgumentList $arguments `
        -WindowStyle Minimized `
        -PassThru

    $started.Add([pscustomobject]@{
        Worker = $shardIndex + 1
        ShardIndex = $shardIndex
        ProcessId = $process.Id
        Log = Join-Path $resolvedRoot ('_wallman_check_database_worker_{0:D2}_of_{1:D2}.csv' -f ($shardIndex + 1), $Workers)
    })
}

Write-Host "Started $Workers disjoint WallMan workers."
Write-Host "Concurrent Save As slots: $SaveConcurrency"
Write-Host 'The worker windows were opened minimized; restore them from the taskbar to monitor or press Ctrl+C.'
$started | Format-Table Worker, ShardIndex, ProcessId, Log -AutoSize
