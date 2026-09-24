param(
    [string]$BaseDir = "",
    [int]$StartTile = 11,
    [int]$EndTile = 27360,
    [int]$ThrottleLimit = 4,
    [string]$HeightField = "HEIGHT_M",
    [string]$Ogr2Ogr = "C:\Program Files\QGIS 4.0.3\bin\ogr2ogr.exe",
    [switch]$OverwriteOdb,
    [switch]$KeepOda
)

$ErrorActionPreference = "Stop"

if ($ThrottleLimit -lt 1) {
    throw "ThrottleLimit must be >= 1"
}

if ([string]::IsNullOrWhiteSpace($BaseDir)) {
    $desktopName = -join @([char]0x684C, [char]0x9762)
    $BaseDir = Join-Path "D:\$desktopName\dac" "512mdata"
}

$scriptDir = Split-Path -Parent $PSCommandPath
$generator = Join-Path $scriptDir "generate_winprop_odb.ps1"
if (-not (Test-Path -LiteralPath $generator)) {
    throw "ODB generator script not found: $generator"
}
if (-not (Test-Path -LiteralPath $BaseDir)) {
    throw "BaseDir does not exist: $BaseDir"
}

$logRoot = Join-Path $scriptDir "logs\odb_parallel_000011_027360"
New-Item -ItemType Directory -Path $logRoot -Force | Out-Null

$successLog = Join-Path $logRoot "success.txt"
$failedLog = Join-Path $logRoot "failed.txt"
$skippedLog = Join-Path $logRoot "skipped.txt"
$summaryLog = Join-Path $logRoot "summary.txt"

function Get-TileName {
    param([int]$Index)
    return ("tile_{0:D6}" -f $Index)
}

function Add-LogLine {
    param(
        [string]$Path,
        [string]$Value
    )

    for ($attempt = 1; $attempt -le 20; $attempt++) {
        $stream = $null
        $writer = $null
        try {
            $encoding = New-Object System.Text.UTF8Encoding($false)
            $stream = [System.IO.File]::Open($Path, [System.IO.FileMode]::Append, [System.IO.FileAccess]::Write, [System.IO.FileShare]::ReadWrite)
            $writer = New-Object System.IO.StreamWriter($stream, $encoding)
            $writer.WriteLine($Value)
            return
        }
        catch {
            Start-Sleep -Milliseconds (100 * $attempt)
        }
        finally {
            if ($null -ne $writer) {
                $writer.Dispose()
            }
            elseif ($null -ne $stream) {
                $stream.Dispose()
            }
        }
    }

    throw "Could not write log after retries: $Path"
}

function Complete-Job {
    param([System.Management.Automation.Job]$Job)

    $result = Receive-Job -Job $Job
    Remove-Job -Job $Job

    $tile = $result.Tile
    if ($result.Success) {
        Add-LogLine $successLog ("{0}`t{1}" -f $tile, $result.OdbPath)
        Write-Host ("[{0}] created ODB" -f $tile)
    }
    else {
        Add-LogLine $failedLog ("{0}`t{1}" -f $tile, $result.Message)
        Write-Warning ("[{0}] failed: {1}" -f $tile, $result.Message)
    }

    return $result
}

$startedAt = Get-Date
Write-Host "BaseDir: $BaseDir"
Write-Host "Range: tile_$("{0:D6}" -f $StartTile) - tile_$("{0:D6}" -f $EndTile)"
Write-Host "ThrottleLimit: $ThrottleLimit"
Write-Host "LogDir: $logRoot"
Write-Host "OverwriteOdb: $([bool]$OverwriteOdb)"
Write-Host ""

Add-LogLine $summaryLog ("START {0:yyyy-MM-dd HH:mm:ss} range={1}-{2} throttle={3} overwrite={4}" -f $startedAt, $StartTile, $EndTile, $ThrottleLimit, [bool]$OverwriteOdb)

$jobs = New-Object System.Collections.Generic.List[System.Management.Automation.Job]
$ok = 0
$skipped = 0
$failed = 0
$submitted = 0
$total = $EndTile - $StartTile + 1

for ($i = $StartTile; $i -le $EndTile; $i++) {
    while ($jobs.Count -ge $ThrottleLimit) {
        $done = Wait-Job -Job $jobs -Any
        $result = Complete-Job $done
        [void]$jobs.Remove($done)
        if ($result.Success) { $ok++ } else { $failed++ }
    }

    $tile = Get-TileName $i
    $tileDir = Join-Path $BaseDir $tile
    $odbPath = Join-Path $tileDir "$tile.odb"
    $tileLog = Join-Path $logRoot "$tile.log"

    if (-not (Test-Path -LiteralPath $tileDir)) {
        $failed++
        $msg = "missing tile directory"
        Add-LogLine $failedLog ("{0}`t{1}" -f $tile, $msg)
        Write-Warning "[$tile] $msg"
        continue
    }

    if ((Test-Path -LiteralPath $odbPath) -and -not $OverwriteOdb) {
        $skipped++
        Add-LogLine $skippedLog ("{0}`t{1}" -f $tile, $odbPath)
        if (($skipped % 100) -eq 0 -or $i -eq $StartTile) {
            Write-Host ("[{0}/{1}] skipped existing ODB up to {2}" -f ($i - $StartTile + 1), $total, $tile)
        }
        continue
    }

    $submitted++
    Write-Host ("[{0}/{1}] queued {2}" -f ($i - $StartTile + 1), $total, $tile)

    $job = Start-Job -ArgumentList @(
        $tile, $i, $generator, $BaseDir, $HeightField, $Ogr2Ogr, [bool]$OverwriteOdb, [bool]$KeepOda, $tileLog, $odbPath
    ) -ScriptBlock {
        param($tile, $index, $generator, $baseDir, $heightField, $ogr2ogr, $overwriteOdb, $keepOda, $tileLog, $odbPath)

        $args = @(
            "-NoProfile",
            "-ExecutionPolicy", "Bypass",
            "-File", $generator,
            "-BaseDir", $baseDir,
            "-StartTile", $index,
            "-EndTile", $index,
            "-HeightField", $heightField,
            "-Ogr2Ogr", $ogr2ogr,
            "-OverwriteOda"
        )

        if ($overwriteOdb) {
            $args += "-OverwriteOdb"
        }
        if ($keepOda) {
            $args += "-KeepOda"
        }

        $output = & powershell @args 2>&1
        $exitCode = $LASTEXITCODE
        $output | Set-Content -LiteralPath $tileLog -Encoding UTF8

        $exists = Test-Path -LiteralPath $odbPath
        [pscustomobject]@{
            Tile = $tile
            Success = ($exitCode -eq 0 -and $exists)
            OdbPath = $odbPath
            Message = "exit=$exitCode odbExists=$exists log=$tileLog"
        }
    }

    [void]$jobs.Add($job)
}

while ($jobs.Count -gt 0) {
    $done = Wait-Job -Job $jobs -Any
    $result = Complete-Job $done
    [void]$jobs.Remove($done)
    if ($result.Success) { $ok++ } else { $failed++ }
}

$finishedAt = Get-Date
$elapsed = $finishedAt - $startedAt
$summary = "FINISH {0:yyyy-MM-dd HH:mm:ss} submitted={1} ok={2} skipped={3} failed={4} elapsed={5}" -f $finishedAt, $submitted, $ok, $skipped, $failed, $elapsed
Add-LogLine $summaryLog $summary

Write-Host ""
Write-Host $summary
Write-Host "SuccessLog: $successLog"
Write-Host "SkippedLog: $skippedLog"
Write-Host "FailedLog: $failedLog"
