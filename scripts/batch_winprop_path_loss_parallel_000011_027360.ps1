param(
    [string]$BaseDir = "",
    [string]$WinPropCli = "C:\Program Files\Altair\2020\feko\bin\WinPropCLI.exe",
    [int]$StartTile = 11,
    [int]$EndTile = 27360,
    [int]$ThrottleLimit = 2,
    [int]$WinPropMultiThreading = 1,
    [string]$TemplateTile = "tile_000001",
    [string]$ProjectSuffix = "3500MHz_a",
    [string]$HeightField = "HEIGHT_M",
    [double]$AntennaAboveMaxBuilding = 5.0,
    [switch]$OverwriteProjects,
    [switch]$OverwriteResults
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
$runner = Join-Path $scriptDir "batch_winprop_path_loss.ps1"
if (-not (Test-Path -LiteralPath $runner)) {
    throw "Path loss runner script not found: $runner"
}
if (-not (Test-Path -LiteralPath $BaseDir)) {
    throw "BaseDir does not exist: $BaseDir"
}
if (-not (Test-Path -LiteralPath $WinPropCli)) {
    throw "WinPropCLI does not exist: $WinPropCli"
}

$logRoot = Join-Path $scriptDir "logs\path_loss_000011_027360"
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

function Complete-PathLossJob {
    param([System.Management.Automation.Job]$Job)

    $result = Receive-Job -Job $Job
    Remove-Job -Job $Job

    $tile = $result.Tile
    if ($result.Success) {
        Add-LogLine $successLog ("{0}`t{1}" -f $tile, $result.ResultDir)
        Write-Host ("[{0}] created path loss" -f $tile)
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
Write-Host "WinPropMultiThreading: $WinPropMultiThreading"
Write-Host "LogDir: $logRoot"
Write-Host "OverwriteResults: $([bool]$OverwriteResults)"
Write-Host ""

Add-LogLine $summaryLog ("START {0:yyyy-MM-dd HH:mm:ss} range={1}-{2} throttle={3} winpropThreads={4} overwriteResults={5}" -f $startedAt, $StartTile, $EndTile, $ThrottleLimit, $WinPropMultiThreading, [bool]$OverwriteResults)

$jobs = New-Object System.Collections.Generic.List[System.Management.Automation.Job]
$ok = 0
$skipped = 0
$failed = 0
$submitted = 0
$total = $EndTile - $StartTile + 1

for ($i = $StartTile; $i -le $EndTile; $i++) {
    while ($jobs.Count -ge $ThrottleLimit) {
        $done = Wait-Job -Job $jobs -Any
        $result = Complete-PathLossJob $done
        [void]$jobs.Remove($done)
        if ($result.Success) { $ok++ } else { $failed++ }
    }

    $tile = Get-TileName $i
    $tileDir = Join-Path $BaseDir $tile
    $odbPath = Join-Path $tileDir "$tile.odb"
    $resultDir = Join-Path $tileDir "${tile}_result1"
    $pathLossTxt = Join-Path $resultDir "Site  1 Antenna 1 Path Loss.txt"
    $pathLossFpl = Join-Path $resultDir "Site  1 Antenna 1 Path Loss.fpl"
    $tileLog = Join-Path $logRoot "$tile.log"

    if (-not (Test-Path -LiteralPath $tileDir)) {
        $failed++
        $msg = "missing tile directory"
        Add-LogLine $failedLog ("{0}`t{1}" -f $tile, $msg)
        Write-Warning "[$tile] $msg"
        continue
    }

    if (-not (Test-Path -LiteralPath $odbPath)) {
        $failed++
        $msg = "missing odb: $odbPath"
        Add-LogLine $failedLog ("{0}`t{1}" -f $tile, $msg)
        Write-Warning "[$tile] $msg"
        continue
    }

    if (((Test-Path -LiteralPath $pathLossTxt) -or (Test-Path -LiteralPath $pathLossFpl)) -and -not $OverwriteResults) {
        $skipped++
        Add-LogLine $skippedLog ("{0}`t{1}" -f $tile, $resultDir)
        if (($skipped % 100) -eq 0 -or $i -eq $StartTile) {
            Write-Host ("[{0}/{1}] skipped existing result up to {2}" -f ($i - $StartTile + 1), $total, $tile)
        }
        continue
    }

    $submitted++
    Write-Host ("[{0}/{1}] queued {2}" -f ($i - $StartTile + 1), $total, $tile)

    $job = Start-Job -ArgumentList @(
        $tile, $i, $runner, $BaseDir, $WinPropCli, $TemplateTile, $ProjectSuffix, $HeightField,
        $AntennaAboveMaxBuilding, $WinPropMultiThreading, [bool]$OverwriteProjects, [bool]$OverwriteResults,
        $tileLog, $resultDir, $pathLossTxt, $pathLossFpl
    ) -ScriptBlock {
        param(
            $tile, $index, $runner, $baseDir, $winPropCli, $templateTile, $projectSuffix, $heightField,
            $antennaAboveMaxBuilding, $winPropMultiThreading, $overwriteProjects, $overwriteResults,
            $tileLog, $resultDir, $pathLossTxt, $pathLossFpl
        )

        $args = @(
            "-NoProfile",
            "-ExecutionPolicy", "Bypass",
            "-File", $runner,
            "-BaseDir", $baseDir,
            "-WinPropCli", $winPropCli,
            "-StartTile", $index,
            "-EndTile", $index,
            "-TemplateTile", $templateTile,
            "-ProjectSuffix", $projectSuffix,
            "-HeightField", $heightField,
            "-AntennaAboveMaxBuilding", $antennaAboveMaxBuilding,
            "-MultiThreading", $winPropMultiThreading
        )

        if ($overwriteProjects) {
            $args += "-OverwriteProjects"
        }
        if ($overwriteResults) {
            $args += "-OverwriteResults"
        }

        $output = & powershell @args 2>&1
        $exitCode = $LASTEXITCODE
        $output | Set-Content -LiteralPath $tileLog -Encoding UTF8

        $hasResult = (Test-Path -LiteralPath $pathLossTxt) -or (Test-Path -LiteralPath $pathLossFpl)
        [pscustomobject]@{
            Tile = $tile
            Success = ($exitCode -eq 0 -and $hasResult)
            ResultDir = $resultDir
            Message = "exit=$exitCode hasResult=$hasResult log=$tileLog"
        }
    }

    [void]$jobs.Add($job)
}

while ($jobs.Count -gt 0) {
    $done = Wait-Job -Job $jobs -Any
    $result = Complete-PathLossJob $done
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
