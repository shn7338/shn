param(
    [string]$BaseDir = "",
    [int]$StartTile = 11,
    [int]$EndTile = 27360,
    [string]$HeightField = "HEIGHT_M",
    [string]$Ogr2Ogr = "C:\Program Files\QGIS 4.0.3\bin\ogr2ogr.exe",
    [switch]$OverwriteOdb,
    [switch]$KeepOda,
    [switch]$StopOnError
)

$ErrorActionPreference = "Stop"

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

$logRoot = Join-Path $scriptDir "logs\odb_000011_027360"
New-Item -ItemType Directory -Path $logRoot -Force | Out-Null

$successLog = Join-Path $logRoot "success.txt"
$failedLog = Join-Path $logRoot "failed.txt"
$summaryLog = Join-Path $logRoot "summary.txt"

function Get-TileName {
    param([int]$Index)
    return ("tile_{0:D6}" -f $Index)
}

function Write-LogLine {
    param(
        [string]$Path,
        [string]$Line
    )
    Add-Content -LiteralPath $Path -Value $Line -Encoding UTF8
}

$startedAt = Get-Date
Write-Host "BaseDir: $BaseDir"
Write-Host "Range: tile_$("{0:D6}" -f $StartTile) - tile_$("{0:D6}" -f $EndTile)"
Write-Host "LogDir: $logRoot"
Write-Host "OverwriteOdb: $([bool]$OverwriteOdb)"
Write-Host ""

Write-LogLine $summaryLog ("START {0:yyyy-MM-dd HH:mm:ss} range={1}-{2} overwrite={3}" -f $startedAt, $StartTile, $EndTile, [bool]$OverwriteOdb)

$total = $EndTile - $StartTile + 1
$ok = 0
$skipped = 0
$failed = 0

for ($i = $StartTile; $i -le $EndTile; $i++) {
    $tile = Get-TileName $i
    $tileDir = Join-Path $BaseDir $tile
    $odbPath = Join-Path $tileDir "$tile.odb"
    $tileLog = Join-Path $logRoot "$tile.log"
    $index = $i - $StartTile + 1

    Write-Host ("[{0}/{1}] {2}" -f $index, $total, $tile)

    if (-not (Test-Path -LiteralPath $tileDir)) {
        $failed++
        $msg = "missing tile directory"
        Write-Warning "[$tile] $msg"
        Write-LogLine $failedLog ("{0}`t{1}" -f $tile, $msg)
        if ($StopOnError) { throw "[$tile] $msg" }
        continue
    }

    if ((Test-Path -LiteralPath $odbPath) -and -not $OverwriteOdb) {
        $skipped++
        Write-Host "[$tile] ODB exists, skipped"
        continue
    }

    $args = @(
        "-NoProfile",
        "-ExecutionPolicy", "Bypass",
        "-File", $generator,
        "-BaseDir", $BaseDir,
        "-StartTile", $i,
        "-EndTile", $i,
        "-HeightField", $HeightField,
        "-Ogr2Ogr", $Ogr2Ogr,
        "-OverwriteOda"
    )

    if ($OverwriteOdb) {
        $args += "-OverwriteOdb"
    }
    if ($KeepOda) {
        $args += "-KeepOda"
    }

    $previousErrorActionPreference = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try {
        $output = & powershell @args 2>&1
        $exitCode = $LASTEXITCODE
    }
    finally {
        $ErrorActionPreference = $previousErrorActionPreference
    }
    $output | Set-Content -LiteralPath $tileLog -Encoding UTF8

    if ($exitCode -eq 0 -and (Test-Path -LiteralPath $odbPath)) {
        $ok++
        Write-Host "[$tile] created ODB"
        Write-LogLine $successLog ("{0}`t{1}" -f $tile, $odbPath)
    }
    else {
        $failed++
        $msg = "generator exit=$exitCode odbExists=$((Test-Path -LiteralPath $odbPath)) log=$tileLog"
        Write-Warning "[$tile] failed: $msg"
        Write-LogLine $failedLog ("{0}`t{1}" -f $tile, $msg)
        if ($StopOnError) { throw "[$tile] failed: $msg" }
    }
}

$finishedAt = Get-Date
$elapsed = $finishedAt - $startedAt
$summary = "FINISH {0:yyyy-MM-dd HH:mm:ss} ok={1} skipped={2} failed={3} elapsed={4}" -f $finishedAt, $ok, $skipped, $failed, $elapsed
Write-LogLine $summaryLog $summary

Write-Host ""
Write-Host $summary
Write-Host "SuccessLog: $successLog"
Write-Host "FailedLog: $failedLog"
