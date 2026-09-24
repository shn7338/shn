$ErrorActionPreference = 'Stop'

$toolRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$profilePath = Join-Path $toolRoot 'profile'
$chromePath = 'C:\Program Files\Google\Chrome\Application\chrome.exe'
$debugPort = 9222

if (-not (Test-Path -LiteralPath $chromePath)) {
    throw "Chrome not found at $chromePath"
}

New-Item -ItemType Directory -Force -Path $profilePath | Out-Null

$existing = Get-NetTCPConnection -State Listen -LocalPort $debugPort -ErrorAction SilentlyContinue
if ($existing) {
    Write-Output "Controlled Chrome is already listening at http://127.0.0.1:$debugPort"
    exit 0
}

$arguments = @(
    "--remote-debugging-address=127.0.0.1"
    "--remote-debugging-port=$debugPort"
    "--user-data-dir=$profilePath"
    '--no-first-run'
    '--no-default-browser-check'
    'https://www.cnki.net/'
)

Start-Process -FilePath $chromePath -ArgumentList $arguments

$deadline = (Get-Date).AddSeconds(20)
do {
    Start-Sleep -Milliseconds 500
    $listener = Get-NetTCPConnection -State Listen -LocalPort $debugPort -ErrorAction SilentlyContinue
} until ($listener -or (Get-Date) -ge $deadline)

if (-not $listener) {
    throw "Chrome opened, but its debugging endpoint did not start on port $debugPort."
}

Write-Output "Controlled Chrome started at http://127.0.0.1:$debugPort"
Write-Output "Complete the BJTU/CNKI login in the visible Chrome window."
