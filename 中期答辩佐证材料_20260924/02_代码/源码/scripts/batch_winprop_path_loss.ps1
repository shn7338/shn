param(
    [string]$BaseDir = "",
    [string]$WinPropCli = "C:\Program Files\Altair\2020\feko\bin\WinPropCLI.exe",
    [int]$StartTile = 1,
    [int]$EndTile = 10,
    [string]$TemplateTile = "tile_000001",
    [string]$ProjectSuffix = "3500MHz_a",
    [string]$HeightField = "HEIGHT_M",
    [double]$AntennaAboveMaxBuilding = 5.0,
    [switch]$PrepareOnly,
    [switch]$OverwriteProjects,
    [switch]$OverwriteResults,
    [int]$MultiThreading = -1
)

$ErrorActionPreference = "Stop"

if ([string]::IsNullOrWhiteSpace($BaseDir)) {
    $desktopName = -join @([char]0x684C, [char]0x9762)
    $BaseDir = Join-Path "D:\$desktopName\dac" "512mdata"
}

function Get-TileName {
    param([int]$Index)
    return ("tile_{0:D6}" -f $Index)
}

function Get-ShapefileBounds {
    param([string]$ShpPath)

    if (-not (Test-Path -LiteralPath $ShpPath)) {
        throw "Missing shapefile: $ShpPath"
    }

    $bytes = [System.IO.File]::ReadAllBytes($ShpPath)
    if ($bytes.Length -lt 68) {
        throw "Invalid shapefile header: $ShpPath"
    }

    return [pscustomobject]@{
        XMin = [BitConverter]::ToDouble($bytes, 36)
        YMin = [BitConverter]::ToDouble($bytes, 44)
        XMax = [BitConverter]::ToDouble($bytes, 52)
        YMax = [BitConverter]::ToDouble($bytes, 60)
    }
}

function Get-MaxDbfNumericValue {
    param(
        [string]$DbfPath,
        [string]$FieldName
    )

    if (-not (Test-Path -LiteralPath $DbfPath)) {
        throw "Missing DBF file: $DbfPath"
    }

    $bytes = [System.IO.File]::ReadAllBytes($DbfPath)
    if ($bytes.Length -lt 33) {
        throw "Invalid DBF header: $DbfPath"
    }

    $recordCount = [BitConverter]::ToUInt32($bytes, 4)
    $headerLength = [BitConverter]::ToUInt16($bytes, 8)
    $recordLength = [BitConverter]::ToUInt16($bytes, 10)
    $fieldOffset = 32
    $recordOffset = 1
    $targetStart = $null
    $targetLength = $null

    while ($fieldOffset + 32 -le $bytes.Length -and $bytes[$fieldOffset] -ne 0x0D) {
        $nameBytes = $bytes[$fieldOffset..($fieldOffset + 10)]
        $zeroIndex = [Array]::IndexOf($nameBytes, [byte]0)
        if ($zeroIndex -ge 0) {
            $nameBytes = $nameBytes[0..($zeroIndex - 1)]
        }
        $name = [System.Text.Encoding]::ASCII.GetString($nameBytes).Trim()
        $length = [int]$bytes[$fieldOffset + 16]

        if ($name -eq $FieldName) {
            $targetStart = $recordOffset
            $targetLength = $length
        }

        $recordOffset += $length
        $fieldOffset += 32
    }

    if ($null -eq $targetStart) {
        throw "Field '$FieldName' not found in $DbfPath"
    }

    $maxValue = $null
    for ($i = 0; $i -lt $recordCount; $i++) {
        $rowStart = $headerLength + ($i * $recordLength)
        if ($rowStart + $recordLength -gt $bytes.Length) {
            break
        }
        if ($bytes[$rowStart] -eq [byte][char]'*') {
            continue
        }

        $valueBytes = $bytes[($rowStart + $targetStart)..($rowStart + $targetStart + $targetLength - 1)]
        $raw = [System.Text.Encoding]::ASCII.GetString($valueBytes).Trim()
        if ([string]::IsNullOrWhiteSpace($raw)) {
            continue
        }

        $value = [double]::Parse($raw, [System.Globalization.CultureInfo]::InvariantCulture)
        if ($null -eq $maxValue -or $value -gt $maxValue) {
            $maxValue = $value
        }
    }

    if ($null -eq $maxValue) {
        throw "No numeric values found in field '$FieldName' in $DbfPath"
    }

    return $maxValue
}

function Read-ProjectText {
    param([string]$Path)
    return [System.IO.File]::ReadAllText($Path, [System.Text.Encoding]::Default)
}

function Write-ProjectText {
    param(
        [string]$Path,
        [string]$Text
    )
    [System.IO.File]::WriteAllText($Path, $Text, [System.Text.Encoding]::Default)
}

function Convert-PathForRegex {
    param([string]$Path)
    return [regex]::Escape($Path)
}

if (-not (Test-Path -LiteralPath $BaseDir)) {
    throw "BaseDir does not exist: $BaseDir"
}

if (-not (Test-Path -LiteralPath $WinPropCli)) {
    throw "WinPropCLI does not exist: $WinPropCli"
}

$templateDir = Join-Path $BaseDir $TemplateTile
$templateBase = Join-Path $templateDir "${TemplateTile}_${ProjectSuffix}"
$templateNet = "$templateBase.net"
$templateNup = "$templateBase.nup"
$templateWpi = "$templateBase.wpi"
$templateMic = "$templateBase.mic"

foreach ($path in @($templateNet, $templateNup, $templateWpi, $templateMic)) {
    if (-not (Test-Path -LiteralPath $path)) {
        throw "Missing template project file: $path"
    }
}

$templateNetText = Read-ProjectText $templateNet
$templateBounds = Get-ShapefileBounds (Join-Path $templateDir "$TemplateTile.shp")

$areaMatch = [regex]::Match(
    $templateNetText,
    '(?m)^COORDINATES_AREA\s+([0-9.+-]+)\s+([0-9.+-]+)\s+([0-9.+-]+)\s+([0-9.+-]+)'
)
if (-not $areaMatch.Success) {
    throw "Cannot find COORDINATES_AREA in template net file."
}

$templateAreaXMin = [double]$areaMatch.Groups[1].Value
$templateAreaYMin = [double]$areaMatch.Groups[2].Value
$templateAreaXMax = [double]$areaMatch.Groups[3].Value
$templateAreaYMax = [double]$areaMatch.Groups[4].Value
$areaWidth = $templateAreaXMax - $templateAreaXMin
$areaHeight = $templateAreaYMax - $templateAreaYMin

$antennaMatch = [regex]::Match(
    $templateNetText,
    '(?m)^ANTENNA 1 POSITION\s+([0-9.+-]+),\s*([0-9.+-]+),\s*([0-9.+-]+)'
)
if (-not $antennaMatch.Success) {
    throw "Cannot find ANTENNA 1 POSITION in template net file."
}

$siteMatch = [regex]::Match(
    $templateNetText,
    '(?m)^SITE 1 SITE_LOCATION\s+([0-9.+-]+)\s+([0-9.+-]+)\s+([0-9.+-]+)'
)
if (-not $siteMatch.Success) {
    throw "Cannot find SITE 1 SITE_LOCATION in template net file."
}

$antennaOffsetX = [double]$antennaMatch.Groups[1].Value - $templateAreaXMin
$antennaOffsetY = [double]$antennaMatch.Groups[2].Value - $templateAreaYMin
$antennaZ = [double]$antennaMatch.Groups[3].Value
$siteOffsetX = [double]$siteMatch.Groups[1].Value - $templateAreaXMin
$siteOffsetY = [double]$siteMatch.Groups[2].Value - $templateAreaYMin
$siteZ = [double]$siteMatch.Groups[3].Value

Write-Host "BaseDir: $BaseDir"
Write-Host "WinPropCLI: $WinPropCli"
Write-Host "Template: $templateBase"
Write-Host ("Template area size: {0} x {1}" -f $areaWidth, $areaHeight)
Write-Host ""

for ($i = $StartTile; $i -le $EndTile; $i++) {
    $tile = Get-TileName $i
    $tileDir = Join-Path $BaseDir $tile
    $tileBase = Join-Path $tileDir "${tile}_${ProjectSuffix}"
    $tileNet = "$tileBase.net"
    $tileNup = "$tileBase.nup"
    $tileWpi = "$tileBase.wpi"
    $tileMic = "$tileBase.mic"
    $tileShp = Join-Path $tileDir "$tile.shp"
    $tileDbf = Join-Path $tileDir "$tile.dbf"

    if (-not (Test-Path -LiteralPath $tileDir)) {
        throw "Missing tile directory: $tileDir"
    }

    $bounds = Get-ShapefileBounds $tileShp
    $xMin = $bounds.XMin
    $yMin = $bounds.YMin
    $xMax = $xMin + $areaWidth
    $yMax = $yMin + $areaHeight

    $antennaX = $xMin + $antennaOffsetX
    $antennaY = $yMin + $antennaOffsetY
    $maxBuildingHeight = Get-MaxDbfNumericValue $tileDbf $HeightField
    $antennaZForTile = $maxBuildingHeight + $AntennaAboveMaxBuilding
    $siteX = $xMin + $siteOffsetX
    $siteY = $yMin + $siteOffsetY

    if ($i -eq 1) {
        $resultName = "${tile}_result2"
    }
    else {
        $resultName = "${tile}_result1"
    }
    $resultDir = Join-Path $tileDir $resultName

    $projectFilesExist = @($tileNet, $tileNup, $tileWpi, $tileMic) | Where-Object {
        Test-Path -LiteralPath $_
    }
    if ($projectFilesExist.Count -gt 0 -and -not $OverwriteProjects) {
        if (Test-Path -LiteralPath $tileNet) {
            $existingNetText = Read-ProjectText $tileNet
            $updatedNetText = $existingNetText -replace '(?m)^ANTENNA 1 POSITION\s+([0-9.+-]+),\s*([0-9.+-]+),\s*[0-9.+-]+\s*$', ("ANTENNA 1 POSITION `$1, `$2, {0:F10} " -f $antennaZForTile)
            if ($updatedNetText -ne $existingNetText) {
                Write-ProjectText $tileNet $updatedNetText
                Write-Host ("[$tile] updated antenna height -> max {0:F3} + {1:F3} = {2:F3} m" -f $maxBuildingHeight, $AntennaAboveMaxBuilding, $antennaZForTile)
            }
            else {
                Write-Host ("[$tile] project exists, antenna height already checked -> {0:F3} m" -f $antennaZForTile)
            }
        }
        if (Test-Path -LiteralPath $tileNup) {
            $existingNupText = Read-ProjectText $tileNup
            $updatedNupText = $existingNupText -replace '(?m)^DATABASE_FILE\s+".*"', "DATABASE_FILE `"$tile`""
            if ($updatedNupText -ne $existingNupText) {
                Write-ProjectText $tileNup $updatedNupText
                Write-Host "[$tile] updated database reference -> $tile"
            }
        }
        else {
            Write-Host "[$tile] project exists, keeping existing files. Use -OverwriteProjects to regenerate."
        }
    }
    else {
        $netText = Read-ProjectText $templateNet
        $netText = $netText.Replace($templateBase, $tileBase)
        $netText = $netText -replace [regex]::Escape($TemplateTile), $tile
        $netText = $netText -replace '(?m)^OUTPUT_PROPAGATION_FILES\s+".*"', "OUTPUT_PROPAGATION_FILES `"$resultName`""
        $netText = $netText -replace '(?m)^COORDINATES_AREA\s+.+$', ("COORDINATES_AREA {0:F10} {1:F10} {2:F10} {3:F10}" -f $xMin, $yMin, $xMax, $yMax)
        $netText = $netText -replace '(?m)^ANTENNA 0 PREDICTION_AREA COORDINATES_AREA\s+.+$', ("ANTENNA 0 PREDICTION_AREA COORDINATES_AREA {0:F10} {1:F10} {2:F10} {3:F10}" -f $xMin, $yMin, $xMax, $yMax)
        $netText = $netText -replace '(?m)^ANTENNA -1 PREDICTION_AREA COORDINATES_AREA\s+.+$', ("ANTENNA -1 PREDICTION_AREA COORDINATES_AREA {0:F10} {1:F10} {2:F10} {3:F10}" -f $xMin, $yMin, $xMax, $yMax)
        $netText = $netText -replace '(?m)^SITE 1 SITE_LOCATION\s+.+$', ("SITE 1 SITE_LOCATION {0:F10} {1:F10} {2:F10}" -f $siteX, $siteY, $siteZ)
        $netText = $netText -replace '(?m)^ANTENNA 1 POSITION\s+.+$', ("ANTENNA 1 POSITION {0:F10}, {1:F10}, {2:F10} " -f $antennaX, $antennaY, $antennaZForTile)
        Write-ProjectText $tileNet $netText

        $nupText = Read-ProjectText $templateNup
        $nupText = $nupText.Replace($templateBase, $tileBase)
        $nupText = $nupText -replace [regex]::Escape($TemplateTile), $tile
        $nupText = $nupText -replace '(?m)^DATABASE_FILE\s+".*"', "DATABASE_FILE `"$tile`""
        Write-ProjectText $tileNup $nupText

        foreach ($copy in @(
            @{ Source = $templateWpi; Target = $tileWpi },
            @{ Source = $templateMic; Target = $tileMic }
        )) {
            Copy-Item -LiteralPath $copy.Source -Destination $copy.Target -Force
        }

        Write-Host ("[$tile] prepared project -> $tileNet; antenna height = max {0:F3} + {1:F3} = {2:F3} m" -f $maxBuildingHeight, $AntennaAboveMaxBuilding, $antennaZForTile)
    }

    if ($PrepareOnly) {
        continue
    }

    $odbPath = Join-Path $tileDir "$tile.odb"
    if (-not (Test-Path -LiteralPath $odbPath)) {
        throw "Missing WinProp urban database: $odbPath. Open/import the tile shapefile in WallMan first, or provide a .odb for this tile."
    }

    if ((Test-Path -LiteralPath $resultDir) -and -not $OverwriteResults) {
        Write-Host "[$tile] result exists, skipping: $resultDir. Use -OverwriteResults to recompute."
        continue
    }

    if ((Test-Path -LiteralPath $resultDir) -and $OverwriteResults) {
        Remove-Item -LiteralPath $resultDir -Recurse -Force
    }

    Write-Host "[$tile] running network prediction..."
    $args = @(
        "-F", $tileNet,
        "-P",
        "--results-pro", $resultDir,
        "--multi-threading", $MultiThreading
    )
    & $WinPropCli @args
    if ($LASTEXITCODE -ne 0) {
        throw "WinPropCLI failed for $tile with exit code $LASTEXITCODE"
    }

    $pathLossTxt = Join-Path $resultDir "Site  1 Antenna 1 Path Loss.txt"
    $pathLossFpl = Join-Path $resultDir "Site  1 Antenna 1 Path Loss.fpl"
    if ((Test-Path -LiteralPath $pathLossTxt) -or (Test-Path -LiteralPath $pathLossFpl)) {
        Write-Host "[$tile] done -> $resultDir"
    }
    else {
        Write-Warning "[$tile] finished, but Path Loss output was not found in $resultDir"
    }
}

Write-Host ""
Write-Host "All requested tiles processed."
