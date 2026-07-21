param(
    [string]$BaseDir = "",
    [int]$StartTile = 2,
    [int]$EndTile = 10,
    [string]$HeightField = "HEIGHT_M",
    [string]$Ogr2Ogr = "C:\Program Files\QGIS 4.0.3\bin\ogr2ogr.exe",
    [switch]$OverwriteOda,
    [switch]$OverwriteOdb,
    [switch]$KeepOda,
    [switch]$LocalTileCoordinates
)

$ErrorActionPreference = "Stop"

if ([string]::IsNullOrWhiteSpace($BaseDir)) {
    $desktopName = -join @([char]0x684C, [char]0x9762)
    $BaseDir = Join-Path "D:\$desktopName\dac" "512mdata"
}

$repoRoot = Split-Path -Parent (Split-Path -Parent $PSCommandPath)
$pythonConverter = Join-Path $repoRoot "tools\winprop_convert_oda_to_odb.py"
$winPropApiBin = "C:\Program Files\Altair\2020\feko\api\winprop\bin"

function Get-TileName {
    param([int]$Index)
    return ("tile_{0:D6}" -f $Index)
}

function Get-ShapefileBounds {
    param([string]$ShpPath)

    $bytes = [System.IO.File]::ReadAllBytes($ShpPath)
    if ($bytes.Length -lt 100) {
        throw "Invalid shapefile header: $ShpPath"
    }

    return [pscustomobject]@{
        XMin = [BitConverter]::ToDouble($bytes, 36)
        YMin = [BitConverter]::ToDouble($bytes, 44)
        XMax = [BitConverter]::ToDouble($bytes, 52)
        YMax = [BitConverter]::ToDouble($bytes, 60)
    }
}

function Get-DbfFieldMap {
    param([byte[]]$Bytes)

    $headerLength = [BitConverter]::ToUInt16($Bytes, 8)
    $recordLength = [BitConverter]::ToUInt16($Bytes, 10)
    $fieldOffset = 32
    $recordOffset = 1
    $fields = @{}

    while ($fieldOffset + 32 -le $Bytes.Length -and $Bytes[$fieldOffset] -ne 0x0D) {
        $nameBytes = $Bytes[$fieldOffset..($fieldOffset + 10)]
        $zeroIndex = [Array]::IndexOf($nameBytes, [byte]0)
        if ($zeroIndex -gt 0) {
            $nameBytes = $nameBytes[0..($zeroIndex - 1)]
        }
        elseif ($zeroIndex -eq 0) {
            $nameBytes = @()
        }

        $name = [System.Text.Encoding]::ASCII.GetString($nameBytes).Trim()
        $length = [int]$Bytes[$fieldOffset + 16]
        $fields[$name] = [pscustomobject]@{
            Start = $recordOffset
            Length = $length
        }
        $recordOffset += $length
        $fieldOffset += 32
    }

    return [pscustomobject]@{
        HeaderLength = $headerLength
        RecordLength = $recordLength
        Fields = $fields
    }
}

function Get-DbfRecords {
    param(
        [string]$DbfPath,
        [string]$RequiredField
    )

    $bytes = [System.IO.File]::ReadAllBytes($DbfPath)
    if ($bytes.Length -lt 33) {
        throw "Invalid DBF header: $DbfPath"
    }

    $recordCount = [BitConverter]::ToUInt32($bytes, 4)
    $map = Get-DbfFieldMap $bytes
    if (-not $map.Fields.ContainsKey($RequiredField)) {
        throw "Field '$RequiredField' not found in $DbfPath"
    }

    $records = New-Object System.Collections.Generic.List[object]
    $field = $map.Fields[$RequiredField]
    for ($i = 0; $i -lt $recordCount; $i++) {
        $rowStart = $map.HeaderLength + ($i * $map.RecordLength)
        if ($rowStart + $map.RecordLength -gt $bytes.Length) {
            break
        }
        if ($bytes[$rowStart] -eq [byte][char]'*') {
            $records.Add($null)
            continue
        }

        $valueBytes = $bytes[($rowStart + $field.Start)..($rowStart + $field.Start + $field.Length - 1)]
        $raw = [System.Text.Encoding]::ASCII.GetString($valueBytes).Trim()
        if ([string]::IsNullOrWhiteSpace($raw)) {
            $records.Add($null)
            continue
        }

        $height = [double]::Parse($raw, [System.Globalization.CultureInfo]::InvariantCulture)
        $records.Add($height)
    }
    return $records
}

function Read-BigEndianInt32 {
    param([byte[]]$Bytes, [int]$Offset)

    return (($Bytes[$Offset] -shl 24) -bor ($Bytes[$Offset + 1] -shl 16) -bor ($Bytes[$Offset + 2] -shl 8) -bor $Bytes[$Offset + 3])
}

function Get-ShpPolygonRecords {
    param([string]$ShpPath)

    $bytes = [System.IO.File]::ReadAllBytes($ShpPath)
    $shxPath = [System.IO.Path]::ChangeExtension($ShpPath, ".shx")
    if (-not (Test-Path -LiteralPath $shxPath)) {
        throw "Missing shapefile index: $shxPath"
    }
    $shxBytes = [System.IO.File]::ReadAllBytes($shxPath)
    $records = New-Object System.Collections.Generic.List[object]

    for ($indexOffset = 100; $indexOffset + 8 -le $shxBytes.Length; $indexOffset += 8) {
        $recordOffsetWords = Read-BigEndianInt32 $shxBytes $indexOffset
        $contentLengthWords = Read-BigEndianInt32 $shxBytes ($indexOffset + 4)
        $offset = $recordOffsetWords * 2
        $contentLengthBytes = $contentLengthWords * 2
        if ($offset + 8 -gt $bytes.Length) {
            $records.Add($null)
            continue
        }
        $recordNumber = Read-BigEndianInt32 $bytes $offset
        $contentStart = $offset + 8
        if ($contentStart + $contentLengthBytes -gt $bytes.Length) {
            $records.Add($null)
            continue
        }

        $shapeType = [BitConverter]::ToInt32($bytes, $contentStart)
        if (($shapeType -eq 5 -or $shapeType -eq 15 -or $shapeType -eq 25) -and $contentLengthBytes -ge 44) {
            $numParts = [BitConverter]::ToInt32($bytes, $contentStart + 36)
            $numPoints = [BitConverter]::ToInt32($bytes, $contentStart + 40)
            $minimumBytes = 44 + ($numParts * 4) + ($numPoints * 16)
            if ($numParts -lt 1 -or $numPoints -lt 3 -or $minimumBytes -gt $contentLengthBytes) {
                $records.Add($null)
                continue
            }
            $parts = @()
            for ($p = 0; $p -lt $numParts; $p++) {
                $parts += [BitConverter]::ToInt32($bytes, $contentStart + 44 + ($p * 4))
            }

            $pointsStart = $contentStart + 44 + ($numParts * 4)
            $points = @()
            for ($pt = 0; $pt -lt $numPoints; $pt++) {
                $x = [BitConverter]::ToDouble($bytes, $pointsStart + ($pt * 16))
                $y = [BitConverter]::ToDouble($bytes, $pointsStart + ($pt * 16) + 8)
                $points += ,@($x, $y)
            }

            $rings = @()
            for ($p = 0; $p -lt $numParts; $p++) {
                $start = $parts[$p]
                $end = if ($p + 1 -lt $numParts) { $parts[$p + 1] } else { $numPoints }
                if ($end -gt $start) {
                    $rings += ,($points[$start..($end - 1)])
                }
            }

            $records.Add($rings)
        }
        else {
            $records.Add($null)
        }
    }

    return $records
}

function Get-RingArea {
    param([object[]]$Ring)

    $area = 0.0
    for ($i = 0; $i -lt $Ring.Count; $i++) {
        $j = ($i + 1) % $Ring.Count
        $area += ([double]$Ring[$i][0] * [double]$Ring[$j][1]) - ([double]$Ring[$j][0] * [double]$Ring[$i][1])
    }
    return $area / 2.0
}

function Normalize-Ring {
    param(
        [object[]]$Ring,
        [double]$XOrigin,
        [double]$YOrigin
    )

    if ($Ring.Count -gt 1) {
        $first = $Ring[0]
        $last = $Ring[$Ring.Count - 1]
        if ([math]::Abs([double]$first[0] - [double]$last[0]) -lt 0.000001 -and [math]::Abs([double]$first[1] - [double]$last[1]) -lt 0.000001) {
            $Ring = $Ring[0..($Ring.Count - 2)]
        }
    }

    if ($Ring.Count -lt 3) {
        return $null
    }

    if ((Get-RingArea $Ring) -lt 0) {
        [array]::Reverse($Ring)
    }

    $points = @()
    foreach ($point in $Ring) {
        $points += ,@(([double]$point[0] - $XOrigin), ([double]$point[1] - $YOrigin))
    }
    return $points
}

function Write-OdaFromTile {
    param(
        [string]$TileDir,
        [string]$TileName,
        [string]$OdaPath,
        [string]$HeightField
    )

    $shpPath = Join-Path $TileDir "$TileName.shp"
    if (-not (Test-Path -LiteralPath $shpPath)) {
        throw "Missing shapefile: $shpPath"
    }
    if (-not (Test-Path -LiteralPath $Ogr2Ogr)) {
        throw "ogr2ogr not found: $Ogr2Ogr"
    }

    $bounds = Get-ShapefileBounds $shpPath
    $tempGeoJson = Join-Path $env:TEMP "$TileName.geojson"
    if (Test-Path -LiteralPath $tempGeoJson) {
        Remove-Item -LiteralPath $tempGeoJson -Force
    }

    & $Ogr2Ogr -f GeoJSON $tempGeoJson $shpPath -lco COORDINATE_PRECISION=3
    if ($LASTEXITCODE -ne 0) {
        throw "ogr2ogr failed for $TileName with exit code $LASTEXITCODE"
    }

    $geoJson = Get-Content -Raw -LiteralPath $tempGeoJson | ConvertFrom-Json
    $buildingLines = New-Object System.Collections.Generic.List[string]
    $buildingId = 1
    foreach ($feature in $geoJson.features) {
        $height = $feature.properties.$HeightField
        if ($null -eq $height -or $null -eq $feature.geometry) {
            continue
        }

        $polygonList = @()
        if ($feature.geometry.type -eq "Polygon") {
            $polygonList += ,$feature.geometry.coordinates
        }
        elseif ($feature.geometry.type -eq "MultiPolygon") {
            foreach ($polygon in $feature.geometry.coordinates) {
                $polygonList += ,$polygon
            }
        }

        foreach ($polygon in $polygonList) {
            if ($null -eq $polygon -or $polygon.Count -lt 1) {
                continue
            }
            $outerRing = @($polygon[0])
            if ($outerRing.Count -lt 4) {
                continue
            }
            $xOrigin = if ($LocalTileCoordinates) { $bounds.XMin } else { 0.0 }
            $yOrigin = if ($LocalTileCoordinates) { $bounds.YMin } else { 0.0 }
            $points = Normalize-Ring $outerRing $xOrigin $yOrigin
            if ($null -eq $points -or $points.Count -lt 3) {
                continue
            }

            $line = " {0} {1}" -f $buildingId, $points.Count
            foreach ($point in $points) {
                $line += ("    {0,10:F3}, {1,10:F3}" -f [double]$point[0], [double]$point[1])
            }
            $line += ("    {0,6:F3} 1 0 " -f [double]$height)
            $buildingLines.Add($line)
            $buildingId++
        }
    }

    if ($buildingLines.Count -eq 0) {
        throw "No valid buildings found in $TileName"
    }

    if ($LocalTileCoordinates) {
        $maxX = $bounds.XMax - $bounds.XMin
        $maxY = $bounds.YMax - $bounds.YMin
    }
    else {
        $maxX = $bounds.XMax
        $maxY = $bounds.YMax
    }
    $header = @(
        "Database generated by Codex from $TileName shapefile",
        "Size of the area: ",
        ("  Max. x = {0:F3}" -f $maxX),
        ("  Max. y = {0:F3}" -f $maxY),
        "This line is intentionally left blank",
        ("{0} 0 5 {1:F3} {2:F3} WINPROP " -f $buildingLines.Count, $maxX, $maxY)
    )

    [System.IO.File]::WriteAllLines($OdaPath, $header + $buildingLines, [System.Text.Encoding]::ASCII)
    Remove-Item -LiteralPath $tempGeoJson -Force
    return $buildingLines.Count
}

if (-not (Test-Path -LiteralPath $BaseDir)) {
    throw "BaseDir does not exist: $BaseDir"
}
if (-not (Test-Path -LiteralPath $winPropApiBin)) {
    throw "WinProp API bin directory does not exist: $winPropApiBin"
}
if (-not (Test-Path -LiteralPath $Ogr2Ogr)) {
    throw "ogr2ogr not found: $Ogr2Ogr"
}
if (-not (Test-Path -LiteralPath $pythonConverter)) {
    throw "Python converter does not exist: $pythonConverter"
}

$env:PATH = "$winPropApiBin;$env:PATH"
$env:RADFLEX_PATH = $winPropApiBin

for ($i = $StartTile; $i -le $EndTile; $i++) {
    $tile = Get-TileName $i
    $tileDir = Join-Path $BaseDir $tile
    $sourceBase = Join-Path $tileDir $tile
    $odaPath = "$sourceBase.oda"
    $odbPath = "$sourceBase.odb"

    if (-not (Test-Path -LiteralPath $tileDir)) {
        throw "Missing tile directory: $tileDir"
    }
    if ((Test-Path -LiteralPath $odbPath) -and -not $OverwriteOdb) {
        Write-Host "[$tile] odb exists, skipping: $odbPath. Use -OverwriteOdb to regenerate."
        continue
    }
    if ((Test-Path -LiteralPath $odaPath) -and -not $OverwriteOda) {
        Write-Host "[$tile] reusing existing ODA: $odaPath"
    }
    else {
        $count = Write-OdaFromTile $tileDir $tile $odaPath $HeightField
        Write-Host "[$tile] wrote ODA with $count buildings -> $odaPath"
    }

    if ((Test-Path -LiteralPath $odbPath) -and $OverwriteOdb) {
        Remove-Item -LiteralPath $odbPath -Force
    }

    Write-Host "[$tile] converting ODA to ODB..."
    $tempDestBase = Join-Path $tileDir "${tile}_converted_tmp"
    $tempOdbPath = "$tempDestBase.odb"
    if (Test-Path -LiteralPath $tempOdbPath) {
        Remove-Item -LiteralPath $tempOdbPath -Force
    }

    python $pythonConverter $sourceBase $tempDestBase $winPropApiBin
    if ($LASTEXITCODE -ne 0) {
        if (Test-Path -LiteralPath $tempOdbPath) {
            Write-Warning "[$tile] converter exited with code $LASTEXITCODE after writing ODB; continuing because output exists."
        }
        else {
            throw "ODB conversion failed for $tile with exit code $LASTEXITCODE"
        }
    }
    if (-not (Test-Path -LiteralPath $tempOdbPath)) {
        throw "ODB was not created: $tempOdbPath"
    }
    Move-Item -LiteralPath $tempOdbPath -Destination $odbPath -Force

    Write-Host "[$tile] created ODB -> $odbPath"
    if (-not $KeepOda) {
        Remove-Item -LiteralPath $odaPath -Force
    }
}

Write-Host ""
Write-Host "ODB generation completed."
