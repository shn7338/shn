<#
.SYNOPSIS
    Prepares and runs a recoverable WinProp prediction smoke test.

.DESCRIPTION
    Clones the existing 3.5 GHz ProMan template, redirects every project to the
    canonical <tile>_checked.odb, runs propagation through WinPropCLI, and
    validates the ASCII path-loss grid. Existing historical projects/results
    are not overwritten.
#>

[CmdletBinding()]
param(
    [string]$BaseDir = '',

    [string]$ReadyManifestPath = '',

    [string]$TileListPath = '',

    [string]$OutputRoot = '',

    [string]$WinPropCli = 'C:\Program Files\Altair\2020\feko\bin\WinPropCLI.exe',

    [string]$TemplateTile = 'tile_000001',

    [string]$TemplateProjectSuffix = '3500MHz_a',

    [string]$ProjectSuffix = '3500MHz_checked_smoke',

    [string]$HeightField = 'HEIGHT_M',

    [double]$AntennaAboveMaxBuilding = 5.0,

    [ValidateRange(-1, 64)]
    [int]$MultiThreading = 1,

    [bool]$AllowOriginalOdbFallback = $true,

    [switch]$OverwriteProjects,

    [switch]$OverwriteResults
)

Set-StrictMode -Version 2.0
$ErrorActionPreference = 'Stop'

$workspaceRoot = Split-Path -Parent $PSScriptRoot
if ([string]::IsNullOrWhiteSpace($BaseDir)) {
    $desktopName = -join @([char]0x684C, [char]0x9762)
    $BaseDir = Join-Path "D:\$desktopName\dac" '512mdata'
}
if ([string]::IsNullOrWhiteSpace($ReadyManifestPath)) {
    $ReadyManifestPath = Join-Path $workspaceRoot 'prediction_ready_all_odb.csv'
}
if ([string]::IsNullOrWhiteSpace($TileListPath)) {
    $TileListPath = Join-Path $workspaceRoot 'prediction_smoke_10_tiles.csv'
}
if ([string]::IsNullOrWhiteSpace($OutputRoot)) {
    $OutputRoot = Join-Path $workspaceRoot 'prediction_smoke_results'
}

foreach ($directory in @($BaseDir)) {
    if (-not (Test-Path -LiteralPath $directory -PathType Container)) {
        throw "Directory does not exist: $directory"
    }
}
foreach ($file in @($ReadyManifestPath, $TileListPath, $WinPropCli)) {
    if (-not (Test-Path -LiteralPath $file -PathType Leaf)) {
        throw "Required file does not exist: $file"
    }
}
New-Item -ItemType Directory -Path $OutputRoot -Force | Out-Null

$resolvedBaseDir = (Resolve-Path -LiteralPath $BaseDir).Path
$resolvedOutputRoot = (Resolve-Path -LiteralPath $OutputRoot).Path

function Get-ShapefileBounds {
    param([string]$ShpPath)

    $bytes = [IO.File]::ReadAllBytes($ShpPath)
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

    $bytes = [IO.File]::ReadAllBytes($DbfPath)
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
            if ($zeroIndex -eq 0) {
                $nameBytes = @()
            }
            else {
                $nameBytes = $nameBytes[0..($zeroIndex - 1)]
            }
        }
        $name = [Text.Encoding]::ASCII.GetString($nameBytes).Trim()
        $length = [int]$bytes[$fieldOffset + 16]
        if ($name.Equals($FieldName, [StringComparison]::OrdinalIgnoreCase)) {
            $targetStart = $recordOffset
            $targetLength = $length
        }
        $recordOffset += $length
        $fieldOffset += 32
    }
    if ($null -eq $targetStart) {
        throw "Field '$FieldName' was not found in $DbfPath"
    }

    $maximum = $null
    for ($index = 0; $index -lt $recordCount; $index++) {
        $rowStart = $headerLength + ($index * $recordLength)
        if ($rowStart + $recordLength -gt $bytes.Length) {
            break
        }
        if ($bytes[$rowStart] -eq [byte][char]'*') {
            continue
        }
        $valueBytes = $bytes[
            ($rowStart + $targetStart)..
            ($rowStart + $targetStart + $targetLength - 1)
        ]
        $raw = [Text.Encoding]::ASCII.GetString($valueBytes).Trim()
        if ([string]::IsNullOrWhiteSpace($raw)) {
            continue
        }
        $value = [double]::Parse(
            $raw,
            [Globalization.CultureInfo]::InvariantCulture
        )
        if ($null -eq $maximum -or $value -gt $maximum) {
            $maximum = $value
        }
    }
    if ($null -eq $maximum) {
        throw "No numeric '$FieldName' values were found in $DbfPath"
    }
    return [double]$maximum
}

function Read-ProjectText {
    param([string]$Path)
    return [IO.File]::ReadAllText($Path, [Text.Encoding]::Default)
}

function Write-ProjectText {
    param(
        [string]$Path,
        [string]$Text
    )
    [IO.File]::WriteAllText($Path, $Text, [Text.Encoding]::Default)
}

function Replace-RequiredLine {
    param(
        [string]$Text,
        [string]$Pattern,
        [string]$Replacement,
        [string]$Description
    )
    if (-not [regex]::IsMatch($Text, $Pattern)) {
        throw "Could not locate $Description in the project template."
    }
    return [regex]::Replace($Text, $Pattern, $Replacement)
}

function Set-ProjectDatabase {
    param(
        [string]$NupPath,
        [string]$NetPath,
        [string]$DatabaseBase,
        [string]$DatabaseOdbPath
    )

    $nupText = Read-ProjectText $NupPath
    $nupText = Replace-RequiredLine $nupText `
        '(?m)^DATABASE_FILE\s+".*"[ \t\r]*$' `
        ('DATABASE_FILE "{0}"' -f $DatabaseBase) `
        'DATABASE_FILE'
    Write-ProjectText $NupPath $nupText

    $netText = Read-ProjectText $NetPath
    $netText = Replace-RequiredLine $netText `
        '(?m)^CLUTTER_DATABASE_TRAFFIC\s+".*"[ \t\r]*$' `
        ('CLUTTER_DATABASE_TRAFFIC "{0}"' -f $DatabaseOdbPath) `
        'traffic database'
    Write-ProjectText $NetPath $netText
}

function Remove-ScopedResultDirectory {
    param(
        [string]$ResultDirectory,
        [string]$AllowedRoot
    )

    if (-not (Test-Path -LiteralPath $ResultDirectory)) {
        return
    }
    $resolvedCandidate = [IO.Path]::GetFullPath($ResultDirectory)
    $resolvedParent = [IO.Path]::GetFullPath($AllowedRoot)
    if (-not $resolvedCandidate.StartsWith(
            $resolvedParent + [IO.Path]::DirectorySeparatorChar,
            [StringComparison]::OrdinalIgnoreCase
        )) {
        throw "Refusing to delete result directory outside OutputRoot: $ResultDirectory"
    }
    Remove-Item -LiteralPath $ResultDirectory -Recurse -Force
}

function Get-RequiredDouble {
    param(
        [string]$Text,
        [string]$Pattern,
        [string]$Description
    )
    $match = [regex]::Match($Text, $Pattern)
    if (-not $match.Success) {
        throw "Could not read $Description from the project template."
    }
    return [double]::Parse(
        $match.Groups[1].Value,
        [Globalization.CultureInfo]::InvariantCulture
    )
}

function Get-RequiredInteger {
    param(
        [string]$Text,
        [string]$Pattern,
        [string]$Description
    )
    $match = [regex]::Match($Text, $Pattern)
    if (-not $match.Success) {
        throw "Could not read $Description from the project template."
    }
    return [int]$match.Groups[1].Value
}

function Measure-PathLossText {
    param([string]$Path)

    $frequency = $null
    $resolution = $null
    $height = $null
    $gridPoints = 0
    $notComputed = 0
    $minimum = $null
    $maximum = $null
    $xValues = New-Object 'System.Collections.Generic.HashSet[string]'
    $yValues = New-Object 'System.Collections.Generic.HashSet[string]'
    $inData = $false

    foreach ($line in [IO.File]::ReadLines($Path)) {
        if (-not $inData) {
            if ($line -eq 'BEGIN_DATA') {
                $inData = $true
                continue
            }
            if ($line -match '^ANTENNA 1\s+FREQUENCY\s+([0-9.+-]+)') {
                $frequency = [double]$matches[1]
            }
            elseif ($line -match '^RESOLUTION\s+([0-9.+-]+)') {
                $resolution = [double]$matches[1]
            }
            elseif ($line -match '^HEIGHT\s+([0-9.+-]+)') {
                $height = [double]$matches[1]
            }
            continue
        }

        if ([string]::IsNullOrWhiteSpace($line)) {
            continue
        }
        $parts = @($line.Trim() -split '\s+')
        if ($parts.Count -lt 3) {
            continue
        }
        $gridPoints++
        [void]$xValues.Add($parts[0])
        [void]$yValues.Add($parts[1])
        if ($parts[2] -eq 'N.C.') {
            $notComputed++
            continue
        }

        $value = 0.0
        if ([double]::TryParse(
                $parts[2],
                [Globalization.NumberStyles]::Float,
                [Globalization.CultureInfo]::InvariantCulture,
                [ref]$value
            )) {
            if ($null -eq $minimum -or $value -lt $minimum) {
                $minimum = $value
            }
            if ($null -eq $maximum -or $value -gt $maximum) {
                $maximum = $value
            }
        }
    }
    if (-not $inData -or $gridPoints -eq 0) {
        throw "Path-loss ASCII output contains no grid data: $Path"
    }

    return [pscustomobject]@{
        FrequencyMHz = $frequency
        ResolutionM = $resolution
        ReceiverHeightM = $height
        GridPoints = $gridPoints
        GridWidth = $xValues.Count
        GridHeight = $yValues.Count
        ValidPoints = $gridPoints - $notComputed
        NotComputed = $notComputed
        MinimumDb = $minimum
        MaximumDb = $maximum
    }
}

$readyRows = @(Import-Csv -LiteralPath $ReadyManifestPath)
$readyByTile = @{}
foreach ($row in $readyRows) {
    if ($row.Status -eq 'READY') {
        $readyByTile[$row.Tile] = $row
    }
}
$selectedRows = @(Import-Csv -LiteralPath $TileListPath)
if ($selectedRows.Count -eq 0) {
    throw "Tile list is empty: $TileListPath"
}
$selectedNames = @($selectedRows | ForEach-Object { $_.Tile })
$duplicateNames = @(
    $selectedNames |
        Group-Object |
        Where-Object Count -gt 1 |
        Select-Object -ExpandProperty Name
)
if ($duplicateNames.Count -gt 0) {
    throw "Tile list contains duplicates: $($duplicateNames -join ', ')"
}

$templateDir = Join-Path $resolvedBaseDir $TemplateTile
$templateBase = Join-Path $templateDir "${TemplateTile}_${TemplateProjectSuffix}"
$templateNet = "$templateBase.net"
$templateNup = "$templateBase.nup"
$templateWpi = "$templateBase.wpi"
$templateMic = "$templateBase.mic"
foreach ($path in @($templateNet, $templateNup, $templateWpi, $templateMic)) {
    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
        throw "Template project file is missing: $path"
    }
}

$templateNetText = Read-ProjectText $templateNet
$templateNupText = Read-ProjectText $templateNup
$areaMatch = [regex]::Match(
    $templateNetText,
    '(?m)^COORDINATES_AREA\s+([0-9.+-]+)\s+([0-9.+-]+)\s+([0-9.+-]+)\s+([0-9.+-]+)'
)
if (-not $areaMatch.Success) {
    throw 'Could not read COORDINATES_AREA from the template.'
}
$areaXMin = [double]$areaMatch.Groups[1].Value
$areaYMin = [double]$areaMatch.Groups[2].Value
$areaWidth = [double]$areaMatch.Groups[3].Value - $areaXMin
$areaHeight = [double]$areaMatch.Groups[4].Value - $areaYMin

$antennaMatch = [regex]::Match(
    $templateNetText,
    '(?m)^ANTENNA 1 POSITION\s+([0-9.+-]+),\s*([0-9.+-]+),\s*([0-9.+-]+)'
)
$siteMatch = [regex]::Match(
    $templateNetText,
    '(?m)^SITE 1 SITE_LOCATION\s+([0-9.+-]+)\s+([0-9.+-]+)\s+([0-9.+-]+)'
)
if (-not $antennaMatch.Success -or -not $siteMatch.Success) {
    throw 'Could not read antenna/site positions from the template.'
}
$antennaOffsetX = [double]$antennaMatch.Groups[1].Value - $areaXMin
$antennaOffsetY = [double]$antennaMatch.Groups[2].Value - $areaYMin
$siteOffsetX = [double]$siteMatch.Groups[1].Value - $areaXMin
$siteOffsetY = [double]$siteMatch.Groups[2].Value - $areaYMin
$siteZ = [double]$siteMatch.Groups[3].Value

$frequencyMHz = Get-RequiredDouble $templateNetText `
    '(?m)^ANTENNA 1 FREQUENCY\s+([0-9.+-]+)' 'frequency'
$receiverHeightM = Get-RequiredDouble $templateNetText `
    '(?m)^HEIGHT\s+([0-9.+-]+)' 'receiver height'
$resolutionM = Get-RequiredDouble $templateNetText `
    '(?m)^RESOLUTION\s+([0-9.+-]+)' 'prediction resolution'
$databaseMode = Get-RequiredInteger $templateNupText `
    '(?m)^DATABASE_MODE\s+(\d+)' 'database mode'
$predictionModelId = Get-RequiredInteger $templateNupText `
    '(?m)^PREDICTION_MODEL_ID\s+(\d+)' 'prediction model'

$parameterSummary = [ordered]@{
    generated_at = (Get-Date).ToString('yyyy-MM-dd HH:mm:ss zzz')
    template_project = $templateBase
    checked_database_suffix = '_checked'
    database_policy = if ($AllowOriginalOdbFallback) {
        'Try checked ODB first; use original ODB only when WinPropCLI 2020 rejects the checked representation.'
    }
    else {
        'Checked ODB only.'
    }
    frequency_mhz = $frequencyMHz
    scenario = 'urban'
    database_mode = $databaseMode
    prediction_model_id = $predictionModelId
    prediction_model = if ($predictionModelId -eq 5) {
        'Urban Dominant Path Model (existing project setting)'
    }
    else {
        'Existing project model'
    }
    area_width_m = $areaWidth
    area_height_m = $areaHeight
    resolution_m = $resolutionM
    receiver_height_m = $receiverHeightM
    transmitter_xy_offset_m = @($antennaOffsetX, $antennaOffsetY)
    transmitter_height_rule = "max($HeightField) + $AntennaAboveMaxBuilding m"
    output = @('Path Loss', 'Power')
    ascii_output = $true
    winprop_cli = $WinPropCli
    winprop_multi_threading = $MultiThreading
    selected_tiles = $selectedNames
}
$parameterPath = Join-Path $resolvedOutputRoot '_prediction_smoke_parameters.json'
[IO.File]::WriteAllText(
    $parameterPath,
    ($parameterSummary | ConvertTo-Json -Depth 6),
    (New-Object Text.UTF8Encoding($false))
)

Write-Host "Selected tiles : $($selectedRows.Count)"
Write-Host "Template       : $templateBase"
Write-Host "Database       : <tile>_checked.odb"
Write-Host "Model          : ID $predictionModelId (Urban DPM in the existing project)"
Write-Host "Frequency      : $frequencyMHz MHz"
Write-Host "Grid           : $areaWidth x $areaHeight m at $resolutionM m"
Write-Host "Receiver       : $receiverHeightM m"
Write-Host "Output root    : $resolvedOutputRoot"
Write-Host ''

$results = New-Object System.Collections.Generic.List[object]
for ($selectionIndex = 0; $selectionIndex -lt $selectedRows.Count; $selectionIndex++) {
    $selection = $selectedRows[$selectionIndex]
    $tile = [string]$selection.Tile
    $timer = [Diagnostics.Stopwatch]::StartNew()
    $status = 'FAILED'
    $message = ''
    $analysis = $null
    $transmitterHeight = $null
    $databaseUsed = ''
    $checkedAttempt = ''
    $resultDir = Join-Path $resolvedOutputRoot $tile
    $tileDir = Join-Path $resolvedBaseDir $tile
    $tileBase = Join-Path $tileDir "${tile}_${ProjectSuffix}"
    $tileNet = "$tileBase.net"
    $tileNup = "$tileBase.nup"
    $tileWpi = "$tileBase.wpi"
    $tileMic = "$tileBase.mic"
    $cliLog = Join-Path $resolvedOutputRoot "$tile`_winpropcli.log"

    Write-Progress `
        -Activity 'WinProp checked-ODB prediction smoke test' `
        -Status "$($selectionIndex + 1)/$($selectedRows.Count): $tile" `
        -PercentComplete ([int](100.0 * ($selectionIndex + 1) / $selectedRows.Count))

    try {
        if ($tile -notmatch '^tile_\d{6}$') {
            throw "Invalid tile name: $tile"
        }
        if (-not $readyByTile.ContainsKey($tile)) {
            throw "$tile is not READY in $ReadyManifestPath"
        }
        $ready = $readyByTile[$tile]
        $checkedOdb = [string]$ready.OdbPath
        if (-not (Test-Path -LiteralPath $checkedOdb -PathType Leaf)) {
            throw "Checked ODB is missing: $checkedOdb"
        }
        if ([IO.Path]::GetFileName($checkedOdb) -ne "$tile`_checked.odb") {
            throw "Manifest points to a non-canonical ODB: $checkedOdb"
        }

        $tileShp = Join-Path $tileDir "$tile.shp"
        $tileDbf = Join-Path $tileDir "$tile.dbf"
        foreach ($path in @($tileShp, $tileDbf)) {
            if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
                throw "Required source file is missing: $path"
            }
        }

        $projectExists = @(
            @($tileNet, $tileNup, $tileWpi, $tileMic) |
                Where-Object { Test-Path -LiteralPath $_ -PathType Leaf }
        )
        if ($projectExists.Count -eq 4 -and -not $OverwriteProjects) {
            Write-Host "[$tile] reusing smoke project."
        }
        elseif ($projectExists.Count -gt 0 -and -not $OverwriteProjects) {
            throw "Smoke project is incomplete for $tile; use -OverwriteProjects."
        }
        else {
            $bounds = Get-ShapefileBounds $tileShp
            $xMin = $bounds.XMin
            $yMin = $bounds.YMin
            $xMax = $xMin + $areaWidth
            $yMax = $yMin + $areaHeight
            $antennaX = $xMin + $antennaOffsetX
            $antennaY = $yMin + $antennaOffsetY
            $siteX = $xMin + $siteOffsetX
            $siteY = $yMin + $siteOffsetY
            $maximumBuildingHeight = Get-MaxDbfNumericValue $tileDbf $HeightField
            $transmitterHeight = $maximumBuildingHeight + $AntennaAboveMaxBuilding

            $netText = $templateNetText.Replace($templateBase, $tileBase)
            $netText = $netText.Replace($TemplateTile, $tile)
            $netText = Replace-RequiredLine $netText `
                '(?m)^OUTPUT_PROPAGATION_FILES\s+".*"[ \t\r]*$' `
                ('OUTPUT_PROPAGATION_FILES "{0}\"' -f $resultDir) `
                'OUTPUT_PROPAGATION_FILES'
            $netText = Replace-RequiredLine $netText `
                '(?m)^PROJECT_FILE\s+".*"[ \t\r]*$' `
                ('PROJECT_FILE "{0}"' -f $tileBase) `
                'PROJECT_FILE'
            $netText = Replace-RequiredLine $netText `
                '(?m)^COORDINATES_AREA\s+.+$' `
                ('COORDINATES_AREA {0:F10} {1:F10} {2:F10} {3:F10}' -f
                    $xMin, $yMin, $xMax, $yMax) `
                'COORDINATES_AREA'
            $netText = Replace-RequiredLine $netText `
                '(?m)^ANTENNA 0 PREDICTION_AREA COORDINATES_AREA\s+.+$' `
                ('ANTENNA 0 PREDICTION_AREA COORDINATES_AREA {0:F10} {1:F10} {2:F10} {3:F10}' -f
                    $xMin, $yMin, $xMax, $yMax) `
                'ANTENNA 0 prediction area'
            $netText = Replace-RequiredLine $netText `
                '(?m)^ANTENNA -1 PREDICTION_AREA COORDINATES_AREA\s+.+$' `
                ('ANTENNA -1 PREDICTION_AREA COORDINATES_AREA {0:F10} {1:F10} {2:F10} {3:F10}' -f
                    $xMin, $yMin, $xMax, $yMax) `
                'ANTENNA -1 prediction area'
            $netText = Replace-RequiredLine $netText `
                '(?m)^SITE 1 SITE_LOCATION\s+.+$' `
                ('SITE 1 SITE_LOCATION {0:F10} {1:F10} {2:F10}' -f
                    $siteX, $siteY, $siteZ) `
                'site location'
            $netText = Replace-RequiredLine $netText `
                '(?m)^ANTENNA 1 POSITION\s+.+$' `
                ('ANTENNA 1 POSITION {0:F10}, {1:F10}, {2:F10} ' -f
                    $antennaX, $antennaY, $transmitterHeight) `
                'antenna position'
            $netText = Replace-RequiredLine $netText `
                '(?m)^CLUTTER_DATABASE_TRAFFIC\s+".*"[ \t\r]*$' `
                ('CLUTTER_DATABASE_TRAFFIC "{0}"' -f $checkedOdb) `
                'traffic database'
            Write-ProjectText $tileNet $netText

            $nupText = $templateNupText.Replace($templateBase, $tileBase)
            $nupText = $nupText.Replace($TemplateTile, $tile)
            $nupText = Replace-RequiredLine $nupText `
                '(?m)^DATABASE_FILE\s+".*"[ \t\r]*$' `
                ('DATABASE_FILE "{0}_checked"' -f $tile) `
                'DATABASE_FILE'
            Write-ProjectText $tileNup $nupText
            Copy-Item -LiteralPath $templateWpi -Destination $tileWpi -Force
            Copy-Item -LiteralPath $templateMic -Destination $tileMic -Force
            Write-Host (
                "[$tile] prepared; Tx height = {0:F3} + {1:F3} = {2:F3} m" -f
                $maximumBuildingHeight,
                $AntennaAboveMaxBuilding,
                $transmitterHeight
            )
        }

        $pathLossTxt = Join-Path $resultDir 'Site  1 Antenna 1 Path Loss.txt'
        $pathLossFpl = Join-Path $resultDir 'Site  1 Antenna 1 Path Loss.fpl'
        if ((Test-Path -LiteralPath $resultDir) -and $OverwriteResults) {
            Remove-ScopedResultDirectory $resultDir $resolvedOutputRoot
        }

        if ((Test-Path -LiteralPath $pathLossTxt) -and -not $OverwriteResults) {
            Write-Host "[$tile] reusing existing smoke result."
            $nupExisting = Read-ProjectText $tileNup
            if ($nupExisting -match '(?m)^DATABASE_FILE\s+"([^"]+)"') {
                $databaseBase = $matches[1]
                if ($databaseBase -eq $tile) {
                    $databaseUsed = Join-Path $tileDir "$tile.odb"
                    $status = 'OK_ORIGINAL_FALLBACK'
                    $priorCheckedLog = Join-Path $resolvedOutputRoot (
                        "$tile`_winpropcli_checked.log"
                    )
                    if (Test-Path -LiteralPath $priorCheckedLog -PathType Leaf) {
                        $checkedAttempt = "Previous checked attempt failed; log=$priorCheckedLog"
                    }
                }
                else {
                    $databaseUsed = $checkedOdb
                }
            }
        }
        else {
            $originalOdb = Join-Path $tileDir "$tile.odb"
            if (-not (Test-Path -LiteralPath $originalOdb -PathType Leaf)) {
                throw "Original ODB is missing: $originalOdb"
            }
            $candidates = New-Object System.Collections.Generic.List[object]
            $candidates.Add([pscustomobject]@{
                Label = 'CHECKED'
                Base = "${tile}_checked"
                Odb = $checkedOdb
            })
            if ($AllowOriginalOdbFallback) {
                $candidates.Add([pscustomobject]@{
                    Label = 'ORIGINAL_FALLBACK'
                    Base = $tile
                    Odb = $originalOdb
                })
            }

            $predictionSucceeded = $false
            $attemptMessages = New-Object System.Collections.Generic.List[string]
            foreach ($candidate in $candidates) {
                if ($candidate.Label -eq 'ORIGINAL_FALLBACK') {
                    Remove-ScopedResultDirectory $resultDir $resolvedOutputRoot
                }
                New-Item -ItemType Directory -Path $resultDir -Force | Out-Null
                Set-ProjectDatabase `
                    -NupPath $tileNup `
                    -NetPath $tileNet `
                    -DatabaseBase $candidate.Base `
                    -DatabaseOdbPath $candidate.Odb

                $arguments = @(
                    '-F', $tileNet,
                    '-P',
                    '--results-pro', $resultDir,
                    '--multi-threading', $MultiThreading,
                    '--disable-project-update'
                )
                Write-Host "[$tile] running WinProp prediction with $($candidate.Label)..."
                $previousErrorActionPreference = $ErrorActionPreference
                $ErrorActionPreference = 'Continue'
                try {
                    $cliOutput = & $WinPropCli @arguments 2>&1
                    $cliExit = $LASTEXITCODE
                }
                finally {
                    $ErrorActionPreference = $previousErrorActionPreference
                }
                $attemptLog = Join-Path $resolvedOutputRoot (
                    '{0}_winpropcli_{1}.log' -f
                    $tile,
                    $candidate.Label.ToLowerInvariant()
                )
                [IO.File]::WriteAllLines(
                    $attemptLog,
                    [string[]]@($cliOutput | ForEach-Object { [string]$_ }),
                    (New-Object Text.UTF8Encoding($true))
                )
                $hasOutputs = (
                    (Test-Path -LiteralPath $pathLossTxt -PathType Leaf) -and
                    (Test-Path -LiteralPath $pathLossFpl -PathType Leaf)
                )
                $attemptMessages.Add(
                    "$($candidate.Label): exit=$cliExit outputs=$hasOutputs log=$attemptLog"
                )
                if ($candidate.Label -eq 'CHECKED') {
                    $checkedAttempt = $attemptMessages[$attemptMessages.Count - 1]
                }
                if ($cliExit -eq 0 -and $hasOutputs) {
                    $predictionSucceeded = $true
                    $databaseUsed = $candidate.Odb
                    if ($candidate.Label -eq 'ORIGINAL_FALLBACK') {
                        $status = 'OK_ORIGINAL_FALLBACK'
                    }
                    break
                }
            }
            if (-not $predictionSucceeded) {
                throw "All WinPropCLI attempts failed. $($attemptMessages -join '; ')"
            }
        }

        if ($null -eq $transmitterHeight) {
            $heightMatch = [regex]::Match(
                (Read-ProjectText $tileNet),
                '(?m)^ANTENNA 1 POSITION\s+[0-9.+-]+,\s*[0-9.+-]+,\s*([0-9.+-]+)'
            )
            if (-not $heightMatch.Success) {
                throw "Could not read transmitter height from $tileNet"
            }
            $transmitterHeight = [double]$heightMatch.Groups[1].Value
        }

        if (-not (Test-Path -LiteralPath $pathLossTxt -PathType Leaf) -or
            -not (Test-Path -LiteralPath $pathLossFpl -PathType Leaf)) {
            throw "Path-loss outputs are incomplete in $resultDir"
        }
        $analysis = Measure-PathLossText $pathLossTxt
        $expectedAxisPoints = [int][math]::Round($areaWidth / $resolutionM)
        $expectedGridPoints = $expectedAxisPoints * $expectedAxisPoints
        if ($analysis.GridWidth -ne $expectedAxisPoints -or
            $analysis.GridHeight -ne $expectedAxisPoints -or
            $analysis.GridPoints -ne $expectedGridPoints) {
            throw (
                "Unexpected grid $($analysis.GridWidth)x$($analysis.GridHeight) " +
                "($($analysis.GridPoints) points); expected " +
                "${expectedAxisPoints}x${expectedAxisPoints}."
            )
        }
        if ($analysis.FrequencyMHz -ne $frequencyMHz -or
            $analysis.ResolutionM -ne $resolutionM -or
            $analysis.ReceiverHeightM -ne $receiverHeightM) {
            throw "Output header parameters do not match the template."
        }
        if ($status -eq 'FAILED') {
            $status = 'OK'
        }
        Write-Host (
            "[$tile] OK: {0}x{1}, valid={2}, N.C.={3}" -f
            $analysis.GridWidth,
            $analysis.GridHeight,
            $analysis.ValidPoints,
            $analysis.NotComputed
        )
    }
    catch {
        $message = $_.Exception.Message
        Write-Warning "[$tile] $message"
    }
    finally {
        $timer.Stop()
    }

    $results.Add([pscustomobject]@{
        Tile = $tile
        SelectionReason = [string]$selection.SelectionReason
        RepairCategory = if ($readyByTile.ContainsKey($tile)) {
            [string]$readyByTile[$tile].RepairCategory
        }
        else {
            ''
        }
        Status = $status
        DurationSeconds = $timer.Elapsed.TotalSeconds.ToString(
            '0.000',
            [Globalization.CultureInfo]::InvariantCulture
        )
        CheckedOdb = if ($readyByTile.ContainsKey($tile)) {
            [string]$readyByTile[$tile].OdbPath
        }
        else {
            ''
        }
        DatabaseUsed = $databaseUsed
        CheckedAttempt = $checkedAttempt
        Project = $tileNet
        ResultDir = $resultDir
        FrequencyMHz = if ($null -ne $analysis) { $analysis.FrequencyMHz } else { '' }
        ResolutionM = if ($null -ne $analysis) { $analysis.ResolutionM } else { '' }
        ReceiverHeightM = if ($null -ne $analysis) { $analysis.ReceiverHeightM } else { '' }
        TransmitterHeightM = if ($null -ne $transmitterHeight) {
            $transmitterHeight.ToString(
                '0.000',
                [Globalization.CultureInfo]::InvariantCulture
            )
        }
        else {
            ''
        }
        GridWidth = if ($null -ne $analysis) { $analysis.GridWidth } else { '' }
        GridHeight = if ($null -ne $analysis) { $analysis.GridHeight } else { '' }
        GridPoints = if ($null -ne $analysis) { $analysis.GridPoints } else { '' }
        ValidPoints = if ($null -ne $analysis) { $analysis.ValidPoints } else { '' }
        NotComputed = if ($null -ne $analysis) { $analysis.NotComputed } else { '' }
        MinimumDb = if ($null -ne $analysis) { $analysis.MinimumDb } else { '' }
        MaximumDb = if ($null -ne $analysis) { $analysis.MaximumDb } else { '' }
        Message = $message
    })
}
Write-Progress -Activity 'WinProp checked-ODB prediction smoke test' -Completed

$resultManifest = Join-Path $resolvedOutputRoot '_prediction_smoke_results.csv'
$results | Export-Csv -LiteralPath $resultManifest -NoTypeInformation -Encoding UTF8
$okCount = @($results | Where-Object Status -like 'OK*').Count
$failedCount = @($results | Where-Object Status -eq 'FAILED').Count
Write-Host ''
Write-Host "Prediction smoke test finished."
Write-Host "OK       : $okCount"
Write-Host "Failed   : $failedCount"
Write-Host "Results  : $resultManifest"
Write-Host "Parameters: $parameterPath"

if ($failedCount -gt 0) {
    exit 1
}
