param(
    [Parameter(Mandatory = $true)][string]$SourceUrl,
    [Parameter(Mandatory = $true)][string]$Destination,
    [Parameter(Mandatory = $true)][long]$TotalBytes,
    [int]$ChunkMiB = 32,
    [int]$MaxRetries = 20
)

$ErrorActionPreference = 'Stop'
$destinationPath = [System.IO.Path]::GetFullPath($Destination)
$parent = [System.IO.Path]::GetDirectoryName($destinationPath)
if (-not [System.IO.Directory]::Exists($parent)) {
    [System.IO.Directory]::CreateDirectory($parent) | Out-Null
}

while ($true) {
    $current = if ([System.IO.File]::Exists($destinationPath)) {
        [System.IO.FileInfo]::new($destinationPath).Length
    } else {
        0L
    }
    if ($current -eq $TotalBytes) {
        Write-Output "complete bytes=$current"
        break
    }
    if ($current -gt $TotalBytes) {
        throw "Destination is larger than expected: $current > $TotalBytes"
    }

    $end = [Math]::Min($TotalBytes - 1, $current + ([long]$ChunkMiB * 1MB) - 1)
    $succeeded = $false
    for ($attempt = 1; $attempt -le $MaxRetries -and -not $succeeded; $attempt++) {
        try {
            $request = [System.Net.HttpWebRequest]::Create($SourceUrl)
            $request.UserAgent = 'Mozilla/5.0'
            $request.AddRange($current, $end)
            $response = $request.GetResponse()
            if ([int]$response.StatusCode -ne 206) {
                throw "Expected HTTP 206, got $([int]$response.StatusCode)"
            }
            $inputStream = $response.GetResponseStream()
            $outputStream = [System.IO.File]::Open(
                $destinationPath,
                [System.IO.FileMode]::Append,
                [System.IO.FileAccess]::Write,
                [System.IO.FileShare]::Read
            )
            try {
                $inputStream.CopyTo($outputStream)
            } finally {
                $outputStream.Dispose()
                $inputStream.Dispose()
                $response.Dispose()
            }
            $newLength = [System.IO.FileInfo]::new($destinationPath).Length
            $percent = 100.0 * $newLength / $TotalBytes
            Write-Output ("downloaded={0} total={1} percent={2:N1}" -f $newLength, $TotalBytes, $percent)
            $succeeded = $true
        } catch {
            Write-Output "retry=$attempt start=$current error=$($_.Exception.Message)"
            Start-Sleep -Seconds ([Math]::Min(10, $attempt + 1))
        }
    }
    if (-not $succeeded) {
        throw "Unable to download range $current-$end after $MaxRetries attempts"
    }
}
