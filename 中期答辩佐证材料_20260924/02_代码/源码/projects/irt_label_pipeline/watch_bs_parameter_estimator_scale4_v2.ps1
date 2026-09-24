$ErrorActionPreference = "SilentlyContinue"

$DesktopDir = [Environment]::GetFolderPath("Desktop")
$OutputDir = Join-Path $DesktopDir "dac\models\bs_parameter_estimator_scale4_v2"
$HistoryPath = Join-Path $OutputDir "history.csv"
$FinalPath = Join-Path $OutputDir "test_metrics.json"
$Host.UI.RawUI.WindowTitle = "BS estimator Scale-4 training monitor"

while ($true) {
    Clear-Host
    Write-Host "BS estimator Scale-4 training monitor" -ForegroundColor Cyan
    Write-Host "Updated: $((Get-Date).ToString('yyyy-MM-dd HH:mm:ss'))"
    Write-Host "Output:  $OutputDir"
    Write-Host ""

    if (Test-Path -LiteralPath $HistoryPath) {
        $Rows = @(Import-Csv -LiteralPath $HistoryPath)
        if ($Rows.Count -gt 0) {
            $Last = $Rows[-1]
            Write-Host ("Epoch: {0}/25    LR: {1:E3}    Patience: {2}" -f `
                [int]$Last.epoch, [double]$Last.learning_rate, [int]$Last.patience_used) -ForegroundColor Yellow
            Write-Host ("Validation location: {0:N3} px  (P90 {1:N3} px)" -f `
                [double]$Last.val_location_error_px_mean, [double]$Last.val_location_error_px_p90)
            Write-Host ("Validation power:    {0:N3} dB" -f `
                [double]$Last.val_power_abs_error_db_mean)
            Write-Host ("Validation azimuth:  {0:N3} deg" -f `
                [double]$Last.val_direction_abs_error_deg_mean)
            Write-Host ("Validation map RMSE: {0:N3} dB" -f `
                [double]$Last.val_map_rmse_db)
            Write-Host ("Best objective:      {0:N6}" -f `
                [double]$Last.best_val_objective)
        }
    } else {
        Write-Host "Waiting for history.csv ..." -ForegroundColor DarkYellow
    }

    $GpuLine = & nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu `
        --format=csv,noheader,nounits 2>$null
    if ($GpuLine) {
        $Gpu = $GpuLine -split ",\s*"
        Write-Host ""
        Write-Host ("GPU: {0}%    VRAM: {1}/{2} MiB    Temp: {3} C" -f `
            $Gpu[0], $Gpu[1], $Gpu[2], $Gpu[3]) -ForegroundColor Green
    }

    if (Test-Path -LiteralPath $FinalPath) {
        $Final = Get-Content -LiteralPath $FinalPath -Raw -Encoding UTF8 | ConvertFrom-Json
        Write-Host ""
        Write-Host "Training finished." -ForegroundColor Green
        Write-Host ("Status: {0}    Best epoch: {1}" -f $Final.status, $Final.best_epoch)
        Write-Host ("Test location: {0:N3} px" -f [double]$Final.test_metrics.location_error_px_mean)
        Write-Host ("Test power:    {0:N3} dB" -f [double]$Final.test_metrics.power_abs_error_db_mean)
        Write-Host ("Test azimuth:  {0:N3} deg" -f [double]$Final.test_metrics.direction_abs_error_deg_mean)
        Write-Host ""
        Write-Host "This window will stay open. Close it after reviewing the result."
        break
    }

    Write-Host ""
    Write-Host "Auto-refresh every 5 seconds. Closing this monitor will NOT stop training." -ForegroundColor DarkGray
    Start-Sleep -Seconds 5
}
