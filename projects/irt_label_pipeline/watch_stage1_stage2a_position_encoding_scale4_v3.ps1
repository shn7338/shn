$ErrorActionPreference = "SilentlyContinue"

$DesktopDir = [Environment]::GetFolderPath("Desktop")
$OutputDir = Join-Path $DesktopDir "dac\models\stage2a_position_encoding_scale4_v3"
$HistoryPath = Join-Path $OutputDir "history.csv"
$FinalPath = Join-Path $OutputDir "test_metrics.json"
$Host.UI.RawUI.WindowTitle = "Stage1 + Stage2A Scale-4 training monitor"

while ($true) {
    Clear-Host
    Write-Host "Stage1 + Stage2A Scale-4 training monitor" -ForegroundColor Cyan
    Write-Host "Updated: $((Get-Date).ToString('yyyy-MM-dd HH:mm:ss'))"
    Write-Host "Output:  $OutputDir"
    Write-Host ""

    if (Test-Path -LiteralPath $HistoryPath) {
        $Rows = @(Import-Csv -LiteralPath $HistoryPath)
        if ($Rows.Count -gt 0) {
            $Last = $Rows[-1]
            $Best = $Rows | Sort-Object { [double]$_.val_rmse_db } | Select-Object -First 1
            Write-Host ("Epoch: {0}/15    Stage1 LR: {1:E3}    Stage2A LR: {2:E3}" -f `
                [int]$Last.epoch, [double]$Last.stage1_learning_rate, [double]$Last.stage2a_learning_rate) -ForegroundColor Yellow
            Write-Host ("Train RMSE: {0:N4} dB    MAE: {1:N4} dB" -f `
                [double]$Last.train_rmse_db, [double]$Last.train_mae_db)
            Write-Host ("Val RMSE:   {0:N4} dB    MAE: {1:N4} dB" -f `
                [double]$Last.val_rmse_db, [double]$Last.val_mae_db)
            Write-Host ("Best val:   {0:N4} dB at epoch {1}" -f `
                [double]$Best.val_rmse_db, [int]$Best.epoch) -ForegroundColor Green
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
        Write-Host ("Test RMSE: {0:N4} dB" -f [double]$Final.test_metrics.rmse_db)
        Write-Host ("Improvement: {0:N4} dB" -f [double]$Final.test_rmse_improvement_db)
        Write-Host ""
        Write-Host "This window will stay open. Close it after reviewing the result."
        break
    }

    Write-Host ""
    Write-Host "Auto-refresh every 5 seconds. Closing this monitor will NOT stop training." -ForegroundColor DarkGray
    Start-Sleep -Seconds 5
}

