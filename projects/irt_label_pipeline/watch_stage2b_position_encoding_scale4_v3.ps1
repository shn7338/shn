$ErrorActionPreference = "SilentlyContinue"

$DesktopDir = [Environment]::GetFolderPath("Desktop")
$OutputDir = Join-Path $DesktopDir "dac\models\stage2b_position_encoding_scale4_v3"
$HistoryPath = Join-Path $OutputDir "history.csv"
$SummaryPath = Join-Path $OutputDir "training_summary.json"
$Host.UI.RawUI.WindowTitle = "Stage2B Scale-4 fine-tuning monitor"

while ($true) {
    Clear-Host
    Write-Host "Stage2B Scale-4 fine-tuning monitor" -ForegroundColor Cyan
    Write-Host "Updated: $((Get-Date).ToString('yyyy-MM-dd HH:mm:ss'))"
    Write-Host "Output:  $OutputDir"
    Write-Host ""

    if (Test-Path -LiteralPath $HistoryPath) {
        $Rows = @(Import-Csv -LiteralPath $HistoryPath)
        if ($Rows.Count -gt 0) {
            $Last = $Rows[-1]
            $Best = $Rows | Sort-Object { [double]$_.val_map_rmse_db } | Select-Object -First 1
            Write-Host ("Epoch: {0}/10    LR: {1:E3}" -f `
                [int]$Last.epoch, [double]$Last.learning_rate) -ForegroundColor Yellow
            Write-Host ("Train RMSE: {0:N4} dB    Unmeasured: {1:N4} dB" -f `
                [double]$Last.train_map_rmse_db, [double]$Last.train_unmeasured_rmse_db)
            Write-Host ("Val RMSE:   {0:N4} dB    Unmeasured: {1:N4} dB" -f `
                [double]$Last.val_map_rmse_db, [double]$Last.val_unmeasured_rmse_db)
            Write-Host ("Best val:   {0:N4} dB at epoch {1}" -f `
                [double]$Best.val_map_rmse_db, [int]$Best.epoch) -ForegroundColor Green
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

    if (Test-Path -LiteralPath $SummaryPath) {
        $Summary = Get-Content -LiteralPath $SummaryPath -Raw -Encoding UTF8 | ConvertFrom-Json
        Write-Host ""
        Write-Host "Training finished; full test-bin evaluation is next." -ForegroundColor Green
        Write-Host ("Best epoch: {0}    Best validation RMSE: {1:N4} dB" -f `
            $Summary.best_epoch, [double]$Summary.best_val_rmse_db)
        Write-Host "This window will stay open. Close it after reviewing the result."
        break
    }

    Write-Host ""
    Write-Host "Auto-refresh every 5 seconds. Closing this monitor will NOT stop training." -ForegroundColor DarkGray
    Start-Sleep -Seconds 5
}

