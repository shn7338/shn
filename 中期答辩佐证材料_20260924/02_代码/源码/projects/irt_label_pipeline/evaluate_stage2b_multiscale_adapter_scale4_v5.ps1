$ErrorActionPreference = "Stop"
$Project = Split-Path -Parent $MyInvocation.MyCommand.Path
$DesktopDir = [Environment]::GetFolderPath("Desktop")
$PythonTrain = "C:\Users\pc\miniconda3\envs\sigmap\python.exe"
$PythonRender = "C:\Users\pc\AppData\Local\Programs\Python\Python312\python.exe"
$ModelDir = Join-Path $DesktopDir "dac\models\stage2b_multiscale_adapter_scale4_v5"
$V5Checkpoint = Join-Path $ModelDir "best.pt"
$V4Checkpoint = Join-Path $DesktopDir "dac\models\stage2b_latent_fusion_scale4_v4\best.pt"
$EstimatorCheckpoint = Join-Path $DesktopDir "dac\models\bs_parameter_estimator_scale4_v2\best.pt"
$ExperimentDir = Join-Path $DesktopDir "dac\experiments"
$V3Report = Join-Path $ExperimentDir "bs_inversion_scale4_stage2b_v3_test_bins.json"
$V4Report = Join-Path $ExperimentDir "bs_inversion_scale4_stage2b_latent_fusion_v4_test_bins.json"
$V5Report = Join-Path $ExperimentDir "bs_inversion_scale4_stage2b_multiscale_adapter_v5_test_bins.json"
$GateReport = Join-Path $ExperimentDir "bs_inversion_scale4_stage2b_multiscale_adapter_v5_gate.json"
$VisualDir = Join-Path $ExperimentDir "scale4_visual_comparison_v5"
$Config = Join-Path $Project "config_stage2b_multiscale_adapter_scale4_v5.json"

if (-not (Test-Path -LiteralPath $V5Checkpoint)) {
    throw "V5 best.pt is absent. Finish training before evaluation: $V5Checkpoint"
}

Write-Host "[1/4] Full V5 test evaluation" -ForegroundColor Cyan
& $PythonTrain (Join-Path $Project "evaluate_pilot_oracle_pipeline.py") `
    --dataset-root (Join-Path $DesktopDir "dac\bs_inversion_scale4_v1") `
    --stage2b-config $Config `
    --estimator-config (Join-Path $Project "config_bs_parameter_estimator_scale4_v2.json") `
    --estimator-checkpoint $EstimatorCheckpoint `
    --split test --batch-size 16 --workers 2 --output $V5Report
if ($LASTEXITCODE -ne 0) { throw "V5 evaluation failed" }

Write-Host "[2/4] V4/V5 representative samples" -ForegroundColor Cyan
& $PythonTrain (Join-Path $Project "visualize_scale4_pipeline_examples.py") `
    --before-config $Config --after-config $Config `
    --estimator-config (Join-Path $Project "config_bs_parameter_estimator_scale4_v2.json") `
    --estimator-checkpoint $EstimatorCheckpoint `
    --before-stage2b-checkpoint $V4Checkpoint `
    --stage2b-checkpoint $V5Checkpoint `
    --output-dir $VisualDir --candidates-per-bin 64 --batch-size 8 --workers 2 `
    --skip-render
if ($LASTEXITCODE -ne 0) { throw "V4/V5 visual inference failed" }

Write-Host "[3/4] Render comparison figures" -ForegroundColor Cyan
& $PythonRender (Join-Path $Project "render_stage2b_v4_comparison.py") `
    --visualization-dir $VisualDir `
    --before-evaluation $V4Report --after-evaluation $V5Report `
    --history (Join-Path $ModelDir "history.csv") `
    --before-label V4 --after-label V5
if ($LASTEXITCODE -ne 0) { throw "V4/V5 rendering failed" }

Write-Host "[4/4] Apply V5 acceptance gate" -ForegroundColor Cyan
& $PythonTrain (Join-Path $Project "assess_stage2b_multiscale_adapter_v5.py") `
    --config $Config --v3-report $V3Report --v4-report $V4Report `
    --v5-report $V5Report --output $GateReport
$GateExitCode = $LASTEXITCODE
if ($GateExitCode -eq 0) {
    Write-Host "V5 accepted." -ForegroundColor Green
} elseif ($GateExitCode -eq 2) {
    Write-Host "V5 did not pass the predeclared promotion gate; reports and figures are still complete." -ForegroundColor Yellow
} else {
    throw "V5 gate script failed with exit code $GateExitCode"
}
