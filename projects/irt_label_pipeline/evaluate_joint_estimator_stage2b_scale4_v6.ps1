$ErrorActionPreference = "Stop"
$Project = Split-Path -Parent $MyInvocation.MyCommand.Path
$DesktopDir = [Environment]::GetFolderPath("Desktop")
$PythonTrain = "C:\Users\pc\miniconda3\envs\sigmap\python.exe"
$PythonRender = "C:\Users\pc\AppData\Local\Programs\Python\Python312\python.exe"
$ModelDir = Join-Path $DesktopDir "dac\models\joint_estimator_stage2b_scale4_v6"
$V6Stage2B = Join-Path $ModelDir "best_stage2b.pt"
$V6Estimator = Join-Path $ModelDir "best_estimator.pt"
$V5Stage2B = Join-Path $DesktopDir "dac\models\stage2b_multiscale_adapter_scale4_v5\best.pt"
$V5Estimator = Join-Path $DesktopDir "dac\models\bs_parameter_estimator_scale4_v2\best.pt"
$ExperimentDir = Join-Path $DesktopDir "dac\experiments"
$V5Report = Join-Path $ExperimentDir "bs_inversion_scale4_stage2b_multiscale_adapter_v5_test_bins.json"
$V6Report = Join-Path $ExperimentDir "bs_inversion_scale4_joint_estimator_stage2b_v6_test_bins.json"
$GateReport = Join-Path $ExperimentDir "bs_inversion_scale4_joint_estimator_stage2b_v6_gate.json"
$VisualDir = Join-Path $ExperimentDir "scale4_visual_comparison_v6"
$Config = Join-Path $Project "config_joint_estimator_stage2b_scale4_v6.json"
$EstimatorConfig = Join-Path $Project "config_bs_parameter_estimator_scale4_v2.json"

foreach ($Required in @($V6Stage2B, $V6Estimator)) {
    if (-not (Test-Path -LiteralPath $Required)) {
        throw "V6 checkpoint is absent; finish training first: $Required"
    }
}

Write-Host "[1/4] Full V6 test evaluation" -ForegroundColor Cyan
& $PythonTrain (Join-Path $Project "evaluate_pilot_oracle_pipeline.py") `
    --dataset-root (Join-Path $DesktopDir "dac\bs_inversion_scale4_v1") `
    --stage2b-config $Config --stage2b-checkpoint $V6Stage2B `
    --estimator-config $EstimatorConfig --estimator-checkpoint $V6Estimator `
    --split test --batch-size 16 --workers 2 --output $V6Report
if ($LASTEXITCODE -ne 0) { throw "V6 evaluation failed" }

Write-Host "[2/4] V5/V6 representative samples" -ForegroundColor Cyan
& $PythonTrain (Join-Path $Project "visualize_scale4_pipeline_examples.py") `
    --before-config $Config --after-config $Config `
    --estimator-config $EstimatorConfig --estimator-checkpoint $V5Estimator `
    --before-estimator-checkpoint $V5Estimator `
    --after-estimator-checkpoint $V6Estimator `
    --before-stage2b-checkpoint $V5Stage2B `
    --stage2b-checkpoint $V6Stage2B `
    --output-dir $VisualDir --candidates-per-bin 64 --batch-size 8 --workers 2 `
    --skip-render
if ($LASTEXITCODE -ne 0) { throw "V5/V6 visual inference failed" }

Write-Host "[3/4] Render comparison figures" -ForegroundColor Cyan
& $PythonRender (Join-Path $Project "render_stage2b_v4_comparison.py") `
    --visualization-dir $VisualDir `
    --before-evaluation $V5Report --after-evaluation $V6Report `
    --history (Join-Path $ModelDir "history.csv") `
    --before-label V5 --after-label V6
if ($LASTEXITCODE -ne 0) { throw "V5/V6 rendering failed" }

Write-Host "[4/4] Apply V6 acceptance gate" -ForegroundColor Cyan
& $PythonTrain (Join-Path $Project "assess_joint_estimator_stage2b_v6.py") `
    --config $Config --v5-report $V5Report --v6-report $V6Report `
    --output $GateReport
$GateExitCode = $LASTEXITCODE
if ($GateExitCode -eq 0) {
    Write-Host "V6 accepted." -ForegroundColor Green
} elseif ($GateExitCode -eq 2) {
    Write-Host "V6 did not pass every promotion check; reports are complete." -ForegroundColor Yellow
} else {
    throw "V6 gate script failed with exit code $GateExitCode"
}
