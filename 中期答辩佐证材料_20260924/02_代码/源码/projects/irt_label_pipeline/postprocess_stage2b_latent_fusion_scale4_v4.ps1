$ErrorActionPreference = "Stop"

$Project = Split-Path -Parent $MyInvocation.MyCommand.Path
$DesktopDir = [Environment]::GetFolderPath("Desktop")
$PythonTrain = "C:\Users\pc\miniconda3\envs\sigmap\python.exe"
$PythonRender = "C:\Users\pc\AppData\Local\Programs\Python\Python312\python.exe"
$ModelDir = Join-Path $DesktopDir "dac\models\stage2b_latent_fusion_scale4_v4"
$Summary = Join-Path $ModelDir "training_summary.json"
$Status = Join-Path $ModelDir "postprocess_status.txt"
$Evaluation = Join-Path $DesktopDir "dac\experiments\bs_inversion_scale4_stage2b_latent_fusion_v4_test_bins.json"
$BaselineEvaluation = Join-Path $DesktopDir "dac\experiments\bs_inversion_scale4_stage2b_v3_test_bins.json"
$GateReport = Join-Path $DesktopDir "dac\experiments\bs_inversion_scale4_stage2b_latent_fusion_v4_gate.json"
$VisualDir = Join-Path $DesktopDir "dac\experiments\scale4_visual_comparison_v4"
$V3Checkpoint = Join-Path $DesktopDir "dac\models\stage2b_position_encoding_scale4_v3\best.pt"
$V4Checkpoint = Join-Path $ModelDir "best.pt"
$EstimatorCheckpoint = Join-Path $DesktopDir "dac\models\bs_parameter_estimator_scale4_v2\best.pt"

$Host.UI.RawUI.WindowTitle = "Stage2B V4 automatic evaluation and visualization"
function Set-Phase([string]$Message) {
    Set-Content -LiteralPath $Status -Encoding UTF8 -Value $Message
    Write-Host ("[{0}] {1}" -f (Get-Date -Format "HH:mm:ss"), $Message) -ForegroundColor Cyan
}

Set-Phase "waiting_for_training"
Write-Host "This tab will automatically evaluate V4 and render V3/V4 comparisons after training."
$TrainingWasSeen = $false
while (-not (Test-Path -LiteralPath $Summary)) {
    $Training = Get-CimInstance Win32_Process | Where-Object {
        $_.Name -eq "python.exe" -and
        $_.CommandLine -match "config_stage2b_latent_fusion_scale4_v4"
    }
    if ($Training) {
        $TrainingWasSeen = $true
    } elseif ($TrainingWasSeen) {
        Set-Phase "failed: training exited without training_summary.json"
        exit 1
    }
    Start-Sleep -Seconds 30
}

Set-Phase "evaluating_full_test_split"
& $PythonTrain (Join-Path $Project "evaluate_pilot_oracle_pipeline.py") `
    --dataset-root (Join-Path $DesktopDir "dac\bs_inversion_scale4_v1") `
    --stage2b-config (Join-Path $Project "config_stage2b_latent_fusion_scale4_v4.json") `
    --estimator-config (Join-Path $Project "config_bs_parameter_estimator_scale4_v2.json") `
    --estimator-checkpoint $EstimatorCheckpoint `
    --split test `
    --batch-size 16 `
    --workers 2 `
    --output $Evaluation
if ($LASTEXITCODE -ne 0) { throw "full-test evaluation failed" }

Set-Phase "collecting_visual_examples"
& $PythonTrain (Join-Path $Project "visualize_scale4_pipeline_examples.py") `
    --before-config (Join-Path $Project "config_stage2b_latent_fusion_scale4_v4.json") `
    --after-config (Join-Path $Project "config_stage2b_latent_fusion_scale4_v4.json") `
    --estimator-config (Join-Path $Project "config_bs_parameter_estimator_scale4_v2.json") `
    --estimator-checkpoint $EstimatorCheckpoint `
    --before-stage2b-checkpoint $V3Checkpoint `
    --stage2b-checkpoint $V4Checkpoint `
    --output-dir $VisualDir `
    --candidates-per-bin 64 `
    --batch-size 8 `
    --workers 2 `
    --skip-render
if ($LASTEXITCODE -ne 0) { throw "visual-example inference failed" }

Set-Phase "rendering_comparison_figures"
& $PythonRender (Join-Path $Project "render_stage2b_v4_comparison.py") `
    --visualization-dir $VisualDir `
    --v3-evaluation $BaselineEvaluation `
    --v4-evaluation $Evaluation `
    --v4-history (Join-Path $ModelDir "history.csv")
if ($LASTEXITCODE -ne 0) { throw "comparison rendering failed" }

Set-Phase "applying_acceptance_gate"
& $PythonTrain (Join-Path $Project "assess_stage2b_latent_fusion_v4.py") `
    --config (Join-Path $Project "config_stage2b_latent_fusion_scale4_v4.json") `
    --baseline-report $BaselineEvaluation `
    --candidate-report $Evaluation `
    --output $GateReport
$GateExitCode = $LASTEXITCODE
if ($GateExitCode -eq 0) {
    Set-Phase "complete: V4 accepted"
} elseif ($GateExitCode -eq 2) {
    Set-Phase "complete: V4 rejected; retain V3"
} else {
    throw "acceptance-gate evaluation failed"
}
