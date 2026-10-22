# ==============================================================================
# Script: scripts/run_4days_suite.ps1
# Description: Windows Native Master 4-Day Benchmark Pipeline on Solar-AL
# ==============================================================================

param(
    [int]$GPU = 0
)

$TOTAL_START = Get-Date

Write-Host "==============================================================================" -ForegroundColor Cyan
Write-Host ">>> [MASTER RUNNER] Launching Windows 4-Day Benchmark & Sensitivity Suite" -ForegroundColor Cyan
Write-Host ">>> Start Time: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') | GPU: $GPU" -ForegroundColor Cyan
Write-Host "==============================================================================" -ForegroundColor Cyan

# Phase 1: Few-Shot Suite
Write-Host ""
Write-Host ">>> [PHASE 1/3] Running Few-Shot Suite (10% ~ 50%)..." -ForegroundColor Yellow
& .\scripts\run_fewshot_il.ps1 -GPU $GPU

# Phase 2: Hyperparameter Sensitivity Suite
Write-Host ""
Write-Host ">>> [PHASE 2/3] Running Hyperparameter Sensitivity Suite..." -ForegroundColor Yellow
& .\scripts\run_hyper_sensitivity.ps1 -GPU $GPU -TargetModule "all"

# Phase 3: Automated Publication Figures
Write-Host ""
Write-Host ">>> [PHASE 3/3] Generating Publication Figures..." -ForegroundColor Yellow
python plot_sensitivity_results.py `
    --fewshot_csv ./results/results_fewshot_al.csv `
    --sensitivity_csv ./results/results_hyper_sensitivity_al.csv `
    --output_dir ./figures/sensitivity

$TOTAL_END = Get-Date
$TOTAL_SPAN = $TOTAL_END - $TOTAL_START

Write-Host ""
Write-Host "==============================================================================" -ForegroundColor Green
Write-Host ">>> [ALL PHASES COMPLETED SUCCESSFULLY]" -ForegroundColor Green
Write-Host ">>> Total Runtime: $($TOTAL_SPAN.Hours) hours $($TOTAL_SPAN.Minutes) minutes" -ForegroundColor Green
Write-Host ">>> High-Res Figures saved in: ./figures/sensitivity" -ForegroundColor Green
Write-Host "==============================================================================" -ForegroundColor Green
