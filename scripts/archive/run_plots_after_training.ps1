# run_plots_after_training.ps1
# ========================================================================================
# 功能：加载刚刚复现完成的基准模型检查点 (0.164985)，执行测试集评估并生成全套概率密度分布图件：
# 1. 步长误差概率密度图 (Step-wise MSE / MAE Profile) -> stepwise_mse_pdf.png, stepwise_mae_pdf.png
# 2. 预测残差直方图与经验 PDF (Residual Histogram & KDE) -> residual_histogram_pdf.png
# 3. 正态 Q-Q 检验图 (Normal Q-Q Plot) -> residual_qq_plot.png
# ========================================================================================

$ErrorActionPreference = "Stop"

$PYTHON_CMD = "python"
if (Test-Path "D:\software\conda\envs\calf\python.exe") {
    $PYTHON_CMD = "D:\software\conda\envs\calf\python.exe"
}

$CKPT_PATH = "./checkpoints/long_term_forecast_CALF_96_96_p100_CALF_Solar_ftM_sl96_ll0_pl96_dm768_nh4_el1_dl1_df3072_fc1_ebtimeF_dtTrue_test_gpt1_rl0.0004_2025_0_feature0.05_output0.214_train_epochs100_bs16_dr0.1_ld0.1_r8/checkpoint.pth"
$OUT_DIR_STEPWISE = "./figures/reproduced_baseline_stepwise_pdf"
$OUT_DIR_RESIDUAL = "./figures/reproduced_baseline_residual_pdf"

Write-Host ">>> [1/2] Generating Step-wise Error Probability Density Plots..." -ForegroundColor Cyan
& $PYTHON_CMD plot_stepwise_error_pdf.py `
    --checkpoint $CKPT_PATH `
    --root_path ./datasets/Solar/ `
    --data_path solar_AL.txt `
    --output_dir $OUT_DIR_STEPWISE `
    --seq_len 96 `
    --pred_len 96 `
    --batch_size 16 `
    --data_split 7:1:2 `
    --num_workers 0 `
    --force_rerun

Write-Host ">>> [2/2] Generating Residual Error Distribution & Normal Q-Q Plots..." -ForegroundColor Cyan
& $PYTHON_CMD plot_residual_histogram.py `
    --checkpoint $CKPT_PATH `
    --root_path ./datasets/Solar/ `
    --data_path solar_AL.txt `
    --output_dir $OUT_DIR_RESIDUAL `
    --seq_len 96 `
    --pred_len 96 `
    --batch_size 16 `
    --data_split 7:1:2 `
    --num_workers 0 `
    --force_rerun

Write-Host ">>> All distribution plots generated successfully!" -ForegroundColor Green
Write-Host ">>> Output 1: $OUT_DIR_STEPWISE" -ForegroundColor Yellow
Write-Host ">>> Output 2: $OUT_DIR_RESIDUAL" -ForegroundColor Yellow
