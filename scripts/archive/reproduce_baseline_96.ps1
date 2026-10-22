# reproduce_baseline_96.ps1
# ========================================================================================
# 功能：复现 develop 分支早期 CALF 原始基准模型 (Horizon 96, Seed 2025)
# 预期目标：MSE ≈ 0.164986, MAE ≈ 0.219895
# 训练完成后将完整模型检查点保存至磁盘
# ========================================================================================

$ErrorActionPreference = "Stop"

# 1. 查找有效 Python 路径
$PYTHON_CMD = "python"
if (Test-Path "D:\software\conda\envs\calf\python.exe") {
    $PYTHON_CMD = "D:\software\conda\envs\calf\python.exe"
} elseif (Get-Command "conda" -ErrorAction SilentlyContinue) {
    $CONDA_ENV_PY = (& conda info --base 2>$null) + "\envs\calf\python.exe"
    if (Test-Path $CONDA_ENV_PY) { $PYTHON_CMD = $CONDA_ENV_PY }
}

Write-Host ">>> Using Python: $PYTHON_CMD" -ForegroundColor Cyan

# 2. 确保日志目录存在
New-Item -ItemType Directory -Force -Path "logs/CALF/Solar" | Out-Null
New-Item -ItemType Directory -Force -Path "checkpoints" | Out-Null

$LOG_FILE = "logs/CALF/Solar/reproduce_p100_96_seed2025.log"
Write-Host ">>> Starting reproduction run for Horizon 96 (Seed 2025)..." -ForegroundColor Green
Write-Host ">>> Target Baseline Metrics: MSE ~ 0.164986, MAE ~ 0.219895" -ForegroundColor Yellow
Write-Host ">>> Logs will be streamed to: $LOG_FILE" -ForegroundColor Cyan

# 3. 执行全量从头训练 (percent=100, patience=5)
& $PYTHON_CMD -u run.py `
    --root_path ./datasets/Solar/ `
    --data_path solar_AL.txt `
    --is_training 1 `
    --task_name long_term_forecast `
    --model_id CALF_96_96_p100 `
    --data Solar `
    --seq_len 96 `
    --label_len 0 `
    --pred_len 96 `
    --percent 100 `
    --batch_size 16 `
    --learning_rate 0.0004 `
    --train_epochs 100 `
    --d_model 768 `
    --n_heads 4 `
    --d_ff 3072 `
    --dropout 0.1 `
    --enc_in 137 `
    --c_out 137 `
    --gpt_layers 1 `
    --itr 1 `
    --model CALF `
    --cos 1 `
    --tmax 10 `
    --r 8 `
    --lora_alpha 32 `
    --lora_dropout 0.1 `
    --patience 5 `
    --num_workers 0 `
    --feature_w 0.05 `
    --output_w 0.214 `
    --task_w 0.736 `
    --use_swa 0 `
    --tq_mode add `
    --use_tq_gate 0 `
    --layer_offset 0 `
    --e_layers 1 `
    --test_branch time `
    --mlp_res_w 0 `
    --out_mlp_layers 3 `
    --cycle_prior_before_denorm 0 `
    --tq_dropout 0.1 `
    --tq_lr_factor 5 `
    --eval_test_every_epoch `
    --use_norm 1 `
    --tq_inject_target q `
    --gpt2_path ./models/gpt2 `
    --task_loss smooth_l1 `
    --feature_loss smooth_l1 `
    --output_loss smooth_l1 `
    --accumulation_steps 4 `
    --random_seed 2025 `
    --cycle 144 `
    *>&1 | Tee-Object -FilePath $LOG_FILE

Write-Host ">>> Reproduction run finished. Check results in $LOG_FILE" -ForegroundColor Green
