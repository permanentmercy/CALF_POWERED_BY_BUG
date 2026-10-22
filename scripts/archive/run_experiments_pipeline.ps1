# run_experiments_pipeline.ps1
# ========================================================================================
# 实验流水线：
# 任务 1: 打开自回归直连趋势跳跃 (--use_direct_trend 1)，其他参数保持不变 (7:1:2 划分)
# 任务 2: 任务 1 完成后，关闭趋势跳跃 (--use_direct_trend 0)，并将数据集划分改为 6:2:2 (--data_split 6:2:2)
# ========================================================================================

$ErrorActionPreference = "Stop"

$PYTHON_CMD = "python"
if (Test-Path "D:\software\conda\envs\calf\python.exe") {
    $PYTHON_CMD = "D:\software\conda\envs\calf\python.exe"
}

Write-Host ">>> Using Python: $PYTHON_CMD" -ForegroundColor Cyan
New-Item -ItemType Directory -Force -Path "logs/CALF/Solar" | Out-Null
New-Item -ItemType Directory -Force -Path "checkpoints" | Out-Null

# ----------------------------------------------------------------------------------------
# 实验 1: 打开自回归直连趋势跳跃 (use_direct_trend = 1)，7:1:2 划分，其他参数不变
# ----------------------------------------------------------------------------------------
$LOG_EXP1 = "logs/CALF/Solar/exp1_direct_trend_712.log"
Write-Host "`n==========================================================================================" -ForegroundColor Green
Write-Host ">>> [Experiment 1/2] CALF with Direct Trend Skip (use_direct_trend=1, split=7:1:2)" -ForegroundColor Green
Write-Host ">>> Streaming output to: $LOG_EXP1" -ForegroundColor Cyan
Write-Host "==========================================================================================`n" -ForegroundColor Green

& $PYTHON_CMD -u run.py `
    --root_path ./datasets/Solar/ `
    --data_path solar_AL.txt `
    --is_training 1 `
    --task_name long_term_forecast `
    --model_id CALF_96_96_direct_trend `
    --des trend_on `
    --data Solar `
    --data_split 7:1:2 `
    --use_direct_trend 1 `
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
    *>&1 | Tee-Object -FilePath $LOG_EXP1

Write-Host "`n>>> [Experiment 1 Finished] Check results in $LOG_EXP1" -ForegroundColor Yellow

# ----------------------------------------------------------------------------------------
# 实验 2: 关闭趋势跳跃 (use_direct_trend = 0)，数据集划分为 6:2:2
# ----------------------------------------------------------------------------------------
$LOG_EXP2 = "logs/CALF/Solar/exp2_notrend_split622.log"
Write-Host "`n==========================================================================================" -ForegroundColor Magenta
Write-Host ">>> [Experiment 2/2] CALF without Direct Trend (use_direct_trend=0, split=6:2:2)" -ForegroundColor Magenta
Write-Host ">>> Streaming output to: $LOG_EXP2" -ForegroundColor Cyan
Write-Host "==========================================================================================`n" -ForegroundColor Magenta

& $PYTHON_CMD -u run.py `
    --root_path ./datasets/Solar/ `
    --data_path solar_AL.txt `
    --is_training 1 `
    --task_name long_term_forecast `
    --model_id CALF_96_96_split622 `
    --des notrend_split622 `
    --data Solar `
    --data_split 6:2:2 `
    --use_direct_trend 0 `
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
    *>&1 | Tee-Object -FilePath $LOG_EXP2

Write-Host "`n>>> [Experiment 2 Finished] Check results in $LOG_EXP2" -ForegroundColor Yellow
Write-Host "`n>>> [All Experiments Pipeline Completed!]" -ForegroundColor Green
