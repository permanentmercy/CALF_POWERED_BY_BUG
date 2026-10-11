$model = "CALF"
$data_name = "Solar"
$python_exe = "D:\software\conda\envs\calf\python.exe"

# 创建日志目录
$log_dir = "logs\$model\$data_name"
if (-not (Test-Path $log_dir)) {
    New-Item -ItemType Directory -Path $log_dir -Force | Out-Null
}

$is_training = 1
$seq_len = 96
$batch_size = 16
$accumulation_steps = 4
$learning_rate = 0.0004
$train_epochs = 20
$patience = 5

$d_model = 768
$n_heads = 4
$d_ff = 3072
$gpt_layers = 1

$task_w = 0.736
$feature_w = 0.05
$output_w = 0.214

$random_seed = 2025
$percent = 100

# 复现 0.15965 历史最佳基准 (pred_len=96, seed=2025)
foreach ($pred_len in @(96)) {
    Write-Host ""
    Write-Host "==========================================================================================" -ForegroundColor Cyan
    Write-Host ">>> [Starting Experiment] Horizon: $pred_len | Seed: $random_seed | Model: $model | Percent: $percent%" -ForegroundColor Green
    Write-Host "==========================================================================================" -ForegroundColor Cyan

    $log_path = "$log_dir\opt_${model}_${seq_len}_${pred_len}_${d_model}_${n_heads}_${learning_rate}_${random_seed}.log"

    & $python_exe -u run.py `
        --root_path ./datasets/Solar/ `
        --data_path solar_AL.txt `
        --is_training $is_training `
        --task_name long_term_forecast `
        --model_id "${model}_opt_${seq_len}_${pred_len}" `
        --data $data_name `
        --seq_len $seq_len `
        --label_len 0 `
        --pred_len $pred_len `
        --batch_size $batch_size `
        --accumulation_steps $accumulation_steps `
        --learning_rate $learning_rate `
        --train_epochs $train_epochs `
        --patience $patience `
        --d_model $d_model `
        --n_heads $n_heads `
        --d_ff $d_ff `
        --dropout 0.1 `
        --enc_in 137 `
        --c_out 137 `
        --gpt_layers $gpt_layers `
        --itr 1 `
        --model $model `
        --cos 1 `
        --tmax 10 `
        --r 8 `
        --lora_alpha 32 `
        --lora_dropout 0.1 `
        --num_workers 0 `
        --feature_w $feature_w `
        --output_w $output_w `
        --task_w $task_w `
        --use_swa 0 `
        --tq_mode add `
        --use_tq_gate 0 `
        --layer_offset 0 `
        --e_layers 1 `
        --test_branch time `
        --mlp_res_w 0 `
        --bestmodel `
        --no_save_checkpoint `
        --save_checkpoint 0 `
        --out_mlp_layers 3 `
        --cycle_prior_before_denorm 0 `
        --tq_dropout 0.1 `
        --tq_lr_factor 5 `
        --use_norm 1 `
        --tq_inject_target q `
        --gpt2_path ./models/gpt2 `
        --task_loss smooth_l1 `
        --feature_loss smooth_l1 `
        --output_loss smooth_l1 `
        --random_seed $random_seed `
        --cycle 144 `
        --eval_test_every_epoch `
        --use_conv_emb 0 `
        --use_tq_time 0 `
        --task_time_ratio 0.5 `
        --lora_target_proj 0 `
        --use_dpg 1 `
        --use_direct_trend 1 `
        --use_harmonic 1 `
        --use_cycle_prior 1 `
        --one_way_distill 1 `
        --data_split 6:2:2 `
        --percent $percent `
        | Tee-Object -FilePath $log_path
}
