$model = "CALF"
$data_name = "Solar"
$python_exe = "D:\software\conda\envs\calf\python.exe"
$al_ckpt = "./checkpoints/AL_pretrained_best_96_96_seed2025.pth"

# 验证基准预训练权重是否存在
if (-not (Test-Path $al_ckpt)) {
    Write-Error "Error: 预训练权重不存在: $al_ckpt"
    exit 1
}

$log_dir = "logs\$model\$data_name"
if (-not (Test-Path $log_dir)) {
    New-Item -ItemType Directory -Path $log_dir -Force | Out-Null
}

$is_training = 1
$seq_len = 96
$pred_len = 96
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

foreach ($percent in @(5, 10)) {
    Write-Host ""
    Write-Host "==========================================================================================" -ForegroundColor Cyan
    Write-Host ">>> [Starting IL Few-Shot Fine-Tuning] Percent: $percent% | Seed: $random_seed | Model: $model" -ForegroundColor Green
    Write-Host "==========================================================================================" -ForegroundColor Cyan

    $log_path = "$log_dir\opt_${model}_IL_ft_p${percent}_${seq_len}_${pred_len}_${random_seed}.log"

    & $python_exe -u run.py `
        --root_path ./datasets/Solar/ `
        --data_path solar_IL.csv `
        --checkpoint_path $al_ckpt `
        --is_training $is_training `
        --task_name long_term_forecast `
        --model_id "${model}_opt_IL_ft_${seq_len}_${pred_len}" `
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
        --percent $percent `
        --dropout 0.1 `
        --enc_in 106 `
        --c_out 106 `
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
        | Tee-Object -FilePath $log_path -Append
}
