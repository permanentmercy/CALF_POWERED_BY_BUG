#!/bin/bash
GPU=0

model=CALF
data_name=Solar

ROOT_LOG_DIR=logs/$model/$data_name

if [ ! -d "./logs" ]; then
    mkdir ./logs
fi
if [ ! -d "./logs/$model" ]; then
    mkdir ./logs/$model
fi
if [ ! -d "./logs/$model/$data_name" ]; then
    mkdir ./logs/$model/$data_name
fi

# 待加入调整的参数：3个loss权重 (output_w 由 task_w 和 feature_w 计算)
is_training=1
batch_size=8
accumulation_steps=1
seq_len=96
tq_lr_factor=5
tq_dropout=0.1
for task_w in 0.736
do
for feature_w in 0.05
do
for d_model in 768
do
for n_heads in 4
do
for random_seed in 2025 2026 2027
do
for pred_len in 96
do
  learning_rate=$(python -c "print('{:.8f}'.format(0.00000625*$batch_size*$accumulation_steps))")

  # 由 task_w 和 feature_w 计算 output_w
  output_w=$(python -c "
o = 1.0 - $task_w - $feature_w
if o < 0:
    print(-1)
else:
    print('{:.6f}'.format(o))
")
  # 如果 output_w <= 0 则跳过该组合
  if python -c "import sys; sys.exit(0 if $output_w < 0 else 1)"; then
    echo "skip (output_w<=0): task_w=$task_w, feature_w=$feature_w -> output_w=$output_w"
    continue
  fi

  echo ""
  echo "=========================================================================================================="
  echo ">>> [Starting Experiment] Data: $data_name | Percent: ${percent}% | Seed: $random_seed | Horizon: $pred_len"
  echo ">>> Loss Weights : task_w=$task_w, feature_w=$feature_w, output_w=$output_w (Loss: smooth_l1)"
  echo ">>> Train Config : batch_size=$batch_size, accum_steps=$accumulation_steps, lr=$learning_rate, epochs=100 (patience=5)"
  echo ">>> Model Config : d_model=$d_model, n_heads=$n_heads, gpt_layers=1, lora(r=8, alpha=32)"
  echo ">>> TQ / Cycle   : cycle=144, mode=add, inject=q, lr_factor=$tq_lr_factor, dropout=$tq_dropout, swa=1"
  echo ">>> Model ID     : ${model}_${seq_len}_${pred_len}_p${percent}"
  echo "=========================================================================================================="

  CUDA_VISIBLE_DEVICES=$GPU \
  python -u run.py \
    --root_path ./datasets/Solar/ \
    --checkpoint_path ./checkpoints/long_term_forecast_CALF_96_96_p100_CALF_Solar_ftM_sl96_ll0_pl96_dm768_nh4_el1_dl1_df3072_fc1_ebtimeF_dtTrue_test_gpt1_rl0.0004_2026_0_feature0.05_output0.214_train_epochs100_bs16_dr0.1_ld0.1_r8/checkpoint.pth \
    --data_path solar_CT.csv \
    --is_training $is_training \
    --task_name long_term_forecast \
    --model_id $model'_'$seq_len'_'$pred_len'_p'$percent \
    --data Solar \
    --seq_len $seq_len \
    --label_len 0 \
    --pred_len $pred_len \
    --batch_size $batch_size \
    --learning_rate $learning_rate \
    --train_epochs 100 \
    --d_model $d_model \
    --n_heads $n_heads \
    --d_ff $((d_model*4)) \
    --dropout 0.1 \
    --enc_in 29 \
    --c_out 29 \
    --gpt_layers 1 \
    --itr 1 \
    --model $model \
    --cos 1 \
    --tmax 10 \
    --r 8 \
    --lora_alpha 32 \
    --lora_dropout 0.1 \
    --patience 5 \
    --num_workers 0 \
    --feature_w $feature_w \
    --output_w $output_w \
    --task_w $task_w \
    --use_swa 0 \
    --tq_mode add \
    --use_tq_gate 0 \
    --layer_offset 0 \
    --e_layers 1 \
    --test_branch time \
    --mlp_res_w 0 \
    --out_mlp_layers 3 \
    --cycle_prior_before_denorm 0 \
    --tq_dropout $tq_dropout \
    --tq_lr_factor $tq_lr_factor \
    --eval_test_every_epoch \
    --bestmodel \
    --use_norm 1 \
    --tq_inject_target q \
    --gpt2_path ./models/gpt2 \
    --task_loss smooth_l1 \
    --feature_loss smooth_l1 \
    --output_loss smooth_l1 \
    --accumulation_steps $accumulation_steps \
    --random_seed $random_seed \
    --cycle 144 \
     | tee -a logs/$model/$data_name/p${percent}_${feature_w}_${output_w}_${model}_${seq_len}_${pred_len}_${d_model}_${n_heads}_${learning_rate}_${random_seed}.logs
  EXIT_CODE=${PIPESTATUS[0]}
  if [ $EXIT_CODE -eq 0 ]; then
    echo "experiment completed"
  else
    echo "run failed (exit $EXIT_CODE)" >&2
  fi
done
done
done
done
done
done
