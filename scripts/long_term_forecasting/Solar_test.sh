GPU=0

model=CALF
data_name=Solar

if [ ! -d "./logs" ]; then
    mkdir ./logs
fi
if [ ! -d "./logs/$model" ]; then
    mkdir ./logs/$model
fi
if [ ! -d "./logs/$model/$data_name" ]; then
    mkdir ./logs/$model/$data_name
fi


# 待加入调整的参数：2个loss系数
seq_len=96
for feature_w in  0.0007
do
for output_w in 1.2
do 
for learning_rate in   0.0001
do
for d_model in 768
do
for n_heads in 4
do
for random_seed in 2025 2026 2027
do
for pred_len in 96
do
  CUDA_VISIBLE_DEVICES=$GPU \
  python -u run.py \
    --root_path ./datasets/Solar/ \
    --data_path solar_AL.txt \
    --is_training 1 \
    --task_name long_term_forecast \
    --model_id $model'_'$seq_len'_'$pred_len \
    --data Solar \
    --seq_len $seq_len \
    --label_len 0 \
    --pred_len $pred_len \
    --batch_size 16 \
    --learning_rate $learning_rate \
    --train_epochs 100 \
    --d_model $d_model \
    --n_heads $n_heads \
    --d_ff $((d_model*4)) \
    --dropout 0.2 \
    --enc_in 137 \
    --c_out 137 \
    --gpt_layers 6 \
    --itr 1 \
    --model $model \
    --cos 1 \
    --tmax 10 \
    --r 8 \
    --lora_alpha 48 \
    --lora_dropout 0.1 \
    --patience 5 \
    --feature_w $feature_w \
    --output_w $output_w \
    --bestmodel \
    --use_amp \
    --gpt2_path ./models/gpt2 \
    --task_loss smooth_l1 \
    --feature_loss smooth_l1 \
    --output_loss smooth_l1 \
    --random_seed $random_seed \
     | tee logs/$model/$data_name/$feature_w'_'$output_w'_'$model'_'$seq_len'_'$pred_len'_'$d_model'_'$n_heads'_'$learning_rate'_'$random_seed.logs
done
done
done
done
done
done
done
# 汇总每组不同 random_seed 的结果，写入 best_seeds_summary.txt
# python3 scripts/long_term_forecasting/select_best_seed.py --logs_dir logs/$model/$data_name --out_file logs/$model/$data_name/best_seeds_summary.txt

