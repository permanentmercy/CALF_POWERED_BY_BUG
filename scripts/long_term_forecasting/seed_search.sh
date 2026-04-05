#!/bin/bash
GPU=0

model=CALF
data_name=Solar
set -o pipefail

# 记录文件和签名
ROOT_LOG_DIR=logs/$model/$data_name
COMPLETED_FILE=$ROOT_LOG_DIR/completed_combos.txt
SIGN_FILE=$ROOT_LOG_DIR/seed_search.signature

# 计算当前脚本签名（用于判断参数或脚本是否改动）
CUR_SIG=$(python - <<'PY'
import hashlib,sys,re
fn=r"f:/DOWNLOAD/参考文献-大模型/CALF-main/CALF-main/scripts/long_term_forecasting/seed_search.sh"
text=open(fn,'rb').read().decode('utf-8',errors='ignore')
# 移除命令行中的 --batch_size <value> 以及任何 batch_size=... 赋值，视为无变化
text=re.sub(r'--batch_size\s+\S+','',text)
text=re.sub(r'(?m)^\s*batch_size\s*=\s*\S+\s*$','',text)
# 规范化空白并移除空行以获得稳定签名
lines=[l.rstrip() for l in text.splitlines() if l.strip()!='']
clean='\n'.join(lines)
print(hashlib.md5(clean.encode('utf-8')).hexdigest())
PY
)

# 如果签名变化则清空已完成记录
if [ -f "$SIGN_FILE" ]; then
  OLD_SIG=$(cat "$SIGN_FILE")
else
  OLD_SIG=""
fi
if [ "$CUR_SIG" != "$OLD_SIG" ]; then
  mkdir -p "$ROOT_LOG_DIR"
  : > "$COMPLETED_FILE"
  echo "$CUR_SIG" > "$SIGN_FILE"
fi

if [ ! -d "./logs" ]; then
  mkdir ./logs
fi
if [ ! -d "./logs/$model" ]; then
  mkdir ./logs/$model
fi
if [ ! -d "./logs/$model/$data_name" ]; then
  mkdir ./logs/$model/$data_name
fi

# 定义要跳过的组合
SKIP_COMBOS=("")

# 待加入调整的参数：2个loss系数
seq_len=96
for feature_w in  $(seq 0.214 0.002 0.3)
do
for output_w in 1.3
do 
# 检查是否是跳过的组合
  combo="${feature_w}_${output_w}"
  skip=false
    
  for skip_combo in "${SKIP_COMBOS[@]}"; do
    if [ "$combo" == "$skip_combo" ]; then
      skip=true
      break
    fi
  done
  
  if [ "$skip" == true ]; then
    echo "skip specific hyperparameters: feature_w=$feature_w, output_w=$output_w"
    continue
  fi

  # 如果已记录完成，则跳过
  if grep -Fxq "$combo" "$COMPLETED_FILE"; then
    echo "already completed: feature_w=$feature_w, output_w=$output_w"
    continue
  fi
for learning_rate in   0.0001
do
for d_model in 768
do
for n_heads in 4
do
for random_seed in 2026
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
    --batch_size 32 \
    --learning_rate $learning_rate \
    --train_epochs 3 \
    --d_model $d_model \
    --n_heads $n_heads \
    --d_ff $((d_model * 4)) \
    --dropout 0.2 \
    --enc_in 137 \
    --c_out 137 \
    --gpt_layers 3 \
    --itr 1 \
    --model $model \
    --cos 1 \
    --tmax 10 \
    --r 8 \
    --lora_alpha 48 \
    --lora_dropout 0.1 \
    --patience 1 \
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
  EXIT_CODE=${PIPESTATUS[0]}
  if [ $EXIT_CODE -eq 0 ]; then
    echo "$combo" >> "$COMPLETED_FILE"
    echo "recorded completed combo: $combo"
  else
    echo "run failed for combo: $combo (exit $EXIT_CODE)" >&2
  fi
done
done
done
done
done
done
done
# 汇总每组不同 random_seed 的结果，写入 best_seeds_summary.txt
# python3 scripts/long_term_forecasting/select_best_seed.py --logs_dir logs/$model/$data_name --out_file logs/$model/$data_name/best_seeds_summary.txt

