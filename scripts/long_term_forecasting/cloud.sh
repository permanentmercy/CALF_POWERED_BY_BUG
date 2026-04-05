#!/bin/bash

# 切换到代码根目录（DLC 环境）
if [ -d "/root/code/CALF_POWERED_BY_BUG" ]; then
    cd /root/code/CALF_POWERED_BY_BUG/
fi
export HF_ENDPOINT=https://hf-mirror.com
GPU=0
model=CALF
data_name=Solar
set -o pipefail
mkdir -p /mnt/data/logs/$model/$data_name
# ==================== 找到可用的 python ====================
if command -v python >/dev/null 2>&1; then
    PY=python
elif command -v python3 >/dev/null 2>&1; then
    PY=python3
else
    echo "ERROR: No python or python3 found in PATH. Please install Python." >&2
    exit 127
fi


# ==================== 签名计算（使用脚本自身路径）====================
ROOT_LOG_DIR=logs/$model/$data_name
COMPLETED_FILE=$ROOT_LOG_DIR/completed_combos.txt
SIGN_FILE=$ROOT_LOG_DIR/seed_search.signature

SCRIPT_PATH=$(realpath "$0")
CUR_SIG=$($PY - "$SCRIPT_PATH" <<'PY'
import hashlib, sys, re, os
fn = sys.argv[1]
try:
    with open(fn, 'rb') as f:
        text = f.read().decode('utf-8', errors='ignore')
except FileNotFoundError:
    print("ERROR: cannot find script file", file=sys.stderr)
    sys.exit(1)
text = re.sub(r'--batch_size\s+\S+', '', text)
text = re.sub(r'(?m)^\s*batch_size\s*=\s*\S+\s*$', '', text)
lines = [l.rstrip() for l in text.splitlines() if l.strip() != '']
clean = '\n'.join(lines)
print(hashlib.md5(clean.encode('utf-8')).hexdigest())
PY
)

if [ $? -ne 0 ]; then
    echo "ERROR: signature calculation failed" >&2
    exit 1
fi

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

# 确保日志目录存在
mkdir -p "./logs/$model/$data_name"

# ==================== 超参数搜索循环 ====================
SKIP_COMBOS=("")
seq_len=96

$PY -u pca.py 

for feature_w in $(seq 0.2 0.002 0.3); do
  for output_w in 1.3; do
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

    if grep -Fxq "$combo" "$COMPLETED_FILE" 2>/dev/null; then
      echo "already completed: feature_w=$feature_w, output_w=$output_w"
      continue
    fi

    for learning_rate in 0.0001; do
      for d_model in 768; do
        for n_heads in 4; do
          for random_seed in 2026; do
            for pred_len in 96; do
              LOG_FILE="/mnt/data/logs/$model/$data_name/${feature_w}_${output_w}_${model}_${seq_len}_${pred_len}_${d_model}_${n_heads}_${learning_rate}_${random_seed}.logs"
              CUDA_VISIBLE_DEVICES=$GPU \
              $PY -u run.py \
                --root_path /mnt/data/ \
                --data_path solar_AL.txt \
                --is_training 1 \
                --task_name long_term_forecast \
                --model_id ${model}_${seq_len}_${pred_len} \
                --data Solar \
                --seq_len $seq_len \
                --label_len 0 \
                --pred_len $pred_len \
                --batch_size 128 \
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
                --gpt2_path /mnt/data/gpt2/ \
                --task_loss smooth_l1 \
                --feature_loss smooth_l1 \
                --output_loss smooth_l1 \
                --random_seed $random_seed \
                | tee "$LOG_FILE"
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

# 可选：汇总结果
# $PY scripts/long_term_forecasting/select_best_seed.py --logs_dir logs/$model/$data_name --out_file logs/$model/$data_name/best_seeds_summary.txt