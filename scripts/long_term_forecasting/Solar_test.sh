#!/bin/bash
GPU=0

model=CALF
data_name=Solar

# 记录文件和签名
ROOT_LOG_DIR=logs/$model/$data_name
COMPLETED_FILE=$ROOT_LOG_DIR/completed_combos.txt
SIGN_FILE=$ROOT_LOG_DIR/Solar_test.signature

# 计算当前脚本签名（用于判断参数或脚本是否改动）
CUR_SIG=$(python - <<'PY'
import hashlib,sys,re
fn=r"f:/DOWNLOAD/参考文献-大模型/CALF-main/CALF-main/scripts/long_term_forecasting/Solar_test.sh"
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

# 待加入调整的参数：3个loss权重 (output_w 由 task_w 和 feature_w 计算)
batch_size=16
seq_len=96
gate_dropout=0.1
gate_lr_factor=0.6
swa_lr=0.00005
for task_w in 0.602
do
for feature_w in 0.01
do
for d_model in 768
do
for n_heads in 4
do
for random_seed in  2026
do
for pred_len in 96 
do
  learning_rate=$(python -c "print('{:.8f}'.format(0.00000625*$batch_size))")

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

  combo="${feature_w}_${output_w}_${task_w}_${learning_rate}_${d_model}_${n_heads}_${random_seed}_${pred_len}_${gate_dropout}_${gate_lr_factor}"

  # 检查是否是显式跳过的组合
  SKIP_COMBOS=("")
  skip=false
  for skip_combo in "${SKIP_COMBOS[@]}"; do
    if [ "$combo" == "$skip_combo" ]; then
      skip=true
      break
    fi
  done
  if [ "$skip" == true ]; then
    echo "skip specific hyperparameters: output_w=$output_w, task_w=$task_w, feature_w=$feature_w"
    continue
  fi

  # 如果已记录完成，则跳过
  if [ -f "$COMPLETED_FILE" ] && grep -Fxq "$combo" "$COMPLETED_FILE"; then
    echo "already completed: $combo"
    continue
  fi
  
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
    --batch_size $batch_size \
    --learning_rate $learning_rate \
    --train_epochs 100 \
    --d_model $d_model \
    --n_heads $n_heads \
    --d_ff $((d_model*4)) \
    --dropout 0.1 \
    --enc_in 137 \
    --c_out 137 \
    --gpt_layers 3 \
    --itr 1 \
    --model $model \
    --cos 1 \
    --tmax 10 \
    --r 16 \
    --lora_alpha 16 \
    --lora_dropout 0.1 \
    --patience 3 \
    --num_workers 0 \
    --feature_w $feature_w \
    --output_w $output_w \
    --task_w $task_w \
    --bestmodel \
    --use_amp \
    --eval_test_every_epoch \
    --gpt2_path ./models/gpt2 \
    --task_loss smooth_l1 \
    --feature_loss smooth_l1 \
    --output_loss smooth_l1 \
    --gate_dropout $gate_dropout \
    --gate_lr_factor $gate_lr_factor \
    --use_swa \
    --swa_lr $swa_lr \
    --random_seed $random_seed \
     | tee logs/$model/$data_name/${feature_w}_${output_w}_${model}_${seq_len}_${pred_len}_${d_model}_${n_heads}_${learning_rate}_${random_seed}.logs

  EXIT_CODE=${PIPESTATUS[0]}
  if [ "$EXIT_CODE" -eq 0 ]; then
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


