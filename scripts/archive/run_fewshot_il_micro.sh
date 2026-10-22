#!/bin/bash
# ==============================================================================
# Script: scripts/run_fewshot_il_micro.sh
# Description: Extreme Micro Few-Shot Suite on Solar-IL (0.5%, 1%, 2%, 3%, 4%, 5%, 10%)
# ==============================================================================

set -e

GPU=${1:-0}
CHECKPOINT_PATH=${2:-"0"}
export CUDA_VISIBLE_DEVICES=$GPU
export HF_ENDPOINT="https://hf-mirror.com"
export PYTHONWARNINGS="ignore"

MODEL="CALF"
DATA="Solar"
DATA_PATH="solar_IL.csv"
DATA_SPLIT="7:1:2"
ENC_IN=106
C_OUT=106
SEQ_LEN=96
PRED_LEN=96
BATCH_SIZE=16
ACCUM_STEPS=1
CYCLE=144
GPT2_PATH="./models/gpt2"

LR=$(python -c "print('{:.8f}'.format(0.00000625 * $BATCH_SIZE * $ACCUM_STEPS))")

RESULTS_DIR="./results"
RESULTS_CSV="${RESULTS_DIR}/results_fewshot_il_micro.csv"
LOG_DIR="./logs/${MODEL}/Solar_IL_Micro_Fewshot"

mkdir -p "$RESULTS_DIR"
mkdir -p "$LOG_DIR"

if [ ! -f "$RESULTS_CSV" ]; then
    echo "percent,seed,test_mse,test_mae,elapsed_sec,status,timestamp" > "$RESULTS_CSV"
fi

echo "=============================================================================="
echo ">>> [CALF] Starting Solar-IL Micro Few-Shot Suite (0.5% ~ 10%)"
echo ">>> GPU: $GPU | Data: $DATA_PATH | Pretrained Checkpoint: $CHECKPOINT_PATH"
echo ">>> Logging to: $RESULTS_CSV"
echo "=============================================================================="

PERCENTS=(0.5 1 2 3 4 5 10)
SEEDS=(2025 2026 2027)

TOTAL_EXPS=$((${#PERCENTS[@]} * ${#SEEDS[@]}))
CURR_EXP=0

for percent in "${PERCENTS[@]}"; do
    for seed in "${SEEDS[@]}"; do
        CURR_EXP=$((CURR_EXP + 1))

        if python -c "
import pandas as pd, sys
try:
    df = pd.read_csv('$RESULTS_CSV')
    match = df[(abs(df['percent'] - $percent) < 1e-4) & (df['seed'] == $seed) & (df['status'] == 'SUCCESS') & (df['test_mse'].notna())]
    sys.exit(0 if len(match) > 0 and str(match.iloc[0]['test_mse']).lower() != 'nan' else 1)
except:
    sys.exit(1)
"; then
            echo "[$CURR_EXP/$TOTAL_EXPS] Percent: ${percent}% | Seed: ${seed} -> ALREADY FINISHED (Skipping)"
            continue
        fi

        echo ""
        echo "------------------------------------------------------------------------------"
        echo "[$CURR_EXP/$TOTAL_EXPS] Starting Percent: ${percent}% | Seed: ${seed}"
        echo "------------------------------------------------------------------------------"

        LOG_FILE="${LOG_DIR}/il_micro_p${percent}_seed${seed}.log"
        START_TIME=$(date +%s)

        set +e
        python -u run.py \
            --root_path ./datasets/Solar/ \
            --data_path "$DATA_PATH" \
            --is_training 1 \
            --task_name long_term_forecast \
            --model_id "${MODEL}_IL_${SEQ_LEN}_${PRED_LEN}_p${percent}" \
            --data "$DATA" \
            --data_split "$DATA_SPLIT" \
            --percent "$percent" \
            --checkpoint_path "$CHECKPOINT_PATH" \
            --seq_len "$SEQ_LEN" \
            --label_len 0 \
            --pred_len "$PRED_LEN" \
            --batch_size "$BATCH_SIZE" \
            --learning_rate "$LR" \
            --train_epochs 100 \
            --patience 5 \
            --d_model 768 \
            --n_heads 4 \
            --e_layers 1 \
            --d_layers 1 \
            --d_ff 3072 \
            --dropout 0.1 \
            --enc_in "$ENC_IN" \
            --c_out "$C_OUT" \
            --dec_in "$ENC_IN" \
            --gpt_layers 1 \
            --itr 1 \
            --model "$MODEL" \
            --cos 1 \
            --tmax 10 \
            --r 8 \
            --lora_alpha 32 \
            --lora_dropout 0.1 \
            --num_workers 0 \
            --task_w 0.736 \
            --feature_w 0.05 \
            --output_w 0.214 \
            --feature_loss smooth_l1 \
            --output_loss smooth_l1 \
            --task_loss smooth_l1 \
            --use_swa 0 \
            --tq_mode add \
            --tq_inject_target q \
            --use_tq 1 \
            --use_tq_gate 0 \
            --cycle_prior_before_denorm 0 \
            --tq_dropout 0.1 \
            --tq_lr_factor 5 \
            --cycle "$CYCLE" \
            --test_branch time \
            --out_mlp_layers 3 \
            --mlp_res_w 0 \
            --use_norm 1 \
            --use_direct_trend 0 \
            --bestmodel \
            --gpt2_path "$GPT2_PATH" \
            --random_seed "$seed" \
            2>&1 | tee "$LOG_FILE"
        
        EXIT_CODE=$?
        set -e
        END_TIME=$(date +%s)
        ELAPSED=$((END_TIME - START_TIME))

        if [ $EXIT_CODE -ne 0 ]; then
            echo ">>> [ERROR] Run failed for percent=${percent}, seed=${seed}"
            echo "${percent},${seed},NaN,NaN,${ELAPSED},FAILED,$(date '+%Y-%m-%d %H:%M:%S')" >> "$RESULTS_CSV"
            continue
        fi

        METRICS=$(python -c "
import re
raw = open('${LOG_FILE}', 'rb').read()
text = ''
for enc in ['utf-16', 'utf-8', 'gbk', 'latin1']:
    try:
        c = raw.decode(enc)
        if 'mse:' in c:
            text = c
            break
    except Exception:
        pass
if not text:
    text = raw.decode('latin1', errors='ignore')
matches = re.findall(r'mse:([0-9\.]+),\s*mae:([0-9\.]+)', text)
if matches:
    print(f'{matches[-1][0]},{matches[-1][1]}')
else:
    print('NaN,NaN')
")

        TEST_MSE=$(echo "$METRICS" | cut -d',' -f1)
        TEST_MAE=$(echo "$METRICS" | cut -d',' -f2)
        TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')

        echo ">>> [SUCCESS] Finished in ${ELAPSED}s | Test MSE: ${TEST_MSE} | Test MAE: ${TEST_MAE}"
        echo "${percent},${seed},${TEST_MSE},${TEST_MAE},${ELAPSED},SUCCESS,${TIMESTAMP}" >> "$RESULTS_CSV"

        # 权重不落盘: 立即删除本轮微调生成的 checkpoint 目录与权重
        find ./checkpoints -maxdepth 1 -type d -name "*CALF_IL*p${percent}*_${seed}_0_*" -exec rm -rf {} + 2>/dev/null || true
    done
done

echo ""
echo "=============================================================================="
echo ">>> [ALL RUNS FINISHED] Aggregating Results and Generating Plots..."
echo "=============================================================================="

python plot_fewshot_il_micro.py \
    --csv_path "$RESULTS_CSV" \
    --summary_csv "./results/fewshot_il_micro_summary.csv" \
    --output_dir "./figures/fewshot"

echo ""
echo ">>> Done! Summary table: ./results/fewshot_il_micro_summary.csv"
echo ">>> Figure generated: ./figures/fewshot/fewshot_il_micro_curve.png"
