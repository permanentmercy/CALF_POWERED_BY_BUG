#!/bin/bash
# ==============================================================================
# Script: scripts/run_fewshot_il.sh
# Description: AL Dataset Few-Shot Training Suite (10% to 50%, step 10%, 3 seeds)
# Features:
#   1. Zero-Disk Footprint: Weights are evaluated in-memory (--bestmodel) and 
#      checkpoints are immediately deleted after evaluation (rm -rf checkpoints/...)
#   2. Idempotent Resume: Automatically skips already completed experiments
#   3. Metrics Logging: Appends test MSE & MAE directly into CSV
# ==============================================================================

set -e

# Default GPU (can be overridden by argument: bash run_fewshot_il.sh 1)
GPU=${1:-0}
export CUDA_VISIBLE_DEVICES=$GPU

MODEL="CALF"
DATA="Solar"
DATA_PATH="solar_AL.txt"
DATA_SPLIT="6:2:2"
ENC_IN=137
C_OUT=137
SEQ_LEN=96
PRED_LEN=96
BATCH_SIZE=16
ACCUM_STEPS=1
CYCLE=144

# Learning rate scaled by batch size
LR=$(python -c "print('{:.8f}'.format(0.00000625 * $BATCH_SIZE * $ACCUM_STEPS))")

RESULTS_DIR="./results"
RESULTS_CSV="${RESULTS_DIR}/results_fewshot_al.csv"
LOG_DIR="./logs/${MODEL}/Solar_AL_Fewshot"

mkdir -p "$RESULTS_DIR"
mkdir -p "$LOG_DIR"

# Initialize CSV header if not present
if [ ! -f "$RESULTS_CSV" ]; then
    echo "percent,seed,test_mse,test_mae,train_epochs,status,timestamp" > "$RESULTS_CSV"
fi

echo "=============================================================================="
echo ">>> [CALF] Starting AL Dataset Few-Shot Training Suite (Zero-Disk Footprint)"
echo ">>> GPU Device: $GPU | Data: $DATA_PATH (EncIn=$ENC_IN, Split=$DATA_SPLIT)"
echo ">>> Horizon: $SEQ_LEN -> $PRED_LEN | Batch Size: $BATCH_SIZE | Base LR: $LR"
echo ">>> Results CSV: $RESULTS_CSV"
echo "=============================================================================="

PERCENTS=(10 20 30 40 50)
SEEDS=(2025 2026 2027)

TOTAL_EXPS=$((${#PERCENTS[@]} * ${#SEEDS[@]}))
CURR_EXP=0

for percent in "${PERCENTS[@]}"; do
    for seed in "${SEEDS[@]}"; do
        CURR_EXP=$((CURR_EXP + 1))
        
        # Check if already completed (Resume protection)
        if grep -q "^${percent},${seed}," "$RESULTS_CSV" 2>/dev/null; then
            echo "[${CURR_EXP}/${TOTAL_EXPS}] Percent: ${percent}% | Seed: ${seed} -> ALREADY FINISHED (Skipping)"
            continue
        fi

        echo ""
        echo "------------------------------------------------------------------------------"
        echo "[${CURR_EXP}/${TOTAL_EXPS}] Starting Percent: ${percent}% | Random Seed: ${seed}"
        echo "------------------------------------------------------------------------------"

        LOG_FILE="${LOG_DIR}/al_fewshot_p${percent}_seed${seed}.log"
        START_TIME=$(date +%s)

        # Run scratch few-shot training (no pre-trained checkpoint passed)
        set +e
        python -u run.py \
            --root_path ./datasets/Solar/ \
            --data_path "$DATA_PATH" \
            --is_training 1 \
            --task_name long_term_forecast \
            --model_id "${MODEL}_AL_${SEQ_LEN}_${PRED_LEN}_p${percent}" \
            --data "$DATA" \
            --data_split "$DATA_SPLIT" \
            --percent "$percent" \
            --seq_len "$SEQ_LEN" \
            --label_len 0 \
            --pred_len "$PRED_LEN" \
            --batch_size "$BATCH_SIZE" \
            --learning_rate "$LR" \
            --train_epochs 100 \
            --patience 5 \
            --d_model 768 \
            --n_heads 4 \
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
            --task_w 0.70 \
            --feature_w 0.05 \
            --output_w 0.25 \
            --use_swa 0 \
            --tq_mode add \
            --tq_inject_target q \
            --use_tq 1 \
            --cycle "$CYCLE" \
            --test_branch time \
            --out_mlp_layers 3 \
            --mlp_res_w 0 \
            --use_norm 1 \
            --use_direct_trend 0 \
            --bestmodel \
            --random_seed "$seed" \
            2>&1 | tee "$LOG_FILE"
        
        EXIT_CODE=$?
        set -e
        END_TIME=$(date +%s)
        ELAPSED=$((END_TIME - START_TIME))

        if [ $EXIT_CODE -ne 0 ]; then
            echo ">>> [ERROR] Run failed for percent=${percent}, seed=${seed} (Exit code: $EXIT_CODE)"
            echo "${percent},${seed},NaN,NaN,0,FAILED,$(date '+%Y-%m-%d %H:%M:%S')" >> "$RESULTS_CSV"
            continue
        fi

        # Parse MSE and MAE from log
        METRICS=$(python -c "
import re
with open('${LOG_FILE}', 'r', encoding='utf-8', errors='ignore') as f:
    text = f.read()
matches = re.findall(r'mse:([0-9\.]+),\s*mae:([0-9\.]+)', text)
if matches:
    mse, mae = matches[-1]
    print(f'{mse},{mae}')
else:
    print('NaN,NaN')
")

        TEST_MSE=$(echo "$METRICS" | cut -d',' -f1)
        TEST_MAE=$(echo "$METRICS" | cut -d',' -f2)
        TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')

        echo ">>> [SUCCESS] Finished in ${ELAPSED}s | Test MSE: ${TEST_MSE} | Test MAE: ${TEST_MAE}"
        echo "${percent},${seed},${TEST_MSE},${TEST_MAE},${ELAPSED},SUCCESS,${TIMESTAMP}" >> "$RESULTS_CSV"

        # --------------------------------------------------------------------------
        # 权重不落盘 (Zero-Disk Footprint): 立即清空该次实验生成的权重目录
        # --------------------------------------------------------------------------
        echo ">>> [CLEANUP] Deleting checkpoint files to prevent disk saturation..."
        # 匹配 run.py 生成的 checkpoint 目录
        rm -rf ./checkpoints/*_${MODEL}_*${percent}*seed${seed}* 2>/dev/null || true
        rm -rf ./checkpoints/*p${percent}*seed${seed}* 2>/dev/null || true
        find ./checkpoints -type f -name "*.pth" -mmin -60 -delete 2>/dev/null || true
        echo ">>> [CLEANUP] Checkpoint storage purged successfully."
    done
done

echo ""
echo "=============================================================================="
echo ">>> [ALL COMPLETED] AL Few-Shot Suite Finished!"
echo ">>> Results saved in: $RESULTS_CSV"
echo "=============================================================================="
