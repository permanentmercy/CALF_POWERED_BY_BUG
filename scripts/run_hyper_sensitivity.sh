#!/bin/bash
# ==============================================================================
# Script: scripts/run_hyper_sensitivity.sh
# Description: Comprehensive Hyperparameter Sensitivity Suite (Solar-AL Full Model)
# Modules:
#   1. depth:        gpt_layers (1..4) x out_mlp_layers (1..4) [4x4 Heatmap A]
#   2. capacity:     r (2,4,8,16) x d_ff (768,1536,3072,4096) [4x4 Heatmap B]
#   3. distillation: Simplex loss weights (task_w + feat_w + out_w = 1.0) [Heatmap C]
#
# Features:
#   - Zero-Disk Footprint: Immediate purge of .pth files after each run
#   - Idempotent Resume: Checks CSV before launching; skips completed jobs
#   - Baseline Sharing: Avoids rerunning baseline across multiple modules
#   - Selective Execution: bash run_hyper_sensitivity.sh <GPU> <all|depth|capacity|distillation>
# ==============================================================================

set -e

GPU=${1:-0}
TARGET_MODULE=${2:-all}
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

# Default Baseline Hyperparameters
BASE_GPT_LAYERS=1
BASE_MLP_LAYERS=3
BASE_LORA_R=8
BASE_LORA_ALPHA=32
BASE_D_FF=3072
BASE_TASK_W=0.70
BASE_FEAT_W=0.05
BASE_OUT_W=0.25

SEEDS=(2025 2026 2027)

RESULTS_DIR="./results"
RESULTS_CSV="${RESULTS_DIR}/results_hyper_sensitivity_al.csv"
LOG_DIR="./logs/${MODEL}/Solar_AL_Sensitivity"

mkdir -p "$RESULTS_DIR"
mkdir -p "$LOG_DIR"

# Initialize CSV header if not present
if [ ! -f "$RESULTS_CSV" ]; then
    echo "module,gpt_layers,out_mlp_layers,lora_r,d_ff,task_w,feature_w,output_w,seed,test_mse,test_mae,elapsed_sec,status,timestamp" > "$RESULTS_CSV"
fi

echo "=============================================================================="
echo ">>> [CALF] Starting Hyperparameter Sensitivity Suite (Zero-Disk Footprint)"
echo ">>> GPU: $GPU | Module Filter: $TARGET_MODULE | Dataset: $DATA_PATH"
echo ">>> Logging Results to: $RESULTS_CSV"
echo "=============================================================================="

# Helper function to execute one experiment
run_single_experiment() {
    local mod=$1
    local gpt_l=$2
    local mlp_l=$3
    local r=$4
    local dff=$5
    local tw=$6
    local fw=$7
    local ow=$8
    local seed=$9

    # Check if already completed (Match exact hyperparameter signature)
    if python -c "
import pandas as pd, sys
csv_path = '$RESULTS_CSV'
try:
    df = pd.read_csv(csv_path)
    match = df[
        (df['gpt_layers'] == $gpt_l) & 
        (df['out_mlp_layers'] == $mlp_l) & 
        (df['lora_r'] == $r) & 
        (df['d_ff'] == $dff) & 
        (abs(df['task_w'] - $tw) < 1e-4) & 
        (abs(df['feature_w'] - $fw) < 1e-4) & 
        (abs(df['output_w'] - $ow) < 1e-4) & 
        (df['seed'] == $seed) &
        (df['status'] == 'SUCCESS')
    ]
    if len(match) > 0:
        sys.exit(0)
    else:
        sys.exit(1)
except Exception:
    sys.exit(1)
"; then
        echo ">>> [SKIP] ${mod} (gpt=${gpt_l}, mlp=${mlp_l}, r=${r}, dff=${dff}, w=[${tw},${fw},${ow}], seed=${seed}) ALREADY DONE"
        return 0
    fi

    echo ""
    echo "------------------------------------------------------------------------------"
    echo ">>> [RUN] Module: ${mod} | GPT:${gpt_l} | MLP:${mlp_l} | r:${r} | d_ff:${dff}"
    echo ">>> Loss Weights: task=${tw}, feat=${fw}, out=${ow} | Seed: ${seed}"
    echo "------------------------------------------------------------------------------"

    local lr=$(python -c "print('{:.8f}'.format(0.00000625 * $BATCH_SIZE * $ACCUM_STEPS))")
    local alpha=$((r * 4)) # standard scaling alpha = 4 * r
    local log_file="${LOG_DIR}/${mod}_gpt${gpt_l}_mlp${mlp_l}_r${r}_dff${dff}_tw${tw}_fw${fw}_s${seed}.log"
    local start_time=$(date +%s)

    set +e
    python -u run.py \
        --root_path ./datasets/Solar/ \
        --data_path "$DATA_PATH" \
        --is_training 1 \
        --task_name long_term_forecast \
        --model_id "${MODEL}_AL_sens_${mod}" \
        --data "$DATA" \
        --data_split "$DATA_SPLIT" \
        --percent 100 \
        --seq_len "$SEQ_LEN" \
        --label_len 0 \
        --pred_len "$PRED_LEN" \
        --batch_size "$BATCH_SIZE" \
        --learning_rate "$lr" \
        --train_epochs 100 \
        --patience 5 \
        --d_model 768 \
        --n_heads 4 \
        --d_ff "$dff" \
        --dropout 0.1 \
        --enc_in "$ENC_IN" \
        --c_out "$C_OUT" \
        --dec_in "$ENC_IN" \
        --gpt_layers "$gpt_l" \
        --itr 1 \
        --model "$MODEL" \
        --cos 1 \
        --tmax 10 \
        --r "$r" \
        --lora_alpha "$alpha" \
        --lora_dropout 0.1 \
        --num_workers 0 \
        --task_w "$tw" \
        --feature_w "$fw" \
        --output_w "$ow" \
        --use_swa 0 \
        --tq_mode add \
        --tq_inject_target q \
        --use_tq 1 \
        --cycle "$CYCLE" \
        --test_branch time \
        --out_mlp_layers "$mlp_l" \
        --mlp_res_w 0 \
        --use_norm 1 \
        --use_direct_trend 0 \
        --bestmodel \
        --random_seed "$seed" \
        2>&1 | tee "$log_file"
    
    local exit_code=$?
    set -e
    local end_time=$(date +%s)
    local elapsed=$((end_time - start_time))

    if [ $exit_code -ne 0 ]; then
        echo ">>> [ERROR] Run failed (Exit code: $exit_code)"
        echo "${mod},${gpt_l},${mlp_l},${r},${dff},${tw},${fw},${ow},${seed},NaN,NaN,${elapsed},FAILED,$(date '+%Y-%m-%d %H:%M:%S')" >> "$RESULTS_CSV"
        return 0
    fi

    # Parse test MSE and MAE
    local metrics=$(python -c "
import re
with open('${log_file}', 'r', encoding='utf-8', errors='ignore') as f:
    text = f.read()
matches = re.findall(r'mse:([0-9\.]+),\s*mae:([0-9\.]+)', text)
if matches:
    mse, mae = matches[-1]
    print(f'{mse},{mae}')
else:
    print('NaN,NaN')
")

    local test_mse=$(echo "$metrics" | cut -d',' -f1)
    local test_mae=$(echo "$metrics" | cut -d',' -f2)
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')

    echo ">>> [SUCCESS] Finished in ${elapsed}s | Test MSE: ${test_mse} | Test MAE: ${test_mae}"
    echo "${mod},${gpt_l},${mlp_l},${r},${dff},${tw},${fw},${ow},${seed},${test_mse},${test_mae},${elapsed},SUCCESS,${timestamp}" >> "$RESULTS_CSV"

    # --------------------------------------------------------------------------
    # 权重不落盘 (Zero-Disk Footprint): 清理生成的 checkpoint
    # --------------------------------------------------------------------------
    echo ">>> [CLEANUP] Purging .pth checkpoints..."
    rm -rf ./checkpoints/*sens_${mod}* 2>/dev/null || true
    find ./checkpoints -type f -name "*.pth" -mmin -60 -delete 2>/dev/null || true
    echo ">>> [CLEANUP] Purge done."
}

# ==============================================================================
# Module 1: 4x4 Architectural Depth Heatmap (gpt_layers x out_mlp_layers)
# ==============================================================================
if [ "$TARGET_MODULE" = "all" ] || [ "$TARGET_MODULE" = "depth" ]; then
    echo ""
    echo "##############################################################################"
    echo ">>> STARTING MODULE 1: 4x4 Architectural Depth Grid (gpt_layers x mlp_layers)"
    echo "##############################################################################"
    for gpt_l in 1 2 3 4; do
        for mlp_l in 1 2 3 4; do
            for seed in "${SEEDS[@]}"; do
                run_single_experiment "depth" "$gpt_l" "$mlp_l" "$BASE_LORA_R" "$BASE_D_FF" "$BASE_TASK_W" "$BASE_FEAT_W" "$BASE_OUT_W" "$seed"
            done
        done
    done
fi

# ==============================================================================
# Module 2: 4x4 Parameter Capacity Heatmap (LoRA Rank r x FFN Dim d_ff)
# ==============================================================================
if [ "$TARGET_MODULE" = "all" ] || [ "$TARGET_MODULE" = "capacity" ]; then
    echo ""
    echo "##############################################################################"
    echo ">>> STARTING MODULE 2: 4x4 Model Capacity Grid (LoRA Rank r x FFN Dim d_ff)"
    echo "##############################################################################"
    for r in 2 4 8 16; do
        for dff in 768 1536 3072 4096; do
            for seed in "${SEEDS[@]}"; do
                run_single_experiment "capacity" "$BASE_GPT_LAYERS" "$BASE_MLP_LAYERS" "$r" "$dff" "$BASE_TASK_W" "$BASE_FEAT_W" "$BASE_OUT_W" "$seed"
            done
        done
    done
fi

# ==============================================================================
# Module 3: Multi-Task Distillation Simplex Grid (task_w + feat_w + out_w = 1.0)
# ==============================================================================
if [ "$TARGET_MODULE" = "all" ] || [ "$TARGET_MODULE" = "distillation" ]; then
    echo ""
    echo "##############################################################################"
    echo ">>> STARTING MODULE 3: Distillation Loss Balance Grid (Ternary Simplex)"
    echo "##############################################################################"

    # Define 11 Simplex Configurations: [task_w, feature_w, output_w]
    SIMPLEX_CONFIGS=(
        "1.00 0.00 0.00"   # C1: Pure Supervised (No Distillation)
        "0.70 0.00 0.30"   # C2: Output Distillation Only
        "0.70 0.30 0.00"   # C3: Feature Distillation Only
        "0.80 0.05 0.15"   # C4: Light Distillation (Output heavy)
        "0.80 0.10 0.10"   # C5: Light Distillation (Balanced)
        "0.70 0.05 0.25"   # C6: Standard Baseline
        "0.70 0.15 0.15"   # C7: Moderate Distillation (Balanced)
        "0.70 0.25 0.05"   # C8: Moderate Distillation (Feature heavy)
        "0.60 0.10 0.30"   # C9: Heavy Distillation (Output heavy)
        "0.60 0.20 0.20"   # C10: Heavy Distillation (Balanced)
        "0.50 0.25 0.25"   # C11: Ultra-Heavy Distillation (Balanced)
    )

    for cfg in "${SIMPLEX_CONFIGS[@]}"; do
        tw=$(echo "$cfg" | awk '{print $1}')
        fw=$(echo "$cfg" | awk '{print $2}')
        ow=$(echo "$cfg" | awk '{print $3}')
        for seed in "${SEEDS[@]}"; do
            run_single_experiment "distillation" "$BASE_GPT_LAYERS" "$BASE_MLP_LAYERS" "$BASE_LORA_R" "$BASE_D_FF" "$tw" "$fw" "$ow" "$seed"
        done
    done
fi

echo ""
echo "=============================================================================="
echo ">>> [ALL SUITES FINISHED] Sensitivity Analysis Suite Completed Successfully!"
echo ">>> Results saved to: $RESULTS_CSV"
echo "=============================================================================="
