#!/bin/bash
# ==============================================================================
# Script: scripts/run_4days_suite.sh
# Description: Master 4-Day Non-Stop Benchmark Pipeline on Solar-IL
# Pipeline:
#   Phase 1: Few-Shot Generalization (10% to 50%, step 10%, 3 seeds) (~3.5 hrs)
#   Phase 2: Hyperparameter Sensitivity Suite (Depth, Capacity, Distillation) (~84 hrs)
#   Phase 3: Automated Publication-Grade Figure Generation (Heatmaps + Curves)
#
# Usage:
#   bash scripts/run_4days_suite.sh <GPU_ID>
#   e.g. nohup bash scripts/run_4days_suite.sh 0 > 4days_master.log 2>&1 &
# ==============================================================================

set -e

GPU=${1:-0}
START_TIME_TOTAL=$(date +%s)

echo "=============================================================================="
echo ">>> [MASTER RUNNER] Launching 4-Day Benchmark & Sensitivity Suite on Solar-IL"
echo ">>> Start Time: $(date '+%Y-%m-%d %H:%M:%S') | Designated GPU: $GPU"
echo "=============================================================================="

# 1. Ensure scripts are executable
chmod +x scripts/run_fewshot_il.sh 2>/dev/null || true
chmod +x scripts/run_hyper_sensitivity.sh 2>/dev/null || true

# 2. Phase 1: Few-Shot Suite
echo ""
echo ">>> [PHASE 1/3] Running Few-Shot Generalization Suite (10% ~ 50%)..."
bash scripts/run_fewshot_il.sh "$GPU"

# 3. Phase 2: Hyperparameter Sensitivity Suite
echo ""
echo ">>> [PHASE 2/3] Running Full Hyperparameter Sensitivity Suite..."
bash scripts/run_hyper_sensitivity.sh "$GPU" all

# 4. Phase 3: Automated Visualization
echo ""
echo ">>> [PHASE 3/3] Generating Publication Figures..."
python plot_sensitivity_results.py \
    --fewshot_csv ./results/results_fewshot_il.csv \
    --sensitivity_csv ./results/results_hyper_sensitivity.csv \
    --output_dir ./figures/sensitivity

END_TIME_TOTAL=$(date +%s)
TOTAL_ELAPSED=$((END_TIME_TOTAL - START_TIME_TOTAL))
HOURS=$((TOTAL_ELAPSED / 3600))
MINUTES=$(((TOTAL_ELAPSED % 3600) / 60))

echo ""
echo "=============================================================================="
echo ">>> [ALL PHASES COMPLETED SUCCESSFULLY]"
echo ">>> Total Duration: ${HOURS} hours ${MINUTES} minutes"
echo ">>> All Checkpoints Purged | Results & Figures Saved to ./figures/sensitivity"
echo "=============================================================================="
