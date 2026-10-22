# ==============================================================================
# Script: scripts/run_fewshot_il_micro.ps1
# Description: Extreme Micro Few-Shot Suite on Solar-IL (0.5%, 1%, 2%, 3%, 4%, 5%, 10%)
# Features:
#   - Zero-Disk Footprint: Immediate purge of .pth checkpoint files after test
#   - Idempotent Resume: Checks CSV before launching; skips completed jobs
#   - Automated Aggregation: Automatically computes Mean/Std and plots curve at the end
# ==============================================================================

param(
    [int]$GPU = 0,
    [string]$CheckpointPath = "./checkpoints/long_term_forecast_CALF_96_96_split622_CALF_Solar_ftM_sl96_ll0_pl96_dm768_nh4_el1_dl1_df3072_fc1_ebtimeF_dtTrue_notrend_split622_gpt1_rl0.0004_2025_0_feature0.05_output0.214_train_epochs100_bs16_dr0.1_ld0.1_r8/checkpoint.pth"
)

if (Test-Path "D:\software\conda\envs\calf\python.exe") {
    $PYTHON = "D:\software\conda\envs\calf\python.exe"
} else {
    $PYTHON = "python"
}

$env:CUDA_VISIBLE_DEVICES = $GPU
$env:HF_ENDPOINT = "https://hf-mirror.com"
$env:PYTHONWARNINGS = "ignore"

$MODEL = "CALF"
$DATA = "Solar"
$DATA_PATH = "solar_IL.csv"
$DATA_SPLIT = "7:1:2"
$ENC_IN = 106
$C_OUT = 106
$SEQ_LEN = 96
$PRED_LEN = 96
$BATCH_SIZE = 16
$ACCUM_STEPS = 1
$CYCLE = 144
$GPT2_PATH = "./models/gpt2"

$LR = & $PYTHON -c "print('{:.8f}'.format(0.00000625 * $BATCH_SIZE * $ACCUM_STEPS))"

$RESULTS_DIR = "./results"
$RESULTS_CSV = "${RESULTS_DIR}/results_fewshot_il_micro.csv"
$LOG_DIR = "./logs/${MODEL}/Solar_IL_Micro_Fewshot"

if (-not (Test-Path $RESULTS_DIR)) { New-Item -ItemType Directory -Path $RESULTS_DIR -Force | Out-Null }
if (-not (Test-Path $LOG_DIR)) { New-Item -ItemType Directory -Path $LOG_DIR -Force | Out-Null }

if (-not (Test-Path $RESULTS_CSV)) {
    "percent,seed,test_mse,test_mae,elapsed_sec,status,timestamp" | Out-File -FilePath $RESULTS_CSV -Encoding utf8
}

Write-Host "==============================================================================" -ForegroundColor Cyan
Write-Host ">>> [CALF] Starting Solar-IL Micro Few-Shot Suite (0.5% ~ 10%)" -ForegroundColor Cyan
Write-Host ">>> GPU: $GPU | Data: $DATA_PATH | Pretrained Checkpoint: $CheckpointPath" -ForegroundColor Cyan
Write-Host ">>> Subsets: 0.5%, 1%, 2%, 3%, 4%, 5%, 10% (3 Seeds Each)" -ForegroundColor Cyan
Write-Host ">>> Logging to: $RESULTS_CSV" -ForegroundColor Cyan
Write-Host "==============================================================================" -ForegroundColor Cyan

$PERCENTS = @(0.5, 1, 2, 3, 4, 5, 10)
$SEEDS = @(2025, 2026, 2027)

$TOTAL_EXPS = $PERCENTS.Count * $SEEDS.Count
$CURR_EXP = 0

foreach ($percent in $PERCENTS) {
    foreach ($seed in $SEEDS) {
        $CURR_EXP++

        # Idempotent resume check
        $already_done = $false
        if (Test-Path $RESULTS_CSV) {
            $check_res = & $PYTHON -c @"
import pandas as pd, sys
try:
    df = pd.read_csv(r'$RESULTS_CSV')
    match = df[(abs(df['percent'] - $percent) < 1e-4) & (df['seed'] == $seed) & (df['status'] == 'SUCCESS') & (df['test_mse'].notna())]
    if len(match) > 0 and str(match.iloc[0]['test_mse']).lower() != 'nan':
        print('DONE')
except:
    pass
"@
            if ($check_res -match "DONE") { $already_done = $true }
        }

        if ($already_done) {
            Write-Host "[$CURR_EXP/$TOTAL_EXPS] Percent: ${percent}% | Seed: ${seed} -> ALREADY FINISHED (Skipping)" -ForegroundColor Yellow
            continue
        }

        Write-Host ""
        Write-Host "------------------------------------------------------------------------------" -ForegroundColor Green
        Write-Host "[$CURR_EXP/$TOTAL_EXPS] Starting Percent: ${percent}% | Seed: ${seed}" -ForegroundColor Green
        Write-Host "------------------------------------------------------------------------------" -ForegroundColor Green

        $LOG_FILE = "${LOG_DIR}/il_micro_p${percent}_seed${seed}.log"
        $START_TIME = Get-Date

        & $PYTHON -u run.py `
            --root_path ./datasets/Solar/ `
            --data_path $DATA_PATH `
            --is_training 1 `
            --task_name long_term_forecast `
            --model_id "${MODEL}_IL_${SEQ_LEN}_${PRED_LEN}_p${percent}" `
            --data $DATA `
            --data_split $DATA_SPLIT `
            --percent $percent `
            --checkpoint_path $CheckpointPath `
            --seq_len $SEQ_LEN `
            --label_len 0 `
            --pred_len $PRED_LEN `
            --batch_size $BATCH_SIZE `
            --learning_rate $LR `
            --train_epochs 100 `
            --patience 5 `
            --d_model 768 `
            --n_heads 4 `
            --e_layers 1 `
            --d_layers 1 `
            --d_ff 3072 `
            --dropout 0.1 `
            --enc_in $ENC_IN `
            --c_out $C_OUT `
            --dec_in $ENC_IN `
            --gpt_layers 1 `
            --itr 1 `
            --model $MODEL `
            --cos 1 `
            --tmax 10 `
            --r 8 `
            --lora_alpha 32 `
            --lora_dropout 0.1 `
            --num_workers 0 `
            --task_w 0.736 `
            --feature_w 0.05 `
            --output_w 0.214 `
            --feature_loss smooth_l1 `
            --output_loss smooth_l1 `
            --task_loss smooth_l1 `
            --use_swa 0 `
            --tq_mode add `
            --tq_inject_target q `
            --use_tq 1 `
            --use_tq_gate 0 `
            --cycle_prior_before_denorm 0 `
            --tq_dropout 0.1 `
            --tq_lr_factor 5 `
            --cycle $CYCLE `
            --test_branch time `
            --out_mlp_layers 3 `
            --mlp_res_w 0 `
            --use_norm 1 `
            --use_direct_trend 0 `
            --bestmodel `
            --gpt2_path $GPT2_PATH `
            --random_seed $seed 2>&1 | Tee-Object -FilePath $LOG_FILE

        $EXIT_CODE = $LASTEXITCODE
        $END_TIME = Get-Date
        $ELAPSED = [math]::Round(($END_TIME - $START_TIME).TotalSeconds)

        if ($EXIT_CODE -ne 0) {
            Write-Host ">>> [ERROR] Run failed for percent=${percent}, seed=${seed}" -ForegroundColor Red
            "${percent},${seed},NaN,NaN,${ELAPSED},FAILED,$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" | Out-File -FilePath $RESULTS_CSV -Append -Encoding utf8
            continue
        }

        # Parse MSE and MAE
        $METRICS = & $PYTHON -c @"
import re
raw = open(r'${LOG_FILE}', 'rb').read()
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
"@

        $TEST_MSE = ($METRICS -split ",")[0]
        $TEST_MAE = ($METRICS -split ",")[1]
        $TIMESTAMP = Get-Date -Format 'yyyy-MM-dd HH:mm:ss'

        Write-Host ">>> [SUCCESS] Finished in ${ELAPSED}s | Test MSE: ${TEST_MSE} | Test MAE: ${TEST_MAE}" -ForegroundColor Cyan
        "${percent},${seed},${TEST_MSE},${TEST_MAE},${ELAPSED},SUCCESS,${TIMESTAMP}" | Out-File -FilePath $RESULTS_CSV -Append -Encoding utf8

        # 权重不落盘 (Zero-Disk Footprint): 清空生成的 checkpoint 目录与 .pth 文件
        Write-Host ">>> [CLEANUP] Purging checkpoint files from disk..." -ForegroundColor Gray
        Get-ChildItem -Path "./checkpoints" -Directory -ErrorAction SilentlyContinue |
            Where-Object { $_.Name -like "*CALF_IL*p${percent}*" -and $_.Name -like "*_${seed}_0_*" } |
            Remove-Item -Recurse -Force -ErrorAction SilentlyContinue
        Write-Host ">>> [CLEANUP] Purged successfully." -ForegroundColor Gray
    }
}

Write-Host ""
Write-Host "==============================================================================" -ForegroundColor Green
Write-Host ">>> [ALL RUNS FINISHED] Aggregating Results and Generating Plots..." -ForegroundColor Green
Write-Host "==============================================================================" -ForegroundColor Green

# Automatically run aggregation and plotting
& $PYTHON plot_fewshot_il_micro.py `
    --csv_path $RESULTS_CSV `
    --summary_csv "./results/fewshot_il_micro_summary.csv" `
    --output_dir "./figures/fewshot"

Write-Host ""
Write-Host ">>> Done! Summary table: ./results/fewshot_il_micro_summary.csv" -ForegroundColor Green
Write-Host ">>> Figure generated: ./figures/fewshot/fewshot_il_micro_curve.png" -ForegroundColor Green
