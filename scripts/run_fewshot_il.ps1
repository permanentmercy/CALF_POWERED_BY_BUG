# ==============================================================================
# Script: scripts/run_fewshot_il.ps1
# Description: Windows PowerShell Native Few-Shot Suite for Solar-IL
# ==============================================================================

param(
    [int]$GPU = 0
)

$env:CUDA_VISIBLE_DEVICES = $GPU

$MODEL = "CALF"
$DATA = "Solar"
$DATA_PATH = "solar_IL.csv"
$DATA_SPLIT = "6:2:2"
$ENC_IN = 106
$C_OUT = 106
$SEQ_LEN = 96
$PRED_LEN = 96
$BATCH_SIZE = 16
$ACCUM_STEPS = 1
$CYCLE = 144

# Scale learning rate
$LR = python -c "print('{:.8f}'.format(0.00000625 * $BATCH_SIZE * $ACCUM_STEPS))"

$RESULTS_DIR = "./results"
$RESULTS_CSV = "${RESULTS_DIR}/results_fewshot_il.csv"
$LOG_DIR = "./logs/${MODEL}/Solar_IL_Fewshot"

if (-not (Test-Path $RESULTS_DIR)) { New-Item -ItemType Directory -Path $RESULTS_DIR -Force | Out-Null }
if (-not (Test-Path $LOG_DIR)) { New-Item -ItemType Directory -Path $LOG_DIR -Force | Out-Null }

if (-not (Test-Path $RESULTS_CSV)) {
    "percent,seed,test_mse,test_mae,train_epochs,status,timestamp" | Out-File -FilePath $RESULTS_CSV -Encoding utf8
}

Write-Host "==============================================================================" -ForegroundColor Cyan
Write-Host ">>> [CALF] Starting Windows PowerShell IL Few-Shot Suite (Zero-Disk Footprint)" -ForegroundColor Cyan
Write-Host ">>> GPU: $GPU | Data: $DATA_PATH | EncIn: $ENC_IN | Split: $DATA_SPLIT" -ForegroundColor Cyan
Write-Host ">>> Results CSV: $RESULTS_CSV" -ForegroundColor Cyan
Write-Host "==============================================================================" -ForegroundColor Cyan

$PERCENTS = @(10, 20, 30, 40, 50)
$SEEDS = @(2025, 2026, 2027)

$TOTAL_EXPS = $PERCENTS.Count * $SEEDS.Count
$CURR_EXP = 0

foreach ($percent in $PERCENTS) {
    foreach ($seed in $SEEDS) {
        $CURR_EXP++
        
        # Idempotent resume check
        $already_done = $false
        if (Test-Path $RESULTS_CSV) {
            $match = Select-String -Path $RESULTS_CSV -Pattern "^${percent},${seed}," -Quiet
            if ($match) { $already_done = $true }
        }

        if ($already_done) {
            Write-Host "[$CURR_EXP/$TOTAL_EXPS] Percent: ${percent}% | Seed: ${seed} -> ALREADY FINISHED (Skipping)" -ForegroundColor Yellow
            continue
        }

        Write-Host ""
        Write-Host "------------------------------------------------------------------------------" -ForegroundColor Green
        Write-Host "[$CURR_EXP/$TOTAL_EXPS] Starting Percent: ${percent}% | Random Seed: ${seed}" -ForegroundColor Green
        Write-Host "------------------------------------------------------------------------------" -ForegroundColor Green

        $LOG_FILE = "${LOG_DIR}/il_fewshot_p${percent}_seed${seed}.log"
        $START_TIME = Get-Date

        # Run scratch few-shot training
        python -u run.py `
            --root_path ./datasets/Solar/ `
            --data_path $DATA_PATH `
            --is_training 1 `
            --task_name long_term_forecast `
            --model_id "${MODEL}_IL_${SEQ_LEN}_${PRED_LEN}_p${percent}" `
            --data $DATA `
            --data_split $DATA_SPLIT `
            --percent $percent `
            --seq_len $SEQ_LEN `
            --label_len 0 `
            --pred_len $PRED_LEN `
            --batch_size $BATCH_SIZE `
            --learning_rate $LR `
            --train_epochs 100 `
            --patience 5 `
            --d_model 768 `
            --n_heads 4 `
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
            --task_w 0.70 `
            --feature_w 0.05 `
            --output_w 0.25 `
            --use_swa 0 `
            --tq_mode add `
            --tq_inject_target q `
            --use_tq 1 `
            --cycle $CYCLE `
            --test_branch time `
            --out_mlp_layers 3 `
            --mlp_res_w 0 `
            --use_norm 1 `
            --use_direct_trend 1 `
            --bestmodel `
            --random_seed $seed `
            2>&1 | Tee-Object -FilePath $LOG_FILE

        $EXIT_CODE = $LASTEXITCODE
        $END_TIME = Get-Date
        $ELAPSED = [math]::Round(($END_TIME - $START_TIME).TotalSeconds)

        if ($EXIT_CODE -ne 0) {
            Write-Host ">>> [ERROR] Run failed for percent=${percent}, seed=${seed}" -ForegroundColor Red
            "${percent},${seed},NaN,NaN,${ELAPSED},FAILED,$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" | Out-File -FilePath $RESULTS_CSV -Append -Encoding utf8
            continue
        }

        # Parse MSE and MAE
        $METRICS = python -c @"
import re
with open(r'${LOG_FILE}', 'r', encoding='utf-8', errors='ignore') as f:
    text = f.read()
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

        # 权重不落盘 (Zero-Disk Footprint): 清空生成的 checkpoint
        Write-Host ">>> [CLEANUP] Deleting checkpoint weights from disk..." -ForegroundColor Gray
        Get-ChildItem -Path "./checkpoints" -Filter "*p${percent}*seed${seed}*" -Recurse -ErrorAction SilentlyContinue | Remove-Item -Recurse -Force -ErrorAction SilentlyContinue
        Get-ChildItem -Path "./checkpoints" -Filter "*.pth" -Recurse -ErrorAction SilentlyContinue | Where-Object { $_.LastWriteTime -gt (Get-Date).AddMinutes(-60) } | Remove-Item -Force -ErrorAction SilentlyContinue
        Write-Host ">>> [CLEANUP] Checkpoints purged." -ForegroundColor Gray
    }
}

Write-Host ""
Write-Host "==============================================================================" -ForegroundColor Green
Write-Host ">>> [ALL COMPLETED] Windows IL Few-Shot Suite Finished!" -ForegroundColor Green
Write-Host ">>> Results: $RESULTS_CSV" -ForegroundColor Green
Write-Host "==============================================================================" -ForegroundColor Green
