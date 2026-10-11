# ==============================================================================
# Script: scripts/run_hyper_sensitivity.ps1
# Description: Windows PowerShell Native Hyperparameter Sensitivity Suite
# Modules:
#   - depth:        gpt_layers (1..4) x out_mlp_layers (1..4) [4x4 Heatmap A]
#   - capacity:     lora_r (2,4,8,16) x d_ff (768,1536,3072,4096) [4x4 Heatmap B]
#   - distillation: Simplex loss weights (task_w + feat_w + out_w = 1.0) [Heatmap C]
# ==============================================================================

param(
    [int]$GPU = 0,
    [string]$TargetModule = "all"
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

# Baseline defaults
$BASE_GPT_LAYERS = 1
$BASE_MLP_LAYERS = 3
$BASE_LORA_R = 8
$BASE_D_FF = 3072
$BASE_TASK_W = 0.70
$BASE_FEAT_W = 0.05
$BASE_OUT_W = 0.25

$SEEDS = @(2025, 2026, 2027)

$RESULTS_DIR = "./results"
$RESULTS_CSV = "${RESULTS_DIR}/results_hyper_sensitivity.csv"
$LOG_DIR = "./logs/${MODEL}/Solar_IL_Sensitivity"

if (-not (Test-Path $RESULTS_DIR)) { New-Item -ItemType Directory -Path $RESULTS_DIR -Force | Out-Null }
if (-not (Test-Path $LOG_DIR)) { New-Item -ItemType Directory -Path $LOG_DIR -Force | Out-Null }

if (-not (Test-Path $RESULTS_CSV)) {
    "module,gpt_layers,out_mlp_layers,lora_r,d_ff,task_w,feature_w,output_w,seed,test_mse,test_mae,elapsed_sec,status,timestamp" | Out-File -FilePath $RESULTS_CSV -Encoding utf8
}

Write-Host "==============================================================================" -ForegroundColor Cyan
Write-Host ">>> [CALF] Starting Windows PowerShell Sensitivity Suite (Zero-Disk Footprint)" -ForegroundColor Cyan
Write-Host ">>> GPU: $GPU | Module: $TargetModule | Dataset: $DATA_PATH" -ForegroundColor Cyan
Write-Host ">>> Results CSV: $RESULTS_CSV" -ForegroundColor Cyan
Write-Host "==============================================================================" -ForegroundColor Cyan

function Run-SingleExperiment {
    param(
        [string]$Module,
        [int]$GptLayers,
        [int]$MlpLayers,
        [int]$LoraR,
        [int]$DFf,
        [double]$TaskW,
        [double]$FeatW,
        [double]$OutW,
        [int]$Seed
    )

    # Check if already completed (Idempotent Resume)
    $already_done = $false
    if (Test-Path $RESULTS_CSV) {
        $check_res = python -c @"
import pandas as pd, sys
try:
    df = pd.read_csv(r'$RESULTS_CSV')
    match = df[
        (df['gpt_layers'] == $GptLayers) & 
        (df['out_mlp_layers'] == $MlpLayers) & 
        (df['lora_r'] == $LoraR) & 
        (df['d_ff'] == $DFf) & 
        (abs(df['task_w'] - $TaskW) < 1e-4) & 
        (abs(df['feature_w'] - $FeatW) < 1e-4) & 
        (abs(df['output_w'] - $OutW) < 1e-4) & 
        (df['seed'] == $Seed) &
        (df['status'] == 'SUCCESS')
    ]
    if len(match) > 0:
        print('DONE')
except:
    pass
"@
        if ($check_res -match "DONE") { $already_done = $true }
    }

    if ($already_done) {
        Write-Host ">>> [SKIP] ${Module} (gpt=${GptLayers}, mlp=${MlpLayers}, r=${LoraR}, dff=${DFf}, w=[${TaskW},${FeatW},${OutW}], seed=${Seed}) ALREADY DONE" -ForegroundColor Yellow
        return
    }

    Write-Host ""
    Write-Host "------------------------------------------------------------------------------" -ForegroundColor Green
    Write-Host ">>> [RUN] Module: ${Module} | GPT:${GptLayers} | MLP:${MlpLayers} | r:${LoraR} | d_ff:${DFf}" -ForegroundColor Green
    Write-Host ">>> Loss Weights: task=${TaskW}, feat=${FeatW}, out=${OutW} | Seed: ${Seed}" -ForegroundColor Green
    Write-Host "------------------------------------------------------------------------------" -ForegroundColor Green

    $lr = python -c "print('{:.8f}'.format(0.00000625 * $BATCH_SIZE * $ACCUM_STEPS))"
    $alpha = $LoraR * 4
    $log_file = "${LOG_DIR}/${Module}_gpt${GptLayers}_mlp${MlpLayers}_r${LoraR}_dff${DFf}_tw${TaskW}_fw${FeatW}_s${Seed}.log"
    $start_time = Get-Date

    python -u run.py `
        --root_path ./datasets/Solar/ `
        --data_path $DATA_PATH `
        --is_training 1 `
        --task_name long_term_forecast `
        --model_id "${MODEL}_IL_sens_${Module}" `
        --data $DATA `
        --data_split $DATA_SPLIT `
        --percent 100 `
        --seq_len $SEQ_LEN `
        --label_len 0 `
        --pred_len $PRED_LEN `
        --batch_size $BATCH_SIZE `
        --learning_rate $lr `
        --train_epochs 100 `
        --patience 5 `
        --d_model 768 `
        --n_heads 4 `
        --d_ff $DFf `
        --dropout 0.1 `
        --enc_in $ENC_IN `
        --c_out $C_OUT `
        --dec_in $ENC_IN `
        --gpt_layers $GptLayers `
        --itr 1 `
        --model $MODEL `
        --cos 1 `
        --tmax 10 `
        --r $LoraR `
        --lora_alpha $alpha `
        --lora_dropout 0.1 `
        --num_workers 0 `
        --task_w $TaskW `
        --feature_w $FeatW `
        --output_w $OutW `
        --use_swa 0 `
        --tq_mode add `
        --tq_inject_target q `
        --use_tq 1 `
        --cycle $CYCLE `
        --test_branch time `
        --out_mlp_layers $MlpLayers `
        --mlp_res_w 0 `
        --use_norm 1 `
        --use_direct_trend 1 `
        --bestmodel `
        --random_seed $Seed `
        2>&1 | Tee-Object -FilePath $log_file

    $exit_code = $LASTEXITCODE
    $end_time = Get-Date
    $elapsed = [math]::Round(($end_time - $start_time).TotalSeconds)

    if ($exit_code -ne 0) {
        Write-Host ">>> [ERROR] Run failed (Exit code: $exit_code)" -ForegroundColor Red
        "${Module},${GptLayers},${MlpLayers},${LoraR},${DFf},${TaskW},${FeatW},${OutW},${Seed},NaN,NaN,${elapsed},FAILED,$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" | Out-File -FilePath $RESULTS_CSV -Append -Encoding utf8
        return
    }

    # Parse metrics
    $metrics = python -c @"
import re
with open(r'${log_file}', 'r', encoding='utf-8', errors='ignore') as f:
    text = f.read()
matches = re.findall(r'mse:([0-9\.]+),\s*mae:([0-9\.]+)', text)
if matches:
    print(f'{matches[-1][0]},{matches[-1][1]}')
else:
    print('NaN,NaN')
"@

    $test_mse = ($metrics -split ",")[0]
    $test_mae = ($metrics -split ",")[1]
    $timestamp = Get-Date -Format 'yyyy-MM-dd HH:mm:ss'

    Write-Host ">>> [SUCCESS] Finished in ${elapsed}s | Test MSE: ${test_mse} | Test MAE: ${test_mae}" -ForegroundColor Cyan
    "${Module},${GptLayers},${MlpLayers},${LoraR},${DFf},${TaskW},${FeatW},${OutW},${Seed},${test_mse},${test_mae},${elapsed},SUCCESS,${timestamp}" | Out-File -FilePath $RESULTS_CSV -Append -Encoding utf8

    # Zero-Disk cleanup
    Write-Host ">>> [CLEANUP] Purging .pth checkpoint files..." -ForegroundColor Gray
    Get-ChildItem -Path "./checkpoints" -Filter "*sens_${Module}*" -Recurse -ErrorAction SilentlyContinue | Remove-Item -Recurse -Force -ErrorAction SilentlyContinue
    Get-ChildItem -Path "./checkpoints" -Filter "*.pth" -Recurse -ErrorAction SilentlyContinue | Where-Object { $_.LastWriteTime -gt (Get-Date).AddMinutes(-60) } | Remove-Item -Force -ErrorAction SilentlyContinue
    Write-Host ">>> [CLEANUP] Purged successfully." -ForegroundColor Gray
}

# ==============================================================================
# Module 1: 4x4 Architectural Depth Heatmap (gpt_layers x out_mlp_layers)
# ==============================================================================
if ($TargetModule -eq "all" -or $TargetModule -eq "depth") {
    Write-Host ""
    Write-Host "##############################################################################" -ForegroundColor Magenta
    Write-Host ">>> STARTING MODULE 1: 4x4 Architectural Depth Grid (gpt_layers x mlp_layers)" -ForegroundColor Magenta
    Write-Host "##############################################################################" -ForegroundColor Magenta
    foreach ($gpt_l in @(1, 2, 3, 4)) {
        foreach ($mlp_l in @(1, 2, 3, 4)) {
            foreach ($seed in $SEEDS) {
                Run-SingleExperiment -Module "depth" -GptLayers $gpt_l -MlpLayers $mlp_l -LoraR $BASE_LORA_R -DFf $BASE_D_FF -TaskW $BASE_TASK_W -FeatW $BASE_FEAT_W -OutW $BASE_OUT_W -Seed $seed
            }
        }
    }
}

# ==============================================================================
# Module 2: 4x4 Parameter Capacity Heatmap (LoRA Rank r x FFN Dim d_ff)
# ==============================================================================
if ($TargetModule -eq "all" -or $TargetModule -eq "capacity") {
    Write-Host ""
    Write-Host "##############################################################################" -ForegroundColor Magenta
    Write-Host ">>> STARTING MODULE 2: 4x4 Model Capacity Grid (LoRA Rank r x FFN Dim d_ff)" -ForegroundColor Magenta
    Write-Host "##############################################################################" -ForegroundColor Magenta
    foreach ($r in @(2, 4, 8, 16)) {
        foreach ($dff in @(768, 1536, 3072, 4096)) {
            foreach ($seed in $SEEDS) {
                Run-SingleExperiment -Module "capacity" -GptLayers $BASE_GPT_LAYERS -MlpLayers $BASE_MLP_LAYERS -LoraR $r -DFf $dff -TaskW $BASE_TASK_W -FeatW $BASE_FEAT_W -OutW $BASE_OUT_W -Seed $seed
            }
        }
    }
}

# ==============================================================================
# Module 3: Multi-Task Distillation Simplex Grid (task_w + feat_w + out_w = 1.0)
# ==============================================================================
if ($TargetModule -eq "all" -or $TargetModule -eq "distillation") {
    Write-Host ""
    Write-Host "##############################################################################" -ForegroundColor Magenta
    Write-Host ">>> STARTING MODULE 3: Distillation Loss Balance Grid (Ternary Simplex)" -ForegroundColor Magenta
    Write-Host "##############################################################################" -ForegroundColor Magenta

    $SIMPLEX_CONFIGS = @(
        @(1.00, 0.00, 0.00),
        @(0.70, 0.00, 0.30),
        @(0.70, 0.30, 0.00),
        @(0.80, 0.05, 0.15),
        @(0.80, 0.10, 0.10),
        @(0.70, 0.05, 0.25),
        @(0.70, 0.15, 0.15),
        @(0.70, 0.25, 0.05),
        @(0.60, 0.10, 0.30),
        @(0.60, 0.20, 0.20),
        @(0.50, 0.25, 0.25)
    )

    foreach ($cfg in $SIMPLEX_CONFIGS) {
        $tw = $cfg[0]
        $fw = $cfg[1]
        $ow = $cfg[2]
        foreach ($seed in $SEEDS) {
            Run-SingleExperiment -Module "distillation" -GptLayers $BASE_GPT_LAYERS -MlpLayers $BASE_MLP_LAYERS -LoraR $BASE_LORA_R -DFf $BASE_D_FF -TaskW $tw -FeatW $fw -OutW $ow -Seed $seed
        }
    }
}

Write-Host ""
Write-Host "==============================================================================" -ForegroundColor Green
Write-Host ">>> [ALL SUITES FINISHED] Sensitivity Analysis Suite Completed Successfully!" -ForegroundColor Green
Write-Host ">>> Results: $RESULTS_CSV" -ForegroundColor Green
Write-Host "==============================================================================" -ForegroundColor Green
