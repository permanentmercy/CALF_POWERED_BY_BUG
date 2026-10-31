# ==============================================================================
# Script: run_one_click.ps1
# Description: CALF 项目唯一一键启动控制脚本 (Windows PowerShell 终极整合版)
# Features:
#   1. 单一入口：全项目仅保留这一个一键启动脚本，覆盖所有研究实验与绘图
#   2. 规范对齐：全量参数严格对齐 scripts/long_term_forecasting/Solar_test.sh
#   3. 零磁盘占用：测试时模型置于内存 (--bestmodel)，实验结束后自动清空权重
#   4. 断点续跑：基于结果 CSV 自动跳过已完成实验，支持 -Force 强制重跑
#   5. 自动出图：训练完成后自动生成 Mean+-Std 汇总表与 300 DPI 出版级图表
# ==============================================================================

param(
    [string]$Task = "",            # 可选: "1"|"fewshot_il", "2"|"sensitivity", "3"|"baseline", "4"|"plot"
    [int]$GPU = 0,                 # 指定 GPU 编号 (默认 0)
    [switch]$Force = $false,       # 强制重跑 (不跳过已完成项)
    [string]$TargetModule = "all", # 敏感性分析子模块: "all", "depth", "capacity", "distillation"
    [string]$CheckpointPath = "./checkpoints/AL_pretrained_best_96_96_seed2025.pth"
)

# ------------------------------------------------------------------------------
# 1. 环境自检与 Python 路径探测
# ------------------------------------------------------------------------------
try { [Console]::OutputEncoding = [System.Text.Encoding]::UTF8 } catch {}
$OutputEncoding = [System.Text.Encoding]::UTF8
$env:CUDA_VISIBLE_DEVICES = $GPU
$env:HF_ENDPOINT = "https://hf-mirror.com"
$env:PYTHONWARNINGS = "ignore"
$env:PYTHONIOENCODING = "utf-8"

$PYTHON_CMD = "python"
if (Test-Path "D:\software\conda\envs\calf\python.exe") {
    $PYTHON_CMD = "D:\software\conda\envs\calf\python.exe"
} elseif (Get-Command "conda" -ErrorAction SilentlyContinue) {
    $CONDA_BASE = & conda info --base 2>$null
    if ($CONDA_BASE -and (Test-Path "${CONDA_BASE}\envs\calf\python.exe")) {
        $PYTHON_CMD = "${CONDA_BASE}\envs\calf\python.exe"
    }
}

# 确保核心基础目录存在
$DIRS = @("./logs/CALF/Solar", "./checkpoints", "./results", "./figures/fewshot", "./figures/sensitivity")
foreach ($d in $DIRS) {
    if (-not (Test-Path $d)) { New-Item -ItemType Directory -Path $d -Force | Out-Null }
}

# ------------------------------------------------------------------------------
# 2. 交互式菜单引导 (若未指定 -Task 参数)
# ------------------------------------------------------------------------------
if ([string]::IsNullOrWhiteSpace($Task)) {
    Clear-Host
    Write-Host "==============================================================================" -ForegroundColor Cyan
    Write-Host ">>> [CALF] 一键启动主控制台 (One-Click Master Console)" -ForegroundColor Cyan
    Write-Host ">>> 使用 Python: $PYTHON_CMD | 指定 GPU: $GPU" -ForegroundColor Gray
    Write-Host "==============================================================================" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "  [1] Solar-IL 极低样本微调实验 (0.5%, 1%, 2%, 3%, 4%, 5%, 10% x 3种子) [默认]" -ForegroundColor Green
    Write-Host "      - 数据集: solar_IL.csv (7:1:2 划分, 关闭 Trend Gate, 权重不落盘)"
    Write-Host "  [2] Solar-AL 超参敏感性实验 (对齐 Solar_test.sh, 4x4深度/4x4容量/11元蒸馏)" -ForegroundColor Yellow
    Write-Host "      - 数据集: solar_AL.txt (7:1:2 划分, lr=0.0004, accum=4, smooth_l1)"
    Write-Host "  [3] Solar-AL 原始基准复现 (Horizon 96, Seeds 2025/2026/2027, 目标 MSE ~ 0.1645)" -ForegroundColor Cyan
    Write-Host "  [4] 仅运行学术可视化绘图 (生成 300 DPI 曲线图与热力图)" -ForegroundColor Magenta
    Write-Host "  [5] 旧 P1 开关对照已停用（保持 use_tq 不变）" -ForegroundColor Gray
    Write-Host "  [0] 退出控制台" -ForegroundColor Gray
    Write-Host ""
    Write-Host "------------------------------------------------------------------------------" -ForegroundColor Cyan
    
    $input_choice = Read-Host "请输入任务编号 [直接回车默认执行 1]"
    if ([string]::IsNullOrWhiteSpace($input_choice)) {
        $Task = "1"
    } else {
        $Task = $input_choice.Trim()
    }
}

# ------------------------------------------------------------------------------
# 3. 任务 1: Solar-IL 极低样本微调实验 (0.5% ~ 10%)
# ------------------------------------------------------------------------------
if ($Task -in @("1", "fewshot_il", "il", "fewshot")) {
    $CSV_FILE = "./results/results_fewshot_il_micro.csv"
    $LOG_DIR = "./logs/CALF/Solar_IL_Micro_Fewshot"
    if (-not (Test-Path $LOG_DIR)) { New-Item -ItemType Directory -Path $LOG_DIR -Force | Out-Null }
    if (-not (Test-Path $CSV_FILE)) {
        "percent,seed,test_mse,test_mae,elapsed_sec,status,timestamp" | Out-File -FilePath $CSV_FILE -Encoding utf8
    }

    Write-Host "`n==============================================================================" -ForegroundColor Green
    Write-Host ">>> [TASK 1] 启动 Solar-IL 极低样本微调流水线 (0.5% ~ 10% x 3种子)" -ForegroundColor Green
    Write-Host ">>> 预训练检查点: $CheckpointPath" -ForegroundColor Cyan
    Write-Host ">>> 规范配置: Split 7:1:2 | 关闭 Trend Gate (use_direct_trend=0) | 权重不落盘" -ForegroundColor Cyan
    Write-Host "==============================================================================" -ForegroundColor Green

    # 检查预训练权重
    if (-not (Test-Path $CheckpointPath)) {
        $alt_ckpt = "./checkpoints/long_term_forecast_CALF_96_96_p100_CALF_Solar_ftM_sl96_ll0_pl96_dm768_nh4_el1_dl1_df3072_fc1_ebtimeF_dtTrue_test_gpt1_rl0.0004_2025_0_feature0.05_output0.214_train_epochs100_bs16_dr0.1_ld0.1_r8/checkpoint.pth"
        if (Test-Path $alt_ckpt) {
            $CheckpointPath = $alt_ckpt
            Write-Host ">>> [INFO] 自动采用基准复现检查点: $CheckpointPath" -ForegroundColor Yellow
        } else {
            Write-Host ">>> [WARN] 未找到预训练检查点 $CheckpointPath，将从头开始训练 (Scratch Few-Shot)" -ForegroundColor Yellow
            $CheckpointPath = ""
        }
    }

    $PERCENTS = @(0.5, 1, 2, 3, 4, 5, 10)
    $SEEDS = @(2025, 2026, 2027)
    $TOTAL = $PERCENTS.Count * $SEEDS.Count
    $CURR = 0

    foreach ($p in $PERCENTS) {
        foreach ($s in $SEEDS) {
            $CURR++

            # 断点续跑检查 (调用 helper 模块)
            $is_done = "NO"
            if (-not $Force) {
                $is_done = (& $PYTHON_CMD scripts/runner_helper.py check_fewshot $CSV_FILE $p $s).Trim()
            }

            if ($is_done -eq "DONE") {
                Write-Host "[$CURR/$TOTAL] Percent: ${p}% | Seed: ${s} -> 已完成 (跳过)" -ForegroundColor Yellow
                continue
            }

            Write-Host ""
            Write-Host "------------------------------------------------------------------------------" -ForegroundColor Green
            Write-Host "[$CURR/$TOTAL] 正在训练: Percent: ${p}% | Seed: ${s} (GPU: $GPU)" -ForegroundColor Green
            Write-Host "------------------------------------------------------------------------------" -ForegroundColor Green

            $log_path = "${LOG_DIR}/il_micro_p${p}_seed${s}.log"
            $t0 = Get-Date

            $run_args = @(
                "-u", "run.py",
                "--root_path", "./datasets/Solar/",
                "--data_path", "solar_IL.csv",
                "--is_training", "1",
                "--task_name", "long_term_forecast",
                "--model_id", "CALF_IL_96_96_p${p}",
                "--data", "Solar",
                "--data_split", "7:1:2",
                "--percent", "$p",
                "--seq_len", "96",
                "--label_len", "0",
                "--pred_len", "96",
                "--batch_size", "16",
                "--learning_rate", "0.0004",
                "--accumulation_steps", "4",
                "--train_epochs", "100",
                "--patience", "5",
                "--d_model", "768",
                "--n_heads", "4",
                "--e_layers", "1",
                "--d_layers", "1",
                "--d_ff", "3072",
                "--dropout", "0.1",
                "--enc_in", "106",
                "--c_out", "106",
                "--dec_in", "106",
                "--gpt_layers", "1",
                "--itr", "1",
                "--model", "CALF",
                "--cos", "1",
                "--tmax", "10",
                "--r", "8",
                "--lora_alpha", "32",
                "--lora_dropout", "0.1",
                "--num_workers", "0",
                "--task_w", "0.736",
                "--feature_w", "0.05",
                "--output_w", "0.214",
                "--task_loss", "smooth_l1",
                "--feature_loss", "smooth_l1",
                "--output_loss", "smooth_l1",
                "--use_swa", "0",
                "--tq_mode", "add",
                "--tq_inject_target", "q",
                "--use_tq", "1",
                "--use_tq_gate", "0",
                "--cycle_prior_before_denorm", "0",
                "--tq_dropout", "0.1",
                "--tq_lr_factor", "5",
                "--cycle", "144",
                "--test_branch", "time",
                "--out_mlp_layers", "3",
                "--mlp_res_w", "0",
                "--use_norm", "1",
                "--use_direct_trend", "0",
                "--eval_test_every_epoch",
                "--bestmodel",
                "--gpt2_path", "./models/gpt2",
                "--random_seed", "$s"
            )

            if (-not [string]::IsNullOrWhiteSpace($CheckpointPath)) {
                $run_args += @("--checkpoint_path", $CheckpointPath)
            }

            & $PYTHON_CMD $run_args 2>&1 | Tee-Object -FilePath $log_path
            $exit_code = $LASTEXITCODE
            $t1 = Get-Date
            $elapsed = [math]::Round(($t1 - $t0).TotalSeconds)

            if ($exit_code -ne 0) {
                Write-Host ">>> [ERROR] 运行失败: p=${p}%, seed=${s} (Exit: $exit_code)" -ForegroundColor Red
                "${p},${s},NaN,NaN,${elapsed},FAILED,$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" | Out-File -FilePath $CSV_FILE -Append -Encoding utf8
                continue
            }

            $metrics = (& $PYTHON_CMD scripts/runner_helper.py parse_log $log_path).Trim()
            $mse = ($metrics -split ",")[0]
            $mae = ($metrics -split ",")[1]
            $timestamp = Get-Date -Format 'yyyy-MM-dd HH:mm:ss'

            Write-Host ">>> [SUCCESS] 耗时: ${elapsed}s | 测试 MSE: $mse | 测试 MAE: $mae" -ForegroundColor Cyan
            "${p},${s},${mse},${mae},${elapsed},SUCCESS,${timestamp}" | Out-File -FilePath $CSV_FILE -Append -Encoding utf8

            # 权重不落盘清理
            Get-ChildItem -Path "./checkpoints" -Filter "*CALF_IL_96_96_p${p}*" -Recurse -ErrorAction SilentlyContinue | Remove-Item -Recurse -Force -ErrorAction SilentlyContinue
            Get-ChildItem -Path "./checkpoints" -Filter "*.pth" -Recurse -ErrorAction SilentlyContinue | Where-Object { $_.LastWriteTime -gt (Get-Date).AddMinutes(-60) -and $_.Name -notmatch "pretrained|AL_best" } | Remove-Item -Force -ErrorAction SilentlyContinue
        }
    }

    Write-Host "`n>>> [FINISH] Solar-IL 少样本微调全部结束！正在生成汇总表与衰减曲线..." -ForegroundColor Green
    & $PYTHON_CMD plot_fewshot_il_micro.py --csv_path $CSV_FILE --output_dir "./figures/fewshot" --summary_csv "./results/fewshot_il_micro_summary.csv"
    Write-Host ">>> [COMPLETE] 结果汇总表: ./results/fewshot_il_micro_summary.csv" -ForegroundColor Yellow
    Write-Host ">>> [COMPLETE] 学术曲线图: ./figures/fewshot/fewshot_il_micro_curve.png" -ForegroundColor Yellow
}

# ------------------------------------------------------------------------------
# 4. 任务 2: Solar-AL 超参敏感性分析流水线
# ------------------------------------------------------------------------------
elseif ($Task -in @("2", "sensitivity", "sens", "hyper")) {
    $CSV_FILE = "./results/results_hyper_sensitivity_al.csv"
    $LOG_DIR = "./logs/CALF/Solar_AL_Sensitivity"
    if (-not (Test-Path $LOG_DIR)) { New-Item -ItemType Directory -Path $LOG_DIR -Force | Out-Null }
    if (-not (Test-Path $CSV_FILE)) {
        "module,gpt_layers,out_mlp_layers,lora_r,d_ff,task_w,feature_w,output_w,seed,test_mse,test_mae,elapsed_sec,status,timestamp" | Out-File -FilePath $CSV_FILE -Encoding utf8
    }

    Write-Host "`n==============================================================================" -ForegroundColor Green
    Write-Host ">>> [TASK 2] 启动 Solar-AL 超参敏感性分析流水线 (完全对齐 Solar_test.sh)" -ForegroundColor Green
    Write-Host ">>> 目标模块: $TargetModule | 结果文件: $CSV_FILE" -ForegroundColor Cyan
    Write-Host "==============================================================================" -ForegroundColor Green

    function Run-SingleSensExperiment {
        param($Mod, $GptL, $MlpL, $R, $Dff, $Tw, $Fw, $Ow, $Seed)

        $is_done = "NO"
        if (-not $Force) {
            $is_done = (& $PYTHON_CMD scripts/runner_helper.py check_sens $CSV_FILE $GptL $MlpL $R $Dff $Tw $Fw $Ow $Seed).Trim()
        }

        if ($is_done -eq "DONE") {
            Write-Host ">>> [SKIP] ${Mod} (gpt=${GptL}, mlp=${MlpL}, r=${R}, dff=${Dff}, w=[${Tw},${Fw},${Ow}], s=${Seed}) 已完成" -ForegroundColor Yellow
            return
        }

        Write-Host ""
        Write-Host "------------------------------------------------------------------------------" -ForegroundColor Green
        Write-Host ">>> [RUN] ${Mod} | GPT:${GptL} | MLP:${MlpL} | r:${R} | d_ff:${Dff} | w=[${Tw},${Fw},${Ow}] | Seed:${Seed}" -ForegroundColor Green
        Write-Host "------------------------------------------------------------------------------" -ForegroundColor Green

        $alpha = $R * 4
        $log_path = "${LOG_DIR}/${Mod}_gpt${GptL}_mlp${MlpL}_r${R}_dff${Dff}_tw${Tw}_s${Seed}.log"
        $t0 = Get-Date

        $sens_args = @(
            "-u", "run.py",
            "--root_path", "./datasets/Solar/",
            "--data_path", "solar_AL.txt",
            "--is_training", "1",
            "--task_name", "long_term_forecast",
            "--model_id", "CALF_AL_sens_${Mod}",
            "--data", "Solar",
            "--data_split", "7:1:2",
            "--percent", "100",
            "--seq_len", "96",
            "--label_len", "0",
            "--pred_len", "96",
            "--batch_size", "16",
            "--learning_rate", "0.0004",
            "--accumulation_steps", "4",
            "--train_epochs", "100",
            "--patience", "5",
            "--d_model", "768",
            "--n_heads", "4",
            "--e_layers", "1",
            "--d_layers", "1",
            "--d_ff", "$Dff",
            "--dropout", "0.1",
            "--enc_in", "137",
            "--c_out", "137",
            "--dec_in", "137",
            "--gpt_layers", "$GptL",
            "--itr", "1",
            "--model", "CALF",
            "--cos", "1",
            "--tmax", "10",
            "--r", "$R",
            "--lora_alpha", "$alpha",
            "--lora_dropout", "0.1",
            "--num_workers", "0",
            "--task_w", "$Tw",
            "--feature_w", "$Fw",
            "--output_w", "$Ow",
            "--task_loss", "smooth_l1",
            "--feature_loss", "smooth_l1",
            "--output_loss", "smooth_l1",
            "--use_swa", "0",
            "--tq_mode", "add",
            "--tq_inject_target", "q",
            "--use_tq", "1",
            "--use_tq_gate", "0",
            "--cycle_prior_before_denorm", "0",
            "--tq_dropout", "0.1",
            "--tq_lr_factor", "5",
            "--cycle", "144",
            "--test_branch", "time",
            "--out_mlp_layers", "$MlpL",
            "--mlp_res_w", "0",
            "--use_norm", "1",
            "--use_direct_trend", "1",
            "--eval_test_every_epoch",
            "--bestmodel",
            "--gpt2_path", "./models/gpt2",
            "--random_seed", "$Seed"
        )

        & $PYTHON_CMD $sens_args 2>&1 | Tee-Object -FilePath $log_path
        $exit_code = $LASTEXITCODE
        $t1 = Get-Date
        $elapsed = [math]::Round(($t1 - $t0).TotalSeconds)

        if ($exit_code -ne 0) {
            Write-Host ">>> [ERROR] 运行失败 (Exit: $exit_code)" -ForegroundColor Red
            "${Mod},${GptL},${MlpL},${R},${Dff},${Tw},${Fw},${Ow},${Seed},NaN,NaN,${elapsed},FAILED,$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" | Out-File -FilePath $CSV_FILE -Append -Encoding utf8
            return
        }

        $metrics = (& $PYTHON_CMD scripts/runner_helper.py parse_log $log_path).Trim()
        $mse = ($metrics -split ",")[0]
        $mae = ($metrics -split ",")[1]
        $timestamp = Get-Date -Format 'yyyy-MM-dd HH:mm:ss'

        Write-Host ">>> [SUCCESS] 耗时: ${elapsed}s | 测试 MSE: $mse | 测试 MAE: $mae" -ForegroundColor Cyan
        "${Mod},${GptL},${MlpL},${R},${Dff},${Tw},${Fw},${Ow},${Seed},${mse},${mae},${elapsed},SUCCESS,${timestamp}" | Out-File -FilePath $CSV_FILE -Append -Encoding utf8

        # 清除检查点
        Get-ChildItem -Path "./checkpoints" -Filter "*sens_${Mod}*" -Recurse -ErrorAction SilentlyContinue | Remove-Item -Recurse -Force -ErrorAction SilentlyContinue
    }

    $SEEDS = @(2025, 2026, 2027)

    # 模块 1: 架构深度 (Depth 4x4)
    if ($TargetModule -in @("all", "depth")) {
        foreach ($g in 1..4) {
            foreach ($m in 1..4) {
                foreach ($s in $SEEDS) {
                    Run-SingleSensExperiment -Mod "depth" -GptL $g -MlpL $m -R 8 -Dff 3072 -Tw 0.736 -Fw 0.05 -Ow 0.214 -Seed $s
                }
            }
        }
    }

    # 模块 2: 参数容量 (Capacity 4x4)
    if ($TargetModule -in @("all", "capacity")) {
        foreach ($r in @(2, 4, 8, 16)) {
            foreach ($d in @(768, 1536, 3072, 4096)) {
                foreach ($s in $SEEDS) {
                    Run-SingleSensExperiment -Mod "capacity" -GptL 1 -MlpL 3 -R $r -Dff $d -Tw 0.736 -Fw 0.05 -Ow 0.214 -Seed $s
                }
            }
        }
    }

    # 模块 3: 多任务蒸馏损失平衡 (Distillation 11元单纯形)
    if ($TargetModule -in @("all", "distillation")) {
        $SIMPLEX = @(
            @(0.736, 0.05, 0.214), @(1.0, 0.0, 0.0), @(0.8, 0.1, 0.1), @(0.8, 0.05, 0.15),
            @(0.7, 0.3, 0.0), @(0.7, 0.25, 0.05), @(0.7, 0.15, 0.15), @(0.7, 0.0, 0.3),
            @(0.6, 0.2, 0.2), @(0.6, 0.1, 0.3), @(0.5, 0.25, 0.25)
        )
        foreach ($cfg in $SIMPLEX) {
            foreach ($s in $SEEDS) {
                Run-SingleSensExperiment -Mod "distillation" -GptL 1 -MlpL 3 -R 8 -Dff 3072 -Tw $cfg[0] -Fw $cfg[1] -Ow $cfg[2] -Seed $s
            }
        }
    }

    Write-Host "`n>>> [FINISH] 超参敏感性实验完成！调用绘图脚本渲染高分辨率图表..." -ForegroundColor Green
    & $PYTHON_CMD plot_sensitivity_results.py --fewshot_csv "./results/results_fewshot_al.csv" --sensitivity_csv $CSV_FILE --output_dir "./figures/sensitivity"
}

# ------------------------------------------------------------------------------
# 5. 任务 3: Solar-AL 原始基准复现实验 (Horizon 96, Seeds 2025/2026/2027)
# ------------------------------------------------------------------------------
elseif ($Task -in @("3", "baseline", "reproduce")) {
    $SEEDS = @(2025, 2026, 2027)
    $RESULTS_CSV = "./results/baseline_reproduction_results.csv"
    if (-not (Test-Path "./results")) { New-Item -ItemType Directory -Path "./results" -Force | Out-Null }
    if (-not (Test-Path $RESULTS_CSV)) {
        "seed,pred_len,mse,mae,timestamp" | Out-File -FilePath $RESULTS_CSV -Encoding utf8
    }

    Write-Host "`n==============================================================================" -ForegroundColor Green
    Write-Host ">>> [TASK 3] 启动 Solar-AL 原始基准复现 (Horizon 96, Seeds 2025/2026/2027)" -ForegroundColor Green
    Write-Host ">>> 目标指标: 2025(0.164986/0.219895), 2026(0.164929/0.219133), 2027(0.164136/0.218951)" -ForegroundColor Cyan
    Write-Host ">>> 严格对齐 scripts/long_term_forecasting/Solar_test.sh" -ForegroundColor Cyan
    Write-Host "==============================================================================" -ForegroundColor Green

    foreach ($s in $SEEDS) {
        $LOG_FILE = "./logs/CALF/Solar/reproduce_baseline_96_seed${s}.log"
        if (-not $Force -and (Test-Path $RESULTS_CSV)) {
            $existing = Import-Csv $RESULTS_CSV -ErrorAction SilentlyContinue | Where-Object { $_.seed -eq "$s" -and $_.mse -ne "NaN" -and $_.mse -match "^[0-9\.]+$" }
            if ($existing) {
                Write-Host ">>> [SKIP Seed $s] 已完成，历史复现指标: MSE = $($existing.mse), MAE = $($existing.mae)" -ForegroundColor Gray
                continue
            }
        }
        Write-Host "`n>>> [RUN SEED $s] 开始训练与测试... 日志流: $LOG_FILE" -ForegroundColor Green

        $base_args = @(
            "-u", "run.py",
            "--root_path", "./datasets/Solar/",
            "--data_path", "solar_AL.txt",
            "--data_split", "7:1:2",
            "--is_training", "1",
            "--task_name", "long_term_forecast",
            "--model_id", "CALF_96_96_p100",
            "--data", "Solar",
            "--seq_len", "96",
            "--label_len", "0",
            "--pred_len", "96",
            "--percent", "100",
            "--batch_size", "16",
            "--learning_rate", "0.0004",
            "--accumulation_steps", "4",
            "--train_epochs", "100",
            "--patience", "5",
            "--d_model", "768",
            "--n_heads", "4",
            "--e_layers", "1",
            "--d_layers", "1",
            "--d_ff", "3072",
            "--dropout", "0.1",
            "--enc_in", "137",
            "--c_out", "137",
            "--dec_in", "137",
            "--gpt_layers", "1",
            "--itr", "1",
            "--model", "CALF",
            "--cos", "1",
            "--tmax", "10",
            "--r", "8",
            "--lora_alpha", "32",
            "--lora_dropout", "0.1",
            "--num_workers", "0",
            "--bestmodel",
            "--feature_w", "0.05",
            "--output_w", "0.214",
            "--task_w", "0.736",
            "--task_loss", "smooth_l1",
            "--feature_loss", "smooth_l1",
            "--output_loss", "smooth_l1",
            "--use_swa", "0",
            "--tq_mode", "add",
            "--use_tq_gate", "0",
            "--cycle_prior_before_denorm", "0",
            "--tq_dropout", "0.1",
            "--tq_lr_factor", "5",
            "--cycle", "144",
            "--test_branch", "time",
            "--out_mlp_layers", "3",
            "--mlp_res_w", "0",
            "--eval_test_every_epoch",
            "--use_norm", "1",
            "--use_direct_trend", "0",
            "--tq_inject_target", "q",
            "--gpt2_path", "./models/gpt2",
            "--random_seed", "$s"
        )

        try {
            & $PYTHON_CMD @base_args *>&1 | Tee-Object -FilePath $LOG_FILE
        } catch {
            Write-Host ">>> [EXCEPTION Seed $s] $_" -ForegroundColor Red
        }
        $metrics = (& $PYTHON_CMD scripts/runner_helper.py parse_log $LOG_FILE).Trim()
        $mse = ($metrics -split ",")[0]
        $mae = ($metrics -split ",")[1]
        Write-Host ">>> [COMPLETE Seed $s] 测试结果: MSE = $mse, MAE = $mae" -ForegroundColor Yellow
        if ($mse -ne "NaN" -and $mse -match "^[0-9\.]+$") {
            "${s},96,${mse},${mae},$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" | Out-File -FilePath $RESULTS_CSV -Append -Encoding utf8
        } else {
            Write-Host ">>> [WARN Seed $s] 未能获得有效指标，请查看日志 $LOG_FILE" -ForegroundColor Red
        }
    }

    Write-Host "`n==============================================================================" -ForegroundColor Green
    Write-Host ">>> [TASK 3 FINISH] 全部 3 种子复现完成！汇总表保存在: $RESULTS_CSV" -ForegroundColor Green
    Write-Host "==============================================================================" -ForegroundColor Green
}

# ------------------------------------------------------------------------------
# 6. 任务 4: 仅重新生成可视化学术图表
# ------------------------------------------------------------------------------
elseif ($Task -in @("4", "plot", "figure", "figures")) {
    Write-Host "`n==============================================================================" -ForegroundColor Magenta
    Write-Host ">>> [TASK 4] 仅生成学术可视化图表 (300 DPI 出版级)" -ForegroundColor Magenta
    Write-Host "==============================================================================" -ForegroundColor Magenta

    if (Test-Path "./results/results_fewshot_il_micro.csv") {
        Write-Host ">>> 渲染 Solar-IL 极低样本微调衰减曲线..." -ForegroundColor Cyan
        & $PYTHON_CMD plot_fewshot_il_micro.py --csv_path "./results/results_fewshot_il_micro.csv" --output_dir "./figures/fewshot" --summary_csv "./results/fewshot_il_micro_summary.csv"
    }

    if (Test-Path "./results/results_hyper_sensitivity_al.csv") {
        Write-Host ">>> 渲染 Solar-AL 超参敏感性全套热力图与单纯形图..." -ForegroundColor Cyan
        & $PYTHON_CMD plot_sensitivity_results.py --fewshot_csv "./results/results_fewshot_al.csv" --sensitivity_csv "./results/results_hyper_sensitivity_al.csv" --output_dir "./figures/sensitivity"
    }

    Write-Host "`n>>> [ALL DONE] 全部图表已更新至 ./figures/ !" -ForegroundColor Green
}
# ------------------------------------------------------------------------------
# 7. 旧 P1 开关对照已撤下，不再启动训练
# ------------------------------------------------------------------------------
elseif ($Task -in @("5", "tq", "cycle", "p1")) {
    throw "旧 P1 开关对照已停用：按用户要求保持 use_tq 不变，请查看 public/cycle_mechanism/v1.0/completion_plan.md。"
}
elseif ($Task -in @("0", "exit", "quit")) {
    Write-Host ">>> 已退出。" -ForegroundColor Gray
}
else {
    Write-Host ">>> [ERROR] 未知任务编号或指令: $Task" -ForegroundColor Red
    Write-Host ">>> 可用参数: 1 (fewshot_il), 2 (sensitivity), 3 (baseline), 4 (plot), 5 (tq/p1)" -ForegroundColor Yellow
}
