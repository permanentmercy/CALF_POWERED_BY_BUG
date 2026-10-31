# UTF-8 Encoding
throw "P1 配对训练方案已撤下：按用户要求保持 use_tq 不变。已有结果仅作历史记录。"

[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$OutputEncoding = [System.Text.Encoding]::UTF8
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUTF8 = "1"

$PYTHON_EXE = "D:\software\conda\envs\calf\python.exe"
if (-not (Test-Path $PYTHON_EXE)) {
    $PYTHON_EXE = "python"
}

Write-Host "==============================================================================" -ForegroundColor Cyan
Write-Host ">>> [CALF] Starting M1-P1 Paired Training Benchmark (Solar-AL, 7:1:2 Split)" -ForegroundColor Green
Write-Host ">>> Protocol: T0 (use_tq=0) vs T1 (use_tq=1) across Seeds 2025, 2026, 2027" -ForegroundColor Yellow
Write-Host ">>> Constraint: Queue active & identical | Zero-Disk Footprint (Checkpoints Purged)" -ForegroundColor Yellow
Write-Host "==============================================================================" -ForegroundColor Cyan

& $PYTHON_EXE scripts/cycle_mechanism/p1_paired_runner.py --action run_all

$EXIT_CODE = $LASTEXITCODE
if ($EXIT_CODE -eq 0) {
    Write-Host "`n>>> [SUCCESS] P1 Paired Training Suite Completed Successfully!" -ForegroundColor Green
} else {
    Write-Host "`n>>> [ERROR] P1 Paired Training Suite terminated with code $EXIT_CODE" -ForegroundColor Red
}
exit $EXIT_CODE
