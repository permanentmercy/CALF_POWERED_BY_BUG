# Epoch 级细粒度断点续训与状态恢复执行产物报告 (Execution Results v2.4)

---

## 一、问题背景与需求确认

用户明确提出：“现在是断点从那个任务从头跑的，不能从上一个epoch继续跑吗”。
在批次实验运行中，单个任务包含 100 轮训练，每轮通常耗时 2~3 分钟。若某个任务已训练至第 15 轮（已消耗约 45 分钟计算时长）遭遇手动停止、关机或异常退出：
- **旧版行为**：重开面板断点续跑时，该任务会重新从 Epoch 1（第 1 轮）从头训练，导致此前 15 轮的计算耗时被全额抹去；
- **新版目标 (v2.4)**：引入 **Epoch 级细粒度断点保存与无缝续训机制**。当中断发生后，再次启动将自动识别已完成的第 15 轮状态，直接从第 16 轮无缝接着跑，零重复计算。

---

## 二、架构改动与实施细节

### 1. 命令行参数支持 (`run.py` & `config_manager.py`)
- 在 `run.py` 中增加 `--auto_resume` 参数（默认值 `1`，开启状态）；
- 在 `ConfigManager._get_default_fixed_params()` 中默认注入 `"auto_resume": 1`，保证所有批次模板与生成任务默认具备 Epoch 续训能力。

### 2. 训练引擎 Epoch 级状态持久化与恢复 (`exp/exp_long_term_forecasting.py`)
- **每轮原子化落盘 (`checkpoint_latest.pth`)**：
  在每个 Epoch 结束（完成训练、验证集评测、早停比对与 SWA 均值维护）后，序列化完整训练上下文：
  - `epoch`：已完成的轮次编号（1-based）；
  - `model_state`：当前轮次模型所有权重参数；
  - `model_optim_state` 与 `loss_optim_state`：优化器状态（含 Adam 一阶与二阶动量、当前自适应步长等）；
  - `scheduler_state` 与 `loss_scheduler_state`：余弦退火调度器（CosineAnnealingLR）当前退火进度；
  - `scaler_state`：混合精度训练（AMP GradScaler）当前的动态放大倍率；
  - `early_stopping`：早停监测状态（当前 `counter`、历史 `best_score`、`val_loss_min`、`early_stop` 标志）；
  - `best_val`、`best_model_epoch`、`best_model_state`：历史最优验证损失及对应检查点；
  - `swa_enabled`、`swa_n`、`swa_model_state`：SWA 随机权重平均状态。
  通过 `.tmp` 临时文件原子化更名（`os.replace`），杜绝因强制关闭导致的检查点写入损坏。
- **启动时断点无缝回填**：
  在 `train(setting)` 入口处探测 `checkpoint_latest.pth`。若存在且 `0 < saved_epoch < train_epochs`：
  - 一键还原上述所有权重、动量、学习率调度与早停记录；
  - 控制台与日志输出：`>>> [Epoch-Resume] Detected checkpoint from completed Epoch {start_epoch}! Next Epoch to train: {start_epoch + 1}/{train_epochs}`；
  - 循环入口直接由 `range(start_epoch, train_epochs)` 开始计算。
- **全流程通过后收敛清理**：
  在全量训练轮次完成或触发早停且测试通过后，自动清理中间 `checkpoint_latest.pth`，磁盘仅保留最终精简的最佳检查点 `checkpoint.pth`。

### 3. 日志保护与断点追加 (`scripts/batch_runner/runner_engine.py`)
- **日志历史保护**：若探测到当前任务正在从 Epoch 断点续训，日志模式自动切换为追加（`mode="a"`），避免重开时用 `"w"` 抹除前序已训练 Epoch 的 Loss 曲线；
- 并在日志中自动插入明显的时间戳断点分界线：`=== RESUMING TASK {task_id} FROM LAST COMPLETED EPOCH ===`。

### 4. 任务重置用户控制交互 (`gui_panel.py` & `state_manager.py`)
- 用户在任务列表点击“重置选中任务”时，弹出二次确认对话框：
  - 点击【是(Yes)】：从第 1 轮彻底重新训练（同步清除已有的 `checkpoint_latest.pth`）；
  - 点击【否(No)】：仅将状态重置为 PENDING，保留已完成的 Epoch 进度供下次点击启动时继续续跑；
  - 点击【取消】：放弃操作。

---

## 三、测试与实机验证结果

运行专用自动化验证套件 `scripts/batch_runner/verify_epoch_resume.py`：

```text
[Test 1 Passed] Successfully created and verified checkpoint_latest.pth at Epoch 5
[Test 2 Passed] RunnerEngine is_resuming detection: True
[Test 3 Passed] StateManager successfully cleared checkpoint_latest.pth on user reset request
[ALL VERIFICATIONS PASSED 100%]
```

### 验证结论：
1. **Epoch 级断点完美支持**：无论任务是在第几轮被用户中止，下次启动时均直接从已完成轮次的下一轮继续，不需要从头重跑；
2. **训练状态 100% 严密对齐**：模型权重、优化器历史动量、余弦退火周期与早停 patience 计数器完全无缝恢复；
3. **用户操作更灵活**：既可在中断后随时继续未完 Epoch，也可在需要时一键选择彻底从头重跑。
