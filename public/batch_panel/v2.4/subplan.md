# Epoch 级细粒度断点续训与状态恢复规划 (Subplan v2.4)

本文档依照团队规范维护，定义 CALF 批量实验控制面板与运行引擎在任务被中断时，实现**从上一个完成的 Epoch 继续断点续训**（而非从第 1 轮从头重跑）的技术方案。

---

## 一、需求背景与目标

1. **核心痛点**：
   - 用户明确提出：“现在是断点从那个任务从头跑的，不能从上一个epoch继续跑吗”。
   - 目前的断点恢复粒度是“任务级”（Task-Level）：若整个批次有 84 个任务，任务 1 跑完后停止，重新启动会跳过任务 1，直接跑任务 2。
   - 但是，如果任务 2 设定了 100 个 Epoch，每个 Epoch 耗时 2~3 分钟；当任务 2 已经运行了 15 个 Epoch（耗时近 45 分钟）遭遇中断后，重新启动该任务会再次从 Epoch 1 开始从头计算，导致这 45 分钟的计算资源和时间被完全浪费。
2. **目标要求**：
   - 实现**Epoch 级细粒度检查点机制**：每个 Epoch 结束（完成验证、早停计算与损失记录）后，原子化保存当前 Epoch 的完整训练状态（`checkpoint_latest.pth`）；
   - **无缝断点恢复**：当进程重新启动时，自动探测是否存在该任务未跑完的 `checkpoint_latest.pth`。若存在，无缝还原模型权重、优化器、调度器、早停计数器（`patience`）与历史最优模型，直接从下一个 Epoch 开始继续训练；
   - **训练完成自动收敛清理**：当该任务全流程训练及测试通过后，清理临时中间 Epoch 状态，保留最终最优检查点 `checkpoint.pth`。

---

## 二、架构改动与实施方案

### 1. 参数与开关支持 (`run.py`)
- 在命令行参数中增加 `--auto_resume`（类型 `int`，默认 `1`）；
- 默认开启 Epoch 级别断点续训探测。

### 2. 实验引擎 Epoch 级状态持久化与恢复 (`exp/exp_long_term_forecasting.py`)
- **状态序列化**：定义包含完整训练元数据的结构字典：
  - `epoch`：已完成的 Epoch 索引（1-based）；
  - `model_state`：当前轮次模型权重；
  - `model_optim_state` 与 `loss_optim_state`：优化器状态（动量、Adam 历史梯度等）；
  - `scheduler_state` 与 `loss_scheduler_state`：余弦退火等学习率调度器状态；
  - `scaler_state`：混合精度（AMP）梯度缩放器状态；
  - `early_stopping`：早停计数器 `counter`、`best_score`、`val_loss_min`、`early_stop`；
  - `best_val`、`best_model_epoch`、`best_model_state`：历史最优验证集损失与最优模型权重；
  - `swa_enabled`、`swa_n`、`swa_model_state`：SWA 状态（若启用）。
- **原子化落盘**：每个 Epoch 结束时保存至 `checkpoint_latest.pth.tmp` 并重命名为 `checkpoint_latest.pth`，杜绝中途强关导致检查点文件损坏。
- **启动时断点恢复**：
  - 在 `train(setting)` 入口处检查 `checkpoint_latest.pth`；
  - 若存在且 `0 < saved_epoch < train_epochs`，完整回填上述所有训练状态，打印 `>>> [Epoch-Resume] Detected checkpoint from completed Epoch {start_epoch}! Resuming training from Epoch {start_epoch + 1}...`；
  - 循环入口无缝从 `for epoch in range(start_epoch, self.args.train_epochs):` 启动。
- **训练收尾清理**：当早停或达到最大轮次并完成测试后，安全移除 `checkpoint_latest.pth`。

### 3. 日志流与运行引擎适配 (`scripts/batch_runner/runner_engine.py`)
- 中断恢复日志追加：若任务存在未完成的 `checkpoint_latest.pth`，日志文件由覆写改为追加（`mode="a"`），并插入 `=== RESUMING TASK {task_id} FROM BREAKPOINT ===` 标头，保证前序 Epoch 的训练指标与趋势不会被冲掉。

### 4. GUI 面板与状态管理适配 (`gui_panel.py`, `state_manager.py`)
- 在任务树和状态栏能够呈现当前任务的恢复状态；
- 重置任务时提供安全确认：是否清除中间 Epoch 检查点以彻底从头重跑。

---

## 三、实施里程碑 (Milestones)

1. **Phase 1**：在 `run.py` 增加 `--auto_resume` 选项。
2. **Phase 2**：在 `exp/exp_long_term_forecasting.py` 实现每轮保存 `checkpoint_latest.pth` 与启动自动断点续训。
3. **Phase 3**：在 `runner_engine.py` 支持断点日志追加与控制台续训标头高亮。
4. **Phase 4**：自动化测试验证（模拟多轮训练中断与续跑，验证优化器与早停对齐）。
5. **Phase 5**：归档 `execution_results.md` 并更新 `public/plan.md`。
