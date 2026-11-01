# 批量控制面板状态强持久化与无损断点恢复规划 (Subplan v2.3)

本文档依照团队规范维护，定义 CALF 批量实验图形控制面板（GUI）在关闭窗口后重新打开、载入模板时历史任务进度丢失问题的彻底修复方案，建立**多级容错任务对齐、会话记忆自动恢复、以及基于物理磁盘与日志的无损状态找回机制**。

---

## 一、问题根因深度诊断 (Root Cause Analysis)

用户反馈：“关闭窗口再重新打开然后重新载入，之前跑的任务进度没了”。经全流程执行审计，定位到以下四个深层根因：

1. **状态文件路径脱节 (State File Path Disconnection)**：
   - 在 `gui_panel.py` 中，用户载入模板或选择大文件夹后，虽然执行了 `self.state_mgr.checkpoints_root = ...`，但 `StateManager.state_file_path` 是在 `__init__` 时单次计算的字符串。
   - 导致 `state_file_path` 依然死锁在初始的 `./checkpoints/batch_default/batch_state.json`。所有状态读写与保存未写入用户指定的权重目录（如 `./checkpoints/学习率/`）。
2. **应用关闭缺乏会话记忆 (No Session Persistence)**：
   - 当用户关闭 GUI 重新打开 `python batch_panel.py` 时，由于未传参，GUI 默认重新初始化为 `batch_default` 模板（3个默认种子任务），无法记忆用户上次正在跑的配置文件与大文件夹。
3. **单一 task_hash 匹配过于严苛脆弱 (Fragile Single-Tier Hash Matching)**：
   - 任务对齐此前仅依赖 `task_hash == thash`。而 `task_hash` 是将包含完整参数（包括 `checkpoints` 路径字符串自身、斜杠规范等）进行序列化计算的 MD5。任何环境重新载入或路径格式细微差异，都会导致 hash 不一致，从而将已完成的任务误判为新任务置为 `PENDING`。
4. **缺乏物理磁盘与运行日志的状态兜底找回机制 (Absence of Disk-Level State Recovery)**：
   - 实验在执行过程中已在指定目录产生了完整的权重检查点（`checkpoint.pth`、`args.json`）、日志输出（`logs/task_*.log` 中已输出 `mse: ... mae: ...`）和评测结果（`results/`）。
   - 但 `StateManager` 完全依赖 `batch_state.json` 内存记录，一旦 JSON 路径错位或被覆盖，完全没有从物理磁盘发现并恢复已完成任务的能力。

---

## 二、架构改动与实施方案

### 1. 动态属性绑定与路径实时同步 (`state_manager.py`)
- 将 `StateManager.checkpoints_root` 改为带有 `@property` 和 `@checkpoints_root.setter` 的受控属性；
- 无论通过属性赋值还是 `set_checkpoints_root()` 方法，只要 `checkpoints_root` 发生变动，`self.state_file_path` 均自动更新为 `os.path.join(self.checkpoints_root, self.STATE_FILENAME)`；
- 保证任何读取、保存永远作用于当前激活的大文件夹。

### 2. 多级智能任务对齐机制 (`init_or_resume`)
在初始化或重新载入批次时，按四级优先级逐级对齐：
- **Tier 1 (全参数 Hash 对齐)**：`old_rec.get("task_hash") == thash`；
- **Tier 2 (循环签名特征对齐)**：在同一批次中，`loop_signature`（如 `{"random_seed": 2025, "learning_rate": 0.0002, "pred_len": 96}`）是唯一且不变的实验指纹；若循环特征匹配，继承已有记录；
- **Tier 3 (实验标识符 setting 对齐)**：若已有记录中 `setting` 与当前任务推导的 `setting` 一致，继承已有记录；
- **Tier 4 (物理磁盘与日志自动探测恢复)**：见第 3 节。

### 3. 物理磁盘与日志无损探测恢复 (Physical Disk & Log Checkpoint Auto-Discovery)
当某个任务未能通过内存 JSON 识别为 `COMPLETED` 时，触发物理探测：
1. **推导预期实验 setting**：调用与 `run.py` 100% 对齐的 `build_setting_from_params(params)`；
2. **探测权重与检查点**：检查 `{checkpoints_root}/{setting}/checkpoint.pth` 是否存在；
3. **探测评测指标**：
   - 优先读取 `{checkpoints_root}/logs/{task_id}.log` 或匹配该 setting 的日志，通过正则扫描 `mse:\s*([0-9.]+),\s*mae:\s*([0-9.]+)`；
   - 备选探测 `results/{setting}/metrics.json` 或 `results/Solar/*seed{seed}_pl{pred_len}*.csv`；
4. **状态无损回填**：若确认该任务已完成且指标完备，自动标定状态为 `COMPLETED`，回填 `metrics`、`setting`、`checkpoint_dir`，并在终端提示找回成功；若未完成测试但有 checkpoint，标为 `PENDING` 准备断点续跑。
5. **持久化保存**：立即将恢复后的完整状态写入 `{checkpoints_root}/batch_state.json`。

### 4. 上次会话自动记忆与恢复 (`gui_panel.py`)
- 新增 `configs/.last_session.json` 会话缓存机制；
- 在用户保存配置、载入模板、启动运行或关闭窗口时，实时持久化记录当前激活的 `last_config_path`、`last_checkpoints_root` 与 `last_batch_name`；
- 当用户再次启动 `python batch_panel.py` 时，无需任何操作，自动恢复到上次关闭时的模板与权重目录，任务进度表瞬间无缝展现。

---

## 三、实施里程碑 (Milestones)

1. **Phase 1**：在 `state_manager.py` 实现属性 setter 同步、多级匹配与物理磁盘/日志探测恢复引擎。
2. **Phase 2**：在 `gui_panel.py` 补充会话持久化缓存机制与目录切换信号同步。
3. **Phase 3**：实机验证：加载 `./checkpoints/学习率/`，验证 task_001 (MSE=0.1699, MAE=0.2254) 瞬间恢复为 COMPLETED，进度条显示已完成项，关闭并重新打开无缝恢复。
4. **Phase 4**：归档 `execution_results.md` 并更新 `public/plan.md`。
