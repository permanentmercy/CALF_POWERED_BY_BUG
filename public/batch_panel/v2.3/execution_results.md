# 批量控制面板状态强持久化与无损断点恢复执行产物报告 (Execution Results v2.3)

---

## 一、问题背景与根因定位

用户反馈：“关闭窗口再重新打开然后重新载入，之前跑的任务进度没了”。
经全流程链路审查与磁盘日志对齐，发现由于以下核心缺陷导致了进度在重新打开后无法显现：

1. **状态文件路径绑定缺陷**：
   `StateManager.state_file_path` 此前在 `__init__` 中作为静态字符串生成，当用户在界面切换大文件夹或载入模板时，`checkpoints_root` 属性虽被修改，但 `state_file_path` 依然固定在初始的 `./checkpoints/batch_default/batch_state.json`。因此实际训练完成写入的状态未落入指定目录（如 `./checkpoints/学习率/`）。
2. **缺乏应用会话记忆缓存**：
   关闭 GUI 后重新打开，系统无参启动，默认载入了初始的 `batch_default` 模板，未记忆上次正在执行的配置与目录。
3. **单一 task_hash 严苛匹配造成误重置**：
   此前任务对齐只校验全量参数的 MD5 `task_hash`，若重新载入时模板路径格式存在微小差异（如前缀 `./`），Hash 发生改变，所有历史任务直接被判定为新任务而重置为 `PENDING`。
4. **缺乏基于物理磁盘检查点的自动发现机制**：
   即便 JSON 状态文件发生路径错位或丢失，物理磁盘上已保存了完整的 `checkpoint.pth`、`args.json`、`logs/task_*.log` 和评测 CSV，但旧版代码没有任何从磁盘扫描恢复已完成实验的能力。

---

## 二、架构改动与实施细节

### 1. 动态受控属性绑定 (`state_manager.py`)
- 将 `StateManager.checkpoints_root` 改造为 Python `@property` 与受控 `setter`；
- 无论是直接属性赋值还是调用 `set_checkpoints_root(new_root)`，`self.state_file_path` 均强制联动更新为 `os.path.join(new_root, "batch_state.json")`；
- 彻底消除了状态文件落入错误路径的可能。

### 2. 四级容错任务智能对齐体系
在 `StateManager.init_or_resume` 中重构任务匹配逻辑，按优先级分级对齐：
- **Tier 1 (全参数 Hash 对齐)**：`old_rec["task_hash"] == thash`；
- **Tier 2 (实验指纹 loop_signature 对齐)**：同一批次中只要扫描参数（如 `random_seed=2025, learning_rate=0.0002, pred_len=96`）一致，继承历史记录；
- **Tier 3 (实验标识符 setting 对齐)**：推导 setting 与历史一致则继承；
- **Tier 4 (物理磁盘与日志自动探测恢复)**：调用 `probe_disk_checkpoint` 深入物理磁盘探测。

### 3. 物理磁盘检查点与日志自动探测引擎 (`probe_disk_checkpoint`)
- **严格 Setting 映射**：实现与 `run.py` 100% 对齐的 `compute_setting_string`；
- **检查点与日志交叉验证**：
  - 检查 `{checkpoints_root}/{setting}/checkpoint.pth` 是否存在；
  - 检查 `{checkpoints_root}/logs/{task_id}.log` 中是否记录了包含该 setting 的最终测试指标（`mse:\s*([0-9.]+),\s*mae:\s*([0-9.]+)`）；
  - 检查 `results/{setting}/metrics.json`；
- **状态无损回填与保存**：
  - 若已完成且指标完备，自动标定为 `COMPLETED` 并回填 `mse` 与 `mae`；
  - 若存在中断的检查点但未完成测试，标定为 `PENDING`（无缝断点续跑）；
  - 立即原子化写入 `{checkpoints_root}/batch_state.json`。

### 4. 上次会话记忆持久化 (`gui_panel.py`)
- 引入 `configs/.last_session.json` 会话缓存，在载入模板、保存配置、选择文件夹或关闭窗口时自动同步记录；
- 用户再次启动 `python batch_panel.py` 时，自动无缝恢复上次工作的配置模板与大文件夹目录，任务进度表瞬间精准呈现；
- 已将 `configs/.last_session.json` 加入 `.gitignore`，杜绝非必要文件污染代码库。

---

## 三、测试与实机验证结果

在真实工程环境对 `configs/batch_学习率.json` 及 `./checkpoints/学习率` 执行无头会话恢复测试：

```
Active Batch Name: 学习率
Active Checkpoints Root: ./checkpoints/学习率
State file path: ./checkpoints/学习率/batch_state.json
GUI State Summary: {'total': 84, 'completed': 1, 'failed': 0, 'running': 0, 'pending': 83, 'progress_pct': 1.1904761904761905}
Total tasks in GUI Tree: 84
Task 001 in GUI Tree: ('task_001', 'COMPLETED', "{'random_seed': 2025, 'learning_rate': 0.0002, 'pred_len': 96}", '0.169942', '0.225397', '-', 'long_term_forecast_CALF_96_96_p100_..._rl0.0002_2025_0_..._tw0.736_cyc144_omlp3_tqm_add')
Task 002 in GUI Tree: ('task_002', 'PENDING', "{'random_seed': 2025, 'learning_rate': 0.0002, 'pred_len': 192}", '-', '-', '-', 'long_term_forecast_CALF_96_192_p100_..._rl0.0002_2025_0_..._tw0.736_cyc144_omlp3_tqm_add')
GUI Test Passed Successfully!
```

### 验证结论：
1. **历史进度瞬间找回**：已完成的 `task_001` 从物理磁盘与日志自动恢复为 `COMPLETED`，且 MSE (`0.169942`) 与 MAE (`0.225397`) 完美呈现；
2. **断点无缝承接**：曾运行中断的 `task_002` 被正确识别为 `PENDING`，点击“启动 / 断点续跑”将自动跳过 `task_001`，直接从 `task_002` 接着跑；
3. **开箱即用**：关闭图形界面重新启动 `python batch_panel.py`，无需手动重新载入模板，直接还原上一次的全部会话进度。
