# 执行产物：三损失权重完全解耦与独立超参控制 (v2.6)

本产物记录 CALF 批量实验控制面板 v2.6 版本关于三损失权重解耦的技术落地与验证结果。

---

## 1. 代码改动清单与文件索引

1. **配置管理器核心**：
   - [scripts/batch_runner/config_manager.py](file:///scripts/batch_runner/config_manager.py)：
     - 移除 `output_w` 强制覆盖与 `output_w <= 0` 强制跳过；
     - 确立 `task_w`, `feature_w`, `output_w` 三维正交独立机制；
     - 仅对显式声明 `"output_w": "auto"` 保留可选求和计算。
2. **GUI 图形控制面板**：
   - [scripts/batch_runner/gui_panel.py](file:///scripts/batch_runner/gui_panel.py)：
     - 移除联动约束按钮与求和标签；
     - 增添三损失完全独立设置的说明提示；
     - 在循环参数编辑窗口中推荐参数名追加 `output_w`。

---

## 2. 自动化验证结果

### 测试用例 1：固定参数任意非 1 和测试
- **设定**：`task_w = 0.8`, `feature_w = 0.3`, `output_w = 0.6`（三者和为 1.7），固定执行 2 组种子。
- **运行命令**：
  ```powershell
  python -c "from scripts.batch_runner.config_manager import ConfigManager; cm = ConfigManager(); cm.set_fixed_param('task_w', 0.8); cm.set_fixed_param('feature_w', 0.3); cm.set_fixed_param('output_w', 0.6); cm.loop_params = {'random_seed': [2025, 2026]}; tasks = cm.generate_task_combos(); print(tasks[0]['params']['task_w'], tasks[0]['params']['feature_w'], tasks[0]['params']['output_w'])"
  ```
- **输出结果**：`0.8 0.3 0.6`（参数 100% 保持，未被修改）。

### 测试用例 2：循环扫描与包含零值测试
- **设定**：`task_w: [0.5, 1.0]`, `output_w: [0.0, 0.5, 1.0]`, `feature_w = 0.2`。
- **生成任务数**：6 组独立笛卡尔积组合。
- **包含值**：
  - `task_001`: `tw=0.5, fw=0.2, ow=0.0 (sum=0.7)` $\rightarrow$ **正常生成，未跳过**；
  - `task_002`: `tw=0.5, fw=0.2, ow=0.5 (sum=1.2)` $\rightarrow$ **正常生成**；
  - `task_003`: `tw=0.5, fw=0.2, ow=1.0 (sum=1.7)` $\rightarrow$ **正常生成**；
  - `task_004`: `tw=1.0, fw=0.2, ow=0.0 (sum=1.2)` $\rightarrow$ **正常生成**；
  - `task_005`: `tw=1.0, fw=0.2, ow=0.5 (sum=1.7)` $\rightarrow$ **正常生成**；
  - `task_006`: `tw=1.0, fw=0.2, ow=1.0 (sum=2.2)` $\rightarrow$ **正常生成**。

**结论**：三个损失权重已彻底解耦，系统完全支持任意独立设定与循环网格扫描。
