# 子计划：周期先验残差机制 (cycle_queue) 一键开关与消融实验规划 (Subplan v1.0)

本文档依照团队协作规范维护，记录关于 CALF 模型中**物理周期先验队列（cycle_queue）**机制的代码审计现状、一键关闭参数（--use_cycle_prior）的技术实现方案与消融实验执行规划。

---

## 一、代码现状审计结论

经过对全项目代码库（包括 `models/CALF.py`、`run.py`、`exp/exp_long_term_forecasting.py` 以及 `batch_runner` 组件）的深度审计：

### 1. 是否已存在一键关闭 cycle_queue 的参数？
- **结论：当前不存在。**
- **技术细节**：
  - 在 `models/CALF.py` 的 `forecast()` 方法中（第 338~353 行），代码仅根据 `getattr(self, 'cycle_prior_before_denorm', 1)` 判断将 `self.cycle_scale * future_cycle_val` 加在反归一化之前（分支 1）还是反归一化之后（分支 0）；
  - 无论 `cycle_prior_before_denorm` 取 0 还是 1，`cycle_queue` 的残差累加均被**无条件强制执行**；
  - 在 `run.py` 中虽然有 `--use_tq`（控制 TQ 文本时序注意力注入开关），但**没有**控制 `cycle_queue` 物理残差项的一键开关参数；
  - 脚本 `validate_cycle_mechanism.py` 中虽然声明了 `--use_cycle_prior`，但并未接入 `CALF.py` 模型底层。

### 2. 两个机制的明确区分
- **机制 A：TQ 机制 (Temporal Query)**：控制 `self.temporal_query` 在特征投影层（`in_layer`）与提示词 / 键值对的注入，由 `--use_tq` 控制；
- **机制 B：周期先验队列 (Cycle Queue Prior, 即用户所指 cycleque)**：在预测输出端（`forecast()`）将 `self.cycle_queue[future_cycle_index]` 乘上 `cycle_scale` 直接以残差形式加到 `outputs_time` 和 `outputs_text` 上。**这正是用户需要进行消融对比的核心机制**。

---

## 二、架构改动与实施方案

### 1. 模型底层改造 ([models/CALF.py](file:///models/CALF.py))
- 在 `__init__` 中新增配置读取：
  ```python
  self.use_cycle_prior = getattr(configs, 'use_cycle_prior', 1)
  if not self.use_cycle_prior:
      self.cycle_queue.requires_grad = False
      self.cycle_scale.requires_grad = False
  ```
- 在 `forecast()` 中设置门控判断：
  ```python
  # Add physical cycle residual prior if available and enabled (before denorm)
  if self.use_cycle_prior and future_cycle_index is not None and hasattr(self, 'cycle_queue') and getattr(self, 'cycle_prior_before_denorm', 1) == 1:
      future_cycle_val = self.cycle_queue[future_cycle_index]
      outputs_time = outputs_time + self.cycle_scale * future_cycle_val
      outputs_text = outputs_text + self.cycle_scale * future_cycle_val

  # Add physical cycle residual prior if available and enabled (after denorm)
  if self.use_cycle_prior and future_cycle_index is not None and hasattr(self, 'cycle_queue') and getattr(self, 'cycle_prior_before_denorm', 1) == 0:
      future_cycle_val = self.cycle_queue[future_cycle_index]
      outputs_time = outputs_time + self.cycle_scale * future_cycle_val
      outputs_text = outputs_text + self.cycle_scale * future_cycle_val
  ```

### 2. 原生启动入口扩展 ([run.py](file:///run.py))
- 新增命令行参数：
  ```python
  parser.add_argument('--use_cycle_prior', type=int, default=1, choices=[0, 1],
                      help='whether to use physical cycle queue prior (1: enable, 0: disable for ablation)')
  ```
- 实验签名防冲突：
  ```python
  if getattr(args, 'use_cycle_prior', 1) != 1:
      extra_tags.append('nocycq')
  ```
  关闭时自动追加 `_nocycq`，防止消融实验的权重和日志覆盖基线模型。

### 3. 批量实验面板与配置管理器适配
- **[scripts/batch_runner/config_manager.py](file:///scripts/batch_runner/config_manager.py)**：基础参数新增 `"use_cycle_prior": 1`，支持在 `loop_params` 中配置 `"use_cycle_prior": [1, 0]`；
- **[scripts/batch_runner/state_manager.py](file:///scripts/batch_runner/state_manager.py)**：签名计算同步追加 `_nocycq` 标签；
- **[scripts/batch_runner/gui_panel.py](file:///scripts/batch_runner/gui_panel.py)**：在固定参数 Tab 2 的“TQ机制与物理先验”分组中新增 `use_cycle_prior` 输入框，循环超参编辑框推荐参数加入 `use_cycle_prior`。

### 4. 专属消融实验配置模板
- 新建 [configs/batch_templates/cycle_queue_ablation.json](file:///configs/batch_templates/cycle_queue_ablation.json)：
  ```json
  {
    "batch_name": "cycle_queue_ablation",
    "checkpoints_root": "./checkpoints/cycle_queue_ablation",
    "mode": "grid",
    "loop_params": {
      "use_cycle_prior": [1, 0],
      "random_seed": [2025, 2026, 2027]
    },
    "fixed_params": {
      "task_name": "long_term_forecast",
      "model": "CALF",
      "data": "Solar",
      "seq_len": 96,
      "pred_len": 96,
      "learning_rate": 0.0004,
      "train_epochs": 100,
      "patience": 5,
      "task_w": 0.736,
      "feature_w": 0.05,
      "output_w": 0.214,
      "cycle": 144,
      "tq_mode": "add",
      "out_mlp_layers": 3,
      "data_split": "7:1:2"
    }
  }
  ```

---

## 三、验证与测试规划
1. **张量前向断言测试**：构建模拟张量输入，验证 `use_cycle_prior=1` 与 `use_cycle_prior=0` 的输出数值差异；
2. **梯度回传断言测试**：验证在 `use_cycle_prior=0` 时 `cycle_queue` 不参与梯度回传；
3. **命令行与 GUI 联调测试**：运行 `--use_cycle_prior 0` 检查签名与训练流程。
