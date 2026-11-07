# 周期先验残差机制 (cycle_queue) 一键开关与消融实验实施产物 (Execution Results v1.0)

本文档依照团队协作规范维护，记录 CALF 模型中**物理周期先验队列（cycle_queue）**一键开关参数（`--use_cycle_prior`）的完整落地实施、批处理面板适配与单元测试验证结果。

---

## 一、实施概述

- **背景需求**：用户需要对 CALF 模型中的 `cycle_queue` 物理周期残差机制进行消融实验，要求具备一键开关参数，且要求与 `batch_panel.py` 批处理体系及可视化 GUI 完美适配，同时严格保证不破坏 baseline 实验、不覆盖现有 checkpoints 权重。
- **实施状态**：**已全部实施完毕，全套 4 项集成单元测试 100% 通过（Exit Code 0）**。

---

## 二、关键技术改造与代码落地

### 1. 模型底层改造 ([models/CALF.py](file:///models/CALF.py))
- **参数控制与梯度冻结**：
  在 `__init__` 中新增 `use_cycle_prior` 参数读取：
  ```python
  self.use_cycle_prior = getattr(configs, 'use_cycle_prior', 1)
  if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
      self.cycle_queue = nn.Parameter(torch.zeros(configs.cycle, configs.enc_in, device=device))
      self.cycle_scale = nn.Parameter(torch.ones(1, device=device) * 0.1)
      self.cycle_prior_before_denorm = getattr(configs, 'cycle_prior_before_denorm', 1)
      if not self.use_cycle_prior:
          self.cycle_queue.requires_grad = False
          self.cycle_scale.requires_grad = False
  ```
  当 `--use_cycle_prior 0` 时，模型内部将 `cycle_queue` 与 `cycle_scale` 的 `requires_grad` 置为 `False`，防止无效参数消耗显存与反向传播计算。
- **前向预测门控与设备对齐**：
  在 `forecast()` 中对反归一化前后的两个残差分支均包裹严格的开关门控与张量设备自动对齐：
  ```python
  # Add physical cycle residual prior if available and enabled (before denormalization option)
  if self.use_cycle_prior and future_cycle_index is not None and hasattr(self, 'cycle_queue') and getattr(self, 'cycle_prior_before_denorm', 1) == 1:
      future_cycle_val = self.cycle_queue[future_cycle_index.to(self.cycle_queue.device)] # (B, pred_len, M)
      outputs_time = outputs_time + self.cycle_scale * future_cycle_val
      outputs_text = outputs_text + self.cycle_scale * future_cycle_val

  # Add physical cycle residual prior if available and enabled (after denormalization option)
  if self.use_cycle_prior and future_cycle_index is not None and hasattr(self, 'cycle_queue') and getattr(self, 'cycle_prior_before_denorm', 1) == 0:
      future_cycle_val = self.cycle_queue[future_cycle_index.to(self.cycle_queue.device)] # (B, pred_len, M)
      outputs_time = outputs_time + self.cycle_scale * future_cycle_val
      outputs_text = outputs_text + self.cycle_scale * future_cycle_val
  ```

### 2. 原生启动入口扩展 ([run.py](file:///run.py))
- **CLI 参数注册**：
  ```python
  parser.add_argument('--use_cycle_prior', type=int, default=1, choices=[0, 1],
                      help='whether to use physical cycle queue prior (1: enable, 0: disable for ablation)')
  ```
  默认值为 `1`，完全保证原有所有脚本与训练任务向后兼容。
- **实验签名自动隔离**：
  在 `build_setting_string` 中：
  ```python
  if getattr(args, 'use_cycle_prior', 1) != 1:
      extra_tags.append('nocycq')
  ```
  当使用 `--use_cycle_prior 0` 时，自动在模型目录后缀增加 `_nocycq`，保证消融模型的检查点与日志绝对不会覆盖基线目录。

### 3. 批量实验面板与配置适配 ([scripts/batch_runner/](file:///scripts/batch_runner/))
- **配置管理** ([scripts/batch_runner/config_manager.py](file:///scripts/batch_runner/config_manager.py))：
  在固定默认参数中注入 `"use_cycle_prior": 1`，并在超参解析与展开器中支持在 `loop_params` 中定义 `"use_cycle_prior": [1, 0]`。
- **状态追踪与隔离** ([scripts/batch_runner/state_manager.py](file:///scripts/batch_runner/state_manager.py))：
  在生成批任务 setting 签名时同步检测 `use_cycle_prior`，若不为 1 则注入 `_nocycq`。
- **可视化图形界面** ([scripts/batch_runner/gui_panel.py](file:///scripts/batch_runner/gui_panel.py))：
  1. 预设目录下拉菜单新增 `"周期残差消融目录"` (`./checkpoints/cycle_queue_ablation`)；
  2. Tab 2 固定参数“TQ机制与物理先验”分组中新增 `启用周期残差(0/1)` 输入控件；
  3. 循环超参编辑框中的参数快速填充列表加入 `use_cycle_prior` 选项。

### 4. 专属批处理模板 ([configs/batch_templates/cycle_queue_ablation.json](file:///configs/batch_templates/cycle_queue_ablation.json))
提供了开箱即用的消融模板，包含 3 种子（2025, 2026, 2027）在启用与关闭周期残差下的成对对比实验，固定参数完全对齐 Solar-AL 最优基线。

---

## 三、自动化验证测试结果

编写并执行了全覆盖自动化测试套件 [tests/test_cycle_prior_ablation.py](file:///tests/test_cycle_prior_ablation.py)，验证内容与输出如下：

```text
======================================================================
>>> Running cycle_queue Ablation Verification Suite
======================================================================
[Test 1/4] Verifying setting string isolation...
  [PASS] Baseline setting: ..._epochs100_bs16_dr0.1_ld0.1_r8
  [PASS] Ablation setting: ...pochs100_bs16_dr0.1_ld0.1_r8_nocycq
[Test 2/4] Verifying cycle_queue_ablation.json template and task combos...
  [PASS] Successfully expanded 6 tasks with valid --use_cycle_prior args.
[Test 3/4] Verifying model parameter gradient freeze...
  [PASS] Gradient requirements verified (On=True, Off=False).
[Test 4/4] Verifying forward output divergence between On and Off...
  [PASS] Forward output mean difference: time=0.101755, text=0.101755 (Ablation successfully bypassing residual!)
======================================================================
>>> ALL 4 TEST SUITES PASSED PERFECTLY!
======================================================================
```

- **测试 1（目录防冲突隔离）**：确认开启时目录保持原始命名，关闭时自动加上 `_nocycq`，互不干扰；
- **测试 2（批处理模板展开）**：确认 `cycle_queue_ablation.json` 能正确被 `BatchConfigManager` 展开为 6 个任务，各任务正确附带 `--use_cycle_prior` 标记；
- **测试 3（梯度冻结）**：确认关闭后 `cycle_queue.requires_grad == False` 且 `cycle_scale.requires_grad == False`；
- **测试 4（前向旁路完全旁路）**：在相同权重下输入相同数据，关闭与开启产生预期的残差偏置分离（平均差异为 `0.101755`），证明消融机制精准有效切断了周期队列注入。

---

## 四、使用方法指南

### 1. 单任务命令行启动消融
```bash
# 开启周期先验（默认，等同于原 baseline）
python run.py --task_name long_term_forecast --model CALF --data Solar --seq_len 96 --pred_len 96 --use_cycle_prior 1

# 关闭周期先验（消融实验，自动保存至带 _nocycq 的检查点目录）
python run.py --task_name long_term_forecast --model CALF --data Solar --seq_len 96 --pred_len 96 --use_cycle_prior 0
```

### 2. 通过 batch_panel.py 批量消融运行
```bash
# 启动 GUI 面板后直接在模板列表选择 "cycle_queue_ablation.json" 并点击 "加载模板" -> "开始批处理"
python batch_panel.py
```
或直接加载模板执行无头批处理，一键全自动完成消融对比。
