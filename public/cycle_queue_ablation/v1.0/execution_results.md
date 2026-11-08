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

---

## 五、消融实验完整执行结果与数据明细

用户在 `checkpoints/cycle消融/` 目录下完成了针对 4 个预测步长 (`96`, `192`, `336`, `720`) 与 3 个随机种子 (`2025`, `2026`, `2027`) 的无周期先验 (`--use_cycle_prior 0`) 完整消融训练（共 12 组实验）。

### 1. 无周期先验消融实验明细数据表 (Without Cycle Prior, `use_cycle_prior=0`)

| 预测步长 (Pred Len) | 种子 (Seed) | MSE | MAE | 检查点目录 |
| :---: | :---: | :---: | :---: | :--- |
| **96** | 2025 | 0.178523 | 0.215850 | `long_term_forecast_..._2025_..._nocycq` |
| **96** | 2026 | 0.179269 | 0.215119 | `long_term_forecast_..._2026_..._nocycq` |
| **96** | 2027 | 0.177735 | 0.214960 | `long_term_forecast_..._2027_..._nocycq` |
| **96 (均值)** | **Average** | **0.178509** | **0.215310** | - |
| **192** | 2025 | 0.213195 | 0.241435 | `long_term_forecast_..._2025_..._nocycq` |
| **192** | 2026 | 0.212265 | 0.241539 | `long_term_forecast_..._2026_..._nocycq` |
| **192** | 2027 | 0.211629 | 0.240477 | `long_term_forecast_..._2027_..._nocycq` |
| **192 (均值)** | **Average** | **0.212363** | **0.241150** | - |
| **336** | 2025 | 0.228715 | 0.254297 | `long_term_forecast_..._2025_..._nocycq` |
| **336** | 2026 | 0.231215 | 0.256151 | `long_term_forecast_..._2026_..._nocycq` |
| **336** | 2027 | 0.229792 | 0.254783 | `long_term_forecast_..._2027_..._nocycq` |
| **336 (均值)** | **Average** | **0.229907** | **0.255077** | - |
| **720** | 2025 | 0.237678 | 0.259843 | `long_term_forecast_..._2025_..._nocycq` |
| **720** | 2026 | 0.238991 | 0.260686 | `long_term_forecast_..._2026_..._nocycq` |
| **720** | 2027 | 0.240163 | 0.261115 | `long_term_forecast_..._2027_..._nocycq` |
| **720 (均值)** | **Average** | **0.238944** | **0.260548** | - |

---

## 六、有周期先验基准对比与同比百分改变分析

### 1. 成对对比与百分改变表

$$\Delta \text{MSE} \% = \frac{\text{MSE}_{\text{Without Cycle}} - \text{MSE}_{\text{With Cycle}}}{\text{MSE}_{\text{With Cycle}}} \times 100\%$$

| 预测步长 (Pred Len) | 有周期先验基线 MSE (With Cycle) | 无周期先验消融 MSE (Without Cycle) | 绝对误差差值 (\(\Delta\)) | 同比百分改变 (\(\Delta\%\)) | 影响结论 |
| :---: | :---: | :---: | :---: | :---: | :--- |
| **96** | 0.164532 | 0.178509 | +0.013977 | **+8.49%** | 显著劣化（去除周期队列误差上升 8.49%） |
| **192** | 0.190280 | 0.212363 | +0.022083 | **+11.61%** | 极显著劣化（去除周期队列误差上升 11.61%） |
| **336** | 0.201357 | 0.229907 | +0.028550 | **+14.18%** | 严重恶化（去除周期队列误差上升 14.18%） |
| **720** | 0.206460 | 0.238944 | +0.032484 | **+15.73%** | 严重恶化（去除周期队列误差上升 15.73%） |

### 2. 规律分析：物理周期先验队列的尺度递增增益规律

与 TQ 机制（主要在短程 96 步起效 +5.70%，长程趋于平缓）不同，**`cycle_queue` 展现出随预测步长延长而单调递增的重要度增长特征**：
- 在 Horizon 96 下，去除周期残差使 MSE 上升 **+8.49%**；
- 在 Horizon 192 下，去除周期残差使 MSE 上升 **+11.61%**；
- 在 Horizon 336 下，去除周期残差使 MSE 上升 **+14.18%**；
- 在 Horizon 720 下，去除周期残差使 MSE 上升高达 **+15.73%**！

**物理机理阐释**：
在光伏太阳能辐射（Solar-AL）时序数据中，日周期具有极其强烈的物理确定性。在短期预测中，语言大模型的注意力机制尚能捕捉局部的趋势；然而随着预测步长不断延伸至 336 乃至 720 步（多日跨度），序列自回归生成与注意力权重的累积相位漂移（Phase Drift）迅速累加。`cycle_queue` 提供的显式物理日周期先验残差能够充当坚固的“物理锚点”，牢牢约束长程预测的波形中枢，因而**步长越长，物理周期先验越不可或缺**。

---

## 七、学术可视化产物与归档清单

本次消融实验的可视化绘图与数据文件已完整归档在 `checkpoints/cycle消融/` 目录中：

1. **图表文件** (300 DPI PNG、矢量 PDF、交互式 FIG)：
   - `cycle_ablation_mse_without_cycle` (PNG / PDF / FIG)：无周期先验消融实验 3 种子与均值折线图；
   - `cycle_ablation_mse_with_cycle` (PNG / PDF / FIG)：有周期先验基线 3 种子与均值折线图；
   - `cycle_ablation_mse_comparison` (PNG / PDF / FIG)：基线 vs 消融对比折线图（横坐标严格对齐，纯圆点 `'o'`，无横杠）；
   - `cycle_ablation_relative_change_bar` (PNG / PDF / FIG)：同比百分改变柱状图（+8.49% 至 +15.73% 标注）。
2. **绘图脚本**：
   - `checkpoints/cycle消融/plot_cycle_ablation.m`：MATLAB 专业学术绘图脚本；
   - `checkpoints/cycle消融/plot_cycle_ablation.py`：Python / Matplotlib 独立跨平台绘图脚本。
3. **数据文件**：
   - `checkpoints/cycle消融/cycle_ablation_raw_results.csv`：12 组实验原始指标；
   - `checkpoints/cycle消融/cycle_ablation_summary.csv`：多步长聚合统计表。

