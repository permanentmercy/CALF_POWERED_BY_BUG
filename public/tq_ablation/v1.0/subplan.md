# 子计划：TQ (Temporal Query) 消融实验多步长分析与学术绘图 (Subplan v1.0)

本文档依照团队协作规范维护，记录 CALF 模型中 **TQ (Temporal Query) 机制**消融实验（无 TQ / `use_tq=0`）与基线实验（有 TQ / Baseline）在 4 个预测步长（Horizon 96, 192, 336, 720）下的综合对比、同比百分改变分析及 MATLAB/Python 学术绘图技术方案。

---

## 一、需求背景与目标

1. **实验背景**：
   - 用户已通过 `batch_panel.py` 在 `checkpoints/TQ消融/` 完成了 TQ 消融实验（`use_tq=0`，3 种子 x 4 步长 = 12 组任务）；
   - 用户提供了有 TQ 基准实验的 3 种子（2025, 2026, 2027）及均值（average）数据表。
2. **核心目标**：
   - 绘制无 TQ 消融实验的预测步长 vs MSE 折线图（3 随机数种子与均值）；
   - 绘制有 TQ 基准实验的同构预测步长 vs MSE 折线图；
   - 绘制有 TQ vs 无 TQ 直接对比折线图（直观展示两组差距与误差棒）；
   - 绘制柱状图，展示无 TQ 相比有 TQ 情况下在不同步长下的**同比百分改变**（\(\Delta \% = \frac{\text{MSE}_{\text{w/o TQ}} - \text{MSE}_{\text{w/ TQ}}}{\text{MSE}_{\text{w/ TQ}}} \times 100\%\)）；
   - 脚本必须使用 MATLAB 编写（`.m` 脚本），同时配备 Python 脚本保障全环境双向可用；
   - 产物与脚本严格归档至 `checkpoints/TQ消融/`，并输出 300 DPI PNG、矢量 PDF 以及交互式 FIG。

---

## 二、数据矩阵与指标统计

### 1. 实验原始与聚合数据
- **预测步长**：96, 192, 336, 720；
- **随机种子**：2025, 2026, 2027；
- **有 TQ 基线 (With TQ, Baseline)**：
  - Pred 96: 2025=0.164986, 2026=0.164929, 2027=0.164136 | Avg = 0.164532
  - Pred 192: 2025=0.187519, 2026=0.193756, 2027=0.189566 | Avg = 0.190280
  - Pred 336: 2025=0.198770, 2026=0.199735, 2027=0.205567 | Avg = 0.201357
  - Pred 720: 2025=0.206418, 2026=0.207503, 2027=0.205460 | Avg = 0.206460
- **无 TQ 消融 (Without TQ, use_tq=0)**：
  - Pred 96: 2025=0.176508, 2026=0.168637, 2027=0.176556 | Avg = 0.173900
  - Pred 192: 2025=0.189280, 2026=0.194357, 2027=0.191140 | Avg = 0.191592
  - Pred 336: 2025=0.200347, 2026=0.200503, 2027=0.202872 | Avg = 0.201241
  - Pred 720: 2025=0.204681, 2026=0.207425, 2027=0.206440 | Avg = 0.206182

### 2. 同比百分改变计算公式
$$\Delta \text{MSE} \% = \frac{\text{MSE}_{\text{w/o TQ}} - \text{MSE}_{\text{w/ TQ}}}{\text{MSE}_{\text{w/ TQ}}} \times 100\%$$
- **Pred 96**: \(\Delta = +0.009368\) (\(\mathbf{+5.70\%}\)) -> 移除 TQ 导致误差增加 5.70%，证明 TQ 在短预测视界具备显著增益！
- **Pred 192**: \(\Delta = +0.001312\) (\(\mathbf{+0.69\%}\)) -> 移除 TQ 导致误差小幅上升 0.69%；
- **Pred 336**: \(\Delta = -0.000117\) (\(\mathbf{-0.06\%}\)) -> 基本持平；
- **Pred 720**: \(\Delta = -0.000278\) (\(\mathbf{-0.13\%}\)) -> 基本持平。

---

## 三、产物清单与文件规划

1. **数据表**：
   - `checkpoints/TQ消融/tq_ablation_raw_results.csv`：原始明细
   - `checkpoints/TQ消融/tq_ablation_summary.csv`：均值、标准差与同比百分比汇总
2. **绘图代码**：
   - `checkpoints/TQ消融/plot_tq_ablation.m`：MATLAB 主绘图脚本
   - `checkpoints/TQ消融/plot_tq_ablation.py`：Python 绘图脚本
3. **输出图表**：
   - 图 1: `tq_ablation_mse_without_tq` (PNG, PDF, FIG)
   - 图 2: `tq_ablation_mse_with_tq` (PNG, PDF, FIG)
   - 图 3: `tq_ablation_mse_comparison` (PNG, PDF, FIG)
   - 图 4: `tq_ablation_relative_change_bar` (PNG, PDF, FIG)
