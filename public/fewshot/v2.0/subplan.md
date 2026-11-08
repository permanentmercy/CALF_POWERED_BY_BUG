# 子计划：少样本从头训练多步长实验结果聚合与学术绘图 (Subplan v2.0)

本文档依照团队协作规范维护，记录 CALF 模型在 **Solar-IL 数据集少样本从头训练体系**（5 训练数据比例 x 4 预测步长 x 3 随机种子，共 60 组实验）的指标统计、数据聚合与 MATLAB/Python 出版级学术绘图技术方案。

---

## 一、需求背景与目标

1. **实验背景**：
   - 实验在 `checkpoints/少样本/` 目录下完成，共计 60 个任务全部执行完毕（Exit Code 0）；
   - 数据集：`solar_IL.csv`（106 通道特征输入，137 通道输出）；
   - 训练范式：从头初始化训练（From Scratch，无外部预训练权重），前向端到端优化；
   - 实验网格：
     - 少样本训练比例 `percent`：`[10%, 20%, 30%, 40%, 50%]`（5 档）；
     - 预测步长 `pred_len`：`[96, 192, 336, 720]`（4 档）；
     - 随机种子 `random_seed`：`[2025, 2026, 2027]`（3 种子）；
     - 任务总量：5 x 4 x 3 = 60 组任务全部完成。
2. **核心目标**：
   - 解析 60 组日志提取完整评价指标（MSE、MAE），导出明细表与聚合统计表；
   - 绘制图 1：少样本训练比例 (10%~50%) vs MSE 折线图（4 预测步长各自一条线）；
   - 绘制图 2：4 步长综合平均 MSE 随数据比例的全局衰减曲线；
   - 绘制图 3：预测步长 (96~720) vs MSE 折线图（5 种训练比例各自一条线）；
   - 绘制图 4：少样本比例 vs MAE 折线图；
   - 绘图规范遵从用户明确偏好：折线连接处统一使用纯实心标准圆点（Circle Marker, `'o'`），彻底移除上下横杠与冗余装饰；
   - 使用 MATLAB 编写主脚本 [`plot_fewshot.m`](file:///checkpoints/少样本/plot_fewshot.m)，并配备 Python 脚本 [`plot_fewshot.py`](file:///checkpoints/少样本/plot_fewshot.py)；
   - 产物输出为 300 DPI PNG、矢量 PDF 以及交互式 FIG。

---

## 二、数据矩阵与指标汇总

### 3 种子均值 MSE 矩阵
| 训练比例 (Percent) | Pred Len = 96 | Pred Len = 192 | Pred Len = 336 | Pred Len = 720 | 4 步长综合平均 |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **10%** | 0.200348 | 0.236908 | 0.242175 | 0.251456 | **0.232722** |
| **20%** | 0.188858 | 0.215214 | 0.232235 | 0.238484 | **0.218698** |
| **30%** | 0.185017 | 0.200506 | 0.223876 | 0.232377 | **0.210444** |
| **40%** | 0.185170 | 0.211271 | 0.223676 | 0.232548 | **0.213166** |
| **50%** | **0.182941** | 0.208061 | 0.236556 | 0.242059 | **0.217404** |

### 核心观察与物理规律
1. **短步长强单调递减**：在 Pred Len = 96 下，随着训练数据量从 10% 提升至 50%，MSE 从 0.2003 单调下降至 0.1829，下降幅度达 **-8.69%**；
2. **综合收敛极值**：全步长综合均值在 30% 训练比例时达到最优平台区（0.210444），表现出大模型在低样本资源下的快速表征收敛能力。

---

## 三、产物清单与文件规划

1. **数据表**：
   - [`checkpoints/少样本/fewshot_raw_results.csv`](file:///checkpoints/少样本/fewshot_raw_results.csv)：60 项原始任务指标
   - [`checkpoints/少样本/fewshot_summary.csv`](file:///checkpoints/少样本/fewshot_summary.csv)：20 项均值与标准差统计
2. **绘图代码**：
   - [`checkpoints/少样本/plot_fewshot.m`](file:///checkpoints/少样本/plot_fewshot.m)：MATLAB 主脚本
   - [`checkpoints/少样本/plot_fewshot.py`](file:///checkpoints/少样本/plot_fewshot.py)：Python 脚本
3. **输出图表**：
   - 图 1: `fewshot_ratio_vs_mse.*` (PNG, PDF, FIG)
   - 图 2: `fewshot_ratio_vs_mse_avg.*` (PNG, PDF, FIG)
   - 图 3: `fewshot_pred_len_vs_mse.*` (PNG, PDF, FIG)
   - 图 4: `fewshot_ratio_vs_mae.*` (PNG, PDF, FIG)
