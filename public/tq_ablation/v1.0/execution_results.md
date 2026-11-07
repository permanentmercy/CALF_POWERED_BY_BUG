# TQ (Temporal Query) 消融实验多步长分析与学术绘图实施产物 (Execution Results v1.0)

本文档依照团队协作规范维护，记录 CALF 模型中 **TQ (Temporal Query) 机制**消融实验与基线对比、同比百分改变分析及 MATLAB/Python 脚本与图表的完整交付结果。

---

## 一、实施成果概览

根据用户指令，针对 `checkpoints/TQ消融/` 完成的 12 组消融任务与提供的有 TQ 基线数据，已全面完成数据统计、指标对比及学术绘图：

| 产物名称 | 文件类型 | 存放位置 | 说明 |
| :--- | :--- | :--- | :--- |
| **MATLAB 绘图脚本** | `.m` | [`checkpoints/TQ消融/plot_tq_ablation.m`](file:///checkpoints/TQ消融/plot_tq_ablation.m) | 完整支持 4 张图生成、图例置顶防遮挡、3 级容错数据读取、导出 PNG/PDF/FIG |
| **Python 绘图脚本** | `.py` | [`checkpoints/TQ消融/plot_tq_ablation.py`](file:///checkpoints/TQ消融/plot_tq_ablation.py) | 基于 Matplotlib 的对齐脚本，支持全自动独立绘图 |
| **原始数据表** | `.csv` | [`checkpoints/TQ消融/tq_ablation_raw_results.csv`](file:///checkpoints/TQ消融/tq_ablation_raw_results.csv) | 包含两种状态、3 种子、4 步长的全部 24 组原始 MSE/MAE |
| **汇总统计表** | `.csv` | [`checkpoints/TQ消融/tq_ablation_summary.csv`](file:///checkpoints/TQ消融/tq_ablation_summary.csv) | 包含均值、标准差、绝对差值与同比百分比变化 |
| **图 1: 无 TQ 消融折线图** | `.png`/`.pdf`/`.fig` | `checkpoints/TQ消融/tq_ablation_mse_without_tq.*` | 横坐标步长，纵坐标 MSE，展示 3 种子与均值 |
| **图 2: 有 TQ 基线折线图** | `.png`/`.pdf`/`.fig` | `checkpoints/TQ消融/tq_ablation_mse_with_tq.*` | 与图 1 同构，展示有 TQ 基线 3 种子与均值 |
| **图 3: 有无 TQ 对比折线图** | `.png`/`.pdf`/`.fig` | `checkpoints/TQ消融/tq_ablation_mse_comparison.*` | 同图对比 With TQ vs Without TQ，包含误差棒与显著度标记 |
| **图 4: 同比百分改变柱状图** | `.png`/`.pdf`/`.fig` | `checkpoints/TQ消融/tq_ablation_relative_change_bar.*` | 柱状图展示各步长下相比有 TQ 的同比百分改变与数值标签 |

---

## 二、关键数据与结论分析

### 1. 核心对比数据表
| 预测步长 (Pred Len) | 有 TQ 基线 MSE (With TQ) | 无 TQ 消融 MSE (Without TQ) | 绝对差值 (\(\Delta\)) | 同比百分改变 (\(\Delta\%\)) | 结论简述 |
| :---: | :---: | :---: | :---: | :---: | :--- |
| **96** | **0.164532** | 0.173900 | **+0.009368** | **+5.70%** | **显著劣化**（去除 TQ 导致 MSE 上升 5.70%，证明 TQ 核心增益有效） |
| **192** | **0.190280** | 0.191592 | +0.001312 | **+0.69%** | 轻微劣化（去除 TQ 误差小幅上升 0.69%） |
| **336** | 0.201357 | 0.201241 | -0.000117 | **-0.06%** | 基本持平（差异 <0.1%） |
| **720** | 0.206460 | 0.206182 | -0.000278 | **-0.13%** | 基本持平（差异 <0.15%） |

### 2. 学术结论提炼
1. **短视界核心收益**：在短预测视界（Horizon 96）下，TQ 模块对预测精度有极其显著的正向作用，关闭 TQ 使得预测误差同比剧烈恶化了 **+5.70%**（MSE 从 0.1645 上升至 0.1739）；
2. **中长视界衰减效应**：随着预测步长从 96 延伸至 192、336 和 720，TQ 的先验增益逐渐饱和，在 336 和 720 步长下两者误差几乎重合（差异小于 0.15%）。这符合时间序列预测的物理衰减规律，表明 TQ 提示词在近端时序感知中贡献最大。

---

## 三、脚本执行与复现方法

### 1. MATLAB 脚本执行
在 MATLAB 控制台或操作系统终端中直接运行：
```powershell
# 终端静默批量执行
matlab -wait -nosplash -noFigureWindows -batch "addpath('checkpoints/TQ消融'); plot_tq_ablation;"
```

### 2. Python 脚本执行
```powershell
python checkpoints/TQ消融/plot_tq_ablation.py
```
均可无损全自动重绘所有图表并输出 300 DPI 产物。
