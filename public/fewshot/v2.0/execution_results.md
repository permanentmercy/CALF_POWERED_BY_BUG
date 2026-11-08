# 少样本从头训练多步长实验结果聚合与学术绘图实施产物 (Execution Results v2.0)

本文档依照团队协作规范维护，记录 CALF 模型在 **Solar-IL 数据集少样本从头训练体系**（60 组实验）的完整数据解析、指标汇总表及 MATLAB/Python 4 张学术出版级图表的交付产物。

---

## 一、实施成果概览

根据用户指令，针对 `checkpoints/少样本/` 完成的全部 60 个任务，已全面完成数据提取、指标聚合与图表绘制：

| 产物名称 | 文件类型 | 存放位置 | 说明 |
| :--- | :--- | :--- | :--- |
| **MATLAB 绘图脚本** | `.m` | [`checkpoints/少样本/plot_fewshot.m`](file:///checkpoints/少样本/plot_fewshot.m) | 完整支持 4 张学术图表生成、图例置顶防遮挡、纯圆点节点、导出 PNG/PDF/FIG |
| **Python 绘图脚本** | `.py` | [`checkpoints/少样本/plot_fewshot.py`](file:///checkpoints/少样本/plot_fewshot.py) | 基于 Matplotlib 的对齐脚本，支持全自动无依赖独立出图 |
| **原始数据明细表** | `.csv` | [`checkpoints/少样本/fewshot_raw_results.csv`](file:///checkpoints/少样本/fewshot_raw_results.csv) | 包含 60 组任务的完整参数（percent, pred_len, seed）与 MSE/MAE 指标 |
| **统计汇总表** | `.csv` | [`checkpoints/少样本/fewshot_summary.csv`](file:///checkpoints/少样本/fewshot_summary.csv) | 包含 20 组实验配置下的 3 种子均值、标准差及单独各随机种子表现 |
| **图 1: 比例 vs MSE 折线图** | `.png`/`.pdf`/`.fig` | `checkpoints/少样本/fewshot_ratio_vs_mse.*` | 横坐标训练比例 (10%~50%)，纵坐标 MSE，4 条预测步长折线 |
| **图 2: 4 步长综合平均衰减图** | `.png`/`.pdf`/`.fig` | `checkpoints/少样本/fewshot_ratio_vs_mse_avg.*` | 展示全步长平均误差随样本量增加的衰减趋势与最优收敛平台 |
| **图 3: 步长 vs MSE 折线图** | `.png`/`.pdf`/`.fig` | `checkpoints/少样本/fewshot_pred_len_vs_mse.*` | 横坐标预测步长 (96~720)，纵坐标 MSE，5 条少样本比例折线 |
| **图 4: 比例 vs MAE 折线图** | `.png`/`.pdf`/`.fig` | `checkpoints/少样本/fewshot_ratio_vs_mae.*` | 横坐标训练比例，纵坐标 MAE，4 条预测步长折线 |

---

## 二、核心对比数据表

### 3 种子均值与标准差完整表 (Mean ± Std)
| 训练数据比例 | 步长 96 (MSE / MAE) | 步长 192 (MSE / MAE) | 步长 336 (MSE / MAE) | 步长 720 (MSE / MAE) | 4 步长综合均值 MSE |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **10%** | 0.2003 ± 0.0014<br>(0.2213 ± 0.0014) | 0.2369 ± 0.0114<br>(0.2488 ± 0.0068) | 0.2422 ± 0.0103<br>(0.2524 ± 0.0038) | 0.2515 ± 0.0006<br>(0.2591 ± 0.0018) | **0.2327** |
| **20%** | 0.1889 ± 0.0014<br>(0.2162 ± 0.0010) | 0.2152 ± 0.0037<br>(0.2387 ± 0.0013) | 0.2322 ± 0.0139<br>(0.2499 ± 0.0068) | 0.2385 ± 0.0147<br>(0.2550 ± 0.0069) | **0.2187** |
| **30%** | 0.1850 ± 0.0064<br>(0.2228 ± 0.0030) | 0.2005 ± 0.0063<br>(0.2385 ± 0.0031) | 0.2239 ± 0.0024<br>(0.2516 ± 0.0015) | 0.2324 ± 0.0125<br>(0.2563 ± 0.0049) | **0.2104** |
| **40%** | 0.1852 ± 0.0016<br>(0.2255 ± 0.0011) | 0.2113 ± 0.0019<br>(0.2459 ± 0.0015) | 0.2237 ± 0.0081<br>(0.2585 ± 0.0021) | 0.2325 ± 0.0069<br>(0.2632 ± 0.0038) | **0.2132** |
| **50%** | **0.1829 ± 0.0023**<br>(0.2286 ± 0.0016) | 0.2081 ± 0.0019<br>(0.2504 ± 0.0021) | 0.2366 ± 0.0073<br>(0.2731 ± 0.0018) | 0.2421 ± 0.0021<br>(0.2695 ± 0.0007) | **0.2174** |

---

## 三、复现与调用方法

### 1. MATLAB 脚本执行
```powershell
matlab -wait -nosplash -noFigureWindows -batch "addpath('checkpoints/少样本'); plot_fewshot;"
```

### 2. Python 脚本执行
```powershell
python checkpoints/少样本/plot_fewshot.py
```
均可全自动无损重绘并输出 300 DPI 学术级图表。
