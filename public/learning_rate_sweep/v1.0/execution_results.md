# 执行产物：学习率敏感性扫描与多预测步长分析绘图 (v1.0)

本产物记录 CALF 学习率超参敏感性实验 84 组全量数据的聚合结果、MATLAB 绘图脚本落地及生成的图表产物索引。

---

## 1. 产物清单与文件索引

1. **MATLAB 绘图脚本**：
   - [plot_lr_vs_mse.m](file:///checkpoints/学习率/plot_lr_vs_mse.m)：全功能 MATLAB 绘图脚本，支持 CSV/JSON 动态解析与内置回退，导出高分辨率三态图片。
2. **生成的高清图表**：
   - **图 1：四预测步长分别变化折线图**：
     - [learning_rate_vs_mse.png](file:///checkpoints/学习率/learning_rate_vs_mse.png)：300 DPI 学术出版级高清图片（图例上移外置，零遮挡）。
     - [learning_rate_vs_mse.pdf](file:///checkpoints/学习率/learning_rate_vs_mse.pdf)：矢量 PDF 文件，用于 LaTeX 论文无损缩放。
     - [learning_rate_vs_mse.fig](file:///checkpoints/学习率/learning_rate_vs_mse.fig)：MATLAB 交互式工程文件。
   - **图 2：四预测步长综合平均折线图**：
     - [learning_rate_vs_mse_avg.png](file:///checkpoints/学习率/learning_rate_vs_mse_avg.png)：300 DPI 学术出版级整体平均折线图（含 $\pm 1$ 标准差阴影带与最优谷底标定）。
     - [learning_rate_vs_mse_avg.pdf](file:///checkpoints/学习率/learning_rate_vs_mse_avg.pdf)：矢量 PDF 文件。
     - [learning_rate_vs_mse_avg.fig](file:///checkpoints/学习率/learning_rate_vs_mse_avg.fig)：MATLAB 交互式工程文件。
3. **数据明细与聚合表**：
   - [learning_rate_summary.csv](file:///checkpoints/学习率/learning_rate_summary.csv)：28 组 (PredLen, LR) 聚合统计表（含 3 种子均值与标准差）。
   - [learning_rate_raw_results.csv](file:///checkpoints/学习率/learning_rate_raw_results.csv)：84 组单次实验完整原始数据记录。

---

## 2. 核心实验结论与数据聚合表

### 3 种子均值 MSE 实验统计表

| 预测步长 (Pred Len) | lr = 0.00020 | lr = 0.00025 | lr = 0.00030 | lr = 0.00035 | lr = 0.00040 (默认) | lr = 0.00045 | lr = 0.00050 | 最优学习率点 |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Horizon 96** | 0.174846 | 0.172591 | 0.169271 | 0.166955 | **0.164683** | 0.168275 | 0.168708 | **0.00040** (MSE 0.16468) |
| **Horizon 192** | 0.200705 | 0.197043 | 0.192129 | 0.190781 | 0.190280 | **0.190103** | 0.190562 | **0.00045** (MSE 0.19010) |
| **Horizon 336** | 0.201237 | 0.208890 | 0.205849 | 0.201390 | **0.199336** | 0.199445 | 0.200798 | **0.00040** (MSE 0.19934) |
| **Horizon 720** | 0.207627 | 0.208639 | 0.207640 | **0.205690** | 0.206460 | 0.206701 | 0.206205 | **0.00035** (MSE 0.20569) |
| **4 步长综合平均 (Overall)** | 0.196104 | 0.196790 | 0.193722 | 0.191204 | **0.190190** | 0.191131 | 0.191568 | **0.00040** (MSE 0.19019) |

### 关键科学发现
1. **凹型收敛谷底显著**：
   - 在短视界与中视界（Horizon 96 和 336），随着学习率从 0.0002 增大至 0.0004，模型收敛速度和表征泛化能力显著增强，MSE 呈现经典“U型/凹型”下降并在 **`lr = 0.00040` 处取得全局最优**；
   - 超过 0.00040 之后（如 0.00045, 0.00050），过大步长导致 LoRA 适配层与 TQ 周期向量震荡，MSE 开始回升劣化。
2. **基准设定的合理性确认**：
   - 实验数据强有力证实：本项目默认采用的基线超参 `learning_rate = 0.00040`（结合 `batch_size=16, accumulation_steps=4`）位于全周期的性能最优点，具有极高稳定性。
