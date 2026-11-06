# 子计划：学习率敏感性扫描与多预测步长分析绘图 (Subplan v1.0)

本文档依照团队规范维护，定义 CALF 学习率超参敏感性实验（4 个预测步长 x 7 个学习率 x 3 个随机种子，共 84 组实验）的统计聚合与 MATLAB 学术绘图技术方案。

---

## 一、需求背景与目标

1. **实验规模与背景**：
   - 实验目标：评估学习率对 CALF 模型在不同预测视界（Horizon）下的收敛性与预测误差影响；
   - 实验网格：
     - 预测步长 `pred_len`: `[96, 192, 336, 720]`（4 个步长）；
     - 学习率 `learning_rate`: `[0.0002, 0.00025, 0.0003, 0.00035, 0.0004, 0.00045, 0.0005]`（7 档）；
     - 随机种子 `random_seed`: `[2025, 2026, 2027]`（3 种子）；
     - 任务总量：4 x 7 x 3 = 84 组任务全部完成。
2. **用户核心指令**：
   - “学习任务跑完了，现在创建脚本绘制图片”；
   - “三个随机数种子做平均作为一个点”；
   - “横坐标是学习率，纵坐标是mse”；
   - “四个不同预测步长用不同颜色的线画，画折线图，画在同一张图里”；
   - “脚本创建在checkpoints\学习率里，图片也存放在这里，用matlab写”。
3. **技术要求**：
   - 脚本必须基于 MATLAB 语言编写（`.m` 脚本），创建在 `checkpoints/学习率/` 目录下；
   - 具备高容错的三级数据载入通道（优先读取导出的统计汇总 CSV，备选解析 `batch_state.json`，底线支持内置完整基准矩阵）；
   - 输出统一规范的学术折线图，区分清晰的配色、标记符与线型，包含 300 DPI PNG、矢量 PDF 以及交互式 FIG；
   - 产物与代码严格存放在 `checkpoints/学习率/`。

---

## 二、实施方案与技术选型

### 1. 数据统计与导出
- 解析 `checkpoints/学习率/batch_state.json` 中的 84 组已完成任务；
- 导出原始指标明细表：`checkpoints/学习率/learning_rate_raw_results.csv`；
- 导出 3 种子均值与标准差聚合表：`checkpoints/学习率/learning_rate_summary.csv`（28 组均值点）。

### 2. MATLAB 脚本设计 (`checkpoints/学习率/plot_lr_vs_mse.m`)
- **坐标映射**：
  - X 轴：`learning_rate` ($2.0 \times 10^{-4} \sim 5.0 \times 10^{-4}$)；
  - Y 轴：`Mean Squared Error (MSE)`；
- **视觉映射**：
  - Horizon 96: 经典深蓝 (`#1f77b4`), 圆形标记 `'o'`;
  - Horizon 192: 珊瑚橙 (`#d95f02`), 方块标记 `'s'`;
  - Horizon 336: 翡翠绿 (`#2ca02c`), 三角标记 `'^'`;
  - Horizon 720: 宝石红 (`#d62728`), 菱形标记 `'d'`;
- **标注基准**：虚线标出 CALF 默认基准学习率 `lr = 0.00040`；
- **排版规范**：采用 Times New Roman 字体，精细虚线网格，透明图例面板。

---

## 三、验收标准

1. 84 组实验全量提取无缺失，均值计算精确。
2. MATLAB 脚本在 `checkpoints/学习率/plot_lr_vs_mse.m` 下可通过命令行 `matlab -batch "run(...)"` 无头执行通过。
3. 同目录下成功生成 `learning_rate_vs_mse.png` (300 DPI)、`learning_rate_vs_mse.pdf`、`learning_rate_vs_mse.fig`。
