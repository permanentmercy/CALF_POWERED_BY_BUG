# 外部基线模型（CycleNet × TQNet）全量调参及少样本跨模型对比评测子计划 (Subplan v1.0)

本文档隶属于 CALF 项目总体推进计划（`public/plan.md`）中的外部基准对比模块（M7: 外部基线跨模型评测与超参搜索全景）。

---

## 一、模块背景与核心目标

在光伏时序预测任务中，CALF 作为基于预训练大语言模型（GPT-2）与多尺度周期先验的 SOTA 架构，需要与当前时序领域的顶尖非大模型架构进行严格的横向基准对比。
外部基准模型包括：
1. **CycleNet**（ICLR 2024 / IEEE 顶刊候选）：包含 `linear`（周期线性分解）与 `mlp`（周期多层感知机）两种变体；
2. **TQNet**：具有时间提示与周期自适应能力的时序基准。

本模块的核心目标为：
1. **数据跨仓迁移与审计归档**：从外部实验仓库 `D:\github\CycleNet` 与 `D:\github\TQNet` 中完整提取 6 大阶段的超参搜索记录（共 149 组、454 次训练实验）以及极端少样本（10%, 20%, 30%, 40%, 50%, 100%）跨模型评测数据，迁入本地 `checkpoints/外部基线/` 进行永久归档；
2. **少样本极值泛化对比评测**：系统评估大模型架构（CALF）与非大模型时序网络（TQNet、CycleNet-mlp）在 10%~50% 极低训练样本下的抗退化能力与绝对性能优势；
3. **超参数搜索动态规律全景**：解析学习率与批大小在不同地域电站数据集（Solar-AL 与 Solar-IL）上的分布漂移（Distribution Shift）及最优有效学习率规律；
4. **出版级双栈绘图与复现**：提供标准化 Python（`matplotlib`）与 MATLAB（`.m`）双套绘图脚本，统一输出 300 DPI 高清 PNG、矢量 PDF 以及可交互编辑的 MATLAB `.fig` 原生工程文件。

---

## 二、评测任务矩阵与参数空间

### 1. 少样本跨模型评测任务矩阵（Solar-IL 数据集，106 电站）
- **评测模型**：CALF (Ours), TQNet, CycleNet-mlp
- **评测比例梯度**：`10%, 20%, 30%, 40%, 50%, 100%`
- **重复种子**：`2025, 2026, 2027`（三随机种子取均值与方差）
- **核心观测指标**：
  - 测试集均方误差（Test Mean MSE）与标准差（SD）
  - 相对各模型全量（100%）基线的退化率（Degradation %）：$\frac{\text{MSE}_{ratio} - \text{MSE}_{100\%}}{\text{MSE}_{100\%}} \times 100\%$
  - CALF 相对基线模型的超额性能优势（Advantage Margin %）：$\frac{\text{MSE}_{baseline} - \text{MSE}_{CALF}}{\text{MSE}_{CALF}} \times 100\%$

### 2. 全量超参数搜索覆盖矩阵（6 大阶段）
| 阶段 | 模型 | 变体 | 数据集 | 电站数 | 搜索组数 | 训练次数 | 基准配置 (lr/bs) | 基准 MSE | 最优配置 (lr/bs) | 最优 MSE | 优化幅度 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Phase 1 | TQNet | standard | Solar-AL | 137 | 17 | 52 | 0.003 / 64 | 0.176988 | **0.005 / 64** | **0.175281** | -0.96% |
| Phase 2 | TQNet | standard | Solar-IL | 106 | 32 | 97 | 0.003 / 64 | 0.187648 | **0.0006 / 64** | **0.177982** | -5.15% |
| Phase 3 | CycleNet | linear | Solar-AL | 137 | 15 | 45 | 0.01 / 64 | 0.209537 | **0.0045 / 64** | **0.207652** | -0.90% |
| Phase 4 | CycleNet | linear | Solar-IL | 106 | 20 | 65 | 0.01 / 64 | 0.253454 | **0.00138 / 64** | **0.220972** | -12.82% |
| Phase 5 | CycleNet | mlp | Solar-AL | 137 | 42 | 126 | 0.01 / 64 | 0.193624 | **0.0054 / 256** | **0.180680** | -6.69% |
| Phase 6 | CycleNet | mlp | Solar-IL | 106 | 23 | 69 | 0.01 / 64 | 0.193588 | **0.0025 / 128** | **0.193503** | -0.04% |

---

## 三、实施里程碑路线

1. **Step 1: 数据迁移与自动化解析**
   - 编写 `checkpoints/外部基线/extract_and_organize.py`，完整拷贝 `HP_SEARCH_SUMMARY.md`、`fewshot_compare_results.xlsx`、`fewshot_IL_results.xlsx` 及 6 阶段 CSV。
   - 解析导出标准化 CSV 表：`fewshot_cross_model_summary.csv`、`fewshot_cross_model_details.csv`、`hp_tuning_phases_summary.csv`、`hp_tuning_all_sweeps.csv`。
2. **Step 2: 学术图表规范设计与生成**
   - 遵从严谨视觉标准：折线图统一采用标准圆点（`'o'`），横坐标绝对对齐（10%~100%），禁止出现遮挡曲线的漂浮文本框，无误差棒横杠。
   - 输出 6 大出版级图表（Python 与 MATLAB 双栈）：
     - 图 1：少样本跨模型 MSE 综合收敛曲线图
     - 图 2：少样本极值衰减柱状图（相对各自 100% 基准退化率）
     - 图 3：CALF 相对外部基线的性能领先优势柱状图
     - 图 4：6 大超参搜索阶段基线 vs 最优对比柱状图
     - 图 5：学习率敏感性曲线与地域数据集分布漂移分析
     - 图 6：四合一出版级全景母图（Master Dashboard）
3. **Step 3: 团队文档与版本归档**
   - 建立 `public/external_benchmarks/v1.0/`，编写子计划与执行产物总结。
   - 更新主计划 `public/plan.md`，同步维护状态与引用索引。
