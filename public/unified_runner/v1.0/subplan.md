# 子计划：全项目统一一键启动脚本架构整合 (v1.0)

本文档规划 CALF 项目启动控制管线的整合方案，彻底解决多版本启动脚本碎片化、配置不一致与不易维护的问题。

---

## 1. 目标与背景

- **背景**：前期由于多次迭代实验，根目录与 `scripts/` 下分散了多个零碎启动脚本（如 `reproduce_baseline_96.ps1`, `run_experiments_pipeline.ps1`, `run_fewshot_il.ps1`, `run_fewshot_il_micro.ps1`, `run_hyper_sensitivity.ps1`, `run_4days_suite.ps1` 等），存在版本冗余、参数未完全对齐 `Solar_test.sh` 及用户操作繁复的问题。
- **目标**：
  1. 全项目只保留 **唯一一个** 一键启动入口：根目录下的 [run_one_click.ps1](file:///run_one_click.ps1)。
  2. 将零散过往脚本安全归档至 [scripts/archive/](file:///scripts/archive/)，保持代码树整洁。
  3. 支持命令行指定任务与无参数交互式双模运行，默认直达用户正在进行的 Solar-IL 少样本微调任务（0.5% ~ 10%）。
  4. 解耦辅助逻辑至 [scripts/runner_helper.py](file:///scripts/runner_helper.py)，彻底杜绝 PowerShell Here-String 编码转义与重定向冲突。

---

## 2. 实施方案与功能矩阵

| 模块 / 任务 | 功能定义 | 核心参数规范 | 产物输出 |
| :--- | :--- | :--- | :--- |
| **Task 1: Solar-IL 极低样本微调 (默认)** | 0.5%, 1%, 2%, 3%, 4%, 5%, 10% x 3种子微调 | `solar_IL.csv`, split 7:1:2, lr=0.0004, accum=4, smooth_l1, use_direct_trend=0, bestmodel | `results/fewshot_il_micro_summary.csv`<br>`figures/fewshot/fewshot_il_micro_curve.png` |
| **Task 2: Solar-AL 超参敏感性** | 4x4 深度、4x4 容量、11 元蒸馏平衡 | `solar_AL.txt`, 对齐 `Solar_test.sh` 全套最优参数, 零磁盘占用 | `results/results_hyper_sensitivity_al.csv`<br>`figures/sensitivity/*.png` |
| **Task 3: Solar-AL 最优基线复现** | 复现 Horizon 96 基准模型 (Seed 2025) | 对齐 `Solar_test.sh`，目标 MSE ~ 0.165 | `logs/CALF/Solar/reproduce_baseline_96_seed2025.log` |
| **Task 4: 学术可视化图表重绘** | 自动提取现有 CSV，重新渲染 300 DPI 图表 | 支持 IL 衰减曲线与 AL 4 张全量图表 | `figures/**/*.png` |

---

## 3. 验收标准
1. 根目录下无多余杂乱 `.ps1` 脚本，仅保留 `run_one_click.ps1`。
2. 历史脚本安全归档至 `scripts/archive/`。
3. `run_one_click.ps1` 支持 `-Task 1/2/3/4` 与无参数直接回车运行，零语法报错。
