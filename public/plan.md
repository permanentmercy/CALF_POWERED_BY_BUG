# CALF 项目总体推进计划 (Master Plan)

本文档依照团队协作规范维护，记录 CALF 项目各研究阶段的整体规划、子模块划分、里程碑状态及迭代版本。

---

## 一、项目架构与里程碑概览

| 阶段 / 模块 | 功能说明 | 对应版本 | 状态 | 产物目录 |
| :--- | :--- | :--- | :--- | :--- |
| **M1: 周期机制验证与基准复现** | 验证解耦周期残差先验机制，对齐 baseline 结果 | v1.0 | 已完成 | `figures/cycle_mechanism_validation/`, `latex/` |
| **M2: Solar-IL 极低样本微调** | 跨域迁移至 Solar-IL，完成 0.5% ~ 10% 微调 | v1.0 | 进行中 | `results/fewshot_il_micro_summary.csv`, `figures/fewshot/` |
| **M3: Solar-AL 4天超参敏感性基准** | 深度网格、容量网格、多任务蒸馏平衡与少样本衰减 | v1.0 | **已完成 (数据已恢复绘图，参数差异已审计对齐)** | `public/sensitivity_benchmark/v1.0/`, `figures/sensitivity/` |
| **M4: 全项目唯一一键启动控制脚本整合** | 消除多版本启动碎片，统合参数、断点续跑与出图 | v1.0 | **已完成** | [run_one_click.ps1](file:///run_one_click.ps1), `public/unified_runner/v1.0/` |
| **M5: 论文终稿整合与发表产物归档** | 整合图表、统计表与 LaTeX 论文 | v1.0 | 待启动 | `latex/` |

---

## 二、当前执行阶段：M4 唯一一键启动控制整合完成，支持用户本地微调与基准复现

- **背景与目标**：响应用户指令“一键启动的脚本留一个能跑的就行，不要那么多版本的”，将此前分散在根目录和 `scripts/` 的多个临时与冗余脚本安全归档至 `scripts/archive/`，在根目录打造全项目唯一的交互式+命令行双模启动入口 [run_one_click.ps1](file:///run_one_click.ps1)。
- **核心特性**：
  1. 默认直达用户正在进行的 **Solar-IL 少样本微调任务**（0.5% ~ 10% x 3种子，Split 7:1:2，关闭 Trend Gate，Zero-Disk Footprint 权重不落盘）；
  2. 覆盖 Solar-AL 4x4 深度/容量/蒸馏敏感性分析与最优基线复现（全参数 100% 对齐 `Solar_test.sh`）；
  3. 解耦日志解析与状态检查至 `scripts/runner_helper.py`，已通过实测验证（Exit Code 0）。
- **子计划与产物**：详见 [unified_runner/v1.0/subplan.md](unified_runner/v1.0/subplan.md) 与 [unified_runner/v1.0/execution_results.md](unified_runner/v1.0/execution_results.md)。
