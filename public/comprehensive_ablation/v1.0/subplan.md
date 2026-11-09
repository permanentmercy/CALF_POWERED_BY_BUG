# 综合消融实验 (基线、TQ单消融、周期单消融与双消融) 统合分析子计划 (Subplan v1.0)

本文档依照团队协作规范维护，规划对 CALF 模型的 4 种实验配置（基线、TQ 单消融、Cycle Queue 周期队列单消融以及双消融）进行系统性探究、交叉对比与统一图表总结。

---

## 一、背景与目标

1. **背景**：
   - 此前已分别独立跑通并完成了：
     - TQ 机制单消融实验（`use_tq=0`，4 步长 x 3 种子，共 12 组任务，归档于 `checkpoints/TQ消融/`）；
     - Cycle Queue 周期队列单消融实验（`use_cycle_prior=0`，4 步长 x 3 种子，共 12 组任务，归档于 `checkpoints/cycle消融/`）；
     - 完整 CALF 基线（With TQ, With Cycle Prior，4 步长 x 3 种子，基准 MSE: 0.1645 ~ 0.2065）；
     - **双消融实验**（同时移除 TQ 与 Cycle Queue，`use_tq=0` 且 `use_cycle_prior=0`，4 步长 x 3 种子，共 12 组任务，归档于 `checkpoints/双消融/`）。
   - 用户明确要求：“我执行了一个双消融实验，也并入综合消融绘图”。
2. **核心目标**：
   - 深入探究两项消融机制及其联合双消融在不同预测步长（96, 192, 336, 720）下的正交叠加效应与主导权收敛规律；
   - 统合 4 种配置的全量实验指标至单一规范化数据表；
   - 编写 MATLAB 与 Python 出版级学术绘图脚本，将基线、无 TQ、无周期先验、双消融在同一画面中整合展示（支持 1x2 双面板母图、单坐标轴 4 曲线对比图、三柱分组百分比柱状图）；
   - 严格落实用户偏好：折线连接处采用规范圆点 `'o'`、横坐标严格对齐刻度、彻底去除上下误差横杠、图例不遮挡。

---

## 二、工作任务分解

| 任务编号 | 任务名称 | 详细执行内容 | 预期产物 | 状态 |
| :---: | :---: | :--- | :--- | :---: |
| **T1** | 双消融数据解析与矩阵统合 | 读取 `checkpoints/双消融/batch_state.json` 提取 12 组任务指标，生成 `dual_ablation_raw_results.csv` 与 `dual_ablation_summary.csv`，更新 `comprehensive_ablation_summary.csv` | CSV 聚合与明细表 | **已完成** |
| **T2** | 机制交叉探究与 2x2 因子解析 | 探究短期预测下两机制的正交叠加效应（双消融 +12.34%）与长程预测下周期先验的主导收敛规律 | `public/comprehensive_ablation/v1.0/execution_results.md` | **已完成** |
| **T3** | 学术绘图脚本升级 | 编写升级版 MATLAB (`plot_comprehensive_ablation.m`) 与 Python (`plot_comprehensive_ablation.py`) 脚本，支持 4 曲线与三柱并列 | 脚本文件保存在 `checkpoints/综合消融/` | **已完成** |
| **T4** | 出版级图表渲染与导出 | 渲染并导出 300 DPI PNG、矢量 PDF 及交互式 FIG 文件（双面板母图、4 曲线图、三柱分组柱状图） | 6 份图表产物 | **已完成** |
| **T5** | 团队文档与主计划同步 | 完善执行结果文档与 master plan，提交 git 远端 | `public/` 体系更新与 git push | **进行中** |
