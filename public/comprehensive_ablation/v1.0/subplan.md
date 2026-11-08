# 综合双消融实验 (TQ 机制与 Cycle Queue 周期队列) 与基线统合分析子计划 (Subplan v1.0)

本文档依照团队协作规范维护，规划对 CALF 模型的两项核心机制消融实验（TQ 机制与 Cycle Queue 物理周期先验队列）与完整基线进行系统性探究、交叉对比与统一图表总结。

---

## 一、背景与目标

1. **背景**：
   - 此前已分别独立跑通并完成了：
     - TQ 机制消融实验（`use_tq=0`，4 步长 x 3 种子，共 12 组任务，归档于 `checkpoints/TQ消融/`）；
     - Cycle Queue 周期队列消融实验（`use_cycle_prior=0`，4 步长 x 3 种子，共 12 组任务，归档于 `checkpoints/cycle消融/`）；
     - 完整 CALF 基线（With TQ, With Cycle Prior，4 步长 x 3 种子，基准 MSE: 0.1645 ~ 0.2065）。
   - 用户明确要求：“探究两项消融实验然后和基线一起总结到一张图里”。
2. **核心目标**：
   - 深入探究两项消融机制在不同预测步长（96, 192, 336, 720）下的作用边界、敏感度规律与物理互补机理；
   - 提取并统合全量实验指标至单一规范化数据表；
   - 编写 MATLAB 与 Python 出版级学术绘图脚本，将基线、无 TQ、无周期先验在同一画面中整合展示（支持 1x2 双面板母图、单坐标轴 3 曲线对比图、分组百分比柱状图）；
   - 严格落实用户偏好：折线连接处采用规范圆点 `'o'`、横坐标严格对齐刻度、彻底去除上下误差横杠、图例不遮挡。

---

## 二、工作任务分解

| 任务编号 | 任务名称 | 详细执行内容 | 预期产物 | 状态 |
| :---: | :---: | :--- | :--- | :---: |
| **T1** | 数据矩阵统合与交叉计算 | 读取 TQ 消融与 Cycle 消融各步长指标，与基线进行对齐计算，计算绝对误差差值与同比百分改变 (\(\Delta\%\)) | `comprehensive_ablation_summary.csv`, `comprehensive_ablation_raw_data.csv` | **已完成** |
| **T2** | 机制交叉探究与原理解析 | 分析 TQ 机制的短期微观高频提示与 Cycle Queue 的长程宏观低频锚定协同机理 | `public/comprehensive_ablation/v1.0/execution_results.md` | **已完成** |
| **T3** | 学术绘图脚本实现 | 编写 MATLAB (`plot_comprehensive_ablation.m`) 与 Python (`plot_comprehensive_ablation.py`) 脚本 | 脚本文件保存在 `checkpoints/综合消融/` | **已完成** |
| **T4** | 出版级图表渲染与导出 | 渲染并导出 300 DPI PNG、矢量 PDF 及交互式 FIG 文件（综合母图、3 曲线图、分组柱状图） | 6 份图表产物 | **已完成** |
| **T5** | 团队文档与主计划同步 | 完善执行结果文档与 master plan，提交 git 远端 | `public/` 体系更新与 git push | **进行中** |
