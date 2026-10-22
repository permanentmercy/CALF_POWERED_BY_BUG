# 子计划：Solar-AL 超参敏感性与少样本基准数据恢复与可视化 (v1.0)

本子计划针对计算机 B 历时 7 小时 50 分钟完成的 138 组实验进行数据解析修复与学术制图。

---

## 1. 问题定位与技术分析

1. **日志编码特性**：
   - Windows PowerShell 下的 `Tee-Object` 管道默认以 `UTF-16LE`（带 BOM）格式将终端输出重定向到 `.log` 文件。
   - 原 runner 脚本中的提取逻辑采用 `open(..., 'r', encoding='utf-8', errors='ignore')`。UTF-16LE 中的双字节编码在以 UTF-8 忽略错误方式读取时，ASCII 字符之间充斥 `\x00` 空字节（如 `m\x00s\x00e\x00:`），导致正则匹配失败并回退为 `NaN`。
2. **绘图脚本崩溃原因**：
   - `plot_sensitivity_results.py` 中的 `plot_distillation_simplex` 对 `df['test_mse']` 求 min/max。当数据中包含 `NaN` 时，`min_val` 计算为 `NaN`，在调用 Matplotlib `ax.set_ylim(min_val * 0.95, ...)` 时触发：
     ```text
     ValueError: Axis limits cannot be NaN or Inf
     ```
3. **共享基准锚点逻辑补齐**：
   - 敏感性分析脚本中，基准配置为 `gpt=1, mlp=3, r=8, dff=3072, task_w=0.70, feat_w=0.05, out_w=0.25`。
   - 该配置在 Module 1 (Depth) 已经执行（3 个种子），而在 Capacity 与 Distillation 中由于幂等去重跳过了重复训练，导致后两个模块的 CSV 缺少该锚点。需要在数据整合时将基准配置自动同步至 Capacity 和 Distillation，实现矩阵与单纯形配置的全覆盖。

---

## 2. 详细执行方案

| 序号 | 任务内容 | 实施方案 | 产物 |
| :--- | :--- | :--- | :--- |
| **T1** | 原始日志多编码自动嗅探与全量解析 | 开发 `scripts/recover_results_from_logs.py`，支持 `utf-16/utf-8/gbk/latin1` 自动解析 | `results/results_fewshot_al.csv`<br>`results/results_hyper_sensitivity_al.csv` |
| **T2** | 共享基准点数据对齐 | 将 `depth` 中的基准 3 种子实验复制对齐到 `capacity` 和 `distillation` | 保证 Capacity (16×3=48)、Distillation (11×3=33) 100% 完整 |
| **T3** | 绘图脚本防御性重构 | 修复 `plot_sensitivity_results.py` 中的 `set_ylim` 与 `dropna` 边界保护 | `plot_sensitivity_results.py` |
| **T4** | 自动化批量出图 | 执行高分辨率（300 DPI）绘图管线，输出 4 张学术级图表 | `figures/sensitivity/*.png` |
| **T5** | 统计汇总表格生成 | 输出 Mean ± Std 统计表，提取最优配置组合 | `results/*_summary.csv` |
| **T6** | 超参对齐审计与脚本同步 | 对比 `Solar_test.sh` 逐项核验，同步所有 runner 脚本配置 | `scripts/run_*.ps1`, `scripts/run_*.sh` |

---

## 3. 验收标准
1. `results_hyper_sensitivity_al.csv` 包含完整 129 条有效数值记录（Depth 48 + Capacity 48 + Distillation 33），`test_mse` 与 `test_mae` 0 NaN。
2. `results_fewshot_al.csv` 包含完整 15 条有效数值记录（5 比例 × 3 种子），0 NaN。
3. `figures/sensitivity/` 下 4 张高分辨率 PNG 全部成功生成且视觉无缺失。
4. 完成 `Solar_test.sh` 与敏感性分析脚本的超参比对，明确数值差异根因并完成脚本参数完全对齐。
