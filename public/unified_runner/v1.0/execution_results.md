# 全项目唯一一键启动控制脚本落地成果报告 (v1.0)

本项目依照团队工程规范，已成功完成所有启动脚本的归并与重构。

---

## 一、改造前后结构对比

### 改造前 (散乱且易冲突)
```text
根目录:
  ├── reproduce_baseline_96.ps1
  ├── run_experiments_pipeline.ps1
  └── run_plots_after_training.ps1
scripts/:
  ├── run_4days_suite.ps1 / .sh
  ├── run_fewshot_il.ps1 / .sh
  ├── run_fewshot_il_micro.ps1 / .sh
  └── run_hyper_sensitivity.ps1 / .sh
```

### 改造后 (唯一清晰、职责单一)
```text
根目录:
  └── run_one_click.ps1          <-- 全项目唯一一键启动脚本 (交互式/命令行双模)
scripts/:
  ├── runner_helper.py           <-- 跨平台日志多编码安全嗅探与断点检查 helper
  ├── plot_sensitivity_results.py <-- AL 敏感性 4 图绘制管线
  ├── plot_fewshot_il_micro.py   <-- IL 少样本曲线绘制管线
  └── archive/                   <-- 历史与多版本脚本安全归档目录
```

---

## 二、功能验证测试

1. **语法与控制台编码验证**：
   - 为 [run_one_click.ps1](file:///run_one_click.ps1) 注入 UTF-8 BOM，并在脚本头显式绑定 `[Console]::OutputEncoding = UTF8`，根除了 Windows GBK 环境下的引号吞噬与重定向报错。
2. **执行验证 (Task 4 测试)**：
   - 执行 `powershell -ExecutionPolicy Bypass -File .\run_one_click.ps1 -Task 4`
   - 结果：**Exit Code 0**，顺利调用两个 Python 模块并渲染出：
     - `results/fewshot_il_micro_summary.csv`
     - `figures/fewshot/fewshot_il_micro_curve.png`
     - `figures/sensitivity/curve_fewshot_al_decay.png`
     - `figures/sensitivity/heatmap_depth_gpt_vs_mlp_al.png`
     - `figures/sensitivity/heatmap_capacity_lora_vs_dff_al.png`
     - `figures/sensitivity/heatmap_distillation_simplex_al.png`

---

## 三、用户使用指南

用户在终端仅需执行一条指令即可直达所需任务：

```powershell
# 方式 1: 直接回车交互启动 (默认即为 Solar-IL 少样本微调任务)
powershell -ExecutionPolicy Bypass -File .\run_one_click.ps1

# 方式 2: 命令行快捷直达
powershell -ExecutionPolicy Bypass -File .\run_one_click.ps1 -Task 1        # Solar-IL 极低样本微调
powershell -ExecutionPolicy Bypass -File .\run_one_click.ps1 -Task 2        # Solar-AL 超参敏感性
powershell -ExecutionPolicy Bypass -File .\run_one_click.ps1 -Task 3        # Solar-AL 基准复现
powershell -ExecutionPolicy Bypass -File .\run_one_click.ps1 -Task 4        # 仅绘图
```
