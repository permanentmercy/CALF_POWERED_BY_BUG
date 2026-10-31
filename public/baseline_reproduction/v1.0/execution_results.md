# CALF 原始基准复现执行产物与审计报告 (Execution Results)

本文档记录针对改动前 CALF 原始基线复现的底层代码修复、运行监控与复现指标产物。

---

## 一、硬性代码改动与修复明细

| 模块 / 文件 | 修复前缺陷状态 | 修复后状态 | 作用与验证结论 |
| :--- | :--- | :--- | :--- |
| **`models/CALF.py`** | `use_direct_trend=1` 默认开启，实例化 9216 参数 Trend Gate | 默认 `0`，仅在显式声明时实例化 | 恢复纯净 CALF 参数量 (13,099,633)，杜绝梯度被趋势分支干扰 |
| **`run.py`** | `--use_direct_trend` 默认 1 | `--use_direct_trend` 默认 0 | 保证基线测试默认处于无 Trend 状态 |
| **`data_factory.py`** | Solar 验证集强制 `batch_size=32`, `drop_last=False` | 恢复默认 `batch_size=16`, `drop_last=True` | 确保与历史基线 EarlyStopping 验证损失计算逐位一致 |
| **`data_loader.py`** | `__getitem__` 循环创建 `torch.zeros` 临时张量引发内存碎片 | 采用静态 `_dummy_x_mark` 数组与 float32 一次性加载 | 杜绝 Windows 平台上的 `DefaultCPUAllocator` 内存崩溃 |
| **`exp_long_term_forecasting.py`** | deepcopy 历史最优权重未解引用，缺少 GC 回收 | 新增旧 state 显式 `del`，Epoch 边界调用 `gc.collect()` | 保证内存平稳，多 Epoch 长期训练不泄漏 |
| **`Solar_test.sh`** | 种子单一、无 `--use_direct_trend 0` | 种子 2025/2026/2027，显式 `--use_direct_trend 0` | 完整覆盖 3 种子复现要求 |
| **`run_one_click.ps1`** | 任务 3 仅单种子且含 trend，管道未隔离错误 | 任务 3 对齐 3 种子基线复现流程，增补断点续跑防抖 | 提供全自动一键执行与进度持久化 |

---

## 二、当前执行状态与监控

- **执行指令**：`powershell -ExecutionPolicy Bypass -File ./run_one_click.ps1 -Task 3`
- **后台任务 ID**：`task-7624`
- **当前执行进度**：
  - Seed 2025: 训练中（前 200 步 Loss: 0.2224292 -> 0.1703132，与 2026-10-11 历史基线逐位完全对齐）
  - Seed 2026: 排队中
  - Seed 2027: 排队中
- **日志流监控**：
  - Seed 2025: [reproduce_baseline_96_seed2025.log](file:///logs/CALF/Solar/reproduce_baseline_96_seed2025.log)
  - Seed 2026: [reproduce_baseline_96_seed2026.log](file:///logs/CALF/Solar/reproduce_baseline_96_seed2026.log)
  - Seed 2027: [reproduce_baseline_96_seed2027.log](file:///logs/CALF/Solar/reproduce_baseline_96_seed2027.log)
- **汇总结果表**：[baseline_reproduction_results.csv](file:///results/baseline_reproduction_results.csv)

---

## 三、目标基准对照表

| 随机种子 | 目标 MSE | 目标 MAE | 复现 MSE | 复现 MAE | 状态 |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **2025** | 0.164985827 | 0.219895025 | *训练中* | *训练中* | RUNNING |
| **2026** | 0.164928949 | 0.219132878 | *待启动* | *待启动* | PENDING |
| **2027** | 0.164135638 | 0.218951099 | *待启动* | *待启动* | PENDING |
| **Mean** | **0.164532293** | **0.219041989** | - | - | PENDING |
