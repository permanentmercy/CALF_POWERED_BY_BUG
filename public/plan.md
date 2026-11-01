# CALF 项目总体推进计划 (Master Plan)

本文档依照团队协作规范维护，记录 CALF 项目各研究阶段的整体规划、子模块划分、里程碑状态及迭代版本。

---

## 一、项目架构与里程碑概览

| 阶段 / 模块 | 功能说明 | 对应版本 | 状态 | 产物目录 |
| :--- | :--- | :--- | :--- | :--- |
| **M0: Solar-AL 原始基准复现** | 复现改动前 3 种子基线 (2025/2026/2027, MSE~0.1645) | v1.0 | **正在执行全量复现训练** | `public/baseline_reproduction/v1.0/`, `scripts/long_term_forecasting/Solar_test.sh` |
| **M1: 周期机制验证与基准复现** | 验证 TQ 与时间周期机制 (7:1:2 划分) | v1.0 | **P0 推理验证已完成；按用户要求保持 use_tq 不变，P1 开关训练已撤下** | `public/cycle_mechanism/v1.0/`, `results/cycle_mechanism/` |
| **M2: Solar-IL 极低样本微调** | 跨域迁移至 Solar-IL，完成 0.5% ~ 10% 微调 | v1.0 | 进行中 | `results/fewshot_il_micro_summary.csv`, `figures/fewshot/` |
| **M3: Solar-AL 4天超参敏感性基准** | 深度网格、容量网格、多任务蒸馏平衡与少样本衰减 | v1.0 | **已完成 (数据已恢复绘图，参数差异已审计对齐)** | `public/sensitivity_benchmark/v1.0/`, `figures/sensitivity/` |
| **M4: 全项目唯一一键启动控制脚本整合** | 消除多版本启动碎片，统合参数、断点续跑与出图 | v1.0 | **已完成** | [run_one_click.ps1](file:///run_one_click.ps1), `public/unified_runner/v1.0/` |
| **M5: 批量实验面板与权重统一管理** | 断点续跑、全超参循环扫描、默认学习率0.0004独立锁定、会话记忆与无损磁盘状态找回GUI | v2.3 | **已完成 (v2.3 解决重开窗口进度丢失，引入多级对齐与物理磁盘探测)** | [batch_panel.py](file:///batch_panel.py), `scripts/batch_runner/`, `public/batch_panel/v2.3/` |
| **M6: 论文终稿整合与发表产物归档** | 整合图表、统计表与 LaTeX 论文 | v1.0 | 待启动 | `latex/` |

---

## 二、模块进展明细：M0 Solar-AL 原始基线复现与脚本修复
- **目标数据**：
  - Seed 2025: MSE 0.164985827 / MAE 0.219895025
  - Seed 2026: MSE 0.164928949 / MAE 0.219132878
  - Seed 2027: MSE 0.164135638 / MAE 0.218951099
  - 均值: MSE 0.164532293 / MAE 0.219041989
- **审计根因与修复**：
  1. `use_direct_trend` 默认开启（新增 9,216 参数 Trend Gate，导致指标劣化至 0.16587+）：已在 `models/CALF.py` 与 `run.py` 彻底封锁并默认置 0。
  2. 验证集 DataLoader 曾被篡改为 `batch_size=32, drop_last=False`：已恢复默认 `batch_size=16, drop_last=True`，保证 EarlyStopping 损失计算完全对齐。
  3. `data_loader.py` 在 `__getitem__` 中循环创建张量导致内存碎片：已改为静态数组单例，并在 DataLoader 读取阶段转为 float32，添加 Epoch 边界 GC 回收，杜绝 `DefaultCPUAllocator` 内存溢出。
  4. 脚本对齐：`Solar_test.sh` 与 `run_one_click.ps1` (Task 3) 已补齐 3 种子、`--bestmodel`、`--data_split 7:1:2`，清除死锁标记。
- **当前状态**：后台调度已正式启动执行（当前正在运行 Seed 2025）。
- **子计划与产物**：详见 [baseline_reproduction/v1.0/subplan.md](baseline_reproduction/v1.0/subplan.md) 与 [baseline_reproduction/v1.0/execution_results.md](baseline_reproduction/v1.0/execution_results.md)。

---

## 三、模块进展明细：M1 TQ 周期向量验证 (7:1:2 划分)
- **核心权重**：采用 7:1:2 充分训练的最优基线检查点 [checkpoint.pth](file:///checkpoints/long_term_forecast_CALF_96_96_p100_CALF_Solar_ftM_sl96_ll0_pl96_dm768_nh4_el1_dl1_df3072_fc1_ebtimeF_dtTrue_test_gpt1_rl0.0004_2025_0_feature0.05_output0.214_train_epochs100_bs16_dr0.1_ld0.1_r8/checkpoint.pth)（基线 MSE 0.164986）、词嵌入原型 [wte_pca_500.pt](file:///wte_pca_500.pt) 与底座语言模型 [models/gpt2](file:///models/gpt2)。
- **已观察到的证据**：现有 CSV 记录全量测试集（10,417 个重叠窗口）的 7 组训练后干预；TQ 错移 12h 使文本分支 MSE 增加 63.89%，高斯替换 WTE 增加 77.13%。这些结果支持文本分支的条件/字典依赖性；7 组时序分支 MSE 均为 0.164986，尚不能证明加入模块提高最终预测效果或语言语义不可替代。
- **2026-10-25 审查结果**：检查点复算的 12h TQ 相似度为 +0.005782，最低点在 9.5h/14.5h，原报告的强反相拓扑描述不成立；相邻向量存在微弱有序性。已完成 [实验完善性审查](cycle_mechanism/v1.0/experiment_review.md)，并形成 [补充实验方案](cycle_mechanism/v1.0/completion_plan.md)。本次用户要求只验证 TQ，P0 已完成验证扫描和单偏移测试确认；后续保持 `use_tq` 原值，使用已完成的固定检查点 TQ 响应证据；不再安排开关配对训练。
- **子计划与产物**：详见 [cycle_mechanism/v1.0/subplan.md](cycle_mechanism/v1.0/subplan.md) 与 [cycle_mechanism/v1.0/execution_results.md](cycle_mechanism/v1.0/execution_results.md)。
- **阶段 1 完成状态 (P0)**：P0 验证集 5,161 窗口扫描、+2 槽单项测试确认及配对区间核对均已完成。逐值核查表明 Solar CSV 不能提供 TXT 行号的可信时间戳。执行详情见 [cycle_mechanism/v1.0/phase1/execution_results.md](cycle_mechanism/v1.0/phase1/execution_results.md)。
- **阶段 2 范围修正 (P1)**：用户明确要求“不要改动 `use_tq`”，原开关配对训练方案已撤下。已完成的 seed2025 两组结果保留为历史记录，剩余种子不再安排；旧 P1 启动入口已停用。当前有效方案见 [TQ 验证方案](cycle_mechanism/v1.0/completion_plan.md)，不修改模型开关，也不开展 Queue/WTE 消融。

---

## 四、当前执行阶段：M4 唯一一键启动控制整合完成，支持用户本地微调与基准复现

- **背景与目标**：响应用户指令“一键启动的脚本留一个能跑的就行，不要那么多版本的”，将此前分散在根目录和 `scripts/` 的多个临时与冗余脚本安全归档至 `scripts/archive/`，在根目录打造全项目唯一的交互式+命令行双模启动入口 [run_one_click.ps1](file:///run_one_click.ps1)。
- **核心特性**：
  1. 默认直达用户正在进行的 **Solar-IL 少样本微调任务**（0.5% ~ 10% x 3种子，Split 7:1:2，关闭 Trend Gate，Zero-Disk Footprint 权重不落盘）；
  2. 覆盖 Solar-AL 4x4 深度/容量/蒸馏敏感性分析与最优基线复现（全参数 100% 对齐 `Solar_test.sh`）；
  3. 解耦日志解析与状态检查至 `scripts/runner_helper.py`，已通过实测验证（Exit Code 0）。
- **子计划与产物**：详见 [unified_runner/v1.0/subplan.md](unified_runner/v1.0/subplan.md) 与 [unified_runner/v1.0/execution_results.md](unified_runner/v1.0/execution_results.md)。

---

## 五、模块进展明细：M5 批量实验控制面板与权重统一管理
- **背景与目标**：
  响应用户指令，开发支持全超参调整、循环网格搜索、断点续跑、实验大文件夹统一管理及日志防冲突隔离的批量运行控制面板。
- **核心功能**：
  1. **签名防冲突**：`run.py` 补充 `--exp_tag`，并在 `task_w`、`cycle`、`use_direct_trend`、`out_mlp_layers`、`tq_mode`、`data_split` 发生变化时自动注入区分标签，杜绝日志与检查点覆盖。
  2. **统一大文件夹收敛**：解决 `exp` 脚本中 `'./checkpoints/'` 硬编码，用户指定大文件夹后，该批次所有模型的权重自动收敛至指定目录。
  3. **持久化断点续跑**：维护 `batch_state.json`，中断后自动识别并跳过已完成项，继续执行未完成实验。
  4. **全超参循环支持**：支持以 JSON 模板或交互方式定义参数列表，自动生成笛卡尔积网格扫描。
- **子计划与产物**：详见 [batch_panel/v1.0/subplan.md](batch_panel/v1.0/subplan.md) 与 [batch_panel/v1.0/execution_results.md](batch_panel/v1.0/execution_results.md)。
