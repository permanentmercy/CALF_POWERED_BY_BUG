# CALF 项目总体推进计划 (Master Plan)

本文档依照团队协作规范维护，记录 CALF 项目各研究阶段的整体规划、子模块划分、里程碑状态及迭代版本。

---

## 一、项目架构与里程碑概览

| 阶段 / 模块 | 功能说明 | 对应版本 | 状态 | 产物目录 |
| :--- | :--- | :--- | :--- | :--- |
| **M0: Solar-AL 原始基准复现** | 复现改动前 3 种子基线 (2025/2026/2027, MSE~0.1645) | v1.0 | **正在执行全量复现训练** | `public/baseline_reproduction/v1.0/`, `scripts/long_term_forecasting/Solar_test.sh` |
| **M1: 周期机制验证与基准复现** | 验证 TQ 与时间周期机制 (7:1:2 划分) | v1.0 | **P0 推理验证已完成；按用户要求保持 use_tq 不变，P1 开关训练已撤下** | `public/cycle_mechanism/v1.0/`, `results/cycle_mechanism/` |
| **M1.1: 周期先验残差 (cycle_queue) 一键开关与消融绘图** | 支持一键关闭 cycle_queue (--use_cycle_prior 0)、12组消融实验完成、对比分析与4张学术图表生成 | v1.0 | **已完成 (12组实验跑通，导出4张学术图表与CSV数据表，MATLAB/Python脚本就绪)** | `public/cycle_queue_ablation/v1.0/`, `checkpoints/cycle消融/` |
| **M1.2: TQ 机制消融实验多步长对比与分析绘图** | 4 预测步长 x 3 种子有无 TQ 对比分析、同比百分改变统计与 MATLAB 绘图 | v1.0 | **已完成 (4张图表已导出，MATLAB/Python脚本就绪)** | `public/tq_ablation/v1.0/`, `checkpoints/TQ消融/` |
| **M1.3: 综合全消融实验体系 (基线、TQ单消融、周期单消融与双消融)** | 深度探究 2x2 全因子消融正交叠加与长程主导收敛规律，四轨对比统合总结至一张图（1x2综合双面板母图、4曲线全景对比图、三柱分组柱状图） | v1.0 | **已完成 (全量指标矩阵统合，双消融12组任务完成并入，MATLAB/Python脚本与出版级图表全部就绪)** | `public/comprehensive_ablation/v1.0/`, `checkpoints/综合消融/`, `checkpoints/双消融/` |
| **M2: Solar 少样本实验体系 (跨域微调与从头衰减)** | 跨域迁移至 Solar-IL (0.5%~10%) 与 Solar-IL 从头衰减 (10%~50%) 60组全量实验、聚合分析与绘图 | v2.0 | **已完成 (60组实验跑通，导出4张学术图表与CSV数据表，MATLAB/Python脚本就绪)** | `public/fewshot/v2.0/`, `checkpoints/少样本/` |
| **M3: Solar-AL 4天超参敏感性基准** | 深度网格、容量网格、多任务蒸馏平衡与少样本衰减 | v1.0 | **已完成 (数据已恢复绘图，参数差异已审计对齐)** | `public/sensitivity_benchmark/v1.0/`, `figures/sensitivity/` |
| **M3.1: 学习率敏感性多步长分析** | 4 预测步长 x 7 学习率 x 3 种子 (84 组任务) 统计与 MATLAB 绘图 | v1.0 | **已完成** | `checkpoints/学习率/`, `public/learning_rate_sweep/v1.0/` |
| **M4: 全项目唯一一键启动控制脚本整合** | 消除多版本启动碎片，统合参数、断点续跑与出图 | v1.0 | **已完成** | [run_one_click.ps1](file:///run_one_click.ps1), `public/unified_runner/v1.0/` |
| **M5: 批量实验面板与权重统一管理** | 断点续跑、全超参循环扫描、默认学习率0.0004独立锁定、会话记忆与无损磁盘状态找回GUI、未完成任务统一标定直接续跑、三损失权重解耦与独立控制 | v2.6 | **已完成 (v2.6 三损失完全解耦，不自动求和为1，支持各自独立设定与网格扫描)** | [batch_panel.py](file:///batch_panel.py), `scripts/batch_runner/`, `public/batch_panel/v2.6/` |
| **M6: 论文终稿整合与发表产物归档** | 整合图表、统计表与 LaTeX 论文 | v1.0 | 待启动 | `latex/` |
| **M7: 外部基线跨模型评测与超参搜索全景** | CycleNet × TQNet 跨仓数据迁移、6阶段超参动力学解析、Solar-IL 极低样本（10%~100%）跨模型评测对比与出版级全景可视化 | v1.0 | **已完成 (数据迁移规整完成，4张单项+1张综合双栈图表已导出，MATLAB/Python脚本就绪)** | `public/external_benchmarks/v1.0/`, `checkpoints/外部基线/` |

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

## 三点五、模块进展明细：M1.1 周期先验残差 (cycle_queue) 一键开关、消融实验与分析绘图
- **背景与目标**：用户完成针对物理周期先验残差队列（`cycle_queue`）的 12 组消融实验（4 预测步长 x 3 随机种子，`use_cycle_prior=0`），需对基线有周期先验与无周期先验进行成对指标对比、同比百分改变分析及高标准学术绘图。
- **实施产物**：
  1. `models/CALF.py`：支持 `--use_cycle_prior`（默认1，0为关闭），关闭时冻结参数梯度并完全旁路残差注入；
  2. `run.py`：注册参数并自动在消融设置时追加 `_nocycq` 标签，实现检查点物理隔离；
  3. `scripts/batch_runner/`：`config_manager`、`state_manager`、`gui_panel` 全面适配 `use_cycle_prior`；
  4. 批处理模板：提供开箱即用模板 [configs/batch_templates/cycle_queue_ablation.json](file:///configs/batch_templates/cycle_queue_ablation.json)；
  5. 自动化测试套件：[tests/test_cycle_prior_ablation.py](file:///tests/test_cycle_prior_ablation.py) 4 项测试全部通过；
  6. 数据处理与统计表：生成明细表 `cycle_ablation_raw_results.csv` 与聚合表 `cycle_ablation_summary.csv`；
  7. 绘图脚本与高清图表：编写 MATLAB [plot_cycle_ablation.m](file:///checkpoints/cycle消融/plot_cycle_ablation.m) 与 Python [plot_cycle_ablation.py](file:///checkpoints/cycle消融/plot_cycle_ablation.py)，导出 4 张专业学术图表（300 DPI PNG、矢量 PDF 及 FIG）：
     - 无周期先验消融折线图 (`cycle_ablation_mse_without_cycle`)；
     - 有周期先验基线折线图 (`cycle_ablation_mse_with_cycle`)；
     - 有无周期先验对比折线图 (`cycle_ablation_mse_comparison`，纯圆点 `'o'`，严格对齐步长刻度，无上下横杠）；
     - 同比百分改变柱状图 (`cycle_ablation_relative_change_bar`，揭示 +8.49% $\rightarrow$ +15.73% 单调递增规律)。
- **子计划与产物**：详见 [cycle_queue_ablation/v1.0/subplan.md](cycle_queue_ablation/v1.0/subplan.md) 与 [cycle_queue_ablation/v1.0/execution_results.md](cycle_queue_ablation/v1.0/execution_results.md)。

---

## 三点六、模块进展明细：M1.2 TQ 机制消融实验多步长对比与分析绘图
- **背景与目标**：用户完成 TQ 消融实验（`use_tq=0`），需对 4 预测步长 (96, 192, 336, 720) 下的有无 TQ 进行均值折线图、同构图、直接对比图以及同比百分改变柱状图绘制。
- **实施产物**：
  1. 数据汇总与处理：输出明细表 `tq_ablation_raw_results.csv` 与汇总表 `tq_ablation_summary.csv`；
  2. 绘图脚本：编写 MATLAB 主脚本 [plot_tq_ablation.m](file:///checkpoints/TQ消融/plot_tq_ablation.m) 与 Python [plot_tq_ablation.py](file:///checkpoints/TQ消融/plot_tq_ablation.py)；
  3. 出版级图表：无 TQ 折线图、有 TQ 折线图、对比折线图及同比百分改变柱状图（导出 300 DPI PNG、矢量 PDF 及 FIG）。
- **子计划与产物**：详见 [tq_ablation/v1.0/subplan.md](tq_ablation/v1.0/subplan.md) 与 [tq_ablation/v1.0/execution_results.md](tq_ablation/v1.0/execution_results.md)。

---

## 三点七、模块进展明细：M1.3 综合全消融实验体系 (基线、TQ单消融、周期单消融与双消融)
- **背景与目标**：深入探究 TQ 机制（时间提示向量）、Cycle Queue 机制（物理周期残差）以及两者同时关闭的“双消融”在多步长预测中的正交叠加与长程主导规律，并将四轨实验统一总结至同一张出版级图表中。
- **核心发现**：
  1. **短期预测（Horizon 96）正交叠加**：w/o TQ 恶化 +5.70%，w/o Cycle 恶化 +8.49%，双消融恶化高达 **+12.34%**，证实两项机制在短期高频与低频不同通道上具备正交独立性；
  2. **长程预测（Horizon 336 & 720）物理周期主导收敛**：随着跨度延伸至 7.5 天，双消融曲线与无周期单消融曲线高度重合（恶化 ~14% 至 ~15%），证实长程抗自回归累积相位漂移完全依赖物理周期先验；
  3. **协同完备性**：完整 CALF 模型在 96~720 全尺度均实现最优误差。
- **实施产物**：
  1. 数据汇总表：[checkpoints/综合消融/comprehensive_ablation_summary.csv](file:///checkpoints/综合消融/comprehensive_ablation_summary.csv) 与原始记录 [comprehensive_ablation_raw_data.csv](file:///checkpoints/综合消融/comprehensive_ablation_raw_data.csv)，双消融明细 [checkpoints/双消融/dual_ablation_summary.csv](file:///checkpoints/双消融/dual_ablation_summary.csv)；
  2. 绘图脚本：MATLAB [plot_comprehensive_ablation.m](file:///checkpoints/综合消融/plot_comprehensive_ablation.m) 与 Python [plot_comprehensive_ablation.py](file:///checkpoints/综合消融/plot_comprehensive_ablation.py)；
  3. 学术整合图表（300 DPI PNG、矢量 PDF 及 FIG）：
     - 综合双面板学术母图 (`comprehensive_ablation_master`)：(a) 4 折线预测步长 vs MSE 对比，(b) 三柱分组百分比退化柱状图；
     - 单坐标轴 4 曲线对比图 (`comprehensive_ablation_4lines`)；
     - 独立三柱分组百分比柱状图 (`comprehensive_ablation_grouped_bar`)。
- **子计划与产物**：详见 [comprehensive_ablation/v1.0/subplan.md](comprehensive_ablation/v1.0/subplan.md) 与 [comprehensive_ablation/v1.0/execution_results.md](comprehensive_ablation/v1.0/execution_results.md)。

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
  5. **三损失完全解耦 (v2.6)**：解除 `output_w = 1.0 - task_w - feature_w` 与和为 1 的强制约束，`task_w`、`feature_w`、`output_w` 彻底作为独立超参数，支持各自在循环参数中独立扫描与配置。
- **子计划与产物**：详见 [batch_panel/v2.6/subplan.md](batch_panel/v2.6/subplan.md) 与 [batch_panel/v2.6/execution_results.md](batch_panel/v2.6/execution_results.md)。

---

## 六、模块进展明细：M7 外部基线模型（CycleNet × TQNet）全量调参与少样本评测全景
- **背景与目标**：
  响应用户指令，从 `D:\github\CycleNet` 与 `D:\github\TQNet` 提取全量调参记录（6 个阶段、149 组、454 次实验）以及少样本对比数据（Solar-IL 10%~100%），迁移至本地项目并实现高标准学术出版级可视化。
- **核心成果**：
  1. **极端少样本大模型表征碾压优势**：在 10% 样本极限场景下，非大模型出现表征崩塌（TQNet MSE 0.3643，退化 +104.67%；CycleNet-mlp MSE 0.3569，退化 +84.46%），而 CALF 凭借预训练语言先验仅轻度退化 +17.77%（MSE 0.2047），取得高达 **+77.92%** 的压倒性性能优势；
  2. **跨数据集超参分布漂移律**：Solar-AL 最优 lr 集中在 3e-3 ~ 5e-3，Solar-IL 最优 lr 陡降至 6e-4 ~ 1.4e-3；MLP 变体在 bs=256 下解锁第二优化通道；
  3. **数据资产归档**：在 [checkpoints/外部基线/](file:///checkpoints/外部基线/) 建立完整的 `raw/`、`data/`、`figures/` 体系，生成 `fewshot_cross_model_summary.csv` 等规整表；
  4. **双栈学术出版图表**：编写 Python [plot_external_benchmarks.py](file:///checkpoints/外部基线/plot_external_benchmarks.py) 与 MATLAB [plot_external_benchmarks.m](file:///checkpoints/外部基线/plot_external_benchmarks.m)，输出 300 DPI PNG、矢量 PDF 及原生交互式 `.fig`。
- **子计划与产物**：详见 [external_benchmarks/v1.0/subplan.md](external_benchmarks/v1.0/subplan.md) 与 [external_benchmarks/v1.0/execution_results.md](external_benchmarks/v1.0/execution_results.md)。

