# 子阶段计划：P1 训练期 TQ 预测增益配对重训练（v1.0 / phase2）

> **最新状态：旧开关对照方案已撤下。** 用户明确要求不改动 `use_tq`，下文仅保留此前方案/测量以便追溯，不再执行剩余种子。当前有效范围见 [completion_plan.md](../completion_plan.md)。P1 启动入口已停用；当前未发现对应训练进程运行。

日期：2026-10-25。所属功能模块：`cycle_mechanism`；所属版本：`v1.0`。
关联主方案：[completion_plan.md](../completion_plan.md) 第 3 节（P1：最终预测收益的主实验）。

---

## 1. 实验背景与核心目标

在阶段 P0（[phase1/execution_results.md](../phase1/execution_results.md)）中，我们通过固定检查点上的系统相位扫描与置换证实了：
- 在已充分训练的模型中，推理期单纯改动 TQ 会显著剧烈改变文本分支的误差（错移 12h 恶化 +97.42%，置零恶化 +19.44%，随机采样恶化 >+150%）；
- 但因其在前向结构中仅注入文本分支并前向单向蒸馏至时序分支，推理期的 TQ 干预不会直接改变时序分支 `outputs_time` 的瞬时数值（恒为 0.164986）。

因此，**TQ 对最终主时序预测 `outputs_time` 是否真正带来泛化增益，必须通过完整的从头训练（Training from Scratch）配对比较来检验**：
在多任务蒸馏机制（`task_w=0.736, feature_w=0.05, output_w=0.214`）作用下，文本分支学习到的 TQ 周期日内条件是否能通过中间特征和输出监督实质性反哺时序分支。

---

## 2. 严格受控实验设计

严格遵循用户明确约束：“只验证 TQ，不要做 Queue 的，Queue 保持相同开启配置”。

### 2.1 对照组定义

| 组别 | 标识符 | TQ 开关 (`use_tq`) | 词嵌入 WTE | 物理周期队列 (`cycle_queue`) | 训练协议 | 探索问题 |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **对照组 T0** | `tq0` | **0 (完全关闭)** | 固定原型 `wte_pca_500.pt` | **开启且完全一致** (144x137) | 从头训练 100 epochs, patience 5 | 不注入任何日内周期向量时的基线性能 |
| **实验组 T1** | `tq1` | **1 (完全开启)** | 固定原型 `wte_pca_500.pt` | **开启且完全一致** (144x137) | 从头训练 100 epochs, patience 5 | 加入可学习 TQ 周期表后的增量增益 |

### 2.2 配对初始化与参数对齐

- **数据划分**：严格遵循 Solar 官方基准 `7:1:2` 比例划分（训练集 36,792 步，验证集 5,256 步，测试集 10,512 步，对应测试评估 10,417 个全量滚动预测窗口）。
- **配对随机种子**：`[2025, 2026, 2027]` 严格成对执行，总计 **6 次全量训练与测试**。同一种子下 T0 与 T1 的共享模块（GPT-2 基础骨干、LoRA 初始化、Linear 投影、优化器状态初始化）完全一致。
- **超参数基准**（100% 对齐 `scripts/long_term_forecasting/Solar_test.sh` 官方复现配置）：
  - 架构：`seq_len=96, pred_len=96, enc_in=137, c_out=137`
  - 主干：`d_model=768, n_heads=4, e_layers=1, d_ff=3072, gpt_layers=1, r=8, lora_alpha=32`
  - 优化器：`batch_size=16, accumulation_steps=4 (有效批次 64), lr=0.0004, cosine decay, epochs=100, patience=5`
  - 蒸馏与损失：`task_w=0.736, feature_w=0.05, output_w=0.214, loss=smooth_l1`
  - 周期配置：`cycle=144, tq_mode=add, tq_inject_target=q, use_tq_gate=0, tq_dropout=0.1, tq_lr_factor=5`
  - 趋势与归一化：`use_norm=1, use_direct_trend=1 (ReZero)`
- **零磁盘空间约束 (Zero-Disk Footprint)**：启用 `--bestmodel` 内存缓存测试，测试结束后即刻自动清理大体积 `.pth` 检查点权重，仅保留轻量日志与指标 CSV。

---

## 3. 统计评估与验收准则

### 3.1 报告指标体系

1. **主终点 (Primary Endpoint)**：`outputs_time` MSE 与 MAE（以测试集 10,417 窗口的全量均方误差为第一判据）；
2. **辅助终点**：
   - 文本分支 `outputs_text` MSE 与 MAE；
   - **昼间活跃期 vs 夜间零值期细分误差**：依据日内物理分布（槽位 33-108 为太阳能高辐射活跃期，槽位 0-32 及 109-143 为完全零辐射夜间期）分段报告，排除夜间恒零对全天平均的掩盖；
   - 训练收敛 Epoch、早停触发点及耗时。

### 3.2 判定与证伪准则

- **确认支持**：跨 3 个配对随机种子，T1 的 Time MSE 均值及配对差值 $\Delta = \text{MSE}_{T1} - \text{MSE}_{T0} < 0$ 且方向高度稳定，尤其是昼间活跃期具备显著预测提升，方可确认 TQ 在该架构下具备实用预测收益。
- **证伪结论**：若 $\Delta \approx 0$（在波动范围内）或 $\Delta > 0$，或仅在文本分支有改善而时序主分支无收益，则需在学术上如实报告：**在已有物理周期队列（`cycle_queue`）存在的前提下，词表侧的时间查询向量（TQ）未对最终时序预测产生独立统计增益**。

---

## 4. 执行计划路线图

1. **阶段 2.1**：构建配对训练自动化控制脚本 `scripts/cycle_mechanism/run_p1_paired_training.ps1` 与度量收集器 `scripts/cycle_mechanism/p1_paired_runner.py`。
2. **阶段 2.2**：按种子顺序依次运行 3 对实验：
   - Seed 2025: T0 (`use_tq=0`) -> T1 (`use_tq=1`)
   - Seed 2026: T0 (`use_tq=0`) -> T1 (`use_tq=1`)
   - Seed 2027: T0 (`use_tq=0`) -> T1 (`use_tq=1`)
3. **阶段 2.3**：计算配对差异表、昼夜误差分解、置信区间分析，并将全部实测数据写入 [execution_results.md](execution_results.md)。
4. **阶段 2.4**：更新项目总进度主文档 [public/plan.md](../../../plan.md)。
