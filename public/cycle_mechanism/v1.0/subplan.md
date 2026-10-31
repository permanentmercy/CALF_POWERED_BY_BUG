# 子计划：词嵌入分支时间周期建模机制严谨实证与物理意义分析 (v1.0)

本文档规划 CALF 双分支跨模态架构中**词嵌入分支（Text Branch）与时间查询向量（Temporal Query, TQ）物理周期建模机制**的实证方案。

> 最新研究范围：保持 `use_tq` 现值，不通过开关消融进行重训练，也不安排 Queue/WTE 实验。下面保留初始方案作为历史记录；当前以 [有效 TQ 方案](completion_plan.md) 为准，已完成结果见 [phase1/execution_results.md](phase1/execution_results.md)。

> 2026-10-25 审查已完成：原有 7 组实验属于训练后干预，尚不足以完成本页全部假说的验证。本轮在现有 v1.0 内审查证据，不新开代码版本；结论见 [experiment_review.md](experiment_review.md)，待执行补实验见 [completion_plan.md](completion_plan.md)。P0 阶段的计划与本轮执行记录见 [phase1/subplan.md](phase1/subplan.md) 和 [phase1/execution_results.md](phase1/execution_results.md)。

---

## 一、研究背景与核心科学问题

### 1. 架构机理背景
CALF 采用独特的双分支跨模态蒸馏时序预测架构：
- **时序分支（Time Branch）**：负责接收规范化时序特征并叠加物理周期队列（`cycle_queue`）先验，在训练期通过知识蒸馏损失（$\mathcal{L}_{\text{feat}}$ 与 $\mathcal{L}_{\text{out}}$）汲取文本分支的隐层语义。
- **跨模态文本分支（Text Branch）**：搭载从 GPT-2 词表矩阵转置后经 PCA 变换得到的 $500 \times 768$ 固定原型字典（`word_embedding_base`）。当前基线为 `q/add`：144 槽可学习 TQ 按输入末点相位查表，加到时序 Q；PCA 字典提供 K/V。模型也支持其他注入模式，解释应以实际配置为准。

### 2. 待验证的科学假说（Hypotheses）
- **假说 1（周期物理驱动性 vs. 静态 LoRA 伪效应）**：时间查询向量 $TQ \in \mathbb{R}^{144 \times 768}$ 是否真正与太阳时角（Solar Hour Angle, 24h 周期）形成了确定性物理相位绑定，而非仅作为 144 组可自由拟合的类似 LoRA 的静态微调参数？
- **假说 2（预训练语义原型必要性）**：500 个文本词嵌入原型是否为跨模态时序迁移提供了不可替代的高维特征空间？如果将其置零或用高斯噪声替换，模型是否会发生结构性崩溃？
- **假说 3（昼夜相位极化解耦）**：正午高光照时段与夜间零光照时段在词嵌入原型上的交叉注意力激活是否呈现物理一致的双峰分离与专业化分工？

---

## 二、实验环境与权重配置规范

严格对齐项目主规范，采用标准 **7:1:2** 划分及官方复现基线检查点：

| 配置项 | 规范取值 | 说明 |
| :--- | :--- | :--- |
| **数据集** | `solar_AL.txt` (Solar 137 通道) | 太阳能高维多变量辐射数据集 |
| **数据划分** | **`7:1:2`** (`--data_split 7:1:2`) | 标准规范划分，测试集时序窗口 10,417 个 |
| **输入/预测步长** | $96 \to 96$ | 16 小时时序上下文预测未来 16 小时 |
| **核心模型检查点** | `./checkpoints/.../checkpoint.pth` (p100 基线) | 在 7:1:2 划分下训练收敛的最优基线模型 (MSE ~ 0.165) |
| **词嵌入原型文件** | `./wte_pca_500.pt` | GPT-2 词表转置后经 PCA 得到的 $500 \times 768$ 原型字典，语义归因待验证 |
| **底座大语言模型** | `./models/gpt2` | 本地 GPT-2 预训练权重 (保留 `gpt_layers=1`) |

---

## 三、实验设计方案（7 组消融与逆向扰动）

在全量测试集（10,417 样本）上运行以下 7 组对照实验，量化记录文本分支与时序分支的 MSE / MAE：

1. **Clean Baseline**：标准模型正常前向推理，获取未扰动的基准性能。
2. **TQ Phase Invert (+12h)**：索引平移半周期（`(cycle_index + 72) % 144`），测试相位错配敏感性；真实钟点标签需由数据元信息确认。
3. **TQ Random Permute（实现为随机索引）**：各样本独立随机抽取查询索引，破坏原条件配对；与固定一一置换区分。
4. **TQ Jitter (+10min)**：时间步微扰（偏移 1 个步长 / 10 分钟），检验模型对局部时钟偏移的鲁棒性。
5. **TQ Zero Ablation**：置零 `temporal_query`，完全剥离日周期向量调控能力。
6. **WTE Zero Ablation**：置零 `word_embedding_base`，观察固定模型对字典移除的敏感性。
7. **WTE Random Gaussian**：用同全局均值/方差的高斯张量替换字典，观察替换敏感性；不能独立排除几何变化或训练适应关系，来源优势需补训练期对照。

---

## 四、几何与拓扑可视化产物规划

1. **7 组训练后干预柱状图**：`figures/cycle_mechanism_validation/cycle_perturbation_impact.png`
2. **144x144 TQ 余弦相似度热力图**：`figures/cycle_mechanism_validation/cycle_tq_cosine_heatmap.png`
3. **24 小时滞后自相关曲线 $R(\tau)$**：`figures/cycle_mechanism_validation/cycle_autocorrelation_lag.png`
4. **昼夜词嵌入原型激活剖面图**：`figures/cycle_mechanism_validation/wte_day_vs_night_activation.png`
5. **量化评测指标汇总表**：`figures/cycle_mechanism_validation/cycle_verification_results.csv`

---

## 五、实验完善性审查计划（2026-10-25）

本轮目标是回答“加入词嵌入与周期向量是否有意义，现有实验能证明到什么程度”，不把训练后破坏实验当作重新训练消融。

1. 对照模型前向与训练损失，确定 TQ、WTE、输出周期残差及最终测试分支的关系。
2. 核验现存 CSV、图件与文档；用 CPU 读取原检查点复算 TQ 几何，检查日周期结论。
3. 审查相位索引定义、随机干预方式、重复种子及统计单位。
4. 在审查报告中给出按优先级排列的对照组、评价指标及支持/否定结论的条件。
5. 更新总计划和执行结果，区分“已观察到的敏感性”与“尚未证明的实际增益/物理语义”。

执行状态：以上 5 项审查已完成；结果已记入 `experiment_review.md`，分优先级实验与验收标准已记入 `completion_plan.md`，总计划及原执行报告已同步。当前只完成审查与方案，未执行全量重训练，未覆盖原实验 CSV 和图件。
