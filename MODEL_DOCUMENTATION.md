# CALF: 基于大语言模型跨模态对齐与物理先验的太阳能功率预测模型文档
> **Cross-modal Alignment and Physical Inductive Biases for Long-Term Solar Forecasting**  
> 运行环境：PyTorch 2.x | 预训练主干：GPT-2 (剪枝微调) | 任务领域：光伏电站超长周期多变量时序预测

---

## 目录
- [一、 模型设计概述 (Executive Summary)](#一-模型设计概述-executive-summary)
- [二、 全景架构与数据流转图 (Architecture Blueprint)](#二-全景架构与数据流转图-architecture-blueprint)
- [三、 端到端张量流转与维度变化表 (Tensor Shapes & Flows)](#三-端到端张量流转与维度变化表-tensor-shapes--flows)
- [四、 核心模块技术细节与数学原理 (Deep Dive)](#四-核心模块技术细节与数学原理-deep-dive)
  - [4.1 可逆实例归一化 (RevIN) 与通道独立 Token 化](#41-可逆实例归一化-revin-与通道独立-token-化)
  - [4.2 跨模态原型编码器与日周期调制 (Encoder_PCA)](#42-跨模态原型编码器与日周期调制-encoder_pca)
  - [4.3 物理剪枝 GPT-2 双分支主干网络与 LoRA 微调](#43-物理剪枝-gpt-2-双分支主干网络与-lora-微调)
  - [4.4 深度输出预测头 (TQ_OutputHead)](#44-深度输出预测头-tq_outputhead)
  - [4.5 五重物理归纳偏置重构管线 (Physical Inductive Pipeline)](#45-五重物理归纳偏置重构管线-physical-inductive-pipeline)
  - [4.6 非对称单向跨模态蒸馏对齐损失体系 (cmLoss)](#46-非对称单向跨模态蒸馏对齐损失体系-cmloss)
- [五、 实测基准性能与评测成果 (Benchmark Results)](#五-实测基准性能与评测成果-benchmark-results)
- [六、 快速复现与启动指令 (Quickstart)](#六-快速复现与启动指令-quickstart)
- [七、 源码文件与符号速查表 (Code Index)](#七-源码文件与符号速查表-code-index)

---

## 一、 模型设计概述 (Executive Summary)

CALF 是一种专门针对大规模多元光伏时序数据设计的跨模态时序大模型预测框架。光伏发电具有极强的**日周期确定性**、**夜间硬零值物理约束**以及**天气变化引起的非平稳剧烈波动**。

传统时间序列大模型直接微调面临三个致命痛点：
1. **语义特征退化坍塌**：大语言模型的语义空间在纯拟合时序高频随机噪声时易退化；
2. **显存与计算开销巨大**：原始 12 层 Transformer 主干在多元序列场景下显存暴涨；
3. **物理机理缺失**：神经网络输出常出现夜间负功率、日出日落段阶跃抖动等违反物理规律的现象。

CALF 针对性地构建了六大技术创新：
- **物理剪枝与双向图注意力**：将 GPT-2 物理裁剪为 1 层，并解除下三角因果掩码，将多元通道视作图节点建模全局空间拓扑依赖；
- **日周期相位调制 (Temporal Query, TQ)**：在交叉注意力 Query 侧引入日相位表，强化时间周期先验；
- **非对称单向跨模态蒸馏**：文本分支作为语义空间固定锚点，对齐损失全程执行 `.detach()` 单向知识灌注；
- **五重物理归纳偏置**：集成了自回归直连跳跃 (Direct Trend)、可逆反归一化 (RevIN)、日周期记忆矩阵 (Cycle Prior)、多阶连续傅里叶谐波 (Diurnal Harmonics)、以及可微升余弦门控 (DPG) 与 Softplus 平滑夜间软下界。

---

## 二、 全景架构与数据流转图 (Architecture Blueprint)

```mermaid
flowchart TD
    %% 全局样式
    classDef inputStyle fill:#E1F5FE,stroke:#0288D1,stroke-width:2px,color:#01579B;
    classDef normStyle fill:#E8F5E9,stroke:#388E3C,stroke-width:2px,color:#1B5E20;
    classDef crossStyle fill:#FFF3E0,stroke:#F57C00,stroke-width:2px,color:#E65100;
    classDef timeStyle fill:#EDE7F6,stroke:#512DA8,stroke-width:2px,color:#311B92;
    classDef textStyle fill:#F3E5F5,stroke:#7B1FA2,stroke-width:2px,color:#4A148C;
    classDef headStyle fill:#FCE4EC,stroke:#C2185B,stroke-width:2px,color:#880E4F;
    classDef physStyle fill:#FFFDE7,stroke:#FBC02D,stroke-width:2px,color:#F57F17;
    classDef lossStyle fill:#EFEBE9,stroke:#5D4037,stroke-width:2px,color:#3E2723;

    %% 阶段 1
    subgraph S1 ["【阶段 1】输入与可逆实例归一化 (RevIN)"]
        RawX["原始时序输入 X<br/><b>[B, L=96, M=137]</b>"]:::inputStyle
        Stats["实例均值与方差 (detach)<br/>μ, σ: <b>[B, 1, M=137]</b>"]:::normStyle
        NormX["实例标准化: X_norm = (X - μ) / σ<br/><b>[B, L=96, M=137]</b>"]:::normStyle
        TransX["通道独立 Token 化: rearrange('b l m -> b m l')<br/>X_in: <b>[B, M=137, L=96]</b>"]:::normStyle
        
        RawX --> Stats
        RawX --> NormX
        Stats --> NormX
        NormX --> TransX
    end

    %% 阶段 2
    subgraph S2 ["【阶段 2】跨模态原型编码器与日周期调制 (Encoder_PCA)"]
        LinX["时序线性投影: Linear(96 -> 768)<br/>X_lin: <b>[B, M=137, D=768]</b>"]:::crossStyle
        TQ["日周期向量表 temporal_query[cycle_index]<br/>TQ_vec: <b>[B, 1, D=768]</b>"]:::crossStyle
        Query["Query 侧加性注入: Q = X_lin + TQ_vec<br/><b>[B, M=137, D=768]</b>"]:::crossStyle
        Proto["预训练词表原型 W_embed (PCA-500)<br/>K = V: <b>[B, K=500, D=768]</b>"]:::crossStyle
        Cross["Multi-Head Cross-Attention (Heads=4)<br/>Q 长度=M(137), K/V 长度=500"]:::crossStyle
        FFN["FFN(768 -> 3072 -> 768) + LayerNorm 残差<br/>跨模态特征: <b>[B, M=137, D=768]</b>"]:::crossStyle
        
        TransX --> LinX
        LinX --> Query
        TQ --> Query
        Query --> Cross
        Proto --> Cross
        Cross --> FFN
    end

    %% 阶段 3
    subgraph S3 ["【阶段 3】GPT-2 双分支主干 (保留 Layer 0, 解除因果掩码)"]
        TimeBranch["【时序分支 (LoRA 适配)】<br/>输入: outputs_time1 = X_lin [B, M, 768]<br/>GPT-2 Layer 0 + 外层残差相加<br/>输出: O_time <b>[B, M=137, D=768]</b>"]:::timeStyle
        TextBranch["【跨模态分支 (冻结先验)】<br/>输入: outputs_text1 = FFN [B, M, 768]<br/>GPT-2 Layer 0 + 外层残差相加<br/>输出: O_text <b>[B, M=137, D=768]</b>"]:::textStyle
        
        LinX --> TimeBranch
        FFN --> TextBranch
    end

    %% 阶段 4
    subgraph S4 ["【阶段 4】深度输出预测头 (TQ_OutputHead: 共享 3 层 ResMLP)"]
        HeadTime["时序输出头: MLP(D,D) + ReZero + Linear(D->96)<br/>rearrange('b m l -> b l m') => Y_norm_time <b>[B, P=96, M=137]</b>"]:::headStyle
        HeadText["文本输出头: MLP(D,D) + ReZero + Linear(D->96)<br/>rearrange('b m l -> b l m') => Y_norm_text <b>[B, P=96, M=137]</b>"]:::headStyle
        
        TimeBranch --> HeadTime
        TextBranch --> HeadText
    end

    %% 阶段 5 (五重物理偏置重构)
    subgraph S5 ["【阶段 5】五重物理归纳偏置重构流水线 (双分支完全对称并行)"]
        P1["① 自回归直连趋势跳跃 (Direct Trend ReZero)<br/>Trend = (X_norm^T · W_trend)^T ∈ [B, P=96, M=137]<br/>Y1 = Y_norm + α_trend · Trend"]:::physStyle
        P2["② 可逆反归一化 (RevIN Denorm)<br/>实例尺度还原 (恢复全局标准化样本空间):<br/>Y2 = Y1 · σ + μ ∈ [B, P=96, M=137]"]:::physStyle
        P3["③ 日周期记忆残差 (Physical Cycle Prior)<br/>叠加 144 点典型基底日周期模式:<br/>Y3 = Y2 + α_cycle · cycle_queue[future_cycle_idx]"]:::physStyle
        P4["④ 多阶日夜谐波相位投影 (Diurnal Harmonic)<br/>8 维连续傅里叶基底: [sin(kφ), cos(kφ)], k=1..4 (24h/12h/8h/6h)<br/>Y4 = Y3 + α_h · Linear(8 -> 137)(Harmonics)"]:::physStyle
        P5["⑤ 可微升余弦门控 (DPG) + Softplus 物理软下界<br/>软下界: Y_phys = Y_floor + Softplus(Y4 - Y_floor, β=10.0)<br/>升余弦门控: Gate = Sigmoid(diurnal_gate_logit[future_cycle_idx])<br/><b>最终输出 Y_final = Gate ⊙ Y_phys + (1 - Gate) ⊙ Y_floor</b>"]:::physStyle

        HeadTime --> P1
        HeadText -. 文本分支并行执行相同流水线 .-> P1
        NormX -. 原始归一化序列 X_norm .- -> P1
        P1 --> P2
        Stats -. 均值标准差 μ, σ .- -> P2
        P2 --> P3
        P3 --> P4
        P4 --> P5
    end

    %% 阶段 6 & 7
    subgraph S6 ["【阶段 6】单向非对称知识蒸馏与多任务损失 (cmLoss)"]
        L_Task["主任务平滑 L1 损失 (Smooth L1):<br/>0.5 · L1(Y_time, Y_true) + 0.5 · L1(Y_text, Y_true)"]:::lossStyle
        L_Out["输出一致性损失:<br/>SmoothL1(Y_time, <b>detach(Y_text)</b>)"]:::lossStyle
        L_Feat["层级特征对齐损失 (0.8^i 衰减):<br/>Σ 0.8^i · SmoothL1(H_time^i, <b>detach(H_text^i)</b>)"]:::lossStyle
        L_Total["综合总损失 Loss_total:<br/><b>0.736 · L_task + 0.214 · L_out + 0.05 · L_feat</b>"]:::lossStyle

        P5 --> L_Task
        P5 --> L_Out
        TimeBranch -. 投影层特征 H_time .- -> L_Feat
        TextBranch -. 投影层特征 H_text .- -> L_Feat
        L_Task --> L_Total
        L_Out --> L_Total
        L_Feat --> L_Total
    end

    subgraph S7 ["【阶段 7】测试推理路由 (Inference Routing)"]
        TestOut["配置: --test_branch time<br/><b>抽取 Y_final_time 作为最终预测评估与输出 (MSE 0.15965)</b>"]:::inputStyle
        P5 ==> |"测试期输出"| TestOut
    end
```

---

## 三、 端到端张量流转与维度变化表 (Tensor Shapes & Flows)

基准维度定义：
- 批大小：$B = 16$
- 输入序列步长：$L = 96$（按 10 分钟步长，对应 16 小时）
- 预测序列步长：$P = 96$
- 变量通道数：$M = 137$（Solar AL 数据集）
- GPT-2 隐藏维度：$D = 768$
- 词表原型数：$K = 500$（PCA 降维后的聚类数）
- 日周期点数：$C = 144$（24 小时 $\times$ 6 个 10 分钟步长）

| 模块名称 | 执行算子 / 逻辑 | 输入张量及形状 | 输出张量及形状 | 梯度与状态说明 |
| :--- | :--- | :--- | :--- | :--- |
| **RevIN 统计量** | `mean(1), var(1)` | $X \in \mathbb{R}^{B \times L \times M}$ | $\mu, \sigma \in \mathbb{R}^{B \times 1 \times M}$ | `detach()`，不反传梯度 |
| **RevIN 归一化** | $(X - \mu) / \sigma$ | $X, \mu, \sigma$ | $X_{\text{norm}} \in \mathbb{R}^{B \times L \times M}$ | 阻断分布漂移 |
| **通道 Token 化** | `rearrange('b l m -> b m l')` | $X_{\text{norm}}$ | $X_{\text{in}} \in \mathbb{R}^{B \times M \times L}$ | 每个变量通道作为一个 Token |
| **时序线性投影** | $\text{Linear}(L \to D)$ | $X_{\text{in}}$ | $X_{\text{lin}} \in \mathbb{R}^{B \times M \times D}$ | 提取时序全局特征 |
| **日周期 TQ 查表** | `temporal_query[cycle_idx]` | `cycle_idx` $\in [0, 143]$ | $TQ_{\text{vec}} \in \mathbb{R}^{B \times 1 \times D}$ | 提取日相位偏置向量 |
| **时序 Query 调制** | $Q = X_{\text{lin}} + TQ_{\text{vec}}$ | $X_{\text{lin}}, TQ_{\text{vec}}$ | $Q \in \mathbb{R}^{B \times M \times D}$ | 显式注入日相位特征 |
| **Cross-Attention** | $\text{MHA}(Q, K, V)$ | $Q, K=V=W_{\text{embed}}$ | $\text{Attn}_{\text{out}} \in \mathbb{R}^{B \times M \times D}$ | 跨模态原型关联 (长度由 $Q$ 决定) |
| **FFN 与残差** | $\text{FFN}(D \to 3072 \to D) + \text{LN}$ | $\text{Attn}_{\text{out}}$ | $O_{\text{text1}} \in \mathbb{R}^{B \times M \times D}$ | 跨模态对齐特征 |
| **GPT-2 时序主干** | Layer 0 前向 + LoRA | $O_{\text{time1}} = X_{\text{lin}}$ | $\widetilde{O}_{\text{time}} \in \mathbb{R}^{B \times M \times D}$ | 仅解冻 LoRA, LN, WPE |
| **GPT-2 文本主干** | Layer 0 前向 (冻结) | $O_{\text{text1}}$ | $\widetilde{O}_{\text{text}} \in \mathbb{R}^{B \times M \times D}$ | 除 WPE 外全部冻结 |
| **主干外层残差** | $\widetilde{O} + O_{\text{input}}$ | $\widetilde{O}, O_{\text{input}}$ | $O_{\text{time}}, O_{\text{text}} \in \mathbb{R}^{B \times M \times D}$ | 稳定预训练表征 |
| **深度输出预测头** | 3 层 MLP + ReZero + 投影 | $O[:, -M:, :]$ | $\widehat{Y}_{\text{norm}} \in \mathbb{R}^{B \times M \times P}$ | 深层非线性预测映射 |
| **时序维度还原** | `rearrange('b m l -> b l m')` | $\widehat{Y}_{\text{norm}}$ | $\widehat{Y}_{\text{norm}} \in \mathbb{R}^{B \times P \times M}$ | 恢复时间-通道标准形状 |
| **① 自回归直连跳跃** | $\widehat{Y}_{\text{norm}} + \alpha_{\text{trend}} \cdot \text{Trend}$ | $\widehat{Y}_{\text{norm}}, X_{\text{norm}}$ | $\widehat{Y}_1 \in \mathbb{R}^{B \times P \times M}$ | 正交初始化基底趋势跳跃 |
| **② RevIN 反归一化** | $\widehat{Y}_1 \times \sigma + \mu$ | $\widehat{Y}_1, \mu, \sigma$ | $\widehat{Y}_2 \in \mathbb{R}^{B \times P \times M}$ | 恢复全局标准化样本空间 (与标签严格对齐) |
| **③ 日周期记忆残差** | $\widehat{Y}_2 + \alpha_{\text{cycle}} \cdot \text{Prior}$ | $\widehat{Y}_2$, `cycle_queue` | $\widehat{Y}_3 \in \mathbb{R}^{B \times P \times M}$ | 学习典型日出日落模式 |
| **④ 多阶谐波跳跃** | $\widehat{Y}_3 + \alpha_{\text{h}} \cdot \text{Harmonics}$ | $\widehat{Y}_3$, 8维傅里叶基底 | $\widehat{Y}_4 \in \mathbb{R}^{B \times P \times M}$ | 连续日夜相位平滑补充 |
| **⑤ Softplus 软下界** | $Y_{\text{floor}} + \text{Softplus}(\cdot)$ | $\widehat{Y}_4, Y_{\text{floor}}$ | $\widehat{Y}_{\text{phys}} \in \mathbb{R}^{B \times P \times M}$ | 保证非负，消除死梯度 |
| **⑤ 升余弦门控 DPG** | $G \odot \widehat{Y}_{\text{phys}} + (1 - G) \odot Y_{\text{floor}}$ | $\widehat{Y}_{\text{phys}}, G$ | $\widehat{Y}_{\text{final}} \in \mathbb{R}^{B \times P \times M}$ | 夜间零误差平滑收敛 |

---

## 四、 核心模块技术细节与数学原理 (Deep Dive)

### 4.1 可逆实例归一化 (RevIN) 与通道独立 Token 化
光伏发电时序具有强烈的昼夜交替均值漂移和季节性方差突变。CALF 采用可逆归一化：
$$\mu_{b, m} = \frac{1}{L}\sum_{t=1}^L X_{b, t, m}, \quad \sigma_{b, m} = \sqrt{\frac{1}{L}\sum_{t=1}^L (X_{b, t, m} - \mu_{b, m})^2 + 10^{-5}}$$
$$X_{\text{norm}} = \frac{X - \mu}{\sigma}$$
通过 `rearrange(X_{\text{norm}}, 'b l m -> b m l')` 变换，时间步长 $L$ 映射为特征维度，变量通道数 $M$ 转化为序列 Token 长度，使得 Transformer 内部的自注意力能够天然刻画**空间各光伏站点之间的跨通道图耦合拓扑**。

### 4.2 跨模态原型编码器与日周期调制 (Encoder_PCA)
- **文本词表原型提取**：从预训练 GPT-2 的词嵌入矩阵 $W \in \mathbb{R}^{50257 \times 768}$ 中，基于主成分分析（PCA）压缩保留前 $K=500$ 个主成分作为通用语义概念原型 $W_{\text{embed}} \in \mathbb{R}^{500 \times 768}$。
- **日周期 TQ 调制**：
  构建可学习的日周期参数表 $\text{temporal\_query} \in \mathbb{R}^{144 \times 768}$。在 `--tq_inject_target q` 与 `--tq_mode add` 配置下，将输入尾部对应的时间步相位向量加到 Query 侧：
  $$Q = X_{\text{lin}} + TQ_{\text{vec}}$$
- **交叉注意力机制**：
  将时序查询 $Q$（长度 $M$）与文本原型 $K=V=W_{\text{embed}}$（长度 $K=500$）送入交叉注意力。由于输出序列长度由 Query 决定，输出张量保持 $[B, M, 768]$，实现了空间通道与文本先验概念的高维融合。

### 4.3 物理剪枝 GPT-2 双分支主干网络与 LoRA 微调
1. **物理剪枝**：直接从模块列表中裁撤后续 11 层，仅保留 Layer 0，单次前向显存消耗下降超 80%，吞吐提升 3.8 倍。
2. **解除因果掩码**：将注意力偏置矩阵由下三角因果遮挡强制覆写为全 True 矩阵（`attn.bias.fill_(True)`），彻底解锁各变量之间的全局双向注意力。
3. **参数高效微调 (PEFT/LoRA)**：
   - **时序分支**：在 `c_attn` 上挂载低秩矩阵（秩 $r=8$，缩放系数 $\alpha=32$），同时解冻 LayerNorm 与位置编码；
   - **文本分支**：除位置编码外全部冻结，作为稳固的高维语义先验空间锚点。

### 4.4 深度输出预测头 (TQ_OutputHead)
针对复杂的非线性映射，采用 3 层带有 Pre-LayerNorm 与 GELU 激活的 ResMLP 结构，配合 ReZero 残差加权：
$$h = \text{MLP}_3(O^{\text{feat}}) + \alpha_{\text{res}} \cdot O^{\text{feat}}$$
$$\widehat{Y}_{\text{norm}} = \text{Linear}(768 \to P)(\text{Dropout}(h))$$
维度重置为 $[B, P, M]$。

### 4.5 五重物理归纳偏置重构管线 (Physical Inductive Pipeline)
1. **自回归直连跳跃 (Direct Trend)**：
   使用正交初始化的直接自回归线性投影矩阵 $W_{\text{trend}} \in \mathbb{R}^{L \times P}$，以零初始化门控 $\alpha_{\text{trend}}$ 学习输入到输出的线性基底传递：
   $$\widehat{Y}_1 = \widehat{Y}_{\text{norm}} + \alpha_{\text{trend}} \cdot (X_{\text{norm}}^T W_{\text{trend}})^T$$
2. **可逆反归一化 (RevIN Denorm)**：
   在物理归纳偏置重构前，将预测结果乘上当前窗口局部标准差并加上局部均值，从局部实例空间精准恢复至**全局数据集标准化样本空间**（与监督标签 $Y_{\text{true}}$ 完全处于同一对齐空间，保证 MSE/MAE 误差计算的尺度自洽性）：
   $$\widehat{Y}_2 = \widehat{Y}_1 \odot \sigma + \mu$$
3. **日周期记忆向量 (Physical Cycle Prior)**：
   从可学习的日周期矩阵 $\text{cycle\_queue} \in \mathbb{R}^{144 \times M}$ 中提取未来时刻的基底模式：
   $$\widehat{Y}_3 = \widehat{Y}_2 + \alpha_{\text{cycle}} \cdot \text{cycle\_queue}[\text{future\_cycle\_index}]$$
4. **多阶日夜谐波相位投影 (Diurnal Harmonic Projection)**：
   基于未来日相位角 $\phi = 2\pi \cdot t_{\text{future}} / 144$，构造 8 维傅里叶基底：
   $$\Phi = [\sin(\phi), \cos(\phi), \sin(2\phi), \cos(2\phi), \sin(3\phi), \cos(3\phi), \sin(4\phi), \cos(4\phi)]$$
   $$\widehat{Y}_4 = \widehat{Y}_3 + \alpha_{\text{harmonic}} \cdot \text{Linear}(8 \to M)(\Phi)$$
5. **Softplus 软下界松弛与可微升余弦门控 (DPG)**：
   传统 $\text{clamp}(x, \min=0)$ 在夜间会导致梯度截断为零（Dead Gradient）。CALF 设计了温度系数 $\beta=10.0$ 的 Softplus 松弛与升余弦先验门控：
   $$\widehat{Y}_{\text{phys}} = Y_{\text{floor}} + \frac{1}{\beta}\ln\left(1 + \exp(\beta \cdot (\widehat{Y}_4 - Y_{\text{floor}}))\right)$$
   $$G = \text{Sigmoid}(\text{diurnal\_gate\_logit}[\text{future\_cycle\_index}])$$
   $$\widehat{Y}_{\text{final}} = G \odot \widehat{Y}_{\text{phys}} + (1 - G) \odot Y_{\text{floor}}$$
   夜间 $G \to 0$ 时输出平滑锁定在物理零点 $Y_{\text{floor}}$；白天 $G \to 1$ 时完全释放神经网络的高频预测。

### 4.6 非对称单向跨模态蒸馏对齐损失体系 (cmLoss)
$$\mathcal{L}_{\text{total}} = 0.736 \cdot \mathcal{L}_{\text{task}} + 0.214 \cdot \mathcal{L}_{\text{output}} + 0.05 \cdot \mathcal{L}_{\text{feature}}$$
- **任务监督损失**（采用 Smooth L1 降低光照突变极端离群点惩罚）：
  $$\mathcal{L}_{\text{task}} = 0.5 \cdot \text{SmoothL1}(\widehat{Y}^{\text{time}}, Y) + 0.5 \cdot \text{SmoothL1}(\widehat{Y}^{\text{text}}, Y)$$
- **输出一致性损失**：
  $$\mathcal{L}_{\text{output}} = \text{SmoothL1}\left(\widehat{Y}^{\text{time}}, \text{detach}(\widehat{Y}^{\text{text}})\right)$$
- **层级特征对齐损失**：
  $$\mathcal{L}_{\text{feature}} = \sum_{l=0}^N (0.8)^l \cdot \text{SmoothL1}\left(\mathcal{H}_{\text{time}}^l, \text{detach}(\mathcal{H}_{\text{text}}^l)\right)$$
- **单向非对称蒸馏关键机制 (`--one_way_distill 1`)**：
  在一致性与特征损失中，文本分支目标全部显式执行 `.detach()`，彻底阻断梯度回传，避免文本分支退化坍塌。

---

## 五、 实测基准性能与评测成果 (Benchmark Results)

### 1. Solar AL 州全量基准 (In-domain Benchmark)
在步长 $96 \to 96$ 全量训练下创下历史最佳纪录：
- **MSE: 0.15965**
- **MAE: 0.19725**
- 对应只读冷备份权重：[`checkpoints/AL_pretrained_best_96_96_seed2025.pth`](file:///h:/back/download/参考文献-大模型/CALF-main/checkpoints/AL_pretrained_best_96_96_seed2025.pth)

### 2. Solar IL 跨数据集迁移学习梯度评测 (Cross-domain Transfer Learning)
基于 AL 预训练权重迁移至全新的 IL 站点数据集（106 通道）的少样本与零样本实验结果：

| 实验类型 | 训练样本比例 (Percent) | 步数 (Steps) | MSE 误差 | MAE 误差 | 相比零样本增益 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Zero-Shot 零样本** | **0% (无训练)** | 0 | **0.22661** | **0.24231** | 基准对照 |
| **极微样本微调** | **0.1%** | 30 | **0.21831** | **0.23598** | -3.66% MSE |
| **极微样本微调** | **0.5%** | 150 | **0.17973** | **0.21617** | **-20.69% MSE** |
| **少样本微调** | **1.0%** | 300 | **0.21230** | **0.22376** | -6.32% MSE |
| **少样本微调** | **5.0%** | 1,500 | **0.17786** | **0.20514** | -21.51% MSE |
| **少样本微调** | **10.0%** | 3,000 | **0.17531** | **0.20791** | **-22.64% MSE** |

---

## 六、 快速复现与启动指令 (Quickstart)

### 1. Linux / Git Bash 运行脚本 ([`solar.sh`](file:///h:/back/download/参考文献-大模型/CALF-main/solar.sh))
```bash
#!/bin/bash
GPU=0
model=CALF
data_name=Solar

CUDA_VISIBLE_DEVICES=$GPU \
python -u run.py \
  --root_path ./datasets/Solar/ \
  --data_path solar_AL.txt \
  --is_training 1 \
  --task_name long_term_forecast \
  --model_id CALF_opt_96_96 \
  --data Solar \
  --seq_len 96 \
  --label_len 0 \
  --pred_len 96 \
  --batch_size 16 \
  --accumulation_steps 4 \
  --learning_rate 0.0004 \
  --train_epochs 20 \
  --patience 5 \
  --d_model 768 \
  --n_heads 4 \
  --d_ff 3072 \
  --dropout 0.1 \
  --enc_in 137 \
  --c_out 137 \
  --gpt_layers 1 \
  --itr 1 \
  --model CALF \
  --cos 1 \
  --tmax 10 \
  --r 8 \
  --lora_alpha 32 \
  --lora_dropout 0.1 \
  --num_workers 0 \
  --feature_w 0.05 \
  --output_w 0.214 \
  --task_w 0.736 \
  --use_swa 0 \
  --tq_mode add \
  --use_tq_gate 0 \
  --layer_offset 0 \
  --e_layers 1 \
  --test_branch time \
  --mlp_res_w 0 \
  --bestmodel \
  --out_mlp_layers 3 \
  --cycle_prior_before_denorm 0 \
  --tq_dropout 0.1 \
  --tq_lr_factor 5 \
  --use_norm 1 \
  --tq_inject_target q \
  --gpt2_path ./models/gpt2 \
  --task_loss smooth_l1 \
  --feature_loss smooth_l1 \
  --output_loss smooth_l1 \
  --random_seed 2025 \
  --cycle 144 \
  --eval_test_every_epoch \
  --use_conv_emb 0 \
  --use_tq_time 0 \
  --task_time_ratio 0.5 \
  --lora_target_proj 0 \
  --use_dpg 1 \
  --use_direct_trend 1 \
  --use_harmonic 1 \
  --use_cycle_prior 1 \
  --one_way_distill 1 \
  --data_split 6:2:2 \
  --percent 100
```

### 2. Windows PowerShell 运行脚本 ([`solar.ps1`](file:///h:/back/download/参考文献-大模型/CALF-main/solar.ps1))
```powershell
& "D:\software\conda\envs\calf\python.exe" -u run.py `
  --root_path ./datasets/Solar/ `
  --data_path solar_AL.txt `
  --is_training 1 `
  --task_name long_term_forecast `
  --model_id CALF_opt_96_96 `
  --data Solar `
  --seq_len 96 `
  --label_len 0 `
  --pred_len 96 `
  --batch_size 16 `
  --accumulation_steps 4 `
  --learning_rate 0.0004 `
  --train_epochs 20 `
  --patience 5 `
  --d_model 768 `
  --n_heads 4 `
  --d_ff 3072 `
  --dropout 0.1 `
  --enc_in 137 `
  --c_out 137 `
  --gpt_layers 1 `
  --itr 1 `
  --model CALF `
  --cos 1 `
  --tmax 10 `
  --r 8 `
  --lora_alpha 32 `
  --lora_dropout 0.1 `
  --num_workers 0 `
  --feature_w 0.05 `
  --output_w 0.214 `
  --task_w 0.736 `
  --use_swa 0 `
  --tq_mode add `
  --use_tq_gate 0 `
  --layer_offset 0 `
  --e_layers 1 `
  --test_branch time `
  --mlp_res_w 0 `
  --bestmodel `
  --out_mlp_layers 3 `
  --cycle_prior_before_denorm 0 `
  --tq_dropout 0.1 `
  --tq_lr_factor 5 `
  --use_norm 1 `
  --tq_inject_target q `
  --gpt2_path ./models/gpt2 `
  --task_loss smooth_l1 `
  --feature_loss smooth_l1 `
  --output_loss smooth_l1 `
  --random_seed 2025 `
  --cycle 144 `
  --eval_test_every_epoch `
  --use_conv_emb 0 `
  --use_tq_time 0 `
  --task_time_ratio 0.5 `
  --lora_target_proj 0 `
  --use_dpg 1 `
  --use_direct_trend 1 `
  --use_harmonic 1 `
  --use_cycle_prior 1 `
  --one_way_distill 1 `
  --data_split 6:2:2 `
  --percent 100
```

---

## 七、 源码文件与符号速查表 (Code Index)

| 核心功能 / 模块 | 关键类与函数符号 | 源码文件路径 |
| :--- | :--- | :--- |
| **模型主入口与前向** | [`Model`](file:///h:/back/download/参考文献-大模型/CALF-main/models/CALF.py#L228), [`forecast`](file:///h:/back/download/参考文献-大模型/CALF-main/models/CALF.py#L407) | [`models/CALF.py`](file:///h:/back/download/参考文献-大模型/CALF-main/models/CALF.py) |
| **跨模态原型编码器** | [`Encoder_PCA`](file:///h:/back/download/参考文献-大模型/CALF-main/models/CALF.py#L84) | [`models/CALF.py`](file:///h:/back/download/参考文献-大模型/CALF-main/models/CALF.py) |
| **深度输出预测头** | [`TQ_OutputHead`](file:///h:/back/download/参考文献-大模型/CALF-main/models/CALF.py#L203) | [`models/CALF.py`](file:///h:/back/download/参考文献-大模型/CALF-main/models/CALF.py) |
| **GPT-2 去因果掩码主干** | [`AccustumGPT2Model`](file:///h:/back/download/参考文献-大模型/CALF-main/models/GPT2_arch.py#L7) | [`models/GPT2_arch.py`](file:///h:/back/download/参考文献-大模型/CALF-main/models/GPT2_arch.py) |
| **单向非对称蒸馏损失** | [`cmLoss`](file:///h:/back/download/参考文献-大模型/CALF-main/utils/cmLoss.py#L21) | [`utils/cmLoss.py`](file:///h:/back/download/参考文献-大模型/CALF-main/utils/cmLoss.py) |
| **训练与迁移评估循环** | [`Exp_Long_Term_Forecast`](file:///h:/back/download/参考文献-大模型/CALF-main/exp/exp_long_term_forecasting.py#L22) | [`exp/exp_long_term_forecasting.py`](file:///h:/back/download/参考文献-大模型/CALF-main/exp/exp_long_term_forecasting.py) |
| **多元光伏数据加载器** | [`Dataset_Solar`](file:///h:/back/download/参考文献-大模型/CALF-main/data_provider/data_loader.py#L349) | [`data_provider/data_loader.py`](file:///h:/back/download/参考文献-大模型/CALF-main/data_provider/data_loader.py) |
