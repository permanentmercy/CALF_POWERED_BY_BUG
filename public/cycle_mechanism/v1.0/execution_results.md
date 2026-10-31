# 词嵌入分支时间周期干预实验记录 (v1.0)

> **2026-10-25 审查修订**：保留下述七组既有测量数值，但将结论限定为固定已训练模型的文本分支敏感性。原有“严格物理相位同构”“12h 负对跖”“语言语义不可替代”等结论证据不足，已修正。检查点复算、实现问题及结论边界见 [experiment_review.md](experiment_review.md)；训练期消融等待执行工作见 [completion_plan.md](completion_plan.md)。本轮未重跑全量七组实验。

本项目基于标准 **7:1:2** 划分基准模型检查点对 CALF 跨模态文本分支的词嵌入原型（`word_embedding_base`）与时间查询矩阵（`temporal_query`）记录了全量测试集（10,417 个重叠窗口）的训练后干预结果。它们不能替代有/无模块的重新训练比较。

---

## 一、使用的权重与配置清单

| 权重组件 | 物理意义与作用 | 文件路径 / 规范配置 |
| :--- | :--- | :--- |
| **核心模型检查点** | 经全量充分训练的 CALF 模型状态，包含收敛的时间查询矩阵 $TQ$ 与跨模态注意力权重 | [checkpoint.pth](file:///checkpoints/long_term_forecast_CALF_96_96_p100_CALF_Solar_ftM_sl96_ll0_pl96_dm768_nh4_el1_dl1_df3072_fc1_ebtimeF_dtTrue_test_gpt1_rl0.0004_2025_0_feature0.05_output0.214_train_epochs100_bs16_dr0.1_ld0.1_r8/checkpoint.pth) |
| **词嵌入原型矩阵** | GPT-2 词表转置后经 PCA 得到的固定字典，500 行近似正交 ($500 \times 768$)，语义归因待验证 | [wte_pca_500.pt](../../../wte_pca_500.pt) |
| **底座大语言模型** | 预训练 GPT-2 语言模型底座（冻结文本语义流形，裁剪保留首层） | [models/gpt2](file:///models/gpt2) |
| **数据集划分** | **标准 7:1:2 划分**（测试集样本数 10,417） | [datasets/Solar/solar_AL.txt](file:///datasets/Solar/solar_AL.txt) (137 变量全维度辐射通道) |

---

## 二、7:1:2 全量测试集量化对照实验结果

以下为基于标准 7:1:2 划分测试集（10,417 样本）在 7 组对抗扰动与消融条件下的实测指标：

| 实验对照组 | 扰动模式说明 | 文本分支 MSE (相对基准变化) | 文本分支 MAE (相对基准变化) | 时序分支 MSE (基准) | 跨模态分支间 MSE 差异 | 机制验证结论 |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **Clean Baseline** | 正常未扰动推理 | 0.183072 (+0.0%) | 0.243379 (+0.0%) | **0.164986** | 0.028923 | 固定模型的两分支参考误差 |
| **TQ Phase Invert (+12h)** | 查询索引平移半周期（72 槽） | **0.300044 (+63.89%)** | **0.336302 (+38.18%)** | 0.164986 | 0.138660 | 文本预测对已学习相位的错配敏感 |
| **TQ Random Permute（实际为随机索引）** | 各样本独立随机抽取查询索引 | **0.345100 (+88.51%)** | **0.343567 (+41.17%)** | 0.164986 | 0.186089 | 文本分支依赖原条件索引；尚未分离有序性与类别记忆 |
| **TQ Jitter (+10min)** | 查询偏移 1 步 | 0.172125 (-5.98%) | 0.233455 (-4.08%) | 0.164986 | 0.020623 | 改善原因待验证集相位扫描与对齐核查 |
| **TQ Zero Ablation** | 将查询置零 ($TQ=0$) | 0.216616 (+18.32%) | 0.265396 (+9.05%) | 0.164986 | 0.055047 | 文本预测使用了非零 TQ 条件 |
| **WTE Zero Ablation** | 将字典置零 ($WTE=0$) | 0.237119 (+29.52%) | 0.308825 (+26.89%) | 0.164986 | 0.077964 | 文本预测依赖训练时的字典表示 |
| **WTE Random Gaussian** | 全局均值/方差匹配高斯替换 | **0.324284 (+77.13%)** | **0.388722 (+59.72%)** | 0.164986 | 0.182734 | 字典替换敏感；几何变化与来源效应尚未分离 |

---

## 三、核心科学发现与物理意义分析

### 1. 周期查询的条件依赖与实际收益需要分开

错移半周期和随机索引均使文本误差明显增加，支持其使用了相位相关条件。但任意学过的类别查表也可能在索引错配后退化，不能据此排除容量作用或证明连续物理同构。七组 Time MSE 完全相同符合模型前向结构；最终时间分支是否从训练期 TQ 获益，应以有/无 TQ 重新训练比较确认。

检查点复算得到 R(1)=0.009775、R(72)=+0.005782，最低值 −0.006888 在 9.5h/14.5h。探索性行序置换检验提示微弱邻近有序性，但不支持强昼夜反相几何；详细检验条件见审查报告。

---

### 2. 字典替换敏感性尚不能归因为语言语义

高斯替换使文本 MSE 增加 77.13%，但原字典逐行范数差异大且行间近似正交；整体均值/方差匹配没有保持这些几何统计。应从相同底座分别训练预训练字典与几何匹配的固定随机字典，才能评估来源优势；当前结果未直接验证抗噪泛化或语言语义不可替代。

---

### 3. 昼夜分组注意力仅作探索性展示

现图展示以索引槽位划分的两个日内分组在原型字典上的平均注意力差异。手工提取只覆盖当前 q/add 首层逻辑，且有批次数上限；应记录真实样本数并与真实前向核对。原型范数也可能影响注意力聚焦，注意力差异不能单独证明不发生过拟合或某原型执行光照/夜间抑制功能。需通过固定输入/相位的交叉对照和遮蔽恢复实验验证。

---

## 四、图表产物清单（300 DPI 学术出版级）

所有图表均已基于 7:1:2 重新渲染并同步至 `figures/` 与 `latex/figures/`：

1. **扰动影响对比柱状图**：
   - 路径：[figures/cycle_mechanism_validation/cycle_perturbation_impact.png](file:///h:/back/download/参考文献-大模型/CALF-main/figures/cycle_mechanism_validation/cycle_perturbation_impact.png)
   - 内容：双分支在 7 组对抗扰动下的 MSE 对比与衰减百分比标注。
2. **144x144 TQ 余弦相似度热力图**：
   - 路径：[figures/cycle_mechanism_validation/cycle_tq_cosine_heatmap.png](file:///h:/back/download/参考文献-大模型/CALF-main/figures/cycle_mechanism_validation/cycle_tq_cosine_heatmap.png)
   - 内容：TQ 原始余弦相似度矩阵；主对角线为自相似，非对角整体接近 0，不能据此描述强平滑日周期或 12h 负对跖区。
3. **24 小时周期滞后自相关曲线**：
   - 路径：[figures/cycle_mechanism_validation/cycle_autocorrelation_lag.png](file:///h:/back/download/参考文献-大模型/CALF-main/figures/cycle_mechanism_validation/cycle_autocorrelation_lag.png)
   - 内容：0..143 槽的环形平均余弦，最低点约 9.5h/14.5h。图中 24h 为标识线；模 144 的自比较恢复由索引定义保证，不是独立的周期学习证据。
4. **昼夜词嵌入原型注意力极化剖面图**：
   - 路径：[figures/cycle_mechanism_validation/wte_day_vs_night_activation.png](file:///h:/back/download/参考文献-大模型/CALF-main/figures/cycle_mechanism_validation/wte_day_vs_night_activation.png)
   - 内容：两个日内索引分组的平均原型注意力及最大差异项；尚无统计显著性和原型功能干预结论。
5. **数值明细表**：
   - 路径：[figures/cycle_mechanism_validation/cycle_verification_results.csv](file:///h:/back/download/参考文献-大模型/CALF-main/figures/cycle_mechanism_validation/cycle_verification_results.csv)
