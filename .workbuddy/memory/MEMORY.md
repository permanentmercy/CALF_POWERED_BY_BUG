# 项目长期记忆 — CALF-main（大模型 / 长期时间序列预测）

## 用户偏好（必须遵守）
- **禁止在仓库中创建脚本或临时文件**。不要新建 .py 分析脚本、不要新建临时目录；验证数据/权重时用只读的内联命令（`python -c`），产出物放在回复正文里。
- 回答风格：先给结论，再给代码行号证据；涉及数值结论时尽量用真实数据/权重验证而不是推测。

## 环境
- 本机可用 Python 环境：`D:\software\conda\envs\calf\python.exe` → torch 2.4.1+cu124 / numpy 1.22.4 / matplotlib 3.8.3（**唯一带 torch 的环境**）。
- 托管 Python（3.13.12）与 `D:\software\conda\python.exe` 均**没有** torch / numpy，不要用它们跑模型相关代码。

## 代码结构要点
- 唯一模型实现：`models/CALF.py`（`models/gpt2` 为本地 GPT-2 权重，`wte_pca_500.pt` 为 PCA 词嵌入）。
- 实验入口：`exp/exp_long_term_forecasting.py`；损失：`utils/cmLoss.py`；数据：`data_provider/data_loader.py::Dataset_Solar`。
- Solar 数据为 10 分钟采样、列顺序即站点顺序；`cycle=144`（=1440/10）对应一天。

## 关键事实（易错点）
- **没有 per-station 独立输出头**：`out_layer` 是单个 `TQ_OutputHead`，全站点共享权重（checkpoint 中只有一组 `out_layer.*`）。
- 唯一逐站点独立的输出侧参数是 `cycle_queue: (cycle, enc_in)`——"每站点 × 每相位"的加性先验，配一个全局标量 `cycle_scale`（初始 0.1）。
- `temporal_query (cycle, 768)` 是**跨站点共享**的，注入 Encoder 交叉注意力的 Q 侧。
- **【已更正】站点 token 之间是双向可见的，不是因果掩码**：`models/GPT2_arch.py::AccustumGPT2Model.__init__` 把每层 `attn.bias`（bool 下三角 buffer，`persistent=False`）`fill_(True)`，使 `_attn` 里的 `torch.where(causal_mask, attn_weights, mask_value)` 不屏蔽任何位置。实测：只改最后一个 token 的嵌入，第一个 token 输出变化 = **1.29**（全 True）vs **恰好 0.0**（还原下三角）。所以不存在"只能看前序站点"的方向性泄漏；但全注意力仍会跨站点混合信息，且站点身份靠 `wpe` 位置编码 → 结果仍与 CSV 列顺序有关。
- `models/GPT2_arch.py` 只有 221 行、一个类 `AccustumGPT2Model(GPT2Model)`，做三件事：① 关因果掩码；② `accustum_forward` 是 HF `GPT2Model.forward` 的完整拷贝；③ `forward` 返回 `(last_hidden_state, hidden_states)` 二元组。CALF 的时间分支和文本分支都实例化它（CALF.py:185-186，均带 `output_attentions=True, output_hidden_states=True`）。
- ⚠️ `accustum_forward` 调用了 `self.warn_if_padding_and_no_attention_mask`（transformers ≥4.33 才有），但本环境是 **4.30.1** → **`input_ids=` 路径会 AttributeError 崩**。CALF 一直用 `inputs_embeds=`，所以从未触发。
- 归一化链：Dataset 全局 StandardScaler（scale=True 时）→ 模型内部 RevIN（use_norm=1）→ 输出回到**标准化空间**（不是原始物理量纲），除非 scale=False 或 task_loss=log1p_tweedie。
