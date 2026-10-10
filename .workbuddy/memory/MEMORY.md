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

## 损失函数结构（utils/cmLoss.py）
- 总损失 `total_loss = task_w*task_loss + output_w*output_loss + feature_w*feature_loss`（cmLoss.py:98）。
- `--task_loss`：唯一使用真值 batch_y 的监督损失；`--output_loss`：time/text 两分支输出的一致性损失（**不用真值**）；
  `--feature_loss`：两分支 GPT-2 中间层特征对齐蒸馏。默认均为 `l1`（run.py:89-91）。
- 三者在 `loss_dict`（cmLoss.py:9-18）里共用同一份可选集，但**可用子集互斥**：
  长期预测/output_loss 只能用两参数损失（l1/mse/smooth_l1）；短期预测 output_loss 必须用五参数（mape/mase/smape），
  否则 cmLoss.py:59 vs :61 会参数数量不匹配直接报错。`ce` 因 target 类型要求基本不可用。
- output_loss 梯度是对称双向的；但 `cycle_scale*cycle_queue` 与 `trend_gate*trend` 在两分支加的是同一份，
  作差时精确抵消 → 这些参数从 output_loss 拿不到梯度，只靠 task_loss 训练。
  例外：CALF.py:485-495 的昼夜门控是非线性，harmonic_gate / diurnal_gate_logit / night_floor 能拿到非零梯度。
- `--task_time_ratio`（默认 0.5）加权 time/text 两支的 task_loss；分类任务只用 time 分支。
- 验证早停与训练 task_loss 解耦：exp_short_term_forecasting.py:125 硬编码 `_select_vali_criterion("smape")`。

## 关键事实（易错点）
- **没有 per-station 独立输出头**：`out_layer` 是单个 `TQ_OutputHead`，全站点共享权重（checkpoint 中只有一组 `out_layer.*`）。
- 唯一逐站点独立的输出侧参数是 `cycle_queue: (cycle, enc_in)`——"每站点 × 每相位"的加性先验，配一个全局标量 `cycle_scale`（初始 0.1）。
- `temporal_query (cycle, 768)` 是**跨站点共享**的，注入 Encoder 交叉注意力的 Q 侧。
- **【已更正】站点 token 之间是双向可见的，不是因果掩码**：`models/GPT2_arch.py::AccustumGPT2Model.__init__` 把每层 `attn.bias`（bool 下三角 buffer，`persistent=False`）`fill_(True)`，使 `_attn` 里的 `torch.where(causal_mask, attn_weights, mask_value)` 不屏蔽任何位置。实测：只改最后一个 token 的嵌入，第一个 token 输出变化 = **1.29**（全 True）vs **恰好 0.0**（还原下三角）。所以不存在"只能看前序站点"的方向性泄漏；但全注意力仍会跨站点混合信息，且站点身份靠 `wpe` 位置编码 → 结果仍与 CSV 列顺序有关。
- `models/GPT2_arch.py` 只有 221 行、一个类 `AccustumGPT2Model(GPT2Model)`，做三件事：① 关因果掩码；② `accustum_forward` 是 HF `GPT2Model.forward` 的完整拷贝；③ `forward` 返回 `(last_hidden_state, hidden_states)` 二元组。CALF 的时间分支和文本分支都实例化它（CALF.py:185-186，均带 `output_attentions=True, output_hidden_states=True`）。
- ⚠️ `accustum_forward` 调用了 `self.warn_if_padding_and_no_attention_mask`（transformers ≥4.33 才有），但本环境是 **4.30.1** → **`input_ids=` 路径会 AttributeError 崩**。CALF 一直用 `inputs_embeds=`，所以从未触发。
- 归一化链：Dataset 全局 StandardScaler（scale=True 时）→ 模型内部 RevIN（use_norm=1）→ 输出回到**标准化空间**（不是原始物理量纲），除非 scale=False 或 task_loss=log1p_tweedie。
- `--task_loss log1p_tweedie` 时：跳过标准化（data_loader.py:418）、target 做 log1p（:444）、inverse_transform 原样返回（:464）；且 `task_loss` 只在 `args.data == 'Solar'` 时传进 Dataset（data_factory.py:90）。

## Solar 实验的脚本与参数（易踩坑）
- **`solar.sh` / `scripts/long_term_forecasting/solar.sh` 均为 `--patience 5` 硬编码**，pred_len 循环只有 96/192/336/720。
  若看到日志里出现 p10/p20/p30/p40，说明是临时改过 patience 的脚本或直接命令行跑的，**不要以这两个脚本反推日志配置**。
- 标准 Solar 实验固定配置：seq96/pred96、d_model=768、n_heads=4、d_ff=3072、e_layers=1、gpt_layers=1、
  batch=16 + accum=4（等效 bs=64）、lr=0.0004、cos tmax=10、epochs=20、
  task_w=0.736 / output_w=0.214 / feature_w=0.05（脚本里用 `output_w = 1 - task_w - feature_w` 反算）。
- 日志参数块位置：`utils/print_args.py` 输出，Patience/Learning Rate 在 "Run Parameters" 段。
