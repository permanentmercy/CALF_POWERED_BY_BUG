# 全量参数精确对齐与损失权重动态联动规划 (Subplan v2.1)

本文档依照团队规范维护，定义 CALF 批量实验控制面板（GUI & CLI）精确对齐 `Solar_test.sh` 全量超参体系、三损失权重（$task\_w, feature\_w, output\_w$）自动联动计算及无效组合跳过机制的技术实施方案。

---

## 一、用户需求与对齐目标

用户提供了生产环境中的完整实验配置脚本片段（源自 `scripts/long_term_forecasting/Solar_test.sh`），要求将批量面板的基础固定超参、循环参数及计算规则 100% 对齐该配置：

### 1. 全参数对齐清单
| 参数项 | 对齐值 | 原默认值 | 说明 |
| :--- | :--- | :--- | :--- |
| `root_path` | `./datasets/Solar/` | `./dataset/Solar/` | 指向真实存在的 `datasets/` 目录 |
| `data_path` | `solar_AL.txt` | `solar_AL.txt` | 数据集文件 |
| `data_split` | `7:1:2` | `7:1:2` | 训练:验证:测试划分 |
| `task_w` | `0.736` | `1.0` | 时序任务主损失权重 |
| `feature_w` | `0.05` | `0.05` | 隐层特征对齐蒸馏权重 |
| `output_w` | `0.214` | `0.214` | 输出 logits 对齐权重（由 $1.0 - task\_w - feature\_w$ 动态计算） |
| `cycle` | `144` | `24` | 物理周期步长（按 Solar 数据对齐） |
| `tq_mode` | `add` | `mul` | 周期向量注入方式 |
| `tq_inject_target` | `q` | `pca` | 注入目标为时序分支 Q |
| `tq_lr_factor` | `5.0` | `10.0` | TQ 专用学习率倍数 |
| `tq_dropout` | `0.1` | `0.0` | TQ 注入 Dropout |
| `out_mlp_layers` | `3` | `2` | 输出头 MLP 深度 |
| `cycle_prior_before_denorm` | `0` | `1` | 物理周期先验在反归一化后叠加 |
| `patience` | `5` | `3` | 早停容忍 Epoch 数 |
| `tmax` | `10` | `20` | 余弦退火调度周期 |
| `accumulation_steps` | `4` | `1` | 梯度累加步数 |
| `learning_rate` | `0.0004` | `0.0004` | 由 $0.00000625 \times batch\_size \times accumulation\_steps$ 严格计算 |
| `task_loss` | `smooth_l1` | `l1` | 任务损失函数 |
| `feature_loss` | `smooth_l1` | `l1` | 特征损失函数 |
| `output_loss` | `smooth_l1` | `l1` | 输出损失函数 |
| `enc_in` / `c_out` | `137` | `7` | Solar 维度 137 |
| `num_workers` | `0` | `10` | Windows 环境下设为 0 避免 DataLoader 进程死锁 |
| `eval_test_every_epoch` | `True` | `False` | 每轮评估测试集 |
| `gpt2_path` | `./models/gpt2` | `None` | 本地 GPT2 底座路径 |
| `random_seed` | `[2025, 2026, 2027]` | `[2025, 2026, 2027]` | 3 种子循环遍历 |

### 2. 损失权重与学习率动态联动规则
1. **三损失权重约束**：
   $$\text{output\_w} = 1.0 - \text{task\_w} - \text{feature\_w}$$
   若 $\text{output\_w} < 0$，属于无效参数组合，自动跳过（Skip）该组合并记录日志，避免无效运行。
2. **学习率与累加联动**：
   $$\text{learning\_rate} = 0.00000625 \times \text{batch\_size} \times \text{accumulation\_steps}$$
   （当 `batch_size=16`, `accum=4` 时，$\text{lr} = 0.0004$）。
3. **动态 Model ID**：
   $$\text{model\_id} = \text{model}\_\text{seq\_len}\_\text{pred\_len}\_\text{p}\text{percent}$$
   （例如 `CALF_96_96_p100`）。

---

## 二、架构改动与实施方案

1. **更新 `config_manager.py`**：
   - 更新 `_get_default_fixed_params()` 基础字典，将上述所有 24 个参数 100% 对齐为默认值；
   - 在 `generate_task_combos()` 中增加动态权重计算和跳过机制：当设置中含有 `task_w` 和 `feature_w` 且未显式锁定 `output_w` 时，自动计算 `output_w = round(1.0 - task_w - feature_w, 6)`；若 `output_w < 0` 则自动过滤跳过；
   - 自动生成符合规范的 `model_id`。
2. **更新 GUI 界面 `gui_panel.py`**：
   - 在基础固定超参面板中，展示并支持编辑所有新增参数（`accumulation_steps`、`tq_inject_target`、`tq_dropout`、`enc_in`、`c_out`、`smooth_l1` 损失、`tmax`、`num_workers` 等）；
   - 增加辅助联动按钮：`[根据 task_w & feature_w 自动联动 output_w]`；
   - 实时显示计算结果：$0.736 + 0.05 + 0.214 = 1.0$。
3. **更新模板 `configs/batch_templates/solar_al_seeds.json`**：
   - 将模板内容全面升级为该套规范配置。
4. **自动化回归测试**：
   - 校验参数生成的准确性与无效权重组合过滤机制。
