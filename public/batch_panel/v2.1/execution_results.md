# 全量参数精确对齐与损失权重动态联动执行结果报告 (v2.1)

本文档记录 CALF 批量实验控制面板（GUI & CLI）全面对齐 `Solar_test.sh` 生产脚本超参体系、三损失权重（$task\_w, feature\_w, output\_w$）动态联动与自动跳过无效组合的执行验证结果。

---

## 一、对齐与修改明细

依照用户提供的生产配置脚本（`scripts/long_term_forecasting/Solar_test.sh`），本次更新在控制面板与底层配置管理器中全面对齐了以下参数：

### 1. 全量超参数基准值 100% 对齐
- **数据路径**：`root_path = ./datasets/Solar/`，`data_path = solar_AL.txt`，`data_split = 7:1:2`。
- **三损失权重**：
  - `task_w = 0.736`
  - `feature_w = 0.05`
  - `output_w = 0.214`
  - **动态联动规则**：$output\_w = 1.0 - task\_w - feature\_w$。若 $output\_w < 0$，系统判定为无效参数组合，自动跳过（Skip），完全对齐 Shell 脚本中 `if output_w <= 0: continue` 的逻辑。
- **TQ 与周期机制**：
  - `cycle = 144`
  - `tq_mode = add`
  - `tq_inject_target = q`
  - `tq_lr_factor = 5.0`
  - `tq_dropout = 0.1`
  - `use_tq_gate = 0`
  - `cycle_prior_before_denorm = 0`
  - `use_direct_trend = 0`
- **模型网络架构**：
  - `d_model = 768`，`n_heads = 4`，`d_ff = 3072` ($d\_model \times 4$)
  - `gpt_layers = 1`，`e_layers = 1`，`d_layers = 1`
  - `out_mlp_layers = 3`
  - `enc_in = 137`，`c_out = 137`
  - `r = 8`，`lora_alpha = 32`，`lora_dropout = 0.1`
- **优化与训练策略**：
  - `batch_size = 16`，`accumulation_steps = 4`
  - `learning_rate = 0.0004`（公式：$0.00000625 \times batch\_size \times accumulation\_steps$）
  - `train_epochs = 100`，`patience = 5`，`tmax = 10`，`cos = 1`
  - `task_loss = smooth_l1`，`feature_loss = smooth_l1`，`output_loss = smooth_l1`
  - `eval_test_every_epoch = True`
  - `num_workers = 0`
  - `bestmodel = True`
- **循环参数**：
  - `random_seed: [2025, 2026, 2027]`（3 种子遍历）

---

## 二、代码层面的核心改进

1. **[scripts/batch_runner/config_manager.py](file:///scripts/batch_runner/config_manager.py)**：
   - 更新 `_get_default_fixed_params()` 基础字典，将上述全部 24 个参数作为默认值；
   - 在 `generate_task_combos()` 中增加动态权重计算：若循环或固定参数中未显式指定 `output_w`，自动按 $1.0 - task\_w - feature\_w$ 动态补全；若 $output\_w \le 0$ 则自动过滤跳过；
   - 动态同步 `model_id` 为 `CALF_${seq_len}_${pred_len}_p${percent}`；
   - 动态同步 `d_ff = d_model * 4`。
2. **[scripts/batch_runner/gui_panel.py](file:///scripts/batch_runner/gui_panel.py)**：
   - 在 GUI「⚙ 固定基础超参」标签页中完整呈现 4 大分组的全部对齐字段；
   - 增加联动辅助计算按钮：
     - `⚡ 根据 task_w 和 feature_w 自动联动 output_w (1 - tw - fw)`；
     - `⚡ 根据 batch_size 和 accum 联动 lr`；
     - 动态提示标签：`[三损失和: 0.736 + 0.05 + 0.214 = 1.000]`。
3. **[configs/batch_templates/solar_al_seeds.json](file:///configs/batch_templates/solar_al_seeds.json)**：
   - 模板文件全量更新为此套标准对齐配置。

---

## 三、自动化回归测试结果

执行专项验证脚本：
1. **默认参数比对测试**：
   - 对比 `task_w=0.736`、`feature_w=0.05`、`output_w=0.214`、`cycle=144`、`tq_mode=add`、`tq_inject_target=q`、`out_mlp_layers=3`、`accumulation_steps=4`、`smooth_l1` 等全部参数值；
   - **结果：100% 对齐通过 (PASSED)**。
2. **三损失权重动态计算与非法组合跳过测试**：
   - 测试循环：`task_w: [0.7, 0.8, 0.98]`，`feature_w: [0.05]`；
   - 组合 1 ($0.7+0.05=0.75$) $\rightarrow$ 动态产出 $output\_w=0.25$（有效保留）；
   - 组合 2 ($0.8+0.05=0.85$) $\rightarrow$ 动态产出 $output\_w=0.15$（有效保留）；
   - 组合 3 ($0.98+0.05=1.03 > 1.0$) $\rightarrow$ 动态判定 $output\_w < 0$ 并自动跳过；
   - **结果：有效任务数由 3 过滤为 2，逻辑完全正确 (PASSED)**。
3. **GUI 启动与任务清单测试**：
   - 验证 GUI 加载该配置后，3 个种子任务（2025/2026/2027）生成的命令行参数完整包含 `--cycle 144 --tq_mode add --task_w 0.736 --feature_w 0.05 --output_w 0.214 --accumulation_steps 4 --tq_inject_target q --tq_lr_factor 5.0`；
   - **结果：验证通过 (PASSED)**。
