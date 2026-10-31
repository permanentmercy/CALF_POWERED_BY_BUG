# CALF 原始基准复现子计划 (Baseline Reproduction Subplan)

本文档依照团队协作规范维护，记录 Solar-AL 数据集改动前 CALF 原始基准复现的排查诊断、底层根因修复、执行规范与参数定义。

---

## 一、复现目标指标

依据项目历史原始基准（2026-09-28/2026-10-11 原始记录），目标指标为改动前纯净 CALF 在 Solar-AL 上的 3 种子基准测试成绩（Horizon = 96，Split 7:1:2）：

| 随机种子 (Random Seed) | 预测步长 (Pred Len) | 目标 MSE | 目标 MAE |
| :---: | :---: | :---: | :---: |
| **2025** | 96 | **0.164985827** | **0.219895025** |
| **2026** | 96 | **0.164928949** | **0.219132878** |
| **2027** | 96 | **0.164135638** | **0.218951099** |
| **均值 (Mean)** | 96 | **0.164532293** | **0.219041989** |

---

## 二、指标漂移与阻断根本原因诊断

通过全量代码审计与历史日志对齐，定位到导致该指标无法复现的底层根本原因：

1. **数学结构污染：`use_direct_trend` 默认激活**
   - 历史提交引入了线性自回归趋势门控分支（Linear 96->96，引入 9,216 参数及 `trend_gate`），并在 `run.py` 中默认开启 (`--use_direct_trend 1`)。
   - `Solar_test.sh` 脚本未显式传入该开关，导致全网格模型前向传播被线性趋势污染，可训练参数从 13,099,633 膨胀至 13,108,849，最终 MSE 劣化为 0.16587+。
   - **硬性修复**：在 [models/CALF.py](file:///models/CALF.py) 中重构逻辑，当 `use_direct_trend == 0` 时彻底不实例化该模块；在 [run.py](file:///run.py) 中将默认值改为 `0`；在 [Solar_test.sh](file:///scripts/long_term_forecasting/Solar_test.sh) 中显式声明 `--use_direct_trend 0`。

2. **验证集 DataLoader 行为漂移**
   - 工作区曾将 Solar 数据的验证集 `batch_size` 篡改为 32 且未丢弃尾部样本，由于 `Exp_Long_Term_Forecast.vali()` 是按批次平均计算 Loss (`np.average(total_loss)`)，Batch Size 32 改变了验证集损失，导致 `EarlyStopping` 在不同的 Epoch 提前/延后触发并选择了错误的检查点。
   - **硬性修复**：在 [data_provider/data_factory.py](file:///data_provider/data_factory.py) 中恢复验证集为默认 `batch_size=16`, `drop_last=True`。

3. **DataLoader 内存碎片与 C++ DefaultCPUAllocator 溢出**
   - `Dataset_Solar.__getitem__` 中原本在每个 sample 返回时调用 `torch.zeros((seq_x.shape[0], 1))`。在每个 Epoch 36,601 步迭代下，产生超 7 万次临时 CPU Tensor 分配，在 Windows 平台长时间连续训练多种子时引发严重的 C 堆内存碎片，抛出 `DefaultCPUAllocator: not enough memory` 异常。
   - **硬性修复**：在 [data_provider/data_loader.py](file:///data_provider/data_loader.py) 中预分配静态单例 dummy mark 数组，彻底消除每次调用的张量实例化；将数据在读取阶段统一转为 `float32`，消除每次 collate 的 float64 转换与内存开销；并在 [exp/exp_long_term_forecasting.py](file:///exp/exp_long_term_forecasting.py) 的 Epoch 边界添加显式 `gc.collect()` 与 `torch.cuda.empty_cache()`。

4. **脚本执行层死锁与种子覆盖**
   - 原 `Solar_test.sh` 缺少 `2026 2027` 种子循环，并在历史运行后残留了 `completed_combos.txt` 跳过死锁；
   - 显式声明 `--data_split 7:1:2` 与 `--bestmodel`。

---

## 三、执行规范

- **执行入口**：[run_one_click.ps1](file:///run_one_click.ps1) 任务 `[3]` 与 [scripts/long_term_forecasting/Solar_test.sh](file:///scripts/long_term_forecasting/Solar_test.sh)
- **实时日志**：`logs/CALF/Solar/reproduce_baseline_96_seed*.log`
- **汇总持久化**：`results/baseline_reproduction_results.csv`
- **零磁盘空间占用**：训练期最佳模型保留在内存 (`--bestmodel`)，测试完毕后不驻留冗余巨型权重文件。
