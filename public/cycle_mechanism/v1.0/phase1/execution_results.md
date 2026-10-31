# P0 执行记录：数据源核查与诊断准备（v1.0 / phase1）

日期：2026-10-25。关联计划：[subplan.md](subplan.md)。本报告分别记录数据审计、TQ-only 推理验证、统计区间和仍未执行的训练期实验。

## 已完成的数据核查

逐行读取了 `datasets/Solar/solar_AL.txt` 和 `datasets/Solar/solar_AL.csv`，并分块比较了全部数值列：

| 检查项 | TXT | CSV |
| --- | ---: | ---: |
| 数据行数 | 52,560 | 52,560（另有一行表头） |
| 数值通道 | 137 | 137（另有日期列） |
| 首个 CSV 时间 | 无时间列 | 2006-01-01 00:10:00 |
| 末个 CSV 时间 | 无时间列 | 2007-01-01 00:00:00 |
| CSV 相邻记录间隔 | 不可判定 | 全部为 600 秒 |

两文件的 7,200,720 个数值中，3,149,188 个（43.73%）差值大于 `1e-8`；逐元素 MAE 为 `0.300455`、RMSE 为 `0.812856`、最大绝对差为 `31.25`。首个差异在从 0 开始计数的第 42 行、第 0 通道：TXT 为 `0.0`，CSV 为 `0.2`。因此 CSV 的日期序列不能直接挂到 TXT 行号上；训练诊断继续使用 TXT，但只把 `row_index % 144` 记为周期槽位。原有 CSV、图和测试结果未被覆盖或改写。

## 已完成的代码准备

- `data_provider/data_factory.py`：Solar 验证集和测试集均按时间顺序加载，不丢弃尾部窗口。验证阶段现在可完整覆盖其时间切分中的窗口。
- `data_provider/data_loader.py`：`Dataset_Solar` 暴露总行数、切分起止行，用于逐窗口追溯。
- `validate_cycle_mechanism.py`：增加 P0 运行入口和验证集/测试集选择；测试运行必须提供单一 `--confirm_shift`。旧随机条件更名为“独立随机索引”，不再错误称为 permutation；几何图和注意力图只标周期槽位，不标未经核对的钟点。
- `scripts/cycle_mechanism/phase_diagnostics.py`：实现 TQ-only 相位扫描、TQ 置零、三个随机索引种子、逐窗口与分预测步长分组误差、Clean 恢复比较、真实前向注意力 hook、检查点键清单及运行 manifest。当前默认保持 Queue 与 WTE 原值；每次运行写入独立目录。
- `scripts/cycle_mechanism/analyze_paired_errors.py`：对 Clean 与预选 TQ 条件做配对 moving-block bootstrap，并核对每 96 个起点取样的不重叠预测窗口。

预期运行目录包含：

- `manifest.json`：参数、Git 状态、检查点/数据/WTE 文件 SHA-256、加载键、数据切分边界和设备信息。
- `summary_metrics.csv`、`per_window_errors.csv`、`horizon_summary.csv`：整体及逐预测起点的双分支误差，包括四段预测步长的 MSE/MAE。
- `phase_slot_summary.csv`：按输入末点周期槽位汇总的窗口数和误差。
- `attention_by_cycle_slot.csv`：实际前向捕获的原型注意力和每个槽位的样本数。

## 验证集推理结果

运行编号：`solar_al_p0_val_20261025_retry1`。使用项目已有 Conda `calf` 环境的 PyTorch 2.4.1+cu124 和 NVIDIA GeForce RTX 4070 Laptop GPU。DataLoader 顺序加载验证集全部 5,161 个窗口；检查点加载键没有缺失或多余。原始运行共 26 个条件，其中包括用户收窄范围前原方案带入的 Queue 控制条件；逐窗口文件共有 134,186 条记录（另含表头）。本报告只采用 TQ 条件和 Clean 行作为结论依据，不纳入 Queue 结果。末尾 Clean 与开头 Clean 的 Text/Time MSE、MAE 绝对差均为 `0`。

| 验证集条件 | Text MSE | 相对 Clean | Time MSE | 相对 Clean |
| --- | ---: | ---: | ---: | ---: |
| Clean | 0.141909 | — | 0.137759 | — |
| TQ +1 槽 | 0.133890 | −5.65% | 0.137759 | 0.00% |
| **TQ +2 槽（验证集最低）** | **0.131620** | **−7.25%** | 0.137759 | 0.00% |
| TQ −1 槽 | 0.154709 | +9.02% | 0.137759 | 0.00% |
| TQ +72 槽 | 0.280158 | +97.42% | 0.137759 | 0.00% |
| TQ 置零 | 0.169502 | +19.44% | 0.137759 | 0.00% |
| 独立随机索引（3 种子） | 0.362638–0.394766 | +155.54% 至 +178.18% | 0.137759 | 0.00% |

TQ-only 条件改变 Text MSE，但所有 TQ-only 相位条件下 Time MSE 完全相同，符合 TQ 注入发生在文本分支、而 `outputs_time` 不接收该推理时改动的前向路径。以上是单检查点的推理敏感性，不证明重新训练时加入 TQ 会改善最终 Time MSE。

首次验证集运行是在用户收窄范围前，按原 P0 方案也保存了若干 Queue 控制行；此后的执行范围和本报告结论只保留 TQ 条件，且测试集运行始终只改 TQ、Queue/WTE 均保持原检查点状态。

**预先登记的测试确认偏移：`+2` 个周期槽。** 选择规则是在完整验证相位扫描中选取 Text MSE 最低的偏移。测试运行仅包含 Clean 前后与该单一偏移，不再扫描其它值。由于 TXT 没有时间戳，这个偏移只是当前行号索引协议下的经验偏移，不对应已证实的钟点或物理时角。

## 测试集单偏移确认

运行编号：`solar_al_p0_test_confirm_20261025`。测试集按顺序完整评估 10,417 个窗口，检查点加载键没有缺失或多余，Clean 恢复差值为 0。

| 指标 | Clean | TQ +2 槽 | 相对变化 |
| --- | ---: | ---: | ---: |
| Text MSE | 0.183072 | 0.166135 | −9.25% |
| Text MAE | 0.243379 | 0.226988 | −6.73% |
| Time MSE | 0.164986 | 0.164986 | 0.00% |
| Time MAE | 0.219895 | 0.219895 | 0.00% |

用逐窗口配对误差做 5,000 次 non-circular moving-block bootstrap。块长为 288 和 1,008 个连续起点（保留 row-origin 单位；TXT 时间戳未核实，不把它们换称“天”）：

| 主指标 | 块长 | Text MSE 差值（TQ−Clean）95% CI | 相对变化 95% CI |
| --- | ---: | ---: | ---: |
| Text MSE | 288 | −0.01694 [−0.02323, −0.01169] | [−13.76%, −6.14%] |
| Text MSE | 1,008 | −0.01694 [−0.02514, −0.01227] | [−15.20%, −6.45%] |
| Text MAE | 288 | −0.01639 [−0.02198, −0.01177] | [−9.33%, −4.78%] |
| Text MAE | 1,008 | −0.01639 [−0.02373, −0.01221] | [−10.12%, −4.97%] |

Time MSE/MAE 的逐窗口差值恒为 0，两个块长下区间也均为 `[0, 0]`。另外每隔 96 个起点取一个不重叠预测窗口（109 个起点），Text MSE 差值为 −0.02731，Text MAE 差值为 −0.02376；Time MSE/MAE 仍为 0。

这些区间只描述**一个固定检查点**上的测试期误差不确定性，不包括训练种子差异，也不证明重新训练时启用 TQ 能改善最终 Time MSE。TQ +2 是由验证集选出的文本分支相位条件，数据时间戳未对齐前不能把它解释成真实太阳时偏移。

## 运行产物

运行目录：`results/cycle_mechanism/solar_al_p0_val_20261025_retry1/`。其中 `manifest.json` 保存运行参数、Git 状态、数据/检查点/WTE 哈希、完整标准化参数、设备与检查点键；`per_window_errors.csv` 保存每个窗口及四个预测步长区间的 Text/Time MSE/MAE；`summary_metrics.csv`、`horizon_summary.csv`、`phase_slot_summary.csv` 和 `attention_by_cycle_slot.csv` 保存聚合表。注意力表含 72,000 个槽位×原型条目，每个槽位记录了实际进入该槽位的验证窗口数。

测试运行目录：`results/cycle_mechanism/solar_al_p0_test_confirm_20261025/`；逐窗口误差和 manifest 保留同一组运行参数。注意力表覆盖 144 个槽位，槽位样本数为 72–73，合计正好 10,417；每个槽位的 500 个原型注意力权重之和约为 1。配对区间和不重叠起点分析保存在该目录的 `paired_block_bootstrap.json`。

如果要从头复现当前 TQ-only 验证扫描，请在项目 `calf` 环境运行并使用新的 run ID：

```powershell
conda run -n calf python validate_cycle_mechanism.py --run_phase_diagnostics --evaluation_split val --device cuda:0 --run_id solar_al_p0_val_tqonly_20261025
```

本次测试确认已完成。若需要复现，在验证集重新确认同一偏移后使用新的 run ID：

```powershell
conda run -n calf python validate_cycle_mechanism.py --run_phase_diagnostics --evaluation_split test --confirm_shift 2 --device cuda:0 --run_id solar_al_p0_test_confirm_retry_20261025
```

## 结论与后续

本阶段另有一次未完成尝试：首轮运行发现 split 行边界属性未暴露，修复后以新 run ID 完整重跑；失败目录 `results/cycle_mechanism/solar_al_p0_val_20261025/failure.json` 保留原因，未计入上述结果。**P0 的 TQ-only 验证集扫描、预选偏移测试确认及配对区间分析均已完成。** 本次只验证固定检查点下的 TQ 条件敏感性；用户最新要求保持 `use_tq` 不变，原 P1 开关配对训练方案已撤下，不继续剩余种子。P0 已完成结果继续有效；phase2 的已有运行仅作历史记录。Queue/WTE 对照不在活动待办中，历史文件保留。
