# 批量实验控制面板与超参管理系统执行结果报告 (v1.0)

本文档记录 CALF 批量实验控制面板（Batch Panel）、断点续跑机制、防日志/检查点冲突签名扩展以及统一权重根目录管理的实施与测试结果。

---

## 一、交付物总览

依照团队协作规范，本期（v1.0）交付以下核心组件：

1. **统一控制面板入口**：
   - [batch_panel.py](file:///batch_panel.py)：支持终端全交互（TUI）与命令行无头（CLI）双模式。
   - [run_one_click.ps1](file:///run_one_click.ps1)：在主菜单新增选项 `[6] 批量实验控制面板 (Batch Panel)`，支持直接启动。
2. **模块化核心架构**：
   - [scripts/batch_runner/config_manager.py](file:///scripts/batch_runner/config_manager.py)：全超参解析、笛卡尔积（Grid）与并列（Paired）循环生成引擎。
   - [scripts/batch_runner/state_manager.py](file:///scripts/batch_runner/state_manager.py)：`batch_state.json` 状态机、断点续跑判定、指标落盘与 CSV 导出。
   - [scripts/batch_runner/runner_engine.py](file:///scripts/batch_runner/runner_engine.py)：独立子进程调度、实时指标抓取、任务隔离日志输出及 Ctrl+C 优雅中断。
   - [scripts/batch_runner/panel_ui.py](file:///scripts/batch_runner/panel_ui.py)：ANSI 终端交互控制台，支持菜单导航、参数修改、进度监控。
3. **预置实验配置模板**：
   - [configs/batch_templates/solar_al_seeds.json](file:///configs/batch_templates/solar_al_seeds.json)：Solar-AL 3 种子（2025, 2026, 2027）基准复现模板。
   - [configs/batch_templates/solar_fewshot.json](file:///configs/batch_templates/solar_fewshot.json)：极低样本（0.5% ~ 10%）少样本微调循环模板。
   - [configs/batch_templates/hyperparam_grid.json](file:///configs/batch_templates/hyperparam_grid.json)：周期（12, 24, 48）与损失权重（0.5, 1.0, 2.0）网格搜索模板。
4. **底座核心修复**：
   - [run.py](file:///run.py)：新增 `--exp_tag`，重构 `setting` 生成逻辑，当 `task_w`、`cycle`、`use_direct_trend`、`out_mlp_layers`、`tq_mode`、`data_split` 偏离默认值时自动追加唯一特征标签；标准基准参数保持 100% 向下兼容。
   - [exp/exp_long_term_forecasting.py](file:///exp/exp_long_term_forecasting.py)：修复 4 处硬编码 `'./checkpoints/' + setting`，使自定义检查点大文件夹在 `train` 与 `test` 全流程生效；在每个实验目录落盘 `args.json` 与 `metrics.json`；优化结果 CSV 命名前缀避免覆写。

---

## 二、关键痛点解决与技术细节

### 1. 彻底解决 run.py 日志与检查点覆盖冲突 (Setting Collision)
- **原因**：原 `run.py` 仅将部分基础参数拼入 `setting`，如 `task_w`、`cycle`、`use_direct_trend` 等差异不会反映在文件夹名称中，导致两组不同超参的实验覆盖同一 checkpoint 和结果。
- **解决方案**：
  在 `run.py` 中引入 `build_setting_string(args, ii)` 统一生成器：
  - 若为标准基线配置，保持原始名称，确保与历史检查点完全兼容；
  - 若传入 `--exp_tag <name>`，或任何扩展超参偏离基线默认值（如 `task_w!=1.0` -> `_tw0.5`，`cycle!=24` -> `_cyc48`，`use_direct_trend!=0` -> `_trend1`），自动注入差异标签；
  - 在检查点与结果目录中自动保存完整的 `args.json`，确保实验 100% 可复现。

### 2. 统一大文件夹收敛管理模型权重 (Unified Checkpoint Root)
- **使用方法**：
  在控制面板中通过菜单 `[2]` 或启动参数 `--checkpoints <path>` 指定目标大文件夹（例如 `./checkpoints/my_grid_2026/`）。
- **执行效果**：
  - 该批次所有模型权重集中存放在 `<path>/<setting>/checkpoint.pth`；
  - 任务执行状态清单持久化在 `<path>/batch_state.json`；
  - 每个任务的完整实时控制台日志存放在 `<path>/logs/<task_id>.log`；
  - 整个实验批次自包含，便于打包归档、迁移与磁盘空间清理。

### 3. 断点续跑与异常恢复机制 (Breakpoint Continuation)
- **状态流转**：
  `PENDING`（排队待跑） $\rightarrow$ `RUNNING`（正在运行） $\rightarrow$ `COMPLETED`（成功结束） / `FAILED`（异常失败）。
- **中断恢复流程**：
  - 批次启动时读取 `batch_state.json`；
  - 已完成任务（`COMPLETED`）自动校验权重或指标，校验通过直接跳过（Skip）；
  - 若因手动中断（Ctrl+C）、显存溢出或重启导致任务停留在 `RUNNING`，自动重置为 `PENDING`，下次启动从该断点继续；
  - 面板提供 `[6] 重置失败/中断任务` 功能，可随时针对失败项一键重试。

### 4. 任意超参配置与循环扫描 (Writing Loops)
- **配置语法**：
  在 JSON 模板或交互菜单中，将任意参数设置为列表即可触发循环，例如：
  ```json
  "loop_params": {
    "random_seed": [2025, 2026, 2027],
    "cycle": [12, 24, 48],
    "task_w": [0.5, 1.0, 2.0]
  }
  ```
- **生成模式**：
  - `grid`：笛卡尔积全网格扫描（3 seeds x 3 cycles x 3 weights = 27 组实验）；
  - `paired`：并列模式（逐项对应扫描）。

---

## 三、测试与验证结果

通过端到端自动化测试套件（含 4 项专项回归）：

1. **ConfigManager 网格展开测试**：
   - 验证 2 种子 $\times$ 2 周期在 grid 模式下生成 4 个独立 Task，在 paired 模式下生成 2 个 Task；
   - 验证 `--checkpoints` 均正确路由到指定大文件夹；
   - **结果：PASSED**。
2. **StateManager 断点续跑与持久化测试**：
   - 模拟任务 1 成功完成、任务 2 在 RUNNING 状态下意外崩溃；
   - 重新加载状态机：任务 1 保持 COMPLETED 并保留 MSE/MAE 指标；任务 2 自动识别中断并重置为 PENDING 待续跑；任务 3 保持 PENDING；
   - **结果：PASSED**。
3. **Setting 防冲突签名测试**：
   - 分别以 Baseline、`--exp_tag batchA`、`--cycle 48`、`--task_w 0.5`、`--use_direct_trend 1` 启动测试；
   - 5 组实验生成的 `setting` 互不相同，差异标签精确注入；基线配置保持原格式向下兼容；
   - **结果：PASSED**。
4. **batch_panel.py CLI 与导出测试**：
   - 无头模式下加载模板、指定大文件夹、执行导出 CSV 并退出；
   - 状态落盘与汇总 CSV 格式校验完全正确；
   - **结果：PASSED**。

---

## 四、快速使用说明

### 1. 方式一：通过主启动器交互使用（推荐）
在项目根目录运行：
```powershell
.\run_one_click.ps1
```
在控制台菜单中输入 `6` 即可直达【批量实验控制面板】。

### 2. 方式二：直接启动 Python 交互面板
```powershell
& "D:\software\conda\envs\calf\python.exe" batch_panel.py
```
- 按 `[2]` 快速指定保存大文件夹（如选择预选的时间戳目录或输入自定义路径）；
- 按 `[3]` 载入预置模板（如 `solar_al_seeds.json`）；
- 按 `[4]` 增删超参循环；
- 按 `[1]` 开始执行或断点续跑。

### 3. 方式三：命令行无头批量执行（后台/脚本调用）
```powershell
# 1. 运行 3 种子基线批量实验并存入指定大文件夹
& "D:\software\conda\envs\calf\python.exe" batch_panel.py --config configs/batch_templates/solar_al_seeds.json --checkpoints ./checkpoints/my_batch_01/ --run

# 2. 中途断开后继续（直接重新执行相同命令，自动从断点继续）
& "D:\software\conda\envs\calf\python.exe" batch_panel.py --config configs/batch_templates/solar_al_seeds.json --checkpoints ./checkpoints/my_batch_01/ --run

# 3. 导出该批次实验结果到 CSV
& "D:\software\conda\envs\calf\python.exe" batch_panel.py --checkpoints ./checkpoints/my_batch_01/ --export ./results/my_batch_01_summary.csv
```
