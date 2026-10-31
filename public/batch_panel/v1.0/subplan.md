# 批量实验控制面板与超参管理子规划 (Subplan v1.0)

本文档依照团队规范维护，定义 CALF 实验批量运行控制面板（Batch Panel）与超参统一管理系统的架构设计、执行计划与技术规范。

---

## 一、需求背景与目标

在 CALF 模型的消融、敏感性分析与多任务微调中，需频繁在不同超参组合（种子、步长、损失权重、网络结构、周期等）下执行批量训练。当前存在以下核心痛点：
1. **日志与检查点标识冲突 (Setting Collision)**：
   `run.py` 原有的 `setting` 字符串构造仅包含部分基础参数，未覆盖 `task_w`、`cycle`、`use_direct_trend`、`out_mlp_layers`、`tq_mode`、`data_split` 等关键超参。当用户扫描不同周期或损失权重时，生成完全相同的 `setting` 目录名，导致检查点和日志相互覆盖。
2. **缺乏统一大文件夹管理 (Unified Checkpoint Root)**：
   `exp_long_term_forecasting.py` 内部部分逻辑硬编码了 `'./checkpoints/'` 相对路径，无法根据不同实验批次方便地指定统一目标根目录（如 `checkpoints/batch_20261101/`）。
3. **缺乏鲁棒的断点续跑机制 (Breakpoint Continuation)**：
   批量实验一旦因中断（Ctrl+C）、显存异常或机器重启终止，缺乏结构化的任务清单记录（Manifest/State），容易造成重复跑或漏跑。
4. **超参配置繁琐、缺乏灵活循环机制 (Hyperparameter Loops)**：
   Shell 脚本在 Windows PowerShell 环境下存在兼容性与循环编写困难，需要一个在 Python 原生支持全超参调整、网格循环（Cartesian Product/Zip）的交互式与脚本化控制面板。

---

## 二、架构设计与模块划分

依规范对功能进行高内聚、低耦合拆分，杜绝单文件臃肿：

```
CALF-main/
├── batch_panel.py                     # 统一根入口（交互式 TUI 与命令行双模）
├── scripts/batch_runner/              # 批量运行核心模块
│   ├── __init__.py
│   ├── config_manager.py              # 配置解析、超参网格笛卡尔积/循环展开引擎
│   ├── state_manager.py               # batch_state.json 状态持久化与断点续跑控制器
│   ├── runner_engine.py               # 子进程调度、日志捕获、指标自动提取
│   └── panel_ui.py                    # 终端交互面板（ANSI彩色高亮、菜单交互）
├── configs/batch_templates/           # 预置实验模板库
│   ├── solar_al_seeds.json            # 3 种子基准复现模板
│   ├── solar_fewshot.json             # 少样本多比率微调模板
│   └── hyperparam_grid.json           # 损失权重与周期网格搜索模板
└── public/batch_panel/v1.0/           # 阶段成果与文档归档
    ├── subplan.md                     # 本计划文档
    └── execution_results.md           # 交付与验证报告
```

---

## 三、核心技术实现方案

### 1. `run.py` 与 `exp` 签名防冲突与超参完整记录
- **动态差异标签注入**：
  在 `run.py` 中新增 `--exp_tag` 参数；同时检测 `task_w`、`cycle`、`use_direct_trend`、`out_mlp_layers`、`tq_mode`、`data_split` 等参数。当其与基线默认值不同时，自动在 `setting` 中追加唯一特征缩写（如 `_tw0.5_cyc48_trend1`），杜绝覆盖；基线默认参数保持原 setting 不变，确保向下兼容。
- **修复硬编码检查点路径**：
  将 `exp_long_term_forecasting.py` 中 4 处硬编码 `'./checkpoints/' + setting` 统一改为 `os.path.join(self.args.checkpoints, setting)`，使 `--checkpoints <custom_dir>` 完全生效。
- **全超参快照落盘**：
  在模型保存及测试结果目录中保存 `args.json`，确保每组实验所有 50+ 个参数均有确凿可追溯记录。

### 2. 状态机持久化与断点续跑机制
- 每个批量批次在指定的 `--checkpoints <root>` 目录下维护 `batch_state.json`。
- 任务状态枚举：`PENDING`（待运行）、`RUNNING`（正在运行）、`COMPLETED`（已完成）、`FAILED`（失败）。
- 启动检测逻辑：
  - 扫描已有 `batch_state.json`；
  - 若任务标记为 `COMPLETED` 且目标 checkpoint 或结果已生成，自动跳过（Skip）；
  - 若为 `RUNNING`（上次被强制终止）或 `PENDING`，从该保存点无缝继续；
  - 面板提供 `[继续未完成]`、`[仅重试失败]`、`[重新跑全部]` 等选项。

### 3. 灵活循环与超参数网格引擎
- 支持任意 `run.py` 接受的超参；
- 参数值为列表时（如 `"random_seed": [2025, 2026, 2027]`, `"pred_len": [96, 192]`），自动生成网格笛卡尔积（Grid Sweep）；
- 支持指定实验大文件夹路径（如 `checkpoints/grid_test_2026/`），一键指定后该批次所有模型的权重和状态文件统一收敛至该大文件夹。

### 4. 控制面板交互与一键入口集成
- 纯 Python 跨平台终端 UI，无须额外重量级依赖，支持菜单式导航、配置载入/保存、实时进度条、剩余耗时估算、实时指标展示与 CSV 导出；
- 集成进 [run_one_click.ps1](file:///run_one_click.ps1)，新增菜单项 `[6] 批量运行控制面板 (Batch Panel)`。

---

## 四、实施里程碑 (Milestones)

1. **Phase 1: 审计并修补底座参数冲突与路径硬编码**
   - 修改 `run.py`：支持 `--exp_tag` 与动态参数标签扩展。
   - 修改 `exp/exp_long_term_forecasting.py`：统一路径解析与 `args.json` 导出。
2. **Phase 2: 构建批量核心模块 (`scripts/batch_runner/`)**
   - 实现 `config_manager.py`（网格循环展开）。
   - 实现 `state_manager.py`（断点续跑状态持久化）。
   - 实现 `runner_engine.py`（子进程调度与实时解析）。
   - 实现 `panel_ui.py` 与入口 `batch_panel.py`。
3. **Phase 3: 预置实验模板并与 `run_one_click.ps1` 联动**
   - 编写标准基线模板与网格搜索模板。
   - 在主启动器加入控制选项。
4. **Phase 4: 全流程仿真与回归验证**
   - 测试超参循环展开逻辑。
   - 测试模拟中断后断点续跑正确跳过已完成项。
   - 测试不同超参组合权重隔离与大文件夹统一管理。
   - 产出 [execution_results.md](execution_results.md) 并更新 [public/plan.md](../../plan.md)。
