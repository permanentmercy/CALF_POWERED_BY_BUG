# 批量实验图形界面控制面板 (GUI Dashboard) 子规划 (Subplan v2.0)

本文档依照团队规范维护，定义 CALF 批量实验图形界面（Desktop GUI Dashboard）的架构设计、界面交互、多线程实时日志流式推送与断点续跑管理规范。

---

## 一、需求背景与目标

在 v1.0 交付的 CLI/TUI 终端面板基础上，用户明确提出：
1. **纯桌面图形界面（GUI）**：摆脱纯字符终端限制，提供现代、直观的桌面操作面板；
2. **实时查看执行输出**：模型训练的每一行 Epoch Loss、验证集 EarlyStopping、测试集 MSE/MAE 必须在界面控制台实时流式滚动输出，杜绝界面卡死（Not Responding）；
3. **断点续跑无缝集成**：中断（Stop/Ctrl+C/意外崩溃）后再次点击启动，自动检测 `batch_state.json` 并从最后一个保存点继续；
4. **全超参调整与写循环**：支持界面直接编辑固定基础超参，以及配置循环超参（如 `random_seed: [2025, 2026, 2027]`、`cycle: [12, 24, 48]`、`task_w: [0.5, 1.0, 2.0]` 等），自动计算笛卡尔积（Grid）或并列（Paired）组合；
5. **统一大文件夹快捷管理**：支持 Windows 原生目录选择器与快捷预设，将该批次所有模型的权重、独立日志与状态清单集中隔离存放在指定大文件夹中。

---

## 二、架构设计与技术路线

为保证零外部重型依赖（如 PyQt 或 Web 框架可能存在的版本不兼容或端口冲突），采用 Python 内置的标准桌面图形库 **Tkinter + ttk**，具备启动迅速、原生适配 Windows 桌面、系统开销极低的优势。

### 1. 模块组织架构
```
CALF-main/
├── batch_panel.py                     # 统一根入口（默认启动 GUI，支持 --cli / --run 参数）
├── scripts/batch_runner/
│   ├── __init__.py
│   ├── config_manager.py              # 配置管理与循环参数笛卡尔积展开
│   ├── state_manager.py               # batch_state.json 状态机与断点恢复
│   ├── runner_engine.py               # 异步线程子进程调度与实时输出捕获
│   ├── gui_panel.py                   # 【新增 v2.0】Tkinter 现代图形用户界面
│   └── panel_ui.py                    # 终端 CLI/TUI 保留用于无界面环境
├── configs/batch_templates/           # 预置实验模板库
└── public/batch_panel/v2.0/           # v2.0 规划与执行产物
    ├── subplan.md                     # 本计划文档
    └── execution_results.md           # 交付与验证报告
```

### 2. 界面布局规划 (Layout Wireframe)
- **顶部工具栏 (Master Checkpoints & Actions)**:
  - 目标保存大文件夹输入框 + `[浏览文件夹 (Browse)]` 原生目录选择器；
  - 快捷预设按钮：`[新建时间戳批次]`、`[Solar-AL 基准目录]`、`[少样本微调目录]`；
  - 批次控制核心按钮：`▶ 启动 / 断点续跑 (Start/Resume)`、`⏹ 停止运行 (Stop)`、`📂 载入模板`、`💾 保存配置`、`📊 导出 CSV`。
- **左侧配置面板 (Configuration Notebook)**:
  - **标签页 1：超参循环扫描 (Loop Sweeps)**：
    - 参数名与取值列表表格（支持动态添加、修改、删除循环项）；
    - 循环模式单选：`全组合网格 (Cartesian Grid)` vs `并列对齐 (Paired Zip)`；
    - 实时计算并显示即将生成的任务总数。
  - **标签页 2：基础固定超参 (Fixed Parameters)**：
    - 分组卡片（数据与任务、模型架构、TQ机制与门控、训练与优化超参）；
    - 所有 30+ 关键超参均可直接在输入框修改。
- **右侧任务进度列表 (Tasks Treeview)**:
  - 列表展示每个任务的 `Task ID`、`Status`（彩色标签：COMPLETED 绿色、RUNNING 蓝色、PENDING 灰色、FAILED 红色）、`MSE`、`MAE`、`耗时`、`循环参数签名`；
  - 批次总体进度条 (`ttk.Progressbar`) 与百分比指示。
- **底部实时输出控制台 (Real-Time Output Terminal)**:
  - 深色高对比度终端组件（Monospace 等宽字体）；
  - 多线程后台无阻塞推送（Thread-safe Queue），实时滚动显示当前正在运行实验的 stdout 与 stderr；
  - 关键信息关键词高亮（Epoch 进度、Vali Loss 降低、最佳模型保存、最终 MSE/MAE）。

---

## 三、实施里程碑 (Milestones)

1. **Phase 1: 编写 GUI 核心模块 (`scripts/batch_runner/gui_panel.py`)**
   - 构建 Tkinter 响应式主窗口与控件布局；
   - 对接 `ConfigManager` 实现超参及循环配置的双向绑定；
   - 对接 `StateManager` 实现任务树状列表状态刷新与断点进度检测；
   - 构建多线程异步执行器与实时输出队列，实现子进程输出实时流式渲染；
   - 实现原生文件夹浏览、模板载入保存与 CSV 结果导出。
2. **Phase 2: 整合统一入口与启动器**
   - 修改 `batch_panel.py`：默认启动 GUI 界面；传入 `--cli` 时回退至终端 TUI，传入 `--run` 时执行无头批处理；
   - 保持 `run_one_click.ps1` Task [6] 无缝启动 GUI。
3. **Phase 3: 验证与验收测试**
   - 测试 GUI 启动无报错；
   - 测试大文件夹选择器与模板切换；
   - 测试超参循环展开联动；
   - 测试实时日志滚动输出与中断断点续跑；
   - 编写 `execution_results.md` 并更新 `public/plan.md`。
