# 子计划：CALF 少样本实验体系与双范式规范 (v1.0)

本文档规划并规范 CALF 项目中的两类少样本（Few-Shot）实验范式，提供图形界面面板、一键脚本与原生命令行三维度的执行指南与模板支持。

---

## 1. 核心背景与研究目标

在时间序列大模型（CALF）评估中，少样本能力主要评估模型在训练数据匮乏场景下的表征迁移能力与数据利用效率。根据学术界标准和本项目实现，划分为两种研究范式：

1. **范式 A：跨域少样本迁移微调 (Cross-Domain Few-Shot Transfer Fine-Tuning)**
   - **典型场景**：使用在源域（如 Solar-AL 137维）上充分训练好的基础检查点权重，迁移至目标域（如 Solar-IL 106维）并在极低数据比例（0.5%, 1%, 2%, 3%, 4%, 5%, 10%）下微调。
   - **核心机制**：通过 `checkpoint_path` 指定预训练权重（如 `./checkpoints/AL_pretrained_best_96_96_seed2025.pth`），在 `exp/exp_basic.py` 中以 `strict=False` 动态适配通道输入/输出维度（137 -> 106），保留 GPT-2 主干、LoRA 与 TQ 物理周期向量的时序动态表征，仅微调投影层与输出头。

2. **范式 B：同域少样本从头训练/数据衰减 (In-Domain Few-Shot Scratch / Data Efficiency Decay)**
   - **典型场景**：在 Solar-AL 上直接截取部分训练数据（如 10%, 20%, 30%, 40%, 50%），从头训练全新模型，不依赖任何预训练权重（`checkpoint_path = 0`）。
   - **核心机制**：用于评估模型在没有大规模先验权重时，仅凭轻量结构从不同规模有限数据中学习模式的容量极限与衰减曲线。

---

## 2. 实施方案与模板配置

为保证团队协作与快速复现，构建以下标准配置产物：

| 实验范式 | 数据集 | 训练比例 (Percent) | 预训练权重 (checkpoint_path) | 配置文件 / 模板 |
| :--- | :--- | :--- | :--- | :--- |
| **A. 跨域微调 (Solar-IL)** | `solar_IL.csv` (106维) | `0.5, 1, 2, 5, 10` | `./checkpoints/AL_pretrained_best_96_96_seed2025.pth` | `configs/batch_templates/solar_il_fewshot.json` |
| **B. 从头衰减 (Solar-AL)** | `solar_AL.txt` (137维) | `10, 20, 30, 40, 50` | `0` (无) | `configs/batch_templates/solar_fewshot.json` |

---

## 3. 三种执行途径

1. **GUI 控制面板 (推荐，最直观)**：
   - 运行 `python batch_panel.py`。
   - 点击“📂 载入模板”，选择 `solar_il_fewshot.json`（微调）或 `solar_fewshot.json`（从头）。
   - 确认循环参数与权重路径后，点击“▶ 启动 / 断点续跑”。
2. **一键整合脚本**：
   - 运行 `.\run_one_click.ps1 -Task 1`。
   - 自动遍历 7 组微调比例 x 3 随机种子，自动跳过已完成项，实时输出至 `results/results_fewshot_il_micro.csv`。
3. **原生命令行**：
   - 运行 `python run.py --percent 10 --checkpoint_path checkpoints/AL_pretrained_best_96_96_seed2025.pth ...`。
