# CALF: Cross-modal Alignment and Physical Inductive Biases for Long-Term Solar Forecasting

本仓库包含 CALF 模型的完整实现、全量预训练权重管理、零样本/少样本跨数据集迁移学习评测及复现脚本。

---

## 快速导航
- 📖 **[模型全景技术文档与架构剖析 (MODEL_DOCUMENTATION.md)](MODEL_DOCUMENTATION.md)**：包含代码级数据流转、数学原理、五重物理归纳偏置详解与实测指标分析。
- 🚀 **[全量预训练执行脚本 (solar.sh)](solar.sh)** / **[PowerShell 版本 (solar.ps1)](solar.ps1)**：复现 MSE 0.15965 历史最佳纪录。
- 📊 **[基准评测结果目录 (results/Solar/)](results/Solar/)**：包含全量预训练及少样本迁移的评测明细 CSV。

---

## 核心复现实验

### 1. Solar AL 96 步长全量预训练
- 最佳性能：**MSE: 0.15965 / MAE: 0.19725**
- 对应只读冷备份权重：`checkpoints/AL_pretrained_best_96_96_seed2025.pth`

### 2. Solar IL 跨数据集迁移学习评测
- **Zero-Shot (0%)**：MSE 0.22661 / MAE 0.24231
- **0.1% 极微样本**：MSE 0.21831 / MAE 0.23598
- **0.5% 极微样本**：MSE 0.17973 / MAE 0.21617（显著超越零样本）
- **10% 少样本**：MSE 0.17531 / MAE 0.20791

---

## 跨机器复现指南 (Quickstart on Any Machine)

### 1. 环境准备
```bash
conda create -n calf python=3.10 -y
conda activate calf
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install transformers peft pandas numpy scikit-learn einops
```

### 2. 数据准备
将 `solar_AL.txt`（或 `solar_AL.csv`）放置于 `./datasets/Solar/` 目录下。

### 3. 一键复现最优基准 (MSE: 0.15965)
- **Linux / Git Bash**:
  ```bash
  bash solar.sh
  ```
- **Windows PowerShell**:
  ```powershell
  .\solar.ps1
  ```
> **注**：脚本内置 `--bestmodel` 与 `--no_save_checkpoint`，在内存中动态维护最优验证集模型进行测试评估，无需额外占用磁盘空间保存 `.pth` 权重，训练结束自动输出测试集评测指标。