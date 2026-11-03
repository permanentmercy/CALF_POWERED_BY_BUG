# 执行产物：CALF 少样本实验体系与双范式规范 (v1.0)

本产物记录 CALF 少样本实验模块的工程落地、模板对齐验证及各调用入口的联调状态。

---

## 1. 产物清单与文件索引

1. **预训练权重文件**：
   - [AL_pretrained_best_96_96_seed2025.pth](file:///checkpoints/AL_pretrained_best_96_96_seed2025.pth) (419 MB, Solar-AL 预训练最优权重, MSE ~ 0.164986)。
2. **GUI 批量扫描模板**：
   - [solar_il_fewshot.json](file:///configs/batch_templates/solar_il_fewshot.json)：Solar-IL 跨域微调模板（包含 0.5% ~ 10% x 3种子，enc_in=106, dec_in=106, c_out=106，挂载 AL 预训练检查点）。
   - [solar_fewshot.json](file:///configs/batch_templates/solar_fewshot.json)：Solar-AL 从头训练衰减模板（包含 10% ~ 50% x 3种子，enc_in=137, dec_in=137, c_out=137，checkpoint_path=0）。
3. **控制面板与核心组件更新**：
   - [scripts/batch_runner/gui_panel.py](file:///scripts/batch_runner/gui_panel.py)：新增 `checkpoint_path` 与 `dec_in` 基础参数可视化与编辑项。
   - [scripts/batch_runner/config_manager.py](file:///scripts/batch_runner/config_manager.py)：默认参数补齐 `checkpoint_path` 与 `dec_in`，保障跨域维度严格对齐。
4. **一键启动流水线**：
   - [run_one_click.ps1](file:///run_one_click.ps1)：内置 `Task 1 (fewshot_il)`，已集成断点检测、日志持久化与结果聚合。

---

## 2. 验证结果

- **模板解析测试**：
  - 执行 `D:\software\conda\envs\calf\python.exe -c "from scripts.batch_runner.config_manager import ConfigManager; cm = ConfigManager(); cm.load_from_file('configs/batch_templates/solar_il_fewshot.json'); print(len(cm.generate_task_combos()))"`。
  - 输出 `15`，5 个比例 x 3 个随机种子生成 15 组独立任务组合，未发生冲突。
- **权重适配测试**：
  - 验证 `exp/exp_basic.py` 中 `strict=False` 动态权重过滤机制，由于 Solar-AL 为 137 通道，Solar-IL 为 106 通道，输入投影层与输出头尺寸自适应重置，主干网络与 LoRA 权重完整复用。
