# -*- coding: utf-8 -*-
"""
Interactive Terminal UI (TUI) and Menu Controller for CALF Batch Runner.
"""
import os
import sys
import glob
from datetime import datetime
from typing import Optional
from .config_manager import ConfigManager
from .state_manager import StateManager
from .runner_engine import RunnerEngine


class BatchPanelUI:
    # ANSI color codes
    C_RESET = "\033[0m"
    C_BOLD = "\033[1m"
    C_CYAN = "\033[36m"
    C_GREEN = "\033[32m"
    C_YELLOW = "\033[33m"
    C_RED = "\033[31m"
    C_MAGENTA = "\033[35m"
    C_BLUE = "\033[34m"

    def __init__(self, config_path: Optional[str] = None, checkpoints_root: Optional[str] = None):
        self.config_mgr = ConfigManager()
        if config_path and os.path.exists(config_path):
            self.config_mgr.load_from_file(config_path)

        if checkpoints_root:
            self.config_mgr.set_checkpoints_root(checkpoints_root)

        self.state_mgr = StateManager(
            checkpoints_root=self.config_mgr.checkpoints_root,
            batch_name=self.config_mgr.batch_name
        )
        self.engine = RunnerEngine()
        self._refresh_tasks_and_state()

    def _refresh_tasks_and_state(self, reset_interrupted: bool = True):
        """Regenerate tasks from config and sync with StateManager."""
        self.state_mgr.checkpoints_root = self.config_mgr.checkpoints_root
        self.state_mgr.state_file_path = os.path.join(
            self.config_mgr.checkpoints_root, StateManager.STATE_FILENAME
        ).replace("\\", "/")
        tasks = self.config_mgr.generate_task_combos()
        self.state_mgr.init_or_resume(tasks, reset_interrupted=reset_interrupted)

    def print_header(self):
        """Print styled header banner."""
        summary = self.state_mgr.get_summary()
        pct = summary["progress_pct"]
        completed = summary["completed"]
        total = summary["total"]
        pending = summary["pending"]
        failed = summary["failed"]

        print(f"\n{self.C_CYAN}{'='*75}{self.C_RESET}")
        print(f"{self.C_BOLD}{self.C_CYAN}    CALF 批量实验控制面板与超参管理系统 (Batch Experiment Panel)    {self.C_RESET}")
        print(f"{self.C_CYAN}{'='*75}{self.C_RESET}")
        print(f"  {self.C_BOLD}批次名称 (Batch Name):{self.C_RESET} {self.config_mgr.batch_name}")
        print(f"  {self.C_BOLD}权重统一根目录 (Checkpoints Root):{self.C_RESET} {self.C_YELLOW}{self.config_mgr.checkpoints_root}{self.C_RESET}")
        print(f"  {self.C_BOLD}循环参数 (Loop Params):{self.C_RESET} {self.config_mgr.loop_params} ({self.config_mgr.mode} 模式)")
        print(f"  {self.C_BOLD}进度汇总 (Progress):{self.C_RESET} {self.C_GREEN}{completed}{self.C_RESET}/{total} ({pct:.1f}%) | "
              f"待执行: {self.C_YELLOW}{pending}{self.C_RESET} | 失败/异常: {self.C_RED}{failed}{self.C_RESET}")
        print(f"{self.C_CYAN}{'-'*75}{self.C_RESET}")

    def menu_main(self):
        """Main interactive menu loop."""
        while True:
            self.print_header()
            print("  [1] 启动 / 断点继续当前批次实验 (Run / Resume Batch)")
            print("  [2] 指定实验权重保存大文件夹 (Set Checkpoint Master Directory)")
            print("  [3] 载入预置实验模板 (Load Preset Template)")
            print("  [4] 调整 / 添加超参数与循环列表 (Edit Hyperparameters & Loops)")
            print("  [5] 查看当前任务清单与执行状态 (View Tasks & Status)")
            print("  [6] 重置失败 / 中断的任务状态 (Reset Failed / Interrupted Tasks)")
            print("  [7] 导出结果汇总表格 (Export Results Summary to CSV)")
            print("  [8] 保存当前批次配置到文件 (Save Config to File)")
            print("  [0] 退出控制面板 (Exit)")
            print(f"{self.C_CYAN}{'-'*75}{self.C_RESET}")

            choice = input("请选择操作 [0-8]: ").strip()

            if choice == "1":
                self.action_run_batch()
            elif choice == "2":
                self.action_set_checkpoints_root()
            elif choice == "3":
                self.action_load_template()
            elif choice == "4":
                self.action_edit_hyperparams()
            elif choice == "5":
                self.action_view_tasks()
            elif choice == "6":
                self.action_reset_tasks()
            elif choice == "7":
                self.action_export_results()
            elif choice == "8":
                self.action_save_config()
            elif choice == "0":
                print("\n[OK] 退出 CALF 批量实验控制面板。祝实验顺利！\n")
                break
            else:
                print(f"{self.C_RED}无效选项，请重新输入。{self.C_RESET}")

    def action_run_batch(self):
        """Run or resume batch execution."""
        self._refresh_tasks_and_state(reset_interrupted=True)
        pending = self.state_mgr.get_pending_tasks()
        if not pending:
            print(f"\n{self.C_GREEN}[OK] 所有任务均已完成，无需继续运行！若需重新运行可选择菜单 [6] 重置状态。{self.C_RESET}")
            input("\n按回车键返回主菜单...")
            return

        print(f"\n即将执行/续跑 {len(pending)} 个待执行任务...")
        tasks = self.config_mgr.generate_task_combos()
        try:
            self.engine.run_all_pending(tasks, self.state_mgr)
        except KeyboardInterrupt:
            print(f"\n{self.C_YELLOW}[!] 运行已由用户中断，保存点已落盘。下次启动将自动从此断点续跑。{self.C_RESET}")
        self._refresh_tasks_and_state(reset_interrupted=False)
        input("\n按回车键返回主菜单...")

    def action_set_checkpoints_root(self):
        """Prompt to change master checkpoint directory."""
        print(f"\n当前大文件夹 (Current Root): {self.config_mgr.checkpoints_root}")
        print("预选快捷选项:")
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        print(f"  [1] ./checkpoints/batch_{ts}/ (自动时间戳独立隔离目录)")
        print(f"  [2] ./checkpoints/solar_al_baseline/ (Solar-AL 基准复现目录)")
        print(f"  [3] ./checkpoints/solar_fewshot_sweep/ (少样本微调扫描目录)")
        print(f"  [4] 输入自定义路径 (Custom Path)")
        print(f"  [0] 取消返回 (Cancel)")

        choice = input("请选择 [0-4]: ").strip()
        new_path = ""
        if choice == "1":
            new_path = f"./checkpoints/batch_{ts}"
        elif choice == "2":
            new_path = "./checkpoints/solar_al_baseline"
        elif choice == "3":
            new_path = "./checkpoints/solar_fewshot_sweep"
        elif choice == "4":
            new_path = input("请输入目标权重保存大文件夹路径 (例如 ./checkpoints/my_run/): ").strip()
        else:
            return

        if new_path:
            self.config_mgr.set_checkpoints_root(new_path)
            self._refresh_tasks_and_state()
            print(f"{self.C_GREEN}[OK] 已更新实验权重统一大文件夹为: {self.config_mgr.checkpoints_root}{self.C_RESET}")
        input("\n按回车键返回主菜单...")

    def action_load_template(self):
        """Load a preset JSON template."""
        template_dir = "configs/batch_templates"
        files = glob.glob(f"{template_dir}/*.json")
        if not files:
            print(f"{self.C_YELLOW}未在 {template_dir} 发现预置模板。{self.C_RESET}")
            input("\n按回车键返回主菜单...")
            return

        print("\n可用的预置模板列表:")
        for idx, fpath in enumerate(files, 1):
            fname = os.path.basename(fpath)
            print(f"  [{idx}] {fname} ({fpath})")
        print("  [0] 取消返回")

        c = input("请选择模板编号: ").strip()
        if c.isdigit() and 1 <= int(c) <= len(files):
            chosen = files[int(c) - 1]
            try:
                self.config_mgr.load_from_file(chosen)
                self._refresh_tasks_and_state()
                print(f"{self.C_GREEN}[OK] 成功载入模板: {chosen}{self.C_RESET}")
            except Exception as e:
                print(f"{self.C_RED}[!] 载入模板失败: {e}{self.C_RESET}")
        input("\n按回车键返回主菜单...")

    def action_edit_hyperparams(self):
        """View and edit hyperparameters & loops."""
        while True:
            print(f"\n{self.C_BOLD}=== 超参数与循环配置 ==={self.C_RESET}")
            print(f"当前循环模式: {self.C_YELLOW}{self.config_mgr.mode}{self.C_RESET} (grid: 笛卡尔积全网格; paired: 对应行并列)")
            print(f"{self.C_BOLD}循环参数 (Loop Params):{self.C_RESET}")
            if self.config_mgr.loop_params:
                for k, v in self.config_mgr.loop_params.items():
                    print(f"  * {k}: {v}")
            else:
                print("  (无循环参数)")

            print(f"\n{self.C_BOLD}固定基础参数 (Fixed Params 节选):{self.C_RESET}")
            sample_keys = ["data", "model", "seq_len", "pred_len", "learning_rate", "train_epochs", "task_w", "cycle", "use_direct_trend", "data_split"]
            for k in sample_keys:
                if k in self.config_mgr.fixed_params:
                    print(f"  * {k}: {self.config_mgr.fixed_params[k]}")

            print("\n操作选项:")
            print("  [1] 添加/修改循环参数 (例如 random_seed: 2025,2026,2027)")
            print("  [2] 删除某个循环参数")
            print("  [3] 修改固定超参值 (例如 task_w, pred_len, learning_rate)")
            print("  [4] 切换循环模式 (Grid <--> Paired)")
            print("  [0] 返回上级菜单")

            c = input("请选择操作 [0-4]: ").strip()
            if c == "1":
                key = input("请输入要循环的超参名称 (如 random_seed / pred_len / task_w / cycle): ").strip()
                if key:
                    raw_vals = input(f"请输入 {key} 的取值 (支持 '起点:步长:终点' 如 0.1:0.1:0.5 或逗号分隔 如 2025,2026,2027): ").strip()
                    if raw_vals:
                        try:
                            self.config_mgr.set_loop_param(key, raw_vals)
                            self._refresh_tasks_and_state()
                            print(f"{self.C_GREEN}[OK] 已设置循环参数: {key} = {self.config_mgr.loop_params[key]}{self.C_RESET}")
                        except Exception as e:
                            print(f"{self.C_RED}[!] 解析循环参数失败: {e}{self.C_RESET}")
                        print(f"{self.C_GREEN}[OK] 已设置循环参数: {key} = {val_list}{self.C_RESET}")
            elif c == "2":
                key = input("请输入要删除的循环参数名称: ").strip()
                if key in self.config_mgr.loop_params:
                    self.config_mgr.loop_params.pop(key, None)
                    self._refresh_tasks_and_state()
                    print(f"{self.C_GREEN}[OK] 已移除循环参数: {key}{self.C_RESET}")
            elif c == "3":
                key = input("请输入要修改的固定超参名称: ").strip()
                if key:
                    raw_val = input(f"请输入 {key} 的新值: ").strip()
                    if raw_val.isdigit():
                        val = int(raw_val)
                    else:
                        try:
                            val = float(raw_val)
                        except ValueError:
                            if raw_val.lower() == "true":
                                val = True
                            elif raw_val.lower() == "false":
                                val = False
                            else:
                                val = raw_val
                    self.config_mgr.set_fixed_param(key, val)
                    self._refresh_tasks_and_state()
                    print(f"{self.C_GREEN}[OK] 已更新固定超参: {key} = {val}{self.C_RESET}")
            elif c == "4":
                self.config_mgr.mode = "paired" if self.config_mgr.mode == "grid" else "grid"
                self._refresh_tasks_and_state()
                print(f"{self.C_GREEN}[OK] 当前循环模式已切换为: {self.config_mgr.mode}{self.C_RESET}")
            elif c == "0":
                break

    def action_view_tasks(self):
        """View detailed table of tasks."""
        self._refresh_tasks_and_state(reset_interrupted=False)
        tasks = sorted(list(self.state_mgr.state.get("tasks", {}).values()), key=lambda x: x.get("task_idx", 0))

        print(f"\n{self.C_BOLD}{'='*85}{self.C_RESET}")
        print(f"{'Task ID':<10} {'Status':<12} {'MSE':<12} {'MAE':<12} {'Duration':<10} {'Loop Parameters'}")
        print(f"{'-'*85}")

        for t in tasks:
            tid = t.get("task_id", "")
            status = t.get("status", "")
            metrics = t.get("metrics") or {}
            mse_str = f"{metrics['mse']:.6f}" if metrics.get("mse") is not None else "-"
            mae_str = f"{metrics['mae']:.6f}" if metrics.get("mae") is not None else "-"
            dur_str = f"{t.get('duration_sec', 0):.1f}s" if t.get("duration_sec") else "-"
            sig_str = str(t.get("loop_signature", {}))

            if status == "COMPLETED":
                status_colored = f"{self.C_GREEN}{status:<12}{self.C_RESET}"
            elif status == "RUNNING":
                status_colored = f"{self.C_CYAN}{status:<12}{self.C_RESET}"
            elif status == "FAILED":
                status_colored = f"{self.C_RED}{status:<12}{self.C_RESET}"
            else:
                status_colored = f"{self.C_YELLOW}{status:<12}{self.C_RESET}"

            print(f"{tid:<10} {status_colored} {mse_str:<12} {mae_str:<12} {dur_str:<10} {sig_str}")

        print(f"{'='*85}")
        input("\n按回车键返回主菜单...")

    def action_reset_tasks(self):
        """Reset failed or interrupted tasks back to PENDING."""
        print("\n重置任务状态选项:")
        print("  [1] 仅重置失败 (FAILED) 和中断 (RUNNING) 的任务为 PENDING")
        print("  [2] 重置所有任务为 PENDING (重新执行全量实验)")
        print("  [0] 取消返回")

        c = input("请选择 [0-2]: ").strip()
        if c == "1":
            self.state_mgr.reset_all_failed_or_interrupted()
            print(f"{self.C_GREEN}[OK] 已成功重置所有失败/中断任务为 PENDING。{self.C_RESET}")
        elif c == "2":
            confirm = input("确定要重置包括已完成在内的全部任务吗？(y/N): ").strip().lower()
            if confirm == "y":
                for tid in self.state_mgr.state.get("tasks", {}):
                    self.state_mgr.reset_task(tid)
                print(f"{self.C_GREEN}[OK] 已重置所有任务为 PENDING。{self.C_RESET}")
        input("\n按回车键返回主菜单...")

    def action_export_results(self):
        """Export summary table to CSV."""
        default_csv = os.path.join(self.config_mgr.checkpoints_root, "batch_summary.csv").replace("\\", "/")
        print(f"\n默认导出路径: {default_csv}")
        custom = input("请输入导出 CSV 文件路径 (直接回车使用默认路径): ").strip()
        out_path = custom if custom else default_csv

        try:
            self.state_mgr.export_summary_csv(out_path)
            print(f"{self.C_GREEN}[OK] 成功导出实验结果汇总至: {out_path}{self.C_RESET}")
        except Exception as e:
            print(f"{self.C_RED}[!] 导出失败: {e}{self.C_RESET}")
        input("\n按回车键返回主菜单...")

    def action_save_config(self):
        """Save current configuration to a JSON file."""
        default_file = f"configs/batch_{self.config_mgr.batch_name}.json"
        print(f"\n默认保存路径: {default_file}")
        custom = input("请输入保存配置文件路径 (直接回车使用默认路径): ").strip()
        out_path = custom if custom else default_file

        try:
            self.config_mgr.save_to_file(out_path)
            print(f"{self.C_GREEN}[OK] 成功保存批次配置至: {out_path}{self.C_RESET}")
        except Exception as e:
            print(f"{self.C_RED}[!] 保存失败: {e}{self.C_RESET}")
        input("\n按回车键返回主菜单...")
