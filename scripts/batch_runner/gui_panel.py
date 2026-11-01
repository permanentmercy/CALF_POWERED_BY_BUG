# -*- coding: utf-8 -*-
"""
CALF Desktop GUI Dashboard (v2.0)
Full Graphical User Interface for Batch Experiment Runner, Hyperparameter Sweeps,
Breakpoint Resumption, Unified Checkpoint Management, and Real-Time Output Streaming.
"""
import os
import sys
import json
import glob
import time
import queue
import threading
import subprocess
from datetime import datetime
from typing import Dict, Any, List, Optional

import tkinter as tk
from tkinter import ttk, messagebox, filedialog

from .config_manager import ConfigManager, parse_loop_values
from .state_manager import StateManager
from .runner_engine import RunnerEngine


class BatchPanelGUI:
    def __init__(self, root: tk.Tk, initial_config: Optional[str] = None, initial_checkpoints: Optional[str] = None):
        self.root = root
        self.root.title("CALF 批量实验图形控制面板 (Batch Experiment GUI Dashboard)")
        self.root.geometry("1320x860")
        self.root.minsize(1050, 720)

        # Ensure window icon / styling
        self._init_styles()

        # Core state
        self.config_mgr = ConfigManager()
        self.current_config_path: Optional[str] = initial_config

        # Automatic session persistence recovery
        session_file = "configs/.last_session.json"
        if not initial_config and os.path.exists(session_file):
            try:
                with open(session_file, "r", encoding="utf-8") as sf:
                    sdata = json.load(sf)
                    last_cfg = sdata.get("last_config_path")
                    last_ckpt = sdata.get("last_checkpoints_root")
                    if last_cfg and os.path.exists(last_cfg):
                        self.config_mgr.load_from_file(last_cfg)
                        self.current_config_path = last_cfg
                    if last_ckpt:
                        self.config_mgr.set_checkpoints_root(last_ckpt)
                    if sdata.get("last_batch_name"):
                        self.config_mgr.batch_name = sdata["last_batch_name"]
            except Exception:
                pass

        if initial_config and os.path.exists(initial_config):
            self.config_mgr.load_from_file(initial_config)
            self.current_config_path = initial_config
        if initial_checkpoints:
            self.config_mgr.set_checkpoints_root(initial_checkpoints)

        self.state_mgr = StateManager(
            checkpoints_root=self.config_mgr.checkpoints_root,
            batch_name=self.config_mgr.batch_name
        )
        self.engine = RunnerEngine()

        # Threading & Real-time queue
        self.log_queue = queue.Queue()
        self.is_running = False
        self.worker_thread: Optional[threading.Thread] = None

        # Fixed parameter widget variable mapping
        self.fixed_vars: Dict[str, tk.StringVar] = {}

        # Build UI layout
        self._build_top_bar()
        self._build_main_split()
        self._build_log_console()
        self._build_status_bar()

        # Synchronize UI with loaded config
        self._refresh_all_views()

        # Start periodic log queue polling
        self.root.after(50, self._poll_log_queue)

        # Handle window close
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    def _init_styles(self):
        """Configure ttk styles and fonts."""
        self.style = ttk.Style()
        try:
            self.style.theme_use("vista" if sys.platform.startswith("win") else "clam")
        except Exception:
            pass

        self.style.configure("Header.TLabel", font=("Segoe UI", 12, "bold"))
        self.style.configure("SubHeader.TLabel", font=("Segoe UI", 10, "bold"))
        self.style.configure("Action.TButton", font=("Segoe UI", 10, "bold"), padding=4)
        self.style.configure("Run.TButton", font=("Segoe UI", 11, "bold"), padding=6, foreground="#006600")
        self.style.configure("Stop.TButton", font=("Segoe UI", 11, "bold"), padding=6, foreground="#990000")

    # =========================================================================
    # UI Building Blocks
    # =========================================================================

    def _build_top_bar(self):
        """Top frame for Checkpoints Root, Presets, and Main Action Buttons."""
        top_frame = ttk.LabelFrame(self.root, text=" 实验大文件夹管理与批次控制 (Master Checkpoint & Batch Control) ", padding=8)
        top_frame.pack(fill=tk.X, padx=10, pady=(8, 4))

        # Row 1: Checkpoints Root Path + Browse + Presets
        r1 = ttk.Frame(top_frame)
        r1.pack(fill=tk.X, pady=2)

        ttk.Label(r1, text="权重保存大文件夹 (Checkpoints Root):", font=("Segoe UI", 9, "bold")).pack(side=tk.LEFT, padx=(0, 6))
        self.ckpt_root_var = tk.StringVar(value=self.config_mgr.checkpoints_root)
        self.ckpt_entry = ttk.Entry(r1, textvariable=self.ckpt_root_var, width=50)
        self.ckpt_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 6))

        ttk.Button(r1, text="浏览文件夹 (Browse)...", command=self._on_browse_checkpoints).pack(side=tk.LEFT, padx=(0, 6))

        # Presets Combobox
        ttk.Label(r1, text="快捷预设:").pack(side=tk.LEFT, padx=(8, 4))
        self.preset_combo = ttk.Combobox(r1, state="readonly", width=22, values=[
            "新建时间戳隔离目录",
            "Solar-AL 基准复现目录",
            "少样本微调扫描目录"
        ])
        self.preset_combo.pack(side=tk.LEFT, padx=(0, 4))
        self.preset_combo.bind("<<ComboboxSelected>>", self._on_preset_selected)

        # Row 2: Batch Name, Template actions, Run / Stop controls
        r2 = ttk.Frame(top_frame)
        r2.pack(fill=tk.X, pady=(6, 2))

        ttk.Label(r2, text="批次名称 (Batch Name):").pack(side=tk.LEFT, padx=(0, 4))
        self.batch_name_var = tk.StringVar(value=self.config_mgr.batch_name)
        ttk.Entry(r2, textvariable=self.batch_name_var, width=20).pack(side=tk.LEFT, padx=(0, 12))

        ttk.Button(r2, text="📂 载入模板", command=self._on_load_template).pack(side=tk.LEFT, padx=(0, 4))
        ttk.Button(r2, text="💾 保存配置", command=self._on_save_config).pack(side=tk.LEFT, padx=(0, 4))
        ttk.Button(r2, text="📊 导出结果 CSV", command=self._on_export_csv).pack(side=tk.LEFT, padx=(0, 4))
        ttk.Button(r2, text="🔄 刷新状态", command=self._on_refresh_click).pack(side=tk.LEFT, padx=(0, 16))

        # Run / Stop prominent buttons on right
        self.btn_run = ttk.Button(r2, text="▶ 启动 / 断点续跑 (Start / Resume)", style="Run.TButton", command=self._on_start_batch)
        self.btn_run.pack(side=tk.RIGHT, padx=4)

        self.btn_stop = ttk.Button(r2, text="⏹ 停止运行 (Stop)", style="Stop.TButton", command=self._on_stop_batch, state=tk.DISABLED)
        self.btn_stop.pack(side=tk.RIGHT, padx=4)

    def _build_main_split(self):
        """Middle section: PanedWindow with Left (Config Tabs) and Right (Tasks Table)."""
        paned = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        paned.pack(fill=tk.BOTH, expand=True, padx=10, pady=4)

        # Left: Notebook for Loop Sweeps and Fixed Hyperparameters
        left_frame = ttk.Frame(paned)
        paned.add(left_frame, weight=5)

        self.notebook = ttk.Notebook(left_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True)

        # Tab 1: Loop Parameters
        self.tab_loops = ttk.Frame(self.notebook, padding=6)
        self.notebook.add(self.tab_loops, text=" 🔄 循环参数设置 (Loop Sweeps) ")
        self._build_loop_params_tab(self.tab_loops)

        # Tab 2: Fixed Parameters
        self.tab_fixed = ttk.Frame(self.notebook, padding=6)
        self.notebook.add(self.tab_fixed, text=" ⚙ 固定基础超参 (Fixed Params) ")
        self._build_fixed_params_tab(self.tab_fixed)

        # Right: Tasks Treeview & Progress
        right_frame = ttk.LabelFrame(paned, text=" 任务执行清单与断点进度 (Tasks & Breakpoint Status) ", padding=6)
        paned.add(right_frame, weight=6)
        self._build_tasks_table(right_frame)

    def _build_loop_params_tab(self, parent):
        """Construct the loop parameters editor inside Tab 1."""
        top_ctrl = ttk.Frame(parent)
        top_ctrl.pack(fill=tk.X, pady=(0, 6))

        ttk.Label(top_ctrl, text="循环模式 (Sweep Mode):").pack(side=tk.LEFT, padx=(0, 4))
        self.mode_var = tk.StringVar(value=self.config_mgr.mode)
        rb_grid = ttk.Radiobutton(top_ctrl, text="全组合网格 (Cartesian Grid)", variable=self.mode_var, value="grid", command=self._on_mode_change)
        rb_grid.pack(side=tk.LEFT, padx=4)
        rb_pair = ttk.Radiobutton(top_ctrl, text="并列对齐 (Paired Zip)", variable=self.mode_var, value="paired", command=self._on_mode_change)
        rb_pair.pack(side=tk.LEFT, padx=4)

        self.lbl_task_count = ttk.Label(top_ctrl, text="总任务数: 0 组", font=("Segoe UI", 9, "bold"), foreground="#0066cc")
        self.lbl_task_count.pack(side=tk.RIGHT, padx=6)

        # Loop parameters Treeview
        tree_frame = ttk.Frame(parent)
        tree_frame.pack(fill=tk.BOTH, expand=True)

        cols = ("param_name", "param_values", "val_count")
        self.loop_tree = ttk.Treeview(tree_frame, columns=cols, show="headings", height=7, selectmode="browse")
        self.loop_tree.heading("param_name", text="循环参数名 (Parameter)")
        self.loop_tree.heading("param_values", text="循环取值列表 (Values)")
        self.loop_tree.heading("val_count", text="数量 (Count)")

        self.loop_tree.column("param_name", width=140, anchor="w")
        self.loop_tree.column("param_values", width=260, anchor="w")
        self.loop_tree.column("val_count", width=60, anchor="center")

        scroll_y = ttk.Scrollbar(tree_frame, orient=tk.VERTICAL, command=self.loop_tree.yview)
        self.loop_tree.configure(yscrollcommand=scroll_y.set)
        self.loop_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scroll_y.pack(side=tk.RIGHT, fill=tk.Y)

        # Buttons below loop tree
        btn_frame = ttk.Frame(parent)
        btn_frame.pack(fill=tk.X, pady=(6, 0))

        ttk.Button(btn_frame, text="➕ 添加/修改循环参数...", command=self._on_add_or_edit_loop).pack(side=tk.LEFT, padx=(0, 6))
        ttk.Button(btn_frame, text="🗑 删除选中循环参数", command=self._on_delete_loop).pack(side=tk.LEFT, padx=(0, 6))
        ttk.Button(btn_frame, text="清空所有循环", command=self._on_clear_loops).pack(side=tk.LEFT, padx=(0, 6))

    def _build_fixed_params_tab(self, parent):
        """Construct scrollable fixed hyperparameters editor inside Tab 2."""
        # Top toolbar
        tb = ttk.Frame(parent)
        tb.pack(fill=tk.X, pady=(0, 4))
        ttk.Label(tb, text="修改基础参数后，点击右侧保存并更新任务清单：").pack(side=tk.LEFT)
        ttk.Button(tb, text="✔ 应用基础参数更改", command=self._on_apply_fixed_params).pack(side=tk.RIGHT)

        # Scrollable canvas
        canvas = tk.Canvas(parent, highlightthickness=0)
        scrollbar = ttk.Scrollbar(parent, orient=tk.VERTICAL, command=canvas.yview)
        scroll_content = ttk.Frame(canvas)

        scroll_content.bind(
            "<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        canvas.create_window((0, 0), window=scroll_content, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Group 1: Data & Forecasting Task
        g1 = ttk.LabelFrame(scroll_content, text=" 数据与预测任务 (Data & Task) ", padding=6)
        g1.pack(fill=tk.X, expand=True, padx=4, pady=4)
        self._add_param_field(g1, 0, 0, "task_name", "任务类型", "long_term_forecast")
        self._add_param_field(g1, 0, 1, "data", "数据集名称", "Solar")
        self._add_param_field(g1, 1, 0, "data_path", "数据文件名", "solar_AL.txt")
        self._add_param_field(g1, 1, 1, "root_path", "数据根路径", "./datasets/Solar/")
        self._add_param_field(g1, 2, 0, "seq_len", "输入序列长", "96")
        self._add_param_field(g1, 2, 1, "label_len", "前置标签长", "0")
        self._add_param_field(g1, 3, 0, "pred_len", "预测长度", "96")
        self._add_param_field(g1, 3, 1, "percent", "数据比例(%)", "100")
        self._add_param_field(g1, 4, 0, "data_split", "数据划分", "7:1:2")
        self._add_param_field(g1, 4, 1, "gpt2_path", "GPT2底座路径", "./models/gpt2")

        # Group 2: Model Architecture
        g2 = ttk.LabelFrame(scroll_content, text=" 模型网络架构 (Architecture) ", padding=6)
        g2.pack(fill=tk.X, expand=True, padx=4, pady=4)
        self._add_param_field(g2, 0, 0, "model", "模型架构", "CALF")
        self._add_param_field(g2, 0, 1, "d_model", "嵌入维度(d_model)", "768")
        self._add_param_field(g2, 1, 0, "n_heads", "注意力头数", "4")
        self._add_param_field(g2, 1, 1, "d_ff", "前馈维度(d_ff)", "3072")
        self._add_param_field(g2, 2, 0, "gpt_layers", "GPT层数", "1")
        self._add_param_field(g2, 2, 1, "out_mlp_layers", "输出头MLP层数", "3")
        self._add_param_field(g2, 3, 0, "enc_in", "输入特征维度", "137")
        self._add_param_field(g2, 3, 1, "c_out", "输出特征维度", "137")
        self._add_param_field(g2, 4, 0, "r", "LoRA秩(r)", "8")
        self._add_param_field(g2, 4, 1, "lora_alpha", "LoRA缩放alpha", "32")
        self._add_param_field(g2, 5, 0, "dropout", "Dropout比例", "0.1")
        self._add_param_field(g2, 5, 1, "lora_dropout", "LoRA Dropout", "0.1")

        # Group 3: TQ Mechanism & Physical Priors
        g3 = ttk.LabelFrame(scroll_content, text=" TQ周期机制与门控 (TQ Mechanism & Physical Priors) ", padding=6)
        g3.pack(fill=tk.X, expand=True, padx=4, pady=4)
        self._add_param_field(g3, 0, 0, "cycle", "周期长度(cycle)", "144")
        self._add_param_field(g3, 0, 1, "use_tq", "启用TQ机制(0/1)", "1")
        self._add_param_field(g3, 1, 0, "tq_mode", "注入方式(add/mul)", "add")
        self._add_param_field(g3, 1, 1, "tq_inject_target", "注入目标(q/pca)", "q")
        self._add_param_field(g3, 2, 0, "tq_lr_factor", "TQ学习率因子", "5.0")
        self._add_param_field(g3, 2, 1, "tq_dropout", "TQ Dropout", "0.1")
        self._add_param_field(g3, 3, 0, "use_direct_trend", "直连趋势门控(0/1)", "0")
        self._add_param_field(g3, 3, 1, "cycle_prior_before_denorm", "反归一化前加先验(0/1)", "0")
        self._add_param_field(g3, 4, 0, "use_norm", "内部RevIN归一化(0/1)", "1")
        self._add_param_field(g3, 4, 1, "use_tq_gate", "可学习TQ门控(0/1)", "0")

        # Group 4: Training & Three Loss Weights
        g4 = ttk.LabelFrame(scroll_content, text=" 训练、学习率与三损失权重 (Optimization & Loss Weights) ", padding=6)
        g4.pack(fill=tk.X, expand=True, padx=4, pady=4)
        self._add_param_field(g4, 0, 0, "batch_size", "批次大小(Batch)", "16")
        self._add_param_field(g4, 0, 1, "accumulation_steps", "梯度累加步数", "4")
        self._add_param_field(g4, 1, 0, "learning_rate", "学习率(lr)", "0.0004")
        self._add_param_field(g4, 1, 1, "train_epochs", "最大训练轮数", "100")
        self._add_param_field(g4, 2, 0, "patience", "早停容忍度", "5")
        self._add_param_field(g4, 2, 1, "tmax", "余弦退火周期(tmax)", "10")
        self._add_param_field(g4, 3, 0, "task_w", "时序主任务权重(task_w)", "0.736")
        self._add_param_field(g4, 3, 1, "feature_w", "特征对齐权重(feature_w)", "0.05")
        self._add_param_field(g4, 4, 0, "output_w", "输出对齐权重(output_w)", "0.214")
        self._add_param_field(g4, 4, 1, "bestmodel", "测试从内存加载最优模型", "True")
        self._add_param_field(g4, 5, 0, "task_loss", "时序损失函数", "smooth_l1")
        self._add_param_field(g4, 5, 1, "feature_loss", "特征损失函数", "smooth_l1")
        self._add_param_field(g4, 6, 0, "output_loss", "输出损失函数", "smooth_l1")
        self._add_param_field(g4, 6, 1, "eval_test_every_epoch", "每轮评估测试集", "True")

        # Helper buttons for Loss weights and LR auto-calculation
        helper_frame = ttk.Frame(g4)
        helper_frame.grid(row=7, column=0, columnspan=2, sticky="ew", padx=6, pady=6)

        ttk.Button(
            helper_frame,
            text="⚡ 根据 task_w 和 feature_w 自动联动 output_w (1 - tw - fw)",
            command=self._on_auto_calc_output_w
        ).pack(side=tk.LEFT, padx=(0, 8))

        self.lbl_loss_sum_info = ttk.Label(
            helper_frame,
            text="[三损失和: 0.736 + 0.05 + 0.214 = 1.000]",
            font=("Segoe UI", 9, "bold"),
            foreground="#008800"
        )
        self.lbl_loss_sum_info.pack(side=tk.LEFT, padx=10)

    def _add_param_field(self, parent, row, col, key, label_text, default_val):
        """Helper to create labeled entry for fixed parameters."""
        frame = ttk.Frame(parent)
        frame.grid(row=row, column=col, sticky="w", padx=6, pady=2)
        ttk.Label(frame, text=f"{label_text} ({key}):", width=22, anchor="w").pack(side=tk.LEFT)
        val = str(self.config_mgr.fixed_params.get(key, default_val))
        var = tk.StringVar(value=val)
        self.fixed_vars[key] = var
        entry = ttk.Entry(frame, textvariable=var, width=14)
        entry.pack(side=tk.LEFT)

    def _on_auto_calc_output_w(self):
        """Auto calculate output_w = 1.0 - task_w - feature_w."""
        try:
            tw = float(self.fixed_vars.get("task_w", tk.StringVar(value="0.736")).get())
            fw = float(self.fixed_vars.get("feature_w", tk.StringVar(value="0.05")).get())
            ow = round(1.0 - tw - fw, 6)
            if ow < 0:
                messagebox.showerror("无效权重", f"task_w({tw}) + feature_w({fw}) = {tw+fw} > 1.0！\n导致 output_w ({ow}) 小于 0！")
                return
            if "output_w" in self.fixed_vars:
                self.fixed_vars["output_w"].set(f"{ow:.6f}".rstrip("0").rstrip("."))
            if hasattr(self, "lbl_loss_sum_info"):
                self.lbl_loss_sum_info.config(
                    text=f"[三损失和: {tw:g} + {fw:g} + {ow:g} = {tw+fw+ow:.3f}]",
                    foreground="#008800"
                )
        except Exception as e:
            messagebox.showerror("计算错误", f"无法计算 output_w: {e}")

    def _build_tasks_table(self, parent):
        """Build the task execution list and progress indicator."""
        # Progress info bar
        prog_frame = ttk.Frame(parent)
        prog_frame.pack(fill=tk.X, pady=(0, 4))

        self.lbl_progress = ttk.Label(prog_frame, text="进度: 0 / 0 (0.0%)", font=("Segoe UI", 9, "bold"))
        self.lbl_progress.pack(side=tk.LEFT, padx=(0, 8))

        self.progress_bar = ttk.Progressbar(prog_frame, orient=tk.HORIZONTAL, mode="determinate")
        self.progress_bar.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 8))

        # Tasks Treeview
        tree_frame = ttk.Frame(parent)
        tree_frame.pack(fill=tk.BOTH, expand=True)

        cols = ("task_id", "status", "loop_sig", "mse", "mae", "duration", "setting")
        self.task_tree = ttk.Treeview(tree_frame, columns=cols, show="headings", selectmode="browse")
        self.task_tree.heading("task_id", text="Task ID")
        self.task_tree.heading("status", text="状态 (Status)")
        self.task_tree.heading("loop_sig", text="循环参数 (Loop Params)")
        self.task_tree.heading("mse", text="MSE")
        self.task_tree.heading("mae", text="MAE")
        self.task_tree.heading("duration", text="耗时")
        self.task_tree.heading("setting", text="Setting 目录名")

        self.task_tree.column("task_id", width=65, anchor="center")
        self.task_tree.column("status", width=85, anchor="center")
        self.task_tree.column("loop_sig", width=160, anchor="w")
        self.task_tree.column("mse", width=75, anchor="center")
        self.task_tree.column("mae", width=75, anchor="center")
        self.task_tree.column("duration", width=65, anchor="center")
        self.task_tree.column("setting", width=180, anchor="w")

        # Color tags
        self.task_tree.tag_configure("COMPLETED", foreground="#008800")
        self.task_tree.tag_configure("RUNNING", foreground="#0055cc", font=("Segoe UI", 9, "bold"))
        self.task_tree.tag_configure("PENDING", foreground="#666666")
        self.task_tree.tag_configure("FAILED", foreground="#cc0000", font=("Segoe UI", 9, "bold"))

        scroll_y = ttk.Scrollbar(tree_frame, orient=tk.VERTICAL, command=self.task_tree.yview)
        scroll_x = ttk.Scrollbar(tree_frame, orient=tk.HORIZONTAL, command=self.task_tree.xview)
        self.task_tree.configure(yscrollcommand=scroll_y.set, xscrollcommand=scroll_x.set)

        self.task_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scroll_y.pack(side=tk.RIGHT, fill=tk.Y)
        scroll_x.pack(side=tk.BOTTOM, fill=tk.X)

        # Context action buttons below task tree
        act_frame = ttk.Frame(parent)
        act_frame.pack(fill=tk.X, pady=(4, 0))

        ttk.Button(act_frame, text="重置选中任务为待跑 (PENDING)", command=self._on_reset_selected_task).pack(side=tk.LEFT, padx=(0, 6))
        ttk.Button(act_frame, text="重置所有失败/中断任务", command=self._on_reset_failed_tasks).pack(side=tk.LEFT, padx=(0, 6))
        ttk.Button(act_frame, text="定位当前检查点目录", command=self._on_open_task_folder).pack(side=tk.LEFT, padx=(0, 6))

    def _build_log_console(self):
        """Bottom dark terminal console for real-time streaming output."""
        console_frame = ttk.LabelFrame(self.root, text=" 实时控制台输出流 (Real-Time Output Stream) ", padding=4)
        console_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=(2, 4))

        # Controls row
        ctrl_row = ttk.Frame(console_frame)
        ctrl_row.pack(fill=tk.X, pady=(0, 2))

        self.auto_scroll_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(ctrl_row, text="自动滚动到最新日志 (Auto-scroll)", variable=self.auto_scroll_var).pack(side=tk.LEFT)

        self.lbl_task_running_info = ttk.Label(ctrl_row, text="[当前就绪]", font=("Segoe UI", 9, "bold"), foreground="#0066cc")
        self.lbl_task_running_info.pack(side=tk.LEFT, padx=16)

        ttk.Button(ctrl_row, text="清空控制台", command=self._on_clear_console).pack(side=tk.RIGHT)
        ttk.Button(ctrl_row, text="复制全部日志", command=self._on_copy_console).pack(side=tk.RIGHT, padx=4)

        # Text widget
        txt_frame = ttk.Frame(console_frame)
        txt_frame.pack(fill=tk.BOTH, expand=True)

        self.log_text = tk.Text(
            txt_frame,
            wrap=tk.WORD,
            bg="#181818",
            fg="#d0d0d0",
            insertbackground="white",
            font=("Consolas", 9),
            height=10
        )
        scroll_y = ttk.Scrollbar(txt_frame, orient=tk.VERTICAL, command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=scroll_y.set)

        self.log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scroll_y.pack(side=tk.RIGHT, fill=tk.Y)

        # Highlighting tags
        self.log_text.tag_configure("epoch", foreground="#4ec9b0")
        self.log_text.tag_configure("vali", foreground="#b5cea8")
        self.log_text.tag_configure("metric", foreground="#dcdcaa", font=("Consolas", 9, "bold"))
        self.log_text.tag_configure("error", foreground="#f44747", font=("Consolas", 9, "bold"))
        self.log_text.tag_configure("task_banner", foreground="#569cd6", font=("Consolas", 9, "bold"))

    def _build_status_bar(self):
        """Bottom status indicator bar."""
        bar = ttk.Frame(self.root)
        bar.pack(fill=tk.X, padx=10, pady=(0, 4))

        self.lbl_status = ttk.Label(bar, text="准备就绪。支持随时停止和断点无缝续跑。", relief=tk.SUNKEN, anchor="w", padding=2)
        self.lbl_status.pack(fill=tk.X)

    # =========================================================================
    # Event Handlers & Business Logic
    # =========================================================================

    def _save_session_cache(self):
        """Persist last active session parameters to configs/.last_session.json."""
        cache_file = "configs/.last_session.json"
        try:
            os.makedirs("configs", exist_ok=True)
            data = {
                "last_config_path": getattr(self, "current_config_path", None),
                "last_checkpoints_root": self.config_mgr.checkpoints_root,
                "last_batch_name": self.config_mgr.batch_name,
                "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception:
            pass

    def _refresh_all_views(self, reset_interrupted: bool = True):
        """Regenerate tasks, update state, and refresh treeviews and progress."""
        # 1. Sync checkpoints root and batch name
        self.config_mgr.set_checkpoints_root(self.ckpt_root_var.get().strip())
        self.config_mgr.batch_name = self.batch_name_var.get().strip()
        self._save_session_cache()

        # 2. Refresh Loop treeview
        for item in self.loop_tree.get_children():
            self.loop_tree.delete(item)
        for k, v in self.config_mgr.loop_params.items():
            self.loop_tree.insert("", tk.END, values=(k, str(v), len(v) if isinstance(v, list) else 1))

        # 3. Regenerate task combinations
        tasks = self.config_mgr.generate_task_combos()
        self.lbl_task_count.config(text=f"总任务数: {len(tasks)} 组组合 ({self.config_mgr.mode} 模式)")

        # 4. Sync StateManager
        self.state_mgr.set_checkpoints_root(self.config_mgr.checkpoints_root, self.config_mgr.batch_name)
        self.state_mgr.init_or_resume(tasks, reset_interrupted=reset_interrupted)

        # 5. Refresh Tasks Table
        for item in self.task_tree.get_children():
            try:
                self.task_tree.delete(item)
            except Exception:
                pass

        tasks_dict = self.state_mgr.state.get("tasks", {})
        sorted_tasks = sorted(list(tasks_dict.items()), key=lambda x: x[1].get("task_idx", 0))

        for tid, t in sorted_tasks:
            status = t.get("status", "PENDING")
            metrics = t.get("metrics") or {}
            mse_s = f"{metrics.get('mse', 0):.6f}" if metrics.get("mse") is not None else "-"
            mae_s = f"{metrics.get('mae', 0):.6f}" if metrics.get("mae") is not None else "-"
            dur_s = f"{t.get('duration_sec', 0):.1f}s" if t.get("duration_sec") else "-"
            setting_s = t.get("setting", "")

            if self.task_tree.exists(tid):
                self.task_tree.delete(tid)

            self.task_tree.insert(
                "", tk.END,
                iid=tid,
                values=(tid, status, str(t.get("loop_signature", {})), mse_s, mae_s, dur_s, setting_s),
                tags=(status,)
            )

        # 6. Update progress bar
        summary = self.state_mgr.get_summary()
        pct = summary["progress_pct"]
        tot = summary["total"]
        comp = summary["completed"]
        self.lbl_progress.config(text=f"进度: {comp} / {tot} ({pct:.1f}%) | 待跑: {summary['pending']} | 失败: {summary['failed']}")
        self.progress_bar["maximum"] = tot if tot > 0 else 1
        self.progress_bar["value"] = comp

    def _on_browse_checkpoints(self):
        """Open directory picker for master checkpoints root."""
        chosen = filedialog.askdirectory(title="选择权重统一保存大文件夹", initialdir=self.config_mgr.checkpoints_root)
        if chosen:
            self.ckpt_root_var.set(chosen.replace("\\", "/"))
            self._save_session_cache()
            self._refresh_all_views()
            self._log_console(f"[GUI] 已切换权重保存大文件夹为: {chosen}\n", "task_banner")

    def _on_preset_selected(self, event=None):
        """Apply pre-configured master checkpoint directory."""
        sel = self.preset_combo.get()
        if sel == "新建时间戳隔离目录":
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.ckpt_root_var.set(f"./checkpoints/batch_{ts}")
        elif sel == "Solar-AL 基准复现目录":
            self.ckpt_root_var.set("./checkpoints/solar_al_baseline")
        elif sel == "少样本微调扫描目录":
            self.ckpt_root_var.set("./checkpoints/solar_fewshot_sweep")
        self._save_session_cache()
        self._refresh_all_views()

    def _on_mode_change(self):
        """Switch between Grid and Paired sweep mode."""
        self.config_mgr.mode = self.mode_var.get()
        self._refresh_all_views()

    def _on_add_or_edit_loop(self):
        """Dialog to add or edit a loop parameter with 'start:step:stop' and comma syntax."""
        dlg = tk.Toplevel(self.root)
        dlg.title("配置循环超参 (Add / Edit Loop Parameter)")
        dlg.geometry("540x330")
        dlg.transient(self.root)
        dlg.grab_set()

        ttk.Label(dlg, text="超参名称 (Parameter Name):", font=("Segoe UI", 9, "bold")).pack(anchor="w", padx=16, pady=(12, 2))
        ttk.Label(dlg, text="常用: random_seed, pred_len, cycle, task_w, feature_w, percent, learning_rate", foreground="#666666").pack(anchor="w", padx=16)
        param_var = tk.StringVar(value="random_seed")
        e_param = ttk.Entry(dlg, textvariable=param_var, width=38)
        e_param.pack(anchor="w", padx=16, pady=4)

        ttk.Label(dlg, text="循环取值写法 (支持两种语法):", font=("Segoe UI", 9, "bold")).pack(anchor="w", padx=16, pady=(10, 2))
        ttk.Label(dlg, text="1. 起点:步长:终点 (例如 0.1:0.1:0.5 或 24:24:144 或 2025:1:2027)\n2. 逗号分隔枚举 (例如 2025, 2026, 2027 或 96, 192)", foreground="#0055aa").pack(anchor="w", padx=16)
        vals_var = tk.StringVar(value="2025:1:2027")
        e_vals = ttk.Entry(dlg, textvariable=vals_var, width=52)
        e_vals.pack(anchor="w", padx=16, pady=6)

        # 实时解析预览标签
        lbl_preview = ttk.Label(dlg, text="解析预览: [2025, 2026, 2027] (共 3 项)", font=("Segoe UI", 9, "italic"), foreground="#008800")
        lbl_preview.pack(anchor="w", padx=16, pady=(2, 6))

        def update_preview(*args):
            raw_v = vals_var.get().strip()
            if not raw_v:
                lbl_preview.config(text="解析预览: (空)", foreground="#888888")
                return
            try:
                parsed = parse_loop_values(raw_v)
                preview_str = str(parsed[:8]) + ("..." if len(parsed) > 8 else "")
                lbl_preview.config(text=f"解析预览: {preview_str} (共 {len(parsed)} 项)", foreground="#008800")
            except Exception as e:
                lbl_preview.config(text=f"语法提示: {e}", foreground="#cc0000")

        vals_var.trace_add("write", update_preview)
        update_preview()

        def save():
            k = param_var.get().strip()
            raw_v = vals_var.get().strip()
            if not k or not raw_v:
                messagebox.showwarning("提示", "参数名称和取值不能为空！", parent=dlg)
                return
            try:
                parsed = parse_loop_values(raw_v)
                if not parsed:
                    messagebox.showwarning("提示", "未解析出任何有效数值！", parent=dlg)
                    return
                self.config_mgr.set_loop_param(k, parsed)
                dlg.destroy()
                self._refresh_all_views()
            except Exception as e:
                messagebox.showerror("语法错误", f"无法解析循环表达式: {e}", parent=dlg)

        btn_row = ttk.Frame(dlg)
        btn_row.pack(fill=tk.X, padx=16, pady=16)
        ttk.Button(btn_row, text="✔ 确定保存", command=save).pack(side=tk.RIGHT, padx=4)
        ttk.Button(btn_row, text="取消", command=dlg.destroy).pack(side=tk.RIGHT)

    def _on_delete_loop(self):
        """Delete selected loop parameter."""
        selected = self.loop_tree.selection()
        if not selected:
            messagebox.showinfo("提示", "请先在循环参数列表中选择要删除的行。")
            return
        item = self.loop_tree.item(selected[0])
        key = item["values"][0]
        if key in self.config_mgr.loop_params:
            self.config_mgr.loop_params.pop(key, None)
            self._refresh_all_views()

    def _on_clear_loops(self):
        """Clear all loop parameters."""
        if messagebox.askyesno("确认", "确定要清空所有循环参数吗？"):
            self.config_mgr.loop_params.clear()
            self._refresh_all_views()

    def _on_apply_fixed_params(self):
        """Apply edited values from fixed parameter tab."""
        for k, var in self.fixed_vars.items():
            raw_val = var.get().strip()
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
            self.config_mgr.set_fixed_param(k, val)
        self._refresh_all_views()
        messagebox.showinfo("成功", "已应用基础参数更改并同步至任务清单。")

    def _on_load_template(self):
        """Load JSON template from file."""
        chosen = filedialog.askopenfilename(
            title="载入实验配置模板",
            initialdir="configs/batch_templates",
            filetypes=[("JSON Files", "*.json"), ("All Files", "*.*")]
        )
        if chosen:
            try:
                self.config_mgr.load_from_file(chosen)
                self.current_config_path = chosen
                self.ckpt_root_var.set(self.config_mgr.checkpoints_root)
                self.batch_name_var.set(self.config_mgr.batch_name)
                self.mode_var.set(self.config_mgr.mode)
                # Sync fixed vars
                for k, v in self.config_mgr.fixed_params.items():
                    if k in self.fixed_vars:
                        self.fixed_vars[k].set(str(v))
                self._save_session_cache()
                self._refresh_all_views()
                self._log_console(f"[GUI] 成功载入模板: {chosen}\n", "task_banner")
            except Exception as e:
                messagebox.showerror("载入失败", f"无法载入模板: {e}")

    def _on_save_config(self):
        """Save current configuration to JSON file."""
        chosen = filedialog.asksaveasfilename(
            title="保存当前实验配置",
            initialdir="configs",
            initialfile=f"batch_{self.config_mgr.batch_name}.json",
            defaultextension=".json",
            filetypes=[("JSON Files", "*.json")]
        )
        if chosen:
            try:
                self.config_mgr.batch_name = self.batch_name_var.get().strip()
                self.config_mgr.set_checkpoints_root(self.ckpt_root_var.get().strip())
                self.config_mgr.save_to_file(chosen)
                self.current_config_path = chosen
                self._save_session_cache()
                messagebox.showinfo("成功", f"配置已保存至:\n{chosen}")
            except Exception as e:
                messagebox.showerror("保存失败", f"无法保存文件: {e}")

    def _on_export_csv(self):
        """Export completed results to CSV."""
        default_p = os.path.join(self.config_mgr.checkpoints_root, "batch_summary.csv").replace("\\", "/")
        chosen = filedialog.asksaveasfilename(
            title="导出实验结果 CSV",
            initialfile="batch_summary.csv",
            defaultextension=".csv",
            filetypes=[("CSV Files", "*.csv")]
        )
        if chosen:
            try:
                self.state_mgr.export_summary_csv(chosen)
                messagebox.showinfo("导出成功", f"结果汇总表已保存至:\n{chosen}")
            except Exception as e:
                messagebox.showerror("导出失败", f"导出失败: {e}")

    def _on_refresh_click(self):
        """Manual refresh."""
        self._refresh_all_views(reset_interrupted=False)

    def _on_reset_selected_task(self):
        """Reset selected task to PENDING with option to clear intermediate epoch checkpoints."""
        selected = self.task_tree.selection()
        if not selected:
            messagebox.showinfo("提示", "请先在任务清单中选中要重置的任务。")
            return
        tid = selected[0]
        ans = messagebox.askyesnocancel(
            "重置确认",
            f"确定要重置任务 {tid} 的状态吗？\n\n"
            f"- 点【是(Yes)】：从第 1 轮彻底重跑（清除已保存的中间 Epoch 检查点）\n"
            f"- 点【否(No)】：仅重置状态为 PENDING，保留已完成 Epoch 进度供断点续训\n"
            f"- 点【取消】：放弃重置操作"
        )
        if ans is None:
            return
        clear_ckpt = bool(ans)
        self.state_mgr.reset_task(tid, clear_epoch_checkpoint=clear_ckpt)
        self._refresh_all_views(reset_interrupted=False)
        msg_extra = "（已清除中间进度，将从头开始）" if clear_ckpt else "（保留已训练的 Epoch 进度，可继续续训）"
        self._log_console(f"[GUI] 任务 {tid} 已重置为 PENDING {msg_extra}\n", "task_banner")

    def _on_reset_failed_tasks(self):
        """Reset all failed or running tasks to PENDING."""
        self.state_mgr.reset_all_failed_or_interrupted()
        self._refresh_all_views(reset_interrupted=False)
        messagebox.showinfo("提示", "已将所有失败和中断的任务重置为 PENDING。")

    def _on_open_task_folder(self):
        """Open Windows Explorer at checkpoint root."""
        path = os.path.abspath(self.config_mgr.checkpoints_root)
        if os.path.exists(path):
            os.startfile(path)
        else:
            messagebox.showinfo("提示", f"目录尚不存在: {path}")

    # =========================================================================
    # Real-Time Execution Engine (Multi-threaded & Non-blocking)
    # =========================================================================

    def _on_start_batch(self):
        """Start or resume execution in background thread."""
        if self.is_running:
            return

        self._refresh_all_views(reset_interrupted=True)
        pending = self.state_mgr.get_pending_tasks()
        if not pending:
            messagebox.showinfo("完成", "当前所有任务均已完成！\n如需重新执行，请在任务列表中重置状态。")
            return

        self.is_running = True
        self.btn_run.config(state=tk.DISABLED)
        self.btn_stop.config(state=tk.NORMAL)
        self.lbl_status.config(text=f"正在运行批次实验，剩余待跑任务: {len(pending)} 个...")

        # Launch background worker
        tasks = self.config_mgr.generate_task_combos()
        self.worker_thread = threading.Thread(
            target=self._run_worker_thread,
            args=(tasks,),
            daemon=True
        )
        self.worker_thread.start()

    def _run_worker_thread(self, tasks: List[Dict[str, Any]]):
        """Executed inside background thread."""
        pending_tasks = self.state_mgr.get_pending_tasks()
        self.log_queue.put(("system", f"\n{'='*75}\n=== BATCH RUNNER STARTED: {self.config_mgr.batch_name} ===\nCheckpoints: {self.config_mgr.checkpoints_root}\n{'='*75}\n"))

        for task in pending_tasks:
            if not self.is_running:
                break

            tid = task["task_id"]
            self.log_queue.put(("task_start", task))

            def on_line(line):
                self.log_queue.put(("log", f"[{tid}] {line}\n"))

            try:
                success = self.engine.run_task(task, self.state_mgr, on_log_line=on_line)
                self.log_queue.put(("task_end", (tid, success)))
            except KeyboardInterrupt:
                break
            except Exception as e:
                self.log_queue.put(("log", f"[-] Task {tid} Exception: {e}\n"))
                self.log_queue.put(("task_end", (tid, False)))

        self.log_queue.put(("batch_finish", None))

    def _on_stop_batch(self):
        """Gracefully stop running execution."""
        if not self.is_running:
            return
        if messagebox.askyesno("停止确认", "确定要中止当前的批量实验吗？\n当前正在跑的任务将被保存为中断状态，下次启动时可直接断点续跑。"):
            self.lbl_status.config(text="正在停止当前任务并保存断点状态...")
            self.is_running = False
            self.engine.interrupted = True
            if self.engine.current_process:
                try:
                    self.engine.current_process.terminate()
                except Exception:
                    pass

    def _poll_log_queue(self):
        """Periodic drain of real-time queue to update GUI safely on main thread."""
        try:
            while True:
                msg_type, data = self.log_queue.get_nowait()
                if msg_type == "log":
                    self._append_log_line(data)
                elif msg_type == "system":
                    self._append_log_line(data, tag="task_banner")
                elif msg_type == "task_start":
                    tid = data["task_id"]
                    self.lbl_task_running_info.config(text=f"[正在运行: {tid} | {data.get('loop_signature')}]")
                    if self.task_tree.exists(tid):
                        self.task_tree.set(tid, "status", "RUNNING")
                        self.task_tree.item(tid, tags=("RUNNING",))
                elif msg_type == "task_end":
                    tid, success = data
                    self._refresh_all_views(reset_interrupted=False)
                elif msg_type == "batch_finish":
                    self.is_running = False
                    self.btn_run.config(state=tk.NORMAL)
                    self.btn_stop.config(state=tk.DISABLED)
                    self.lbl_task_running_info.config(text="[批次就绪/已暂停]")
                    self._refresh_all_views(reset_interrupted=False)
                    self.lbl_status.config(text="批次执行结束或已暂停。状态已保存至 batch_state.json。")
                    self._append_log_line("\n[OK] 批次执行全部结束或已被用户安全中止。\n", tag="task_banner")
        except queue.Empty:
            pass

        self.root.after(40, self._poll_log_queue)

    def _append_log_line(self, text: str, tag: Optional[str] = None):
        """Append log line to text widget with optional keyword highlighting."""
        # Detect keywords for automatic colorization
        if tag is None:
            if "Epoch: " in text:
                tag = "epoch"
            elif "Validation loss decreased" in text or "Best model stored" in text:
                tag = "vali"
            elif "mse:" in text and "mae:" in text:
                tag = "metric"
            elif "Error" in text or "FAILED" in text or "Exception" in text:
                tag = "error"
            elif "start training :" in text or "testing :" in text or "[Epoch-Resume]" in text or "[Resume Detection]" in text:
                tag = "task_banner"

        if tag:
            self.log_text.insert(tk.END, text, tag)
        else:
            self.log_text.insert(tk.END, text)

        if self.auto_scroll_var.get():
            self.log_text.see(tk.END)

    def _log_console(self, text: str, tag: Optional[str] = None):
        """Direct thread-safe logging."""
        self.log_queue.put(("system" if tag == "task_banner" else "log", text))

    def _on_clear_console(self):
        """Clear terminal text area."""
        self.log_text.delete("1.0", tk.END)

    def _on_copy_console(self):
        """Copy console text to clipboard."""
        self.root.clipboard_clear()
        self.root.clipboard_append(self.log_text.get("1.0", tk.END))
        messagebox.showinfo("复制成功", "已复制全部控制台日志至剪贴板。")

    def _on_close(self):
        """Handle window close event."""
        self._save_session_cache()
        if self.is_running:
            if messagebox.askyesno("退出提示", "后台正在运行实验任务，确定要中止并退出吗？\n当前断点状态将安全保存。"):
                self.is_running = False
                self.engine.interrupted = True
                if self.engine.current_process:
                    try:
                        self.engine.current_process.terminate()
                    except Exception:
                        pass
                self.root.destroy()
        else:
            self.root.destroy()


def launch_gui(initial_config: Optional[str] = None, initial_checkpoints: Optional[str] = None):
    """Entry point to start the Tkinter GUI."""
    root = tk.Tk()
    app = BatchPanelGUI(root, initial_config=initial_config, initial_checkpoints=initial_checkpoints)
    root.mainloop()


if __name__ == "__main__":
    launch_gui()
