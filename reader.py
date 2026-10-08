import os
import glob
import json
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import tkinter as tk
from tkinter import ttk, messagebox
import extract

# 配置信息
DATA_ROOT = "./datasets/Solar/"
DATA_PATH = "solar_AL.txt"
RESULTS_DIR = "./results/Solar/"
CACHE_FILE = os.path.join(RESULTS_DIR, "scaler_cache.json")

def parse_model_label(file_path):
    """从文件名智能提取清晰简短的标签，避免同名或缺失关键信息"""
    fname = os.path.basename(file_path)
    mse_m = re.search(r'mse([\d.]+)', fname)
    mae_m = re.search(r'mae([\d.]+)', fname)
    pl_m = re.search(r'pl(\d+)', fname)
    fw_m = re.search(r'fw([\d.]+)', fname)
    ow_m = re.search(r'ow([\d.]+)', fname)
    
    tags = []
    if '_swa' in fname:
        tags.append('SWA')
    elif '_best' in fname:
        tags.append('best')
    tag_str = f" ({','.join(tags)})" if tags else ""
    
    parts = []
    if pl_m:
        parts.append(f"pl{pl_m.group(1)}")
    if mse_m:
        parts.append(f"MSE:{mse_m.group(1)}")
    if mae_m:
        parts.append(f"MAE:{mae_m.group(1)}")
    if not (mse_m or mae_m):
        if fw_m and ow_m:
            parts.append(f"fw{fw_m.group(1)}_ow{ow_m.group(1)}")
        else:
            parts.append(fname[:22])
            
    label = " | ".join(parts) + tag_str
    return label if label else fname[:22]

class ResultViewer:
    def __init__(self, root):
        self.root = root
        self.root.title("CALF Experiment Comparer - [Click / Double-Click to Select]")
        self.root.geometry("1400x860")
        
        # 状态变量
        self.scaler_cache = {}
        self.cache_loaded = False
        self.results_dir = RESULTS_DIR
        self.files = []
        self.selected_indices = set()
        self.current_station = 0
        self.station_buttons = {}
        self.canvas = None
        self.toolbar = None
        self.debounce_timer = None
        self.is_plotting = False
        
        self.create_layout()
        self.refresh_files()
        self.preload_scalers()
        
        # 启动时自动触发首绘（展示 Ground Truth）
        self.plot_action()
        
        # 绑定退出
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        
        # 监听变量（使用 debounce 防抖）
        self.win_var.trace_add("write", self.auto_refresh)
        self.concat_var.trace_add("write", self.auto_refresh)
        self.inverse_var.trace_add("write", self.auto_refresh)

    def auto_refresh(self, *args):
        """防抖自动刷新，避免快速滑动控件造成卡顿和重绘撕裂"""
        if self.debounce_timer:
            self.root.after_cancel(self.debounce_timer)
        self.debounce_timer = self.root.after(80, self.plot_action)

    def on_close(self):
        if self.debounce_timer:
            self.root.after_cancel(self.debounce_timer)
        plt.close('all')
        self.root.quit()
        self.root.destroy()
        os._exit(0)

    def preload_scalers(self):
        if os.path.exists(CACHE_FILE):
            try:
                with open(CACHE_FILE, 'r') as f:
                    data = json.load(f)
                    self.scaler_cache = {int(k): v for k, v in data.items()}
                self.cache_loaded = True
                return
            except:
                pass

        raw_file = os.path.join(DATA_ROOT, DATA_PATH)
        if os.path.exists(raw_file):
            try:
                raw_data = pd.read_csv(raw_file, header=None).values
                num_train = int(len(raw_data) * 0.7)
                train_data = raw_data[:num_train, :]
                means = np.mean(train_data, axis=0)
                stds = np.std(train_data, axis=0)
                self.scaler_cache = {i: [float(means[i]), float(stds[i])] for i in range(len(means))}
                os.makedirs(os.path.dirname(CACHE_FILE), exist_ok=True)
                with open(CACHE_FILE, 'w') as f:
                    json.dump(self.scaler_cache, f)
                self.cache_loaded = True
            except:
                pass

    def get_scaler_from_cache(self, station_id):
        val = self.scaler_cache.get(station_id)
        return (val[0], val[1]) if val else (None, None)

    def create_layout(self):
        main_paned = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        main_paned.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # ---- 左侧区域：绘图显示 ----
        self.plot_frame = ttk.Frame(main_paned, relief="sunken")
        main_paned.add(self.plot_frame, weight=3)
        
        self.fig = plt.figure(figsize=(9, 6.5))
        self.setup_canvas()

        # ---- 右侧区域：控制面板 ----
        self.right_sidebar = ttk.Frame(main_paned, padding="10")
        main_paned.add(self.right_sidebar, weight=1)

        # 1. 文件列表标题与状态
        header_frame = ttk.Frame(self.right_sidebar)
        header_frame.pack(fill=tk.X, pady=(0, 2))
        ttk.Label(header_frame, text="📂 Result Files:", font=("Arial", 10, "bold")).pack(side=tk.LEFT)
        self.status_lbl = ttk.Label(header_frame, text="(0 selected)", font=("Arial", 9), foreground="#005fb8")
        self.status_lbl.pack(side=tk.RIGHT)

        list_frame = ttk.Frame(self.right_sidebar)
        list_frame.pack(fill=tk.BOTH, expand=True, pady=(2, 6))
        
        sb = ttk.Scrollbar(list_frame)
        sb.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.file_listbox = tk.Listbox(
            list_frame, font=("Consolas", 9), 
            yscrollcommand=sb.set, bg="#fdfdfd", 
            selectmode=tk.MULTIPLE, highlightthickness=0,
            activestyle='none'
        )
        self.file_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        sb.config(command=self.file_listbox.yview)
        
        # 绑定事件：单击与双击均支持灵活切换选中
        self.file_listbox.bind('<ButtonRelease-1>', self.on_listbox_click)

        # 快捷选择按钮栏
        btn_bar = ttk.Frame(self.right_sidebar)
        btn_bar.pack(fill=tk.X, pady=(0, 8))
        ttk.Button(btn_bar, text="Top 1", width=6, command=lambda: self.select_top_n(1)).pack(side=tk.LEFT, padx=1)
        ttk.Button(btn_bar, text="Top 3", width=6, command=lambda: self.select_top_n(3)).pack(side=tk.LEFT, padx=1)
        ttk.Button(btn_bar, text="Clear", width=6, command=self.clear_selection).pack(side=tk.LEFT, padx=1)
        ttk.Button(btn_bar, text="🔄 Refresh", command=self.refresh_files).pack(side=tk.RIGHT, padx=1)

        # 2. 答题卡式电站选择 (0-9)
        ttk.Label(self.right_sidebar, text="📍 Select Station (0-9):", font=("Arial", 10, "bold")).pack(anchor=tk.W, pady=(4, 4))
        grid_frame = ttk.Frame(self.right_sidebar)
        grid_frame.pack(fill=tk.X)

        for i in range(10):
            row = i // 5
            col = i % 5
            btn = tk.Button(
                grid_frame, text=str(i), width=4, height=2, 
                font=("Arial", 10, "bold"),
                command=lambda sid=i: self.switch_station(sid)
            )
            btn.grid(row=row, column=col, padx=2, pady=2)
            self.station_buttons[i] = btn
        self.update_button_colors()

        # 3. 绘图参数
        cfg_frame = ttk.LabelFrame(self.right_sidebar, text="Display Settings", padding="10")
        cfg_frame.pack(fill=tk.X, pady=12)

        ttk.Label(cfg_frame, text="Windows to Compare:").pack(anchor=tk.W)
        self.win_var = tk.IntVar(value=3)
        self.win_scale = tk.Scale(
            cfg_frame, from_=1, to=10, orient=tk.HORIZONTAL, 
            variable=self.win_var, showvalue=True
        )
        self.win_scale.pack(fill=tk.X, pady=(0, 8))

        mode_frame = ttk.Frame(cfg_frame)
        mode_frame.pack(fill=tk.X)
        self.concat_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(mode_frame, text="Concat Mode", variable=self.concat_var).pack(side=tk.LEFT, padx=(0, 10))
        
        self.inverse_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(mode_frame, text="Inverse Scale (MW)", variable=self.inverse_var).pack(side=tk.LEFT)

    def setup_canvas(self):
        if self.canvas:
            self.canvas.get_tk_widget().destroy()
            if self.toolbar:
                self.toolbar.destroy()
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.plot_frame)
        toolbar_frame = ttk.Frame(self.plot_frame)
        toolbar_frame.pack(side=tk.BOTTOM, fill=tk.X)
        self.toolbar = NavigationToolbar2Tk(self.canvas, toolbar_frame)
        self.toolbar.update()
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    def refresh_files(self):
        pattern = os.path.join(self.results_dir, "*.csv")
        self.files = sorted(glob.glob(pattern), key=os.path.getmtime, reverse=True)
        self.file_listbox.delete(0, tk.END)
        for f in self.files:
            lbl = parse_model_label(f)
            self.file_listbox.insert(tk.END, f" {lbl}")
        
        # 过滤可能过期的选中索引
        self.selected_indices = {i for i in self.selected_indices if i < len(self.files)}
        self.sync_listbox_colors()
        self.auto_refresh()

    def select_top_n(self, n):
        self.selected_indices = set(range(min(n, len(self.files))))
        self.sync_listbox_colors()
        self.plot_action()

    def clear_selection(self):
        self.selected_indices.clear()
        self.sync_listbox_colors()
        self.plot_action()

    def on_listbox_click(self, event):
        idx = self.file_listbox.nearest(event.y)
        if 0 <= idx < len(self.files):
            if idx in self.selected_indices:
                self.selected_indices.remove(idx)
            else:
                self.selected_indices.add(idx)
            self.sync_listbox_colors()
            self.auto_refresh()

    def sync_listbox_colors(self):
        self.file_listbox.selection_clear(0, tk.END)
        for i in range(self.file_listbox.size()):
            if i in self.selected_indices:
                self.file_listbox.itemconfig(i, bg="#fff200", fg="black")  # 黄色高亮
                self.file_listbox.selection_set(i)
            else:
                self.file_listbox.itemconfig(i, bg="white", fg="black")
        count = len(self.selected_indices)
        self.status_lbl.config(text=f"({count} selected)")

    def switch_station(self, sid):
        self.current_station = sid
        self.update_button_colors()
        self.plot_action()

    def update_button_colors(self):
        for sid, btn in self.station_buttons.items():
            if sid == self.current_station:
                btn.config(bg="#005fb8", fg="white")
            else:
                btn.config(bg="#e1e1e1", fg="black")

    def plot_action(self):
        """核心绘图逻辑：使用 global_step 精确对齐时间轴"""
        if self.is_plotting:
            return
        self.is_plotting = True

        display_indices = sorted(list(self.selected_indices))
        is_empty_selection = False
        if not display_indices:
            if not self.files:
                self.fig.clf()
                self.canvas.draw_idle()
                self.is_plotting = False
                return
            display_indices = [0]  # 借用首个文件提取 Ground Truth
            is_empty_selection = True
        
        n_windows = max(1, self.win_var.get())
        concat = self.concat_var.get()
        inverse = self.inverse_var.get()
        station_id = self.current_station
        
        try:
            self.fig.clf()
            colors = plt.cm.tab10(np.linspace(0, 1, 10))
            
            # 获取 scaler
            mean, std = self.get_scaler_from_cache(station_id) if inverse else (0.0, 1.0)
            if mean is None:
                mean, std = 0.0, 1.0
            
            # 读取所有选中的文件数据
            models_data = []
            for i, idx in enumerate(display_indices):
                if idx >= len(self.files):
                    continue
                file_path = self.files[idx]
                try:
                    df = pd.read_csv(file_path)
                    pred_col = f"station_{station_id}_pred"
                    true_col = f"station_{station_id}_true"
                    if pred_col not in df.columns or true_col not in df.columns:
                        continue
                    
                    # 全局时间步对齐
                    df["global_step"] = df["window"] + df["step"]
                    
                    # 反归一化
                    df["plot_pred"] = df[pred_col] * std + mean
                    df["plot_true"] = df[true_col] * std + mean
                    
                    label_name = parse_model_label(file_path)
                    models_data.append({
                        "file": file_path,
                        "df": df,
                        "label": label_name,
                        "color": colors[i % 10]
                    })
                except Exception as ex:
                    print(f"Error loading {file_path}: {ex}")

            if not models_data:
                self.fig.text(0.5, 0.5, "No valid data to display", ha='center', va='center', color='gray')
                self.canvas.draw_idle()
                self.is_plotting = False
                return

            y_label = "Power (MW)" if inverse else "Normalized Power (Std)"

            if concat:
                # =================== Concat Mode ===================
                ax = self.fig.add_subplot(111)
                
                # 确定要展示的公共时间跨度（基于首个模型的独特窗口）
                first_df = models_data[0]["df"]
                unique_wins = sorted(first_df["window"].unique())[:n_windows]
                sub_first = first_df[first_df["window"].isin(unique_wins)].sort_values("global_step")
                
                min_step = sub_first["global_step"].min()
                max_step = sub_first["global_step"].max()
                
                # 绘制 Ground Truth 实线（使用全局时间步）
                ax.plot(
                    sub_first["global_step"], sub_first["plot_true"],
                    label="Ground Truth", color="black", linewidth=2.0, alpha=0.55
                )
                
                # 绘制各个模型的预测曲线
                if not is_empty_selection:
                    for m in models_data:
                        m_df = m["df"]
                        # 筛选落在当前时间区间内的预测数据
                        m_sub = m_df[(m_df["global_step"] >= min_step) & (m_df["global_step"] <= max_step)].sort_values("global_step")
                        if not m_sub.empty:
                            ax.plot(
                                m_sub["global_step"], m_sub["plot_pred"],
                                label=m["label"], color=m["color"], linestyle="--", linewidth=1.6
                            )
                
                title = f"Station {station_id} Solar Forecast (Concat {len(unique_wins)} Windows)"
                if is_empty_selection:
                    title += " [Ground Truth Only - Please Select Models]"
                
                ax.set_title(title, fontsize=12, fontweight="bold", pad=10)
                ax.set_xlabel("Time Step (10-min interval)", fontsize=10)
                ax.set_ylabel(y_label, fontsize=10)
                ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
                ax.grid(True, linestyle="--", alpha=0.35)

            else:
                # =================== Subplot Mode (Per Window) ===================
                first_df = models_data[0]["df"]
                unique_wins = sorted(first_df["window"].unique())[:n_windows]
                n_plots = len(unique_wins)
                
                axes = self.fig.subplots(n_plots, 1, sharex=False)
                axes = np.atleast_1d(axes).flatten()
                
                for j, w in enumerate(unique_wins):
                    ax_j = axes[j]
                    w_first = first_df[first_df["window"] == w].sort_values("step")
                    
                    # 绘制当前窗口的 Ground Truth
                    ax_j.plot(
                        w_first["step"], w_first["plot_true"],
                        label="GT", color="black", linewidth=1.8, alpha=0.5
                    )
                    
                    if not is_empty_selection:
                        for m in models_data:
                            m_df = m["df"]
                            m_w = m_df[m_df["window"] == w].sort_values("step")
                            if not m_w.empty:
                                ax_j.plot(
                                    m_w["step"], m_w["plot_pred"],
                                    label=m["label"], color=m["color"], linestyle="--", linewidth=1.4
                                )
                    
                    ax_j.set_title(f"Window {w} (Step {w} ~ {w + len(w_first) - 1})", fontsize=9, loc='left')
                    ax_j.grid(True, linestyle="--", alpha=0.3)
                    ax_j.set_ylabel(y_label, fontsize=8)
                    if not is_empty_selection:
                        ax_j.legend(loc="upper right", fontsize=7, framealpha=0.85)

                axes[-1].set_xlabel("Window Step Index", fontsize=10)
                self.fig.suptitle(f"Station {station_id} Multi-Window Comparison", fontsize=11, fontweight="bold", y=0.995)

            self.fig.tight_layout()
            self.canvas.draw_idle()

        except Exception as e:
            print(f"Plotting error: {e}")
            self.fig.clf()
            self.fig.text(0.5, 0.5, f"Plotting Error:\n{e}", ha='center', va='center', color='red', fontsize=11)
            self.canvas.draw_idle()
        finally:
            self.is_plotting = False

if __name__ == "__main__":
    try:
        extract.main()
    except Exception as ex:
        print(f"Warning: extract.main() failed: {ex}")
    root = tk.Tk()
    app = ResultViewer(root)
    root.mainloop()
