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
SORT_CONFIG_FILE = os.path.join(RESULTS_DIR, "reader_sort_config.json")

class ResultViewer:
    def __init__(self, root):
        self.root = root
        self.root.title("CALF Experiment Comparer - [Double-Click to Toggle Selection]")
        self.root.geometry("1400x850")
        
        # 状态变量
        self.scaler_cache = {}
        self.cache_loaded = False
        self.results_dir = RESULTS_DIR
        self.files = [] # 存储完整路径
        self.selected_indices = set() # 存储选中的索引
        self.current_station = 0
        self.station_buttons = {}
        self.canvas = None
        self.toolbar = None
        sort_settings = self.load_sort_settings()
        self.sort_metric_var = tk.StringVar(value=sort_settings["metric"])
        self.sort_order_var = tk.StringVar(value=sort_settings["order"])
        self.hide_non_swa_var = tk.BooleanVar(value=sort_settings["hide_non_swa"])
        
        self.create_layout()
        self.refresh_files()
        self.preload_scalers()
        
        # 启动时自动触发首绘（展示 Ground Truth）
        self.plot_action()
        
        # 绑定退出
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        
        # 监听变量
        self.win_var.trace_add("write", self.auto_refresh)
        self.concat_var.trace_add("write", self.auto_refresh)
        self.inverse_var.trace_add("write", self.auto_refresh)

    def auto_refresh(self, *args):
        self.root.after(50, self.plot_action)

    def on_close(self):
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
            except: pass

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
            except: pass

    def get_scaler_from_cache(self, station_id):
        val = self.scaler_cache.get(station_id)
        return (val[0], val[1]) if val else (None, None)

    def create_layout(self):
        main_paned = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        main_paned.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # ---- 左侧区域：绘图显示 ----
        self.plot_frame = ttk.Frame(main_paned, relief="sunken")
        main_paned.add(self.plot_frame, weight=3)
        
        self.fig, self.ax = plt.subplots(figsize=(8, 6))
        self.ax.text(0.5, 0.5, "Double-Click files to select for comparison", 
                    ha='center', va='center', color='gray')
        self.setup_canvas()

        # ---- 右侧区域：控制面板 ----
        self.right_sidebar = ttk.Frame(main_paned, padding="10")
        main_paned.add(self.right_sidebar, weight=1)

        # 1. 文件列表 (多选高亮)
        ttk.Label(self.right_sidebar, text="📂 Result Files (Double-Click to Toggle):", font=("Arial", 10, "bold")).pack(anchor=tk.W)

        sort_frame = ttk.Frame(self.right_sidebar)
        sort_frame.pack(fill=tk.X, pady=(2, 5))
        sort_metric = ttk.Combobox(
            sort_frame, textvariable=self.sort_metric_var,
            values=("Time", "MSE", "MAE"), state="readonly", width=8
        )
        sort_metric.pack(side=tk.LEFT, padx=(0, 5))
        sort_order = ttk.Combobox(
            sort_frame, textvariable=self.sort_order_var,
            values=("Ascending", "Descending"), state="readonly", width=12
        )
        sort_order.pack(side=tk.LEFT)
        sort_metric.bind("<<ComboboxSelected>>", self.on_sort_changed)
        sort_order.bind("<<ComboboxSelected>>", self.on_sort_changed)
        ttk.Checkbutton(
            sort_frame, text="Hide non-SWA", variable=self.hide_non_swa_var,
            command=self.on_filter_changed
        ).pack(side=tk.LEFT, padx=(8, 0))

        list_frame = ttk.Frame(self.right_sidebar)
        list_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        sb = ttk.Scrollbar(list_frame)
        sb.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.file_listbox = tk.Listbox(list_frame, font=("Consolas", 9), 
                                       yscrollcommand=sb.set, bg="#fdfdfd", 
                                       selectmode=tk.SINGLE, highlightthickness=0,
                                       activestyle='none', selectbackground="#d9d9d9",
                                       selectforeground="black")
        self.file_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        sb.config(command=self.file_listbox.yview)

        self.file_context_menu = tk.Menu(self.root, tearoff=0)
        self.file_context_menu.add_command(label="Delete File", command=self.delete_selected_file)
        
        # 绑定事件
        self.file_listbox.bind('<Double-1>', self.toggle_selection)
        self.file_listbox.bind('<Button-3>', self.show_file_context_menu)

        # 2. 答题卡式电站选择
        ttk.Label(self.right_sidebar, text="📍 Select Station (0-9):", font=("Arial", 10, "bold")).pack(anchor=tk.W, pady=(10, 5))
        grid_frame = ttk.Frame(self.right_sidebar)
        grid_frame.pack(fill=tk.X)

        for i in range(10):
            row = i // 5
            col = i % 5
            btn = tk.Button(grid_frame, text=str(i), width=4, height=2, 
                            font=("Arial", 10, "bold"),
                            command=lambda sid=i: self.switch_station(sid))
            btn.grid(row=row, column=col, padx=2, pady=2)
            self.station_buttons[i] = btn
        self.update_button_colors()

        # 3. 绘图参数
        cfg_frame = ttk.LabelFrame(self.right_sidebar, text="Global Settings", padding="10")
        cfg_frame.pack(fill=tk.X, pady=15)

        ttk.Label(cfg_frame, text="Windows to Compare:").pack(anchor=tk.W)
        self.win_var = tk.IntVar(value=1)
        self.win_scale = tk.Scale(cfg_frame, from_=1, to=10, orient=tk.HORIZONTAL, 
                                  variable=self.win_var, showvalue=True)
        self.win_scale.pack(fill=tk.X, pady=(0, 10))

        self.concat_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(cfg_frame, text="Concat Mode", variable=self.concat_var).pack(side=tk.LEFT, padx=5)
        
        self.inverse_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(cfg_frame, text="Inverse Scale", variable=self.inverse_var).pack(side=tk.LEFT)

        # 5. 刷新按钮
        ttk.Button(self.right_sidebar, text="🔄 Clear All & Refresh List", command=self.full_refresh).pack(fill=tk.X, pady=(10, 0))

    def setup_canvas(self):
        if self.canvas:
            self.canvas.get_tk_widget().destroy()
            self.toolbar.destroy()
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.plot_frame)
        self.canvas.draw()
        toolbar_frame = ttk.Frame(self.plot_frame)
        toolbar_frame.pack(side=tk.BOTTOM, fill=tk.X)
        self.toolbar = NavigationToolbar2Tk(self.canvas, toolbar_frame)
        self.toolbar.update()
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    @staticmethod
    def load_sort_settings():
        defaults = {"metric": "Time", "order": "Descending", "hide_non_swa": False}
        try:
            with open(SORT_CONFIG_FILE, "r", encoding="utf-8") as file:
                settings = json.load(file)
            if settings.get("metric") in ("Time", "MSE", "MAE"):
                defaults["metric"] = settings["metric"]
            if settings.get("order") in ("Ascending", "Descending"):
                defaults["order"] = settings["order"]
            if isinstance(settings.get("hide_non_swa"), bool):
                defaults["hide_non_swa"] = settings["hide_non_swa"]
        except (OSError, json.JSONDecodeError):
            pass
        return defaults

    def save_sort_settings(self):
        try:
            os.makedirs(os.path.dirname(SORT_CONFIG_FILE), exist_ok=True)
            with open(SORT_CONFIG_FILE, "w", encoding="utf-8") as file:
                json.dump({
                    "metric": self.sort_metric_var.get(),
                    "order": self.sort_order_var.get(),
                    "hide_non_swa": self.hide_non_swa_var.get()
                }, file, ensure_ascii=False, indent=2)
        except OSError:
            pass

    def on_sort_changed(self, event=None):
        self.save_sort_settings()
        self.refresh_files()

    def on_filter_changed(self):
        self.save_sort_settings()
        self.refresh_files()

    def show_file_context_menu(self, event):
        index = self.file_listbox.nearest(event.y)
        if index >= self.file_listbox.size():
            return
        self.file_listbox.selection_clear(0, tk.END)
        self.file_listbox.selection_set(index)
        self.file_listbox.activate(index)
        self.file_context_menu.tk_popup(event.x_root, event.y_root)

    def delete_selected_file(self):
        selection = self.file_listbox.curselection()
        if not selection:
            return

        index = selection[0]
        if index >= len(self.files):
            return
        file_path = self.files[index]
        file_name = os.path.basename(file_path)
        if not messagebox.askyesno("Confirm Delete", f"Delete this file?\n\n{file_name}"):
            return

        try:
            os.remove(file_path)
        except OSError as error:
            messagebox.showerror("Delete Failed", f"Cannot delete file:\n{error}")
            return

        self.selected_indices.discard(index)
        self.refresh_files()
        self.plot_action()

    def refresh_files(self):
        selected_files = {
            self.files[index] for index in self.selected_indices
            if index < len(self.files)
        }
        pattern = os.path.join(self.results_dir, "*.csv")
        files = glob.glob(pattern)
        if self.hide_non_swa_var.get():
            files = [path for path in files if "_swa" in os.path.basename(path).lower()]
        metric = self.sort_metric_var.get()
        reverse = self.sort_order_var.get() == "Descending"
        if metric == "Time":
            self.files = sorted(files, key=os.path.getmtime, reverse=reverse)
        else:
            self.files = sorted(
                files,
                key=lambda path: self.get_sort_key(path, metric, reverse),
                reverse=reverse
            )

        self.selected_indices = {
            index for index, path in enumerate(self.files)
            if path in selected_files
        }
        self.file_listbox.delete(0, tk.END)
        for f in self.files:
            display_name = os.path.basename(f)
            if metric != "Time":
                metric_value = self.get_metric_from_filename(f, metric)
                metric_text = f"{metric_value:.4f}" if metric_value is not None else "N/A"
                display_name = f"{metric}={metric_text} | {display_name}"
            self.file_listbox.insert(tk.END, f" {display_name}")
        # 恢复之前的颜色
        self.sync_listbox_colors()

    @staticmethod
    def get_metric_from_filename(file_path, metric):
        metric_name = metric.lower()
        match = re.search(
            rf"(?:^|_){metric_name}(-?[0-9]+(?:\.[0-9]+)?)(?=_|\.|$)",
            os.path.basename(file_path),
            re.IGNORECASE
        )
        return float(match.group(1)) if match else None

    @classmethod
    def get_sort_key(cls, file_path, metric, reverse):
        value = cls.get_metric_from_filename(file_path, metric)
        if value is None:
            return float("-inf" if reverse else "inf")
        return value

    def full_refresh(self):
        self.selected_indices.clear()
        self.refresh_files()
        self.plot_action()

    def toggle_selection(self, event):
        idx = self.file_listbox.nearest(event.y)
        if idx in self.selected_indices:
            self.selected_indices.remove(idx)
        else:
            self.selected_indices.add(idx)
        self.sync_listbox_colors()
        self.plot_action()

    def sync_listbox_colors(self):
        for i in range(self.file_listbox.size()):
            if i in self.selected_indices:
                self.file_listbox.itemconfig(i, bg="#fff200", fg="black") # 黄色高亮
            else:
                self.file_listbox.itemconfig(i, bg="white", fg="black")

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
        """核心绘图逻辑"""
        # 如果什么都没选，我们默认用第一个文件来展示 Ground Truth
        display_indices = sorted(list(self.selected_indices))
        is_empty_selection = False
        if not display_indices:
            if not self.files:
                self.fig.clf()
                self.canvas.draw()
                return
            display_indices = [0] # 临时借用第一个文件画 GT
            is_empty_selection = True
        
        n_windows = self.win_var.get()
        concat = self.concat_var.get()
        inverse = self.inverse_var.get()
        station_id = self.current_station
        
        try:
            self.fig.clf()
            ax = None
            axes = []
            colors = plt.cm.tab10(np.linspace(0, 1, 10))
            
            first_file = True
            for i, idx in enumerate(display_indices):
                if idx >= len(self.files): continue
                file_path = self.files[idx]
                df = pd.read_csv(file_path)
                pred_col = f"station_{station_id}_pred"
                true_col = f"station_{station_id}_true"
                
                if inverse:
                    mean, std = self.get_scaler_from_cache(station_id)
                    if mean is not None:
                        df[pred_col] = df[pred_col] * std + mean
                        df[true_col] = df[true_col] * std + mean

                unique_wins = sorted(df["window"].unique())[:n_windows]
                
                # 从文件名提取 MSE 数值
                try:
                    mse_val = os.path.basename(file_path).split('mse')[1].split('_')[0]
                    label_name = f"MSE: {mse_val}"
                    if "_swa" in file_path: label_name += " (SWA)"
                except:
                    label_name = os.path.basename(file_path)[:15]

                if concat:
                    if first_file:
                        ax = self.fig.add_subplot(111)
                    
                    all_true, all_pred = [], []
                    for w in unique_wins:
                        w_data = df[df["window"] == w].sort_values("step")
                        if first_file: all_true.extend(w_data[true_col].values)
                        all_pred.extend(w_data[pred_col].values)
                    
                    if first_file:
                        ax.plot(all_true, label="Ground Truth", color="black", linewidth=2.0, alpha=0.5)
                    
                    # 如果不是单纯为了画 GT，才画预测线
                    if not is_empty_selection:
                        ax.plot(all_pred, label=label_name, color=colors[i%10], linestyle="--")
                else:
                    if first_file:
                        axes = self.fig.subplots(len(unique_wins), 1, sharex=True)
                        if len(unique_wins) == 1: axes = [axes]
                    
                    for j, w in enumerate(unique_wins):
                        w_data = df[df["window"] == w].sort_values("step")
                        if first_file:
                            axes[j].plot(w_data[true_col].values, label="GT", color="black", linewidth=1.5, alpha=0.4)
                        
                        if not is_empty_selection:
                            axes[j].plot(w_data[pred_col].values, label=label_name, color=colors[i%10], linestyle="--")
                        
                        if i == 0: axes[j].set_title(f"Window {w}", fontsize=8)

                first_file = False

            if concat:
                title = f"Station {station_id} Comparison" if not is_empty_selection else f"Station {station_id} Ground Truth Only"
                ax.set_title(title, fontsize=11)
                ax.legend(loc="upper right", fontsize=8)
                ax.grid(True, alpha=0.3)
            else:
                axes[0].legend(loc="upper right", fontsize=7)

            self.fig.tight_layout()
            self.canvas.draw()
        except Exception as e:
            print(f"Plotting error: {e}")

if __name__ == "__main__":
    extract.main()
    root = tk.Tk()
    app = ResultViewer(root)
    root.mainloop()
