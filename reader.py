import os
import glob
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import tkinter as tk
from tkinter import ttk, messagebox

# 配置信息
DATA_ROOT = "./datasets/Solar/"
DATA_PATH = "solar_AL.txt"
RESULTS_DIR = "./results/Solar/"
CACHE_FILE = os.path.join(RESULTS_DIR, "scaler_cache.json")

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
        list_frame = ttk.Frame(self.right_sidebar)
        list_frame.pack(fill=tk.BOTH, expand=True, pady=(2, 10))
        
        sb = ttk.Scrollbar(list_frame)
        sb.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.file_listbox = tk.Listbox(list_frame, font=("Consolas", 9), 
                                       yscrollcommand=sb.set, bg="#fdfdfd", 
                                       selectmode=tk.SINGLE, highlightthickness=0,
                                       activestyle='none') # 去掉下划线
        self.file_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        sb.config(command=self.file_listbox.yview)
        
        # 绑定事件
        self.file_listbox.bind('<Button-1>', lambda e: self.root.after(1, lambda: self.file_listbox.selection_clear(0, tk.END))) # 强行去掉蓝色选中
        self.file_listbox.bind('<Double-1>', self.toggle_selection)

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

    def refresh_files(self):
        pattern = os.path.join(self.results_dir, "*.csv")
        self.files = sorted(glob.glob(pattern), key=os.path.getmtime, reverse=True)
        self.file_listbox.delete(0, tk.END)
        for f in self.files:
            self.file_listbox.insert(tk.END, f" {os.path.basename(f)}")
        # 恢复之前的颜色
        self.sync_listbox_colors()

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
                short_name = os.path.basename(file_path).split('_mse')[0]

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
                        ax.plot(all_pred, label=f"Pred: {short_name}", color=colors[i%10], linestyle="--")
                else:
                    if first_file:
                        axes = self.fig.subplots(len(unique_wins), 1, sharex=True)
                        if len(unique_wins) == 1: axes = [axes]
                    
                    for j, w in enumerate(unique_wins):
                        w_data = df[df["window"] == w].sort_values("step")
                        if first_file:
                            axes[j].plot(w_data[true_col].values, label="GT", color="black", linewidth=1.5, alpha=0.4)
                        
                        if not is_empty_selection:
                            axes[j].plot(w_data[pred_col].values, label=short_name, color=colors[i%10], linestyle="--")
                        
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
    root = tk.Tk()
    app = ResultViewer(root)
    root.mainloop()
