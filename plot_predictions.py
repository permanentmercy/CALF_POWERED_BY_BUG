"""
加载 test 阶段保存的 predictions.csv，绘制指定电站的预测值与真实值对比图。

用法:
    python plot_predictions.py <csv_path> --station <station_id> [--n_windows <N>] [--output <img_path>]
    python plot_predictions.py <csv_path> --station <station_id> --concat 1  # 拼接所有窗口为一张长图

新增参数:
    --concat <0|1>  若设为 1，将所有窗口按时间顺序首尾拼接，绘制在一张大图中（默认 0 为分窗口子图）
"""

import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings

# 默认最大窗口数，仅当不是 concat 模式时生效
DEFAULT_MAX_WINDOWS = 9


def plot_predictions(csv_path, station_id, n_windows=None, output_path=None,
                     max_windows=DEFAULT_MAX_WINDOWS, concat=0):
    # 1. 仅读取需要的列
    use_cols = ["window", "step", f"station_{station_id}_pred", f"station_{station_id}_true"]
    try:
        df = pd.read_csv(csv_path, usecols=use_cols)
    except ValueError:
        df = pd.read_csv(csv_path)
        available = [c.replace("_pred", "").replace("_true", "")
                     for c in df.columns if c.endswith("_pred")]
        print(f"Station {station_id} not found. Available stations: {sorted(set(available))}")
        return

    print(f"Loaded {csv_path}: {df.shape} (only necessary columns)")

    pred_col = f"station_{station_id}_pred"
    true_col = f"station_{station_id}_true"

    # 2. 确定实际窗口数
    max_window = int(df["window"].max()) + 1
    if n_windows is not None:
        n_windows = min(n_windows, max_window)
    else:
        n_windows = max_window  # 默认全取，下面再根据模式限制

    # 如果 concat 模式，强制使用所有窗口（或 n_windows 指定的部分窗口），忽略 max_windows 限制
    if concat == 1:
        print(f"Concat mode: plotting {n_windows} windows in one figure.")
        # 估算预测长度
        pred_len = int(df[df["window"] == 0]["step"].max()) + 1

        # 取出指定窗口数（按 window 排序）
        windows_to_plot = sorted(df["window"].unique())[:n_windows]
        true_all = []
        pred_all = []
        time_steps = []

        global_step = 0
        for win_id in windows_to_plot:
            win_data = df[df["window"] == win_id].sort_values("step")
            steps = win_data["step"].values
            true_vals = win_data[true_col].values
            pred_vals = win_data[pred_col].values

            # 生成全局连续步数
            global_steps = steps + global_step
            time_steps.append(global_steps)
            true_all.append(true_vals)
            pred_all.append(pred_vals)
            global_step += pred_len   # 假设所有窗口等长

        true_all = np.concatenate(true_all)
        pred_all = np.concatenate(pred_all)
        time_steps = np.concatenate(time_steps)

        fig, ax = plt.subplots(figsize=(16, 5))
        ax.plot(time_steps, true_all, label="True", color="blue", linewidth=1.5)
        ax.plot(time_steps, pred_all, label="Predicted", color="red", linewidth=1.5)
        ax.fill_between(time_steps, true_all, pred_all, alpha=0.15, color="gray")
        ax.set_title(f"Station {station_id} — Concatenated Prediction vs Ground Truth "
                     f"({len(windows_to_plot)} windows)")
        ax.set_xlabel("Global time step (windows concatenated)")
        ax.set_ylabel("Value")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()

        if output_path:
            fig.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"Figure saved to {output_path}")
        else:
            plt.show()
        plt.close(fig)
        return

    # 3. 原有子图模式（concat=0）
    if n_windows > max_windows and max_windows < DEFAULT_MAX_WINDOWS:
        n_windows = min(n_windows, max_windows)
    else:
        # 如果用户没指定，使用 max_windows 限制显示数
        if n_windows is None:
            n_windows = min(max_window, max_windows)
            if max_window > max_windows:
                print(f"Total windows = {max_window}, limiting plot to first {max_windows}. "
                      f"Use --n_windows to change, or --max_windows to adjust limit.")
        else:
            n_windows = min(n_windows, max_window)

    n_cols = 1
    n_rows = n_windows
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 2.5 * n_rows), squeeze=False)
    axes = axes.flatten()

    pred_len = int(df[df["window"] == 0]["step"].max()) + 1

    for idx in range(n_windows):
        ax = axes[idx]
        win_data = df[df["window"] == idx].sort_values("step")
        steps = win_data["step"].values
        true_vals = win_data[true_col].values
        pred_vals = win_data[pred_col].values

        ax.plot(steps, true_vals, label="True", color="blue", linewidth=1.5)
        ax.plot(steps, pred_vals, label="Predicted", color="red", linewidth=1.5)
        ax.fill_between(steps, true_vals, pred_vals, alpha=0.15, color="gray")

        ax.set_title(f"Window {idx} — Station {station_id}", fontsize=10)
        ax.set_xlabel("Time step in prediction horizon")
        ax.set_ylabel("Value")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle(f"Station {station_id} — Prediction vs Ground Truth ({n_windows} windows)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to {output_path}")
    else:
        plt.show()
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot predicted vs true values for a specific station")
    parser.add_argument("csv_path", type=str, help="Path to predictions.csv")
    parser.add_argument("--station", type=int, required=True, help="Station (feature) index to plot")
    parser.add_argument("--n_windows", type=int, default=None,
                        help="Number of windows to plot (default: up to max_windows in subplot mode, all in concat mode)")
    parser.add_argument("--max_windows", type=int, default=DEFAULT_MAX_WINDOWS,
                        help="Safety limit for windows if --n_windows is not set (only in subplot mode)")
    parser.add_argument("--output", type=str, default=None,
                        help="Save figure to file instead of showing (e.g., plot.png)")
    parser.add_argument("--concat", type=int, default=0,
                        help="Set to 1 to concatenate all windows into one continuous plot")
    args = parser.parse_args()

    plot_predictions(
        csv_path=args.csv_path,
        station_id=args.station,
        n_windows=args.n_windows,
        output_path=args.output,
        max_windows=args.max_windows,
        concat=args.concat
    )