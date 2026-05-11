"""
加载 test 阶段保存的 predictions.csv，绘制指定电站的预测值与真实值对比图。

用法:
    python plot_predictions.py <csv_path> --station <station_id> [--n_windows <N>] [--output <img_path>]

示例:
    python plot_predictions.py results/long_term_forecast_.../predictions.csv --station 0
    python plot_predictions.py results/long_term_forecast_.../predictions.csv --station 5 --n_windows 10 --output plot.png
"""

import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings

# 默认最大窗口数，避免一次性画太多子图
DEFAULT_MAX_WINDOWS = 9


def plot_predictions(csv_path, station_id, n_windows=None, output_path=None, max_windows=DEFAULT_MAX_WINDOWS):
    # 1. 仅读取需要的列（大幅减少 IO 和内存）
    use_cols = ["window", "step", f"station_{station_id}_pred", f"station_{station_id}_true"]
    try:
        df = pd.read_csv(csv_path, usecols=use_cols)
    except ValueError as e:
        # 可能列名不存在，回退到读取全表以便诊断
        df = pd.read_csv(csv_path)
        available = [c.replace("_pred", "").replace("_true", "")
                     for c in df.columns if c.endswith("_pred")]
        print(f"Station {station_id} not found. Available stations: {sorted(set(available))}")
        return

    print(f"Loaded {csv_path}: {df.shape} (only necessary columns)")

    pred_col = f"station_{station_id}_pred"
    true_col = f"station_{station_id}_true"

    # 2. 确定实际窗口数，并限制到合理范围
    max_window = int(df["window"].max()) + 1
    if n_windows is None:
        n_windows = min(max_window, max_windows)
        if max_window > max_windows:
            print(f"Total windows = {max_window}, limiting plot to first {max_windows}. "
                  f"Use --n_windows to change, or --max_windows to adjust limit.")
    else:
        n_windows = min(n_windows, max_window)

    # 3. 创建子图（行数可控，不会爆炸）
    n_cols = 1
    n_rows = n_windows
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 2.5 * n_rows), squeeze=False)
    axes = axes.flatten()

    # 预测长度（假设所有窗口同一个 horizon）
    pred_len = int(df[df["window"] == 0]["step"].max()) + 1

    for idx in range(n_windows):
        ax = axes[idx]
        win_data = df[df["window"] == idx].sort_values("step")

        steps = win_data["step"].values
        true_vals = win_data[true_col].values
        pred_vals = win_data[pred_col].values

        # 4. 绘图：不画 marker 点，仅用线条（速度大幅提升）
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

    # 5. 保存或显示（优先保存文件，避免 GUI 开销）
    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to {output_path}")
    else:
        plt.show()
    plt.close(fig)  # 释放内存


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot predicted vs true values for a specific station")
    parser.add_argument("csv_path", type=str, help="Path to predictions.csv")
    parser.add_argument("--station", type=int, required=True, help="Station (feature) index to plot")
    parser.add_argument("--n_windows", type=int, default=None,
                        help="Number of windows to plot (default: up to max_windows)")
    parser.add_argument("--max_windows", type=int, default=DEFAULT_MAX_WINDOWS,
                        help="Safety limit for windows if --n_windows is not set")
    parser.add_argument("--output", type=str, default=None,
                        help="Save figure to file instead of showing (e.g., plot.png)")
    args = parser.parse_args()

    plot_predictions(args.csv_path, args.station, args.n_windows, args.output, args.max_windows)