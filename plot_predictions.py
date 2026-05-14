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


def plot_predictions(csv_path, station_id, n_windows=None, output_path=None, max_windows=DEFAULT_MAX_WINDOWS, concat=False):
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

    # 2. 确定实际窗口列表，并限制到合理范围
    unique_windows = sorted(df["window"].unique())
    if n_windows is None:
        n_windows = min(len(unique_windows), max_windows)
        if len(unique_windows) > max_windows:
            print(f"Total windows = {len(unique_windows)}, limiting plot to first {max_windows}. "
                  f"Use --n_windows to change, or --max_windows to adjust limit.")
    else:
        n_windows = min(n_windows, len(unique_windows))
    
    target_windows = unique_windows[:n_windows]

    # 3. 创建画布
    if concat:
        fig, ax = plt.subplots(figsize=(15, 5))
        all_true = []
        all_pred = []
        all_steps = []
        offset = 0
        
        for win_id in target_windows:
            win_data = df[df["window"] == win_id].sort_values("step")
            true_vals = win_data[true_col].values
            pred_vals = win_data[pred_col].values
            steps = win_data["step"].values
            
            all_true.append(true_vals)
            all_pred.append(pred_vals)
            # 这里的 steps 是相对偏移，累加 offset 实现连续绘制
            all_steps.append(np.arange(len(steps)) + offset)
            offset += len(steps)
            
        all_true = np.concatenate(all_true)
        all_pred = np.concatenate(all_pred)
        all_steps = np.concatenate(all_steps)

        ax.plot(all_steps, all_true, label="True", color="blue", linewidth=1.5)
        ax.plot(all_steps, all_pred, label="Predicted", color="red", linewidth=1.5)
        ax.fill_between(all_steps, all_true, all_pred, alpha=0.15, color="gray")
        
        ax.set_title(f"Station {station_id} — Concatenated {n_windows} Windows", fontsize=12)
        ax.set_xlabel("Cumulative Time Steps")
        ax.set_ylabel("Value")
        ax.legend()
        ax.grid(True, alpha=0.3)
    else:
        n_cols = 1
        n_rows = n_windows
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 2.5 * n_rows), squeeze=False)
        axes = axes.flatten()

        for idx, win_id in enumerate(target_windows):
            ax = axes[idx]
            win_data = df[df["window"] == win_id].sort_values("step")

            steps = win_data["step"].values
            true_vals = win_data[true_col].values
            pred_vals = win_data[pred_col].values

            # 4. 绘图：不画 marker 点，仅用线条
            ax.plot(steps, true_vals, label="True", color="blue", linewidth=1.5)
            ax.plot(steps, pred_vals, label="Predicted", color="red", linewidth=1.5)
            ax.fill_between(steps, true_vals, pred_vals, alpha=0.15, color="gray")

            ax.set_title(f"Window {win_id} — Station {station_id}", fontsize=10)
            ax.set_xlabel("Time step in prediction horizon")
            ax.set_ylabel("Value")
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

    fig.suptitle(f"Station {station_id} — Prediction vs Ground Truth ({n_windows} windows)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()

    # 5. 保存或显示
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
                        help="Number of windows to plot (default: up to max_windows)")
    parser.add_argument("--max_windows", type=int, default=DEFAULT_MAX_WINDOWS,
                        help="Safety limit for windows if --n_windows is not set")
    parser.add_argument("--concat", type=int, default=0, choices=[0, 1],
                        help="Whether to concatenate windows into a single plot (1 for True, 0 for False)")
    parser.add_argument("--output", type=str, default=None,
                        help="Save figure to file instead of showing (e.g., plot.png)")
    args = parser.parse_args()

    plot_predictions(args.csv_path, args.station, args.n_windows, args.output, args.max_windows, concat=bool(args.concat))