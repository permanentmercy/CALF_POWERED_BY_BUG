#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Script: plot_fewshot_il_micro.py
Description: Aggregate and plot extreme few-shot fine-tuning results (0.5% to 10%) on Solar-IL
Outputs:
  - results/fewshot_il_micro_summary.csv: Aggregated mean & std table
  - figures/fewshot/fewshot_il_micro_curve.png: High-resolution publication curve
"""

import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 11
plt.rcParams['axes.linewidth'] = 1.2
plt.rcParams['axes.unicode_minus'] = False


def main():
    parser = argparse.ArgumentParser(description="Aggregate and plot micro few-shot results")
    parser.add_argument("--csv_path", type=str, default="./results/results_fewshot_il_micro.csv")
    parser.add_argument("--output_dir", type=str, default="./figures/fewshot")
    parser.add_argument("--summary_csv", type=str, default="./results/fewshot_il_micro_summary.csv")
    args = parser.parse_args()

    if not os.path.exists(args.csv_path):
        print(f">>> [WARN] CSV not found: {args.csv_path}")
        return

    df = pd.read_csv(args.csv_path)
    df = df[df['status'] == 'SUCCESS'].copy()
    if len(df) == 0:
        print(">>> [WARN] No successful runs in CSV.")
        return

    df['percent'] = pd.to_numeric(df['percent'], errors='coerce')
    df['test_mse'] = pd.to_numeric(df['test_mse'], errors='coerce')
    df['test_mae'] = pd.to_numeric(df['test_mae'], errors='coerce')

    grouped = df.groupby('percent').agg(
        MSE_Mean=('test_mse', 'mean'),
        MSE_Std=('test_mse', 'std'),
        MAE_Mean=('test_mae', 'mean'),
        MAE_Std=('test_mae', 'std'),
        Seed_Count=('test_mse', 'count'),
        Avg_Elapsed_Sec=('elapsed_sec', 'mean')
    ).reset_index().sort_values(by='percent')

    os.makedirs(os.path.dirname(os.path.abspath(args.summary_csv)), exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)

    grouped.to_csv(args.summary_csv, index=False, float_format='%.6f')
    print("\n" + "=" * 90)
    print(">>> [FEW-SHOT SUMMARY TABLE: SOLAR-IL (0.5% - 10%)]")
    print("=" * 90)
    print(grouped.to_string(index=False))
    print("=" * 90)
    print(f">>> Summary table saved to: {args.summary_csv}\n")

    # Plot
    fig, ax1 = plt.subplots(figsize=(9, 6), dpi=300)
    percents = grouped['percent'].values
    x_indices = np.arange(len(percents))
    x_labels = [f"{p:g}%" for p in percents]

    mse_m = grouped['MSE_Mean'].values
    mse_s = grouped['MSE_Std'].fillna(0).values
    mae_m = grouped['MAE_Mean'].values
    mae_s = grouped['MAE_Std'].fillna(0).values

    color_mse = '#1f77b4'
    color_mae = '#ff7f0e'

    # Left Axis: MSE
    line1 = ax1.plot(x_indices, mse_m, color=color_mse, marker='o', linewidth=2.4, markersize=8, label='Test MSE (Mean ± Std)')
    ax1.fill_between(x_indices, mse_m - mse_s, mse_m + mse_s, color=color_mse, alpha=0.18)
    ax1.set_xlabel('Few-Shot Training Subset Percentage (%)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Test Mean Squared Error (MSE)', color=color_mse, fontsize=12, fontweight='bold')
    ax1.tick_params(axis='y', labelcolor=color_mse)
    ax1.set_xticks(x_indices)
    ax1.set_xticklabels(x_labels, fontsize=11, fontweight='bold')
    ax1.grid(True, linestyle=':', alpha=0.6)

    # Right Axis: MAE
    ax2 = ax1.twinx()
    line2 = ax2.plot(x_indices, mae_m, color=color_mae, marker='s', linewidth=2.4, markersize=8, linestyle='--', label='Test MAE (Mean ± Std)')
    ax2.fill_between(x_indices, mae_m - mae_s, mae_m + mae_s, color=color_mae, alpha=0.18)
    ax2.set_ylabel('Test Mean Absolute Error (MAE)', color=color_mae, fontsize=12, fontweight='bold')
    ax2.tick_params(axis='y', labelcolor=color_mae)

    # Annotate points
    for idx, (xm, ym) in enumerate(zip(x_indices, mse_m)):
        ax1.annotate(f"{ym:.4f}", xy=(xm, ym), xytext=(0, 7), textcoords="offset points",
                     ha='center', fontsize=9.5, fontweight='bold', color=color_mse)

    for idx, (xm, ym) in enumerate(zip(x_indices, mae_m)):
        ax2.annotate(f"{ym:.4f}", xy=(xm, ym), xytext=(0, -14), textcoords="offset points",
                     ha='center', fontsize=9.5, fontweight='bold', color=color_mae)

    # Combined legend
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper right', frameon=True, framealpha=0.9, fontsize=10.5)

    plt.title('Solar-IL Micro Few-Shot Generalization: 0.5% to 10% Subsets', fontsize=13, fontweight='bold', pad=14)
    fig.tight_layout()

    out_img = os.path.join(args.output_dir, "fewshot_il_micro_curve.png")
    fig.savefig(out_img, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f">>> [Saved Plot] {out_img}")


if __name__ == "__main__":
    main()
