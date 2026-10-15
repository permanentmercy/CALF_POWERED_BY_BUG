#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Script: plot_sensitivity_results.py
Description: Automated Publication-Grade Visualizations for AL Few-Shot & Sensitivity Experiments
Generates (pure Matplotlib, zero seaborn dependency):
  1. curve_fewshot_al_decay.png: Few-shot performance degradation curve (10% to 50%)
  2. heatmap_depth_gpt_vs_mlp_al.png: 4x4 Architectural Depth Heatmap (gpt_layers x mlp_layers)
  3. heatmap_capacity_lora_vs_dff_al.png: 4x4 Model Capacity Heatmap (LoRA Rank r x FFN Dim d_ff)
  4. heatmap_distillation_simplex_al.png: Multi-task distillation loss balance comparison
"""

import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Style configuration for scientific publication
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 11
plt.rcParams['axes.linewidth'] = 1.2
plt.rcParams['axes.unicode_minus'] = False


def safe_savefig(fig, path):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches='tight')
    print(f">>> [Saved Plot] {path}")


def draw_annotated_matrix(ax, matrix, row_labels, col_labels, cmap="YlGnBu_r", cbar_label="Test MSE"):
    """Pure matplotlib implementation of annotated 2D heatmap"""
    im = ax.imshow(matrix, cmap=cmap, aspect='auto')
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(cbar_label, fontsize=11, fontweight='bold')

    ax.set_xticks(np.arange(len(col_labels)))
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_xticklabels(col_labels, fontsize=10.5)
    ax.set_yticklabels(row_labels, fontsize=10.5)

    # Annotate numbers
    norm_vals = (matrix - np.nanmin(matrix)) / (np.nanmax(matrix) - np.nanmin(matrix) + 1e-8)
    for i in range(len(row_labels)):
        for j in range(len(col_labels)):
            val = matrix[i, j]
            if np.isnan(val):
                text = "N/A"
                text_color = "gray"
            else:
                text = f"{val:.4f}"
                # Invert text color if background is too dark
                text_color = "white" if norm_vals[i, j] < 0.35 or norm_vals[i, j] > 0.85 else "black"
            ax.text(j, i, text, ha="center", va="center", color=text_color, fontsize=10.5, fontweight="bold")

    return im


def plot_fewshot_curve(fewshot_csv, output_dir):
    if not os.path.exists(fewshot_csv):
        print(f">>> [INFO] Few-shot CSV not found at: {fewshot_csv} (Will be plotted after training completes)")
        return

    df = pd.read_csv(fewshot_csv)
    df = df[df['status'] == 'SUCCESS'].copy()
    if len(df) == 0:
        return

    df['test_mse'] = pd.to_numeric(df['test_mse'], errors='coerce')
    df['test_mae'] = pd.to_numeric(df['test_mae'], errors='coerce')
    grouped = df.groupby('percent').agg(
        mse_mean=('test_mse', 'mean'),
        mse_std=('test_mse', 'std'),
        mae_mean=('test_mae', 'mean'),
        mae_std=('test_mae', 'std')
    ).reset_index()

    fig, ax1 = plt.subplots(figsize=(8.5, 5.5), dpi=300)
    percents = grouped['percent'].values
    mse_m = grouped['mse_mean'].values
    mse_s = grouped['mse_std'].fillna(0).values
    mae_m = grouped['mae_mean'].values
    mae_s = grouped['mae_std'].fillna(0).values

    color_mse = '#1f77b4'
    color_mae = '#ff7f0e'

    # Line 1: MSE
    ax1.plot(percents, mse_m, color=color_mse, marker='o', linewidth=2.2, markersize=7, label='Test MSE')
    ax1.fill_between(percents, mse_m - mse_s, mse_m + mse_s, color=color_mse, alpha=0.18)
    ax1.set_xlabel('Training Data Percentage (%)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Test Mean Squared Error (MSE)', color=color_mse, fontsize=12, fontweight='bold')
    ax1.tick_params(axis='y', labelcolor=color_mse)
    ax1.set_xticks(percents)
    ax1.grid(True, linestyle=':', alpha=0.5)

    # Line 2: MAE on Twin Axis
    ax2 = ax1.twinx()
    ax2.plot(percents, mae_m, color=color_mae, marker='s', linewidth=2.2, markersize=7, linestyle='--', label='Test MAE')
    ax2.fill_between(percents, mae_m - mae_s, mae_m + mae_s, color=color_mae, alpha=0.18)
    ax2.set_ylabel('Test Mean Absolute Error (MAE)', color=color_mae, fontsize=12, fontweight='bold')
    ax2.tick_params(axis='y', labelcolor=color_mae)

    for x, y in zip(percents, mse_m):
        ax1.annotate(f"{y:.4f}", xy=(x, y), xytext=(0, 6), textcoords="offset points", ha='center', fontsize=9.5, fontweight='bold', color=color_mse)

    plt.title('Solar-AL Few-Shot Generalization & Degradation Curve (10% to 50%)', fontsize=13, fontweight='bold', pad=14)
    fig.tight_layout()
    safe_savefig(fig, os.path.join(output_dir, "curve_fewshot_al_decay.png"))
    plt.close(fig)


def plot_depth_heatmap(sensitivity_csv, output_dir):
    if not os.path.exists(sensitivity_csv):
        return

    df = pd.read_csv(sensitivity_csv)
    df = df[(df['status'] == 'SUCCESS') & (df['module'] == 'depth')].copy()
    if len(df) == 0:
        return

    df['test_mse'] = pd.to_numeric(df['test_mse'], errors='coerce')
    pivot = df.groupby(['gpt_layers', 'out_mlp_layers'])['test_mse'].mean().unstack()

    # Reindex to ensure strict 1..4 order
    rows = [1, 2, 3, 4]
    cols = [1, 2, 3, 4]
    pivot = pivot.reindex(index=rows, columns=cols)

    fig, ax = plt.subplots(figsize=(7.5, 6.5), dpi=300)
    draw_annotated_matrix(ax, pivot.values, [f"GPT-{r}" for r in rows], [f"MLP-{c}" for c in cols],
                          cmap="YlGnBu_r", cbar_label="Test MSE (Lower is Better)")

    ax.set_title("Architectural Depth Trade-off: Backbone vs. Predictor Head", fontsize=12.5, fontweight='bold', pad=14)
    ax.set_xlabel("Output MLP Head Layers (out_mlp_layers)", fontsize=11.5, fontweight='bold')
    ax.set_ylabel("Pruned GPT-2 Backbone Layers (gpt_layers)", fontsize=11.5, fontweight='bold')

    fig.tight_layout()
    safe_savefig(fig, os.path.join(output_dir, "heatmap_depth_gpt_vs_mlp_al.png"))
    plt.close(fig)


def plot_capacity_heatmap(sensitivity_csv, output_dir):
    if not os.path.exists(sensitivity_csv):
        return

    df = pd.read_csv(sensitivity_csv)
    df = df[(df['status'] == 'SUCCESS') & (df['module'] == 'capacity')].copy()
    if len(df) == 0:
        return

    df['test_mse'] = pd.to_numeric(df['test_mse'], errors='coerce')
    pivot = df.groupby(['lora_r', 'd_ff'])['test_mse'].mean().unstack()

    rows = [2, 4, 8, 16]
    cols = [768, 1536, 3072, 4096]
    pivot = pivot.reindex(index=rows, columns=cols)

    fig, ax = plt.subplots(figsize=(8.0, 6.5), dpi=300)
    draw_annotated_matrix(ax, pivot.values, [f"r={r}" for r in rows], [f"{c}" for c in cols],
                          cmap="magma_r", cbar_label="Test MSE (Lower is Better)")

    ax.set_title("Model Parameter Capacity Frontier: LoRA Rank vs. FFN Width", fontsize=12.5, fontweight='bold', pad=14)
    ax.set_xlabel("Feed-Forward Network Hidden Dimension (d_ff)", fontsize=11.5, fontweight='bold')
    ax.set_ylabel("LoRA Adaptation Rank (r)", fontsize=11.5, fontweight='bold')

    fig.tight_layout()
    safe_savefig(fig, os.path.join(output_dir, "heatmap_capacity_lora_vs_dff_al.png"))
    plt.close(fig)


def plot_distillation_simplex(sensitivity_csv, output_dir):
    if not os.path.exists(sensitivity_csv):
        return

    df = pd.read_csv(sensitivity_csv)
    df = df[(df['status'] == 'SUCCESS') & (df['module'] == 'distillation')].copy()
    if len(df) == 0:
        return

    df['test_mse'] = pd.to_numeric(df['test_mse'], errors='coerce')
    grouped = df.groupby(['task_w', 'feature_w', 'output_w']).agg(
        mse_mean=('test_mse', 'mean'),
        mse_std=('test_mse', 'std')
    ).reset_index()

    grouped['config_label'] = grouped.apply(
        lambda r: f"Task:{r['task_w']:.2f}\nFeat:{r['feature_w']:.2f}\nOut:{r['output_w']:.2f}", axis=1
    )
    grouped = grouped.sort_values(by='mse_mean', ascending=True).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(11, 6), dpi=300)
    x = np.arange(len(grouped))
    colors = plt.cm.viridis(np.linspace(0.2, 0.85, len(grouped)))

    bars = ax.bar(x, grouped['mse_mean'], yerr=grouped['mse_std'].fillna(0), capsize=4,
                  color=colors, edgecolor='black', linewidth=1.1, alpha=0.9, width=0.6)

    ax.set_xticks(x)
    ax.set_xticklabels(grouped['config_label'], rotation=35, ha='right', fontsize=9)
    ax.set_ylabel("Test Mean Squared Error (MSE)", fontsize=11.5, fontweight='bold')
    ax.set_title("Multi-Task Distillation Loss Balance on Ternary Simplex", fontsize=13, fontweight='bold', pad=14)
    ax.grid(axis='y', linestyle=':', alpha=0.6)

    min_val = grouped['mse_mean'].min()
    ax.set_ylim(min_val * 0.95, grouped['mse_mean'].max() * 1.05)

    for bar, val in zip(bars, grouped['mse_mean']):
        ax.annotate(f"{val:.4f}", xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(0, 4), textcoords="offset points", ha='center', fontsize=8.5, fontweight='bold')

    fig.tight_layout()
    safe_savefig(fig, os.path.join(output_dir, "heatmap_distillation_simplex_al.png"))
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Generate Visualizations for 4-Day Sensitivity & Few-Shot Suites")
    parser.add_argument("--fewshot_csv", type=str, default="./results/results_fewshot_al.csv")
    parser.add_argument("--sensitivity_csv", type=str, default="./results/results_hyper_sensitivity_al.csv")
    parser.add_argument("--output_dir", type=str, default="./figures/sensitivity")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    print("=" * 80)
    print(">>> [PLOT] Rendering Publication Figures for Benchmark Suites...")
    print("=" * 80)

    plot_fewshot_curve(args.fewshot_csv, args.output_dir)
    plot_depth_heatmap(args.sensitivity_csv, args.output_dir)
    plot_capacity_heatmap(args.sensitivity_csv, args.output_dir)
    plot_distillation_simplex(args.sensitivity_csv, args.output_dir)

    print("=" * 80)
    print(f">>> [DONE] All Available Figures Processed for: {args.output_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
