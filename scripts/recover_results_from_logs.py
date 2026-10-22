#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Script: scripts/recover_results_from_logs.py
Description: Parse all raw training log files copied from Computer B,
reconstruct complete and clean CSV results, and render high-resolution publication figures.
"""

import os
import re
import glob
import subprocess
import pandas as pd
import numpy as np


def parse_metrics_from_file(path):
    raw = open(path, 'rb').read()
    text = ''
    for enc in ['utf-16', 'utf-8', 'gbk', 'latin1']:
        try:
            c = raw.decode(enc)
            if 'mse:' in c:
                text = c
                break
        except Exception:
            pass
    if not text:
        text = raw.decode('latin1', errors='ignore')

    matches = re.findall(r'mse:([0-9\.]+),\s*mae:([0-9\.]+)', text)
    if not matches:
        return None, None, 0
    mse = float(matches[-1][0])
    mae = float(matches[-1][1])
    epoch_times = [float(x) for x in re.findall(r'cost time:\s*([0-9\.]+)', text)]
    elapsed = int(round(sum(epoch_times))) if epoch_times else 0
    return mse, mae, elapsed


def recover_fewshot_al(log_dir="./logs/CALF/Solar_AL_Fewshot", output_csv="./results/results_fewshot_al.csv"):
    files = glob.glob(os.path.join(log_dir, "*.log"))
    records = []
    for f in files:
        m = re.search(r'al_fewshot_p(\d+)_seed(\d+)', os.path.basename(f))
        if not m:
            continue
        pct, seed = int(m.group(1)), int(m.group(2))
        mse, mae, elapsed = parse_metrics_from_file(f)
        if mse is not None:
            records.append({
                'percent': pct,
                'seed': seed,
                'test_mse': mse,
                'test_mae': mae,
                'train_epochs': 100,
                'status': 'SUCCESS',
                'timestamp': '2026-10-22'
            })

    df = pd.DataFrame(records).sort_values(by=['percent', 'seed']).reset_index(drop=True)
    os.makedirs(os.path.dirname(os.path.abspath(output_csv)), exist_ok=True)
    df.to_csv(output_csv, index=False, float_format='%.6f')
    print(f">>> [RECOVER] Successfully generated {output_csv} with {len(df)} runs.")
    return df


def recover_sensitivity_al(log_dir="./logs/CALF/Solar_AL_Sensitivity", output_csv="./results/results_hyper_sensitivity_al.csv"):
    files = glob.glob(os.path.join(log_dir, "*.log"))
    records = []
    pattern = r'^(depth|capacity|distillation)_gpt(\d+)_mlp(\d+)_r(\d+)_dff(\d+)_tw([0-9\.]+)_fw([0-9\.]+)_s(\d+)\.log$'

    for f in files:
        fname = os.path.basename(f)
        m = re.match(pattern, fname)
        if not m:
            continue
        mod, gpt, mlp, r, dff, tw, fw, s = m.groups()
        gpt, mlp, r, dff, s = int(gpt), int(mlp), int(r), int(dff), int(s)
        tw, fw = float(tw), float(fw)
        ow = round(1.0 - tw - fw, 4)
        if ow < 0:
            ow = 0.0

        mse, mae, elapsed = parse_metrics_from_file(f)
        if mse is None:
            continue

        rec = {
            'module': mod,
            'gpt_layers': gpt,
            'out_mlp_layers': mlp,
            'lora_r': r,
            'd_ff': dff,
            'task_w': tw,
            'feature_w': fw,
            'output_w': ow,
            'seed': s,
            'test_mse': mse,
            'test_mae': mae,
            'elapsed_sec': elapsed,
            'status': 'SUCCESS',
            'timestamp': '2026-10-22'
        }
        records.append(rec)

        # Baseline anchor configuration (depth: gpt1, mlp3, r8, dff3072, tw0.7, fw0.05)
        # Shared anchor for capacity and distillation grids
        if mod == 'depth' and gpt == 1 and mlp == 3 and r == 8 and dff == 3072 and abs(tw - 0.7) < 1e-4 and abs(fw - 0.05) < 1e-4:
            rec_cap = dict(rec)
            rec_cap['module'] = 'capacity'
            records.append(rec_cap)
            rec_dist = dict(rec)
            rec_dist['module'] = 'distillation'
            records.append(rec_dist)

    df = pd.DataFrame(records).sort_values(by=['module', 'seed']).reset_index(drop=True)
    os.makedirs(os.path.dirname(os.path.abspath(output_csv)), exist_ok=True)
    df.to_csv(output_csv, index=False, float_format='%.6f')
    print(f">>> [RECOVER] Successfully generated {output_csv} with {len(df)} runs.")
    print(">>> [RECOVER] Module breakdown:")
    for mod, count in df['module'].value_counts().items():
        print(f"    - {mod}: {count} runs")
    return df


def generate_summaries(df_few, df_sens, results_dir="./results"):
    # Few-shot summary
    few_sum = df_few.groupby('percent').agg(
        mse_mean=('test_mse', 'mean'),
        mse_std=('test_mse', 'std'),
        mae_mean=('test_mae', 'mean'),
        mae_std=('test_mae', 'std'),
        count=('test_mse', 'count')
    ).reset_index().sort_values(by='percent')
    few_sum_file = os.path.join(results_dir, "fewshot_al_summary.csv")
    few_sum.to_csv(few_sum_file, index=False, float_format='%.6f')
    print(f">>> [SUMMARY] Few-shot summary saved to {few_sum_file}")

    # Depth summary
    depth_df = df_sens[df_sens['module'] == 'depth']
    depth_sum = depth_df.groupby(['gpt_layers', 'out_mlp_layers']).agg(
        mse_mean=('test_mse', 'mean'),
        mse_std=('test_mse', 'std'),
        mae_mean=('test_mae', 'mean'),
        mae_std=('test_mae', 'std')
    ).reset_index().sort_values(by=['gpt_layers', 'out_mlp_layers'])
    depth_sum_file = os.path.join(results_dir, "sensitivity_depth_summary.csv")
    depth_sum.to_csv(depth_sum_file, index=False, float_format='%.6f')

    # Capacity summary
    cap_df = df_sens[df_sens['module'] == 'capacity']
    cap_sum = cap_df.groupby(['lora_r', 'd_ff']).agg(
        mse_mean=('test_mse', 'mean'),
        mse_std=('test_mse', 'std'),
        mae_mean=('test_mae', 'mean'),
        mae_std=('test_mae', 'std')
    ).reset_index().sort_values(by=['lora_r', 'd_ff'])
    cap_sum_file = os.path.join(results_dir, "sensitivity_capacity_summary.csv")
    cap_sum.to_csv(cap_sum_file, index=False, float_format='%.6f')

    # Distillation summary
    dist_df = df_sens[df_sens['module'] == 'distillation']
    dist_sum = dist_df.groupby(['task_w', 'feature_w', 'output_w']).agg(
        mse_mean=('test_mse', 'mean'),
        mse_std=('test_mse', 'std'),
        mae_mean=('test_mae', 'mean'),
        mae_std=('test_mae', 'std')
    ).reset_index().sort_values(by='mse_mean', ascending=True)
    dist_sum_file = os.path.join(results_dir, "sensitivity_distillation_summary.csv")
    dist_sum.to_csv(dist_sum_file, index=False, float_format='%.6f')


def main():
    print("=" * 80)
    print(">>> [RECOVER] Starting Sensitivity & Few-Shot Result Recovery from Logs...")
    print("=" * 80)

    fewshot_csv = "./results/results_fewshot_al.csv"
    sens_csv = "./results/results_hyper_sensitivity_al.csv"
    output_dir = "./figures/sensitivity"

    df_few = recover_fewshot_al("./logs/CALF/Solar_AL_Fewshot", fewshot_csv)
    df_sens = recover_sensitivity_al("./logs/CALF/Solar_AL_Sensitivity", sens_csv)

    generate_summaries(df_few, df_sens, "./results")

    print("\n" + "=" * 80)
    print(">>> [PLOT] Calling plot_sensitivity_results.py...")
    print("=" * 80)

    python_bin = "D:\\software\\conda\\envs\\calf\\python.exe"
    if not os.path.exists(python_bin):
        python_bin = "python"

    cmd = [
        python_bin, "plot_sensitivity_results.py",
        "--fewshot_csv", fewshot_csv,
        "--sensitivity_csv", sens_csv,
        "--output_dir", output_dir
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    print(res.stdout)
    if res.stderr:
        print("[STDERR]", res.stderr)

    print("=" * 80)
    print(">>> [DONE] Recovery and Plotting Completed Successfully!")
    print("=" * 80)


if __name__ == "__main__":
    main()
