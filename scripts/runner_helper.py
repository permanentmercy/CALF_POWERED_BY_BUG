# -*- coding: utf-8 -*-
"""
Helper script for run_one_click.ps1
Provides robust log metric extraction and idempotent check functions.
"""

import os
import sys
import re
import pandas as pd


def parse_log_metrics(log_path):
    if not os.path.exists(log_path):
        return "NaN,NaN"
    with open(log_path, 'rb') as f:
        raw = f.read()
    text = ""
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
    if matches:
        return f"{matches[-1][0]},{matches[-1][1]}"
    return "NaN,NaN"


def check_fewshot_done(csv_path, percent, seed):
    if not os.path.exists(csv_path):
        return "NO"
    try:
        df = pd.read_csv(csv_path)
        m = df[
            (abs(df['percent'] - float(percent)) < 1e-4) &
            (df['seed'] == int(seed)) &
            (df['status'] == 'SUCCESS') &
            (df['test_mse'].notna())
        ]
        if len(m) > 0 and str(m.iloc[0]['test_mse']).lower() != 'nan':
            return "DONE"
    except Exception:
        pass
    return "NO"


def check_sens_done(csv_path, gpt, mlp, r, dff, tw, fw, ow, seed):
    if not os.path.exists(csv_path):
        return "NO"
    try:
        df = pd.read_csv(csv_path)
        m = df[
            (df['gpt_layers'] == int(gpt)) &
            (df['out_mlp_layers'] == int(mlp)) &
            (df['lora_r'] == int(r)) &
            (df['d_ff'] == int(dff)) &
            (abs(df['task_w'] - float(tw)) < 1e-4) &
            (abs(df['feature_w'] - float(fw)) < 1e-4) &
            (abs(df['output_w'] - float(ow)) < 1e-4) &
            (df['seed'] == int(seed)) &
            (df['status'] == 'SUCCESS') &
            (df['test_mse'].notna())
        ]
        if len(m) > 0 and str(m.iloc[0]['test_mse']).lower() != 'nan':
            return "DONE"
    except Exception:
        pass
    return "NO"


if __name__ == "__main__":
    action = sys.argv[1]
    if action == "parse_log":
        print(parse_log_metrics(sys.argv[2]))
    elif action == "check_fewshot":
        print(check_fewshot_done(sys.argv[2], sys.argv[3], sys.argv[4]))
    elif action == "check_sens":
        print(check_sens_done(
            sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5],
            sys.argv[6], sys.argv[7], sys.argv[8], sys.argv[9], sys.argv[10]
        ))
