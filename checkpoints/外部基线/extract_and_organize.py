import shutil
import os
import pandas as pd
import openpyxl

base_dir = os.path.dirname(os.path.abspath(__file__))
raw_dir = os.path.join(base_dir, "raw")
data_dir = os.path.join(base_dir, "data")
os.makedirs(raw_dir, exist_ok=True)
os.makedirs(data_dir, exist_ok=True)

# 1. Copy files
copies = [
    (r"D:\github\CycleNet\HP_SEARCH_SUMMARY.md", os.path.join(raw_dir, "HP_SEARCH_SUMMARY.md")),
    (r"D:\github\CycleNet\fewshot_compare_results.xlsx", os.path.join(raw_dir, "fewshot_compare_results.xlsx")),
    (r"D:\github\CycleNet\fewshot_IL_results.xlsx", os.path.join(raw_dir, "fewshot_IL_results.xlsx")),
    (r"D:\github\CycleNet\tuning\results.csv", os.path.join(raw_dir, "CycleNet_linear_AL_results.csv")),
    (r"D:\github\CycleNet\tuning_IL\results.csv", os.path.join(raw_dir, "CycleNet_linear_IL_results.csv")),
    (r"D:\github\CycleNet\tuning_AL_mlp\results.csv", os.path.join(raw_dir, "CycleNet_mlp_AL_results.csv")),
    (r"D:\github\CycleNet\tuning_IL_mlp\results.csv", os.path.join(raw_dir, "CycleNet_mlp_IL_results.csv")),
    (r"D:\github\TQNet\tuning\results.csv", os.path.join(raw_dir, "TQNet_AL_results.csv")),
    (r"D:\github\TQNet\tuning_IL\results.csv", os.path.join(raw_dir, "TQNet_IL_results.csv")),
]

for src, dst in copies:
    if os.path.exists(src):
        shutil.copy2(src, dst)
        print(f"Copied {src} -> {dst}")
    else:
        print(f"Warning: {src} not found!")

# 2. Extract and format few-shot cross-model benchmark data
wb = openpyxl.load_workbook(os.path.join(raw_dir, "fewshot_compare_results.xlsx"))
ws_models = wb.worksheets[0]  # 各模型总表
model_rows = list(ws_models.iter_rows(values_only=True))

data = []
for r in model_rows[4:]:
    if r[0] in ['CALF', 'TQNet', 'CycleNet']:
        # Normalize model name for CycleNet to 'CycleNet-mlp'
        mname = 'CycleNet-mlp' if r[0] == 'CycleNet' else r[0]
        row_data = [mname] + list(r[1:12])
        data.append(row_data)

df_fewshot = pd.DataFrame(data, columns=['model', 'percent', 'lr', 'batch_size', 'mean_mse', 'sd', 'range', 'seed2025', 'seed2026', 'seed2027', 'mean_mae', 'degradation_pct'])
df_fewshot.to_csv(os.path.join(data_dir, "fewshot_cross_model_details.csv"), index=False, encoding='utf-8-sig')
print(f"Saved {os.path.join(data_dir, 'fewshot_cross_model_details.csv')}")

# Extract summary table (CALF, TQNet, CycleNet-mlp side by side)
ws_summary = wb.worksheets[1]  # 横向对比
summary_rows = list(ws_summary.iter_rows(values_only=True))
comp_data = []
for r in summary_rows[3:9]:
    comp_data.append([r[0], r[1], r[2], r[3], r[5]])

df_comp = pd.DataFrame(comp_data, columns=['percent', 'CALF_MSE', 'TQNet_MSE', 'CycleNet_MSE', 'TQNet_vs_CALF_adv_pct'])
df_comp['CycleNet_vs_CALF_adv_pct'] = (df_comp['CycleNet_MSE'] - df_comp['CALF_MSE']) / df_comp['CALF_MSE'] * 100

deg_map = {r['model']: {} for _, r in df_fewshot.iterrows()}
for _, r in df_fewshot.iterrows():
    deg_map[r['model']][r['percent']] = r['degradation_pct']

df_comp['CALF_deg_pct'] = df_comp['percent'].map(deg_map['CALF'])
df_comp['TQNet_deg_pct'] = df_comp['percent'].map(deg_map['TQNet'])
df_comp['CycleNet_deg_pct'] = df_comp['percent'].map(deg_map['CycleNet-mlp'])

df_comp.to_csv(os.path.join(data_dir, "fewshot_cross_model_summary.csv"), index=False, encoding='utf-8-sig')
print(f"Saved {os.path.join(data_dir, 'fewshot_cross_model_summary.csv')}")

# 3. Extract HP Search Phases Summary
hp_phases = [
    {
        "phase": 1, "model": "TQNet", "type": "standard", "dataset": "Solar-AL", "n_stations": 137,
        "search_groups": 17, "total_runs": 52,
        "base_lr": 0.003, "base_bs": 64, "base_mse": 0.176988,
        "best_lr": 0.005, "best_bs": 64, "best_mse": 0.175281,
        "improvement_pct": -0.96, "seed_range": 0.005324
    },
    {
        "phase": 2, "model": "TQNet", "type": "standard", "dataset": "Solar-IL", "n_stations": 106,
        "search_groups": 32, "total_runs": 97,
        "base_lr": 0.003, "base_bs": 64, "base_mse": 0.187648,
        "best_lr": 0.0006, "best_bs": 64, "best_mse": 0.177982,
        "improvement_pct": -5.15, "seed_range": 0.002467
    },
    {
        "phase": 3, "model": "CycleNet", "type": "linear", "dataset": "Solar-AL", "n_stations": 137,
        "search_groups": 15, "total_runs": 45,
        "base_lr": 0.01, "base_bs": 64, "base_mse": 0.209537,
        "best_lr": 0.0045, "best_bs": 64, "best_mse": 0.207652,
        "improvement_pct": -0.90, "seed_range": 0.001414
    },
    {
        "phase": 4, "model": "CycleNet", "type": "linear", "dataset": "Solar-IL", "n_stations": 106,
        "search_groups": 20, "total_runs": 65,
        "base_lr": 0.01, "base_bs": 64, "base_mse": 0.253454,
        "best_lr": 0.00138, "best_bs": 64, "best_mse": 0.220972,
        "improvement_pct": -12.82, "seed_range": 0.001341
    },
    {
        "phase": 5, "model": "CycleNet", "type": "mlp", "dataset": "Solar-AL", "n_stations": 137,
        "search_groups": 42, "total_runs": 126,
        "base_lr": 0.01, "base_bs": 64, "base_mse": 0.193624,
        "best_lr": 0.0054, "best_bs": 256, "best_mse": 0.180680,
        "improvement_pct": -6.69, "seed_range": 0.002440
    },
    {
        "phase": 6, "model": "CycleNet", "type": "mlp", "dataset": "Solar-IL", "n_stations": 106,
        "search_groups": 23, "total_runs": 69,
        "base_lr": 0.01, "base_bs": 64, "base_mse": 0.193588,
        "best_lr": 0.0025, "best_bs": 128, "best_mse": 0.193503,
        "improvement_pct": -0.04, "seed_range": 0.004722
    }
]
df_hp = pd.DataFrame(hp_phases)
df_hp.to_csv(os.path.join(data_dir, "hp_tuning_phases_summary.csv"), index=False, encoding='utf-8-sig')
print(f"Saved {os.path.join(data_dir, 'hp_tuning_phases_summary.csv')}")

# 4. Consolidate detailed sweeps from each tuning csv
all_sweeps = []
sweep_files = [
    ("CycleNet_linear_AL_results.csv", "CycleNet", "linear", "Solar-AL"),
    ("CycleNet_linear_IL_results.csv", "CycleNet", "linear", "Solar-IL"),
    ("CycleNet_mlp_AL_results.csv", "CycleNet", "mlp", "Solar-AL"),
    ("CycleNet_mlp_IL_results.csv", "CycleNet", "mlp", "Solar-IL"),
    ("TQNet_AL_results.csv", "TQNet", "standard", "Solar-AL"),
    ("TQNet_IL_results.csv", "TQNet", "standard", "Solar-IL")
]

for fname, model, mtype, dataset in sweep_files:
    fpath = os.path.join(raw_dir, fname)
    if os.path.exists(fpath):
        df_sub = pd.read_csv(fpath)
        df_sub.columns = [c.strip() for c in df_sub.columns]
        df_sub['model'] = model
        df_sub['model_type'] = mtype
        df_sub['dataset'] = dataset
        all_sweeps.append(df_sub)

if all_sweeps:
    df_all_sweeps = pd.concat(all_sweeps, ignore_index=True)
    df_all_sweeps.to_csv(os.path.join(data_dir, "hp_tuning_all_sweeps.csv"), index=False, encoding='utf-8-sig')
    print(f"Saved {os.path.join(data_dir, 'hp_tuning_all_sweeps.csv')} with {len(df_all_sweeps)} rows")
