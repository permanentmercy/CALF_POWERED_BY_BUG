import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Set visual style
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.autolayout'] = False

base_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(base_dir, "data")
fig_dir = os.path.join(base_dir, "figures")
os.makedirs(fig_dir, exist_ok=True)

# Also copy to artifacts dir for visual review
artifact_fig_dir = r"C:\Users\sheng\.gemini\antigravity\brain\d006b4c2-088d-4852-a084-081cb9ace8e4\figures"
os.makedirs(artifact_fig_dir, exist_ok=True)

# 1. Load data
df_fewshot_sum = pd.read_csv(os.path.join(data_dir, "fewshot_cross_model_summary.csv"))
df_hp_phases = pd.read_csv(os.path.join(data_dir, "hp_tuning_phases_summary.csv"))
df_sweeps = pd.read_csv(os.path.join(data_dir, "hp_tuning_all_sweeps.csv"))

colors = {
    'CALF': '#1f77b4',       # Deep Blue
    'TQNet': '#d62728',      # Vivid Red
    'CycleNet': '#2ca02c',   # Forest Green
    'CycleNet-mlp': '#2ca02c',
    'Baseline': '#7f7f7f',   # Gray
    'Optimal': '#1f77b4',    # Blue
    'Solar-AL': '#ff7f0e',   # Orange
    'Solar-IL': '#9467bd'    # Purple
}

# ==============================================================================
# Figure 1: Few-Shot Cross-Model MSE Curves (10% to 100%)
# ==============================================================================
fig, ax = plt.subplots(figsize=(8.5, 5.5), dpi=300)

x = df_fewshot_sum['percent'].values
y_calf = df_fewshot_sum['CALF_MSE'].values
y_tq = df_fewshot_sum['TQNet_MSE'].values
y_cycle = df_fewshot_sum['CycleNet_MSE'].values

# Pure circle markers, NO error bar caps, NO floating text blocking curves
ax.plot(x, y_calf, marker='o', markersize=8, linewidth=2.5, color=colors['CALF'], label='CALF (Ours, Pretrained LLM)')
ax.plot(x, y_tq, marker='o', markersize=8, linewidth=2.2, linestyle='--', color=colors['TQNet'], label='TQNet (SOTA Time Series)')
ax.plot(x, y_cycle, marker='o', markersize=8, linewidth=2.2, linestyle='-.', color=colors['CycleNet'], label='CycleNet-mlp (Periodic Baseline)')

ax.set_xlabel('Training Data Ratio (%)', fontsize=12, fontweight='bold')
ax.set_ylabel('Test MSE (Lower is Better)', fontsize=12, fontweight='bold')
ax.set_title('Cross-Model Few-Shot Benchmark on Solar-IL (10% to 100%)', fontsize=14, fontweight='bold', pad=12)
ax.set_xticks([10, 20, 30, 40, 50, 100])
ax.set_xticklabels(['10%', '20%', '30%', '40%', '50%', '100%'], fontsize=11)
ax.grid(True, linestyle='--', alpha=0.5)
ax.set_ylim(0.15, 0.39)

# Clean legend in upper right
ax.legend(loc='upper right', frameon=True, framealpha=0.95, edgecolor='#cccccc', fontsize=11)

plt.tight_layout()
fig1_path = os.path.join(fig_dir, "fewshot_cross_model_mse_curves.png")
fig.savefig(fig1_path, dpi=300)
fig.savefig(os.path.join(fig_dir, "fewshot_cross_model_mse_curves.pdf"))
fig.savefig(os.path.join(artifact_fig_dir, "fewshot_cross_model_mse_curves.png"), dpi=300)
plt.close(fig)
print("Saved Fig 1: fewshot_cross_model_mse_curves")

# ==============================================================================
# Figure 2: Few-Shot Relative Degradation Bar Chart vs 100% Baseline
# ==============================================================================
fig, ax = plt.subplots(figsize=(9, 5.5), dpi=300)

subset = df_fewshot_sum[df_fewshot_sum['percent'] <= 50]
x_sub = np.arange(len(subset))
width = 0.25

rects1 = ax.bar(x_sub - width, subset['CALF_deg_pct'], width, label='CALF (Ours)', color=colors['CALF'], alpha=0.9, edgecolor='black', linewidth=0.8)
rects2 = ax.bar(x_sub, subset['TQNet_deg_pct'], width, label='TQNet', color=colors['TQNet'], alpha=0.9, edgecolor='black', linewidth=0.8)
rects3 = ax.bar(x_sub + width, subset['CycleNet_deg_pct'], width, label='CycleNet-mlp', color=colors['CycleNet'], alpha=0.9, edgecolor='black', linewidth=0.8)

# Value labels on top of bars
def autolabel(rects):
    for rect in rects:
        h = rect.get_height()
        ax.annotate(f'+{h:.1f}%',
                    xy=(rect.get_x() + rect.get_width() / 2, h),
                    xytext=(0, 4), textcoords="offset points",
                    ha='center', va='bottom', fontsize=9, fontweight='bold')

autolabel(rects1)
autolabel(rects2)
autolabel(rects3)

ax.set_xlabel('Training Data Ratio (%)', fontsize=12, fontweight='bold')
ax.set_ylabel('Degradation vs Own 100% Baseline (%)', fontsize=12, fontweight='bold')
ax.set_title('Performance Degradation Under Extreme Low-Data Regimes (vs 100% Baseline)', fontsize=13, fontweight='bold', pad=12)
ax.set_xticks(x_sub)
ax.set_xticklabels(['10%', '20%', '30%', '40%', '50%'], fontsize=11)
ax.grid(True, axis='y', linestyle='--', alpha=0.5)
ax.set_ylim(0, 120)
ax.legend(loc='upper right', frameon=True, framealpha=0.95, edgecolor='#cccccc', fontsize=11)

plt.tight_layout()
fig2_path = os.path.join(fig_dir, "fewshot_cross_model_degradation_bar.png")
fig.savefig(fig2_path, dpi=300)
fig.savefig(os.path.join(fig_dir, "fewshot_cross_model_degradation_bar.pdf"))
fig.savefig(os.path.join(artifact_fig_dir, "fewshot_cross_model_degradation_bar.png"), dpi=300)
plt.close(fig)
print("Saved Fig 2: fewshot_cross_model_degradation_bar")

# ==============================================================================
# Figure 3: CALF Advantage Margin Over Baseline Models
# ==============================================================================
fig, ax = plt.subplots(figsize=(9, 5.5), dpi=300)

x_all = np.arange(len(df_fewshot_sum))
width = 0.35

rects_tq = ax.bar(x_all - width/2, df_fewshot_sum['TQNet_vs_CALF_adv_pct'], width,
                  label='CALF Advantage over TQNet', color='#2b5c8f', alpha=0.9, edgecolor='black', linewidth=0.8)
rects_cyc = ax.bar(x_all + width/2, df_fewshot_sum['CycleNet_vs_CALF_adv_pct'], width,
                   label='CALF Advantage over CycleNet-mlp', color='#439775', alpha=0.9, edgecolor='black', linewidth=0.8)

def autolabel_adv(rects):
    for rect in rects:
        h = rect.get_height()
        ax.annotate(f'+{h:.1f}%',
                    xy=(rect.get_x() + rect.get_width() / 2, h),
                    xytext=(0, 4), textcoords="offset points",
                    ha='center', va='bottom', fontsize=9, fontweight='bold')

autolabel_adv(rects_tq)
autolabel_adv(rects_cyc)

ax.set_xlabel('Training Data Ratio (%)', fontsize=12, fontweight='bold')
ax.set_ylabel('CALF Relative Advantage Margin (%)', fontsize=12, fontweight='bold')
ax.set_title('CALF Superiority Over Baselines Across Training Ratios (Solar-IL)', fontsize=13, fontweight='bold', pad=12)
ax.set_xticks(x_all)
ax.set_xticklabels(['10%', '20%', '30%', '40%', '50%', '100%'], fontsize=11)
ax.grid(True, axis='y', linestyle='--', alpha=0.5)
ax.set_ylim(0, 92)
ax.legend(loc='upper right', frameon=True, framealpha=0.95, edgecolor='#cccccc', fontsize=11)

plt.tight_layout()
fig3_path = os.path.join(fig_dir, "fewshot_calf_advantage_bar.png")
fig.savefig(fig3_path, dpi=300)
fig.savefig(os.path.join(fig_dir, "fewshot_calf_advantage_bar.pdf"))
fig.savefig(os.path.join(artifact_fig_dir, "fewshot_calf_advantage_bar.png"), dpi=300)
plt.close(fig)
print("Saved Fig 3: fewshot_calf_advantage_bar")

# ==============================================================================
# Figure 4: HP Tuning Phases Baseline vs. Optimal MSE Comparison
# ==============================================================================
fig, ax = plt.subplots(figsize=(10, 5.8), dpi=300)

x_p = np.arange(len(df_hp_phases))
width = 0.35

phase_labels = [
    f"Phase 1\nTQNet (AL)",
    f"Phase 2\nTQNet (IL)",
    f"Phase 3\nCycle-Linear (AL)",
    f"Phase 4\nCycle-Linear (IL)",
    f"Phase 5\nCycle-MLP (AL)",
    f"Phase 6\nCycle-MLP (IL)"
]

rects_base = ax.bar(x_p - width/2, df_hp_phases['base_mse'], width, label='Baseline Config', color='#8c96c6', edgecolor='black', linewidth=0.8)
rects_opt = ax.bar(x_p + width/2, df_hp_phases['best_mse'], width, label='Optimal Config', color='#810f7c', edgecolor='black', linewidth=0.8)

for i, r in df_hp_phases.iterrows():
    h_opt = r['best_mse']
    imp = r['improvement_pct']
    ax.annotate(f"{imp:.2f}%\n({r['best_lr']}/{int(r['best_bs'])})",
                xy=(x_p[i] + width/2, h_opt),
                xytext=(0, 4), textcoords="offset points",
                ha='center', va='bottom', fontsize=8.5, fontweight='bold', color='#4d004b')

ax.set_ylabel('Test MSE (Lower is Better)', fontsize=12, fontweight='bold')
ax.set_title('Hyperparameter Tuning Optimization Across 6 Phases (Baseline vs Optimal)', fontsize=13, fontweight='bold', pad=12)
ax.set_xticks(x_p)
ax.set_xticklabels(phase_labels, fontsize=10)
ax.grid(True, axis='y', linestyle='--', alpha=0.5)
ax.set_ylim(0.14, 0.28)
ax.legend(loc='upper right', frameon=True, framealpha=0.95, edgecolor='#cccccc', fontsize=11)

plt.tight_layout()
fig4_path = os.path.join(fig_dir, "hp_tuning_phases_comparison.png")
fig.savefig(fig4_path, dpi=300)
fig.savefig(os.path.join(fig_dir, "hp_tuning_phases_comparison.pdf"))
fig.savefig(os.path.join(artifact_fig_dir, "hp_tuning_phases_comparison.png"), dpi=300)
plt.close(fig)
print("Saved Fig 4: hp_tuning_phases_comparison")

# ==============================================================================
# Figure 5: Learning Rate Sensitivity & Dataset Distribution Shift (AL vs IL)
# ==============================================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.2), dpi=300, sharey=False)

# Group sweeps by model, model_type, dataset, bs, lr to get mean_mse
df_sweeps_grp = df_sweeps.groupby(['model', 'model_type', 'dataset', 'bs', 'lr'], as_index=False)['mse'].mean().rename(
    columns={'mse': 'mean_mse', 'lr': 'learning_rate', 'bs': 'batch_size'}
)

# Filter for standard bs=64 sweeps
sweeps_64 = df_sweeps_grp[df_sweeps_grp['batch_size'] == 64].copy()

# Subplot 1: Solar-AL
al_sweeps = sweeps_64[sweeps_64['dataset'] == 'Solar-AL']
for (m, mt), grp in al_sweeps.groupby(['model', 'model_type']):
    grp_sorted = grp.sort_values('learning_rate')
    lbl = f"{m}-{mt}" if mt != 'standard' else m
    ax1.plot(grp_sorted['learning_rate'], grp_sorted['mean_mse'], marker='o', markersize=6, linewidth=1.8, label=lbl)

ax1.set_xscale('log')
ax1.set_xlabel('Learning Rate (log scale)', fontsize=11, fontweight='bold')
ax1.set_ylabel('Test MSE (bs=64)', fontsize=11, fontweight='bold')
ax1.set_title('Solar-AL Dataset (Optimal lr ~ 3e-3 - 5e-3)', fontsize=12, fontweight='bold')
ax1.grid(True, linestyle='--', alpha=0.5)
ax1.legend(loc='best', frameon=True, framealpha=0.9, fontsize=9.5)

# Subplot 2: Solar-IL
il_sweeps = sweeps_64[sweeps_64['dataset'] == 'Solar-IL']
for (m, mt), grp in il_sweeps.groupby(['model', 'model_type']):
    grp_sorted = grp.sort_values('learning_rate')
    lbl = f"{m}-{mt}" if mt != 'standard' else m
    ax2.plot(grp_sorted['learning_rate'], grp_sorted['mean_mse'], marker='o', markersize=6, linewidth=1.8, label=lbl)

ax2.set_xscale('log')
ax2.set_xlabel('Learning Rate (log scale)', fontsize=11, fontweight='bold')
ax2.set_ylabel('Test MSE (bs=64)', fontsize=11, fontweight='bold')
ax2.set_title('Solar-IL Dataset (Optimal lr ~ 6e-4 - 1.4e-3)', fontsize=12, fontweight='bold')
ax2.grid(True, linestyle='--', alpha=0.5)
ax2.legend(loc='best', frameon=True, framealpha=0.9, fontsize=9.5)

fig.suptitle('Learning Rate Sensitivity & Dataset Distribution Shift (bs=64)', fontsize=14, fontweight='bold', y=0.98)
plt.tight_layout()
fig5_path = os.path.join(fig_dir, "hp_tuning_lr_sensitivity.png")
fig.savefig(fig5_path, dpi=300)
fig.savefig(os.path.join(fig_dir, "hp_tuning_lr_sensitivity.pdf"))
fig.savefig(os.path.join(artifact_fig_dir, "hp_tuning_lr_sensitivity.png"), dpi=300)
plt.close(fig)
print("Saved Fig 5: hp_tuning_lr_sensitivity")

# ==============================================================================
# Figure 6: Master Dashboard (2x2 Panel)
# ==============================================================================
fig, axs = plt.subplots(2, 2, figsize=(15, 11), dpi=300)

# (a) Few-Shot Curves
ax = axs[0, 0]
ax.plot(x, y_calf, marker='o', markersize=7, linewidth=2.2, color=colors['CALF'], label='CALF (Ours)')
ax.plot(x, y_tq, marker='o', markersize=7, linewidth=2.0, linestyle='--', color=colors['TQNet'], label='TQNet')
ax.plot(x, y_cycle, marker='o', markersize=7, linewidth=2.0, linestyle='-.', color=colors['CycleNet'], label='CycleNet-mlp')
ax.set_xlabel('Training Data Ratio (%)', fontsize=11, fontweight='bold')
ax.set_ylabel('Test MSE', fontsize=11, fontweight='bold')
ax.set_title('(a) Few-Shot Test MSE Across Data Ratios', fontsize=12, fontweight='bold')
ax.set_xticks([10, 20, 30, 40, 50, 100])
ax.set_xticklabels(['10%', '20%', '30%', '40%', '50%', '100%'])
ax.grid(True, linestyle='--', alpha=0.5)
ax.set_ylim(0.15, 0.39)
ax.legend(loc='upper right', frameon=True, framealpha=0.9, fontsize=10)

# (b) Degradation Bar Chart
ax = axs[0, 1]
rects1 = ax.bar(x_sub - width*0.8, subset['CALF_deg_pct'], width*0.8, label='CALF', color=colors['CALF'], alpha=0.9, edgecolor='black')
rects2 = ax.bar(x_sub, subset['TQNet_deg_pct'], width*0.8, label='TQNet', color=colors['TQNet'], alpha=0.9, edgecolor='black')
rects3 = ax.bar(x_sub + width*0.8, subset['CycleNet_deg_pct'], width*0.8, label='CycleNet-mlp', color=colors['CycleNet'], alpha=0.9, edgecolor='black')
for rects in [rects1, rects2, rects3]:
    for rect in rects:
        h = rect.get_height()
        ax.annotate(f'+{h:.1f}%', xy=(rect.get_x() + rect.get_width()/2, h), xytext=(0, 2),
                    textcoords="offset points", ha='center', va='bottom', fontsize=7.5, fontweight='bold')
ax.set_xlabel('Training Data Ratio (%)', fontsize=11, fontweight='bold')
ax.set_ylabel('Degradation vs 100% Baseline (%)', fontsize=11, fontweight='bold')
ax.set_title('(b) Performance Degradation in Low-Data Regimes', fontsize=12, fontweight='bold')
ax.set_xticks(x_sub)
ax.set_xticklabels(['10%', '20%', '30%', '40%', '50%'])
ax.grid(True, axis='y', linestyle='--', alpha=0.5)
ax.set_ylim(0, 120)
ax.legend(loc='upper right', frameon=True, framealpha=0.9, fontsize=10)

# (c) CALF Advantage Margin
ax = axs[1, 0]
rects_tq = ax.bar(x_all - width*0.6, df_fewshot_sum['TQNet_vs_CALF_adv_pct'], width*0.6,
                  label='Advantage over TQNet', color='#2b5c8f', alpha=0.9, edgecolor='black')
rects_cyc = ax.bar(x_all + width*0.6, df_fewshot_sum['CycleNet_vs_CALF_adv_pct'], width*0.6,
                   label='Advantage over CycleNet-mlp', color='#439775', alpha=0.9, edgecolor='black')
for rects in [rects_tq, rects_cyc]:
    for rect in rects:
        h = rect.get_height()
        ax.annotate(f'+{h:.1f}%', xy=(rect.get_x() + rect.get_width()/2, h), xytext=(0, 2),
                    textcoords="offset points", ha='center', va='bottom', fontsize=7.5, fontweight='bold')
ax.set_xlabel('Training Data Ratio (%)', fontsize=11, fontweight='bold')
ax.set_ylabel('CALF Advantage Margin (%)', fontsize=11, fontweight='bold')
ax.set_title('(c) CALF Superiority Across Training Ratios', fontsize=12, fontweight='bold')
ax.set_xticks(x_all)
ax.set_xticklabels(['10%', '20%', '30%', '40%', '50%', '100%'])
ax.grid(True, axis='y', linestyle='--', alpha=0.5)
ax.set_ylim(0, 92)
ax.legend(loc='upper right', frameon=True, framealpha=0.9, fontsize=10)

# (d) 6-Phase HP Tuning
ax = axs[1, 1]
ax.bar(x_p - width*0.6, df_hp_phases['base_mse'], width*0.6, label='Baseline', color='#8c96c6', edgecolor='black')
rects_opt = ax.bar(x_p + width*0.6, df_hp_phases['best_mse'], width*0.6, label='Optimal', color='#810f7c', edgecolor='black')
for i, r in df_hp_phases.iterrows():
    h_opt = r['best_mse']
    imp = r['improvement_pct']
    ax.annotate(f"{imp:.1f}%\n({r['best_lr']}/{int(r['best_bs'])})",
                xy=(x_p[i] + width*0.6, h_opt), xytext=(0, 2), textcoords="offset points",
                ha='center', va='bottom', fontsize=7, fontweight='bold', color='#4d004b')
ax.set_ylabel('Test MSE', fontsize=11, fontweight='bold')
ax.set_title('(d) HP Tuning Across 6 Phases (Baseline vs Optimal)', fontsize=12, fontweight='bold')
ax.set_xticks(x_p)
ax.set_xticklabels(['P1:TQ-AL', 'P2:TQ-IL', 'P3:CycL-AL', 'P4:CycL-IL', 'P5:CycM-AL', 'P6:CycM-IL'], fontsize=9)
ax.grid(True, axis='y', linestyle='--', alpha=0.5)
ax.set_ylim(0.14, 0.28)
ax.legend(loc='upper right', frameon=True, framealpha=0.9, fontsize=10)

fig.suptitle('External Benchmarking Master Dashboard: CALF vs TQNet & CycleNet', fontsize=16, fontweight='bold', y=0.99)
plt.tight_layout()
fig6_path = os.path.join(fig_dir, "external_benchmarks_master_dashboard.png")
fig.savefig(fig6_path, dpi=300)
fig.savefig(os.path.join(fig_dir, "external_benchmarks_master_dashboard.pdf"))
fig.savefig(os.path.join(artifact_fig_dir, "external_benchmarks_master_dashboard.png"), dpi=300)
plt.close(fig)
print("Saved Fig 6: external_benchmarks_master_dashboard")
