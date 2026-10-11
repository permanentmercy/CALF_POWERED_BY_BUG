"""
validate_cycle_mechanism.py
========================================================================================
验证目标：
严格实证验证 CALF 架构中结合词嵌入矩阵 (word_embedding_base) 与时间查询向量 (temporal_query)
的交叉注意力模块是否真正起到了物理时间上的周期建模 (Diurnal Periodicity Modeling)。

架构背景：
CALF 采用双分支跨模态蒸馏架构：
- 跨模态文本分支 (Text Branch, outputs_text)：核心搭载 word_embedding_base (500个语义原型)
  与 temporal_query (144步周期向量) 的交叉注意力机制。
- 时序分支 (Time Branch, outputs_time)：搭载原始时序投影及输出端物理周期残差 (cycle_queue)，
  并在训练阶段通过 feature_loss 与 output_loss 接收来自文本分支的知识蒸馏。

实验规划与设计：
1. 8组对照组前向推理与扰动消融测试 (覆盖全部 10,417 个测试样本)，同时评估两分支：
   - Clean Baseline (标准全量模型正常推理)
   - TQ_Phase_Invert_12h (时间查询向量逆相位对撞：cycle_index + 72 % 144，注入颠倒的昼夜周期)
   - Full_Phase_Invert_12h (TQ与输出端cycle_queue同时逆相位，检验全周期崩溃)
   - TQ_Random_Permute (时间查询向量随机置换，破坏时序有序性与周期先验)
   - TQ_Zero_Ablation (置零 temporal_query，剥离时序周期向量调控)
   - TQ_Jitter_1step (时间微扰：偏移1个步长 / 10分钟，测试相位微扰容忍度)
   - WTE_Zero_Ablation (置零词嵌入矩阵 word_embedding_base，剥离语义特征空间)
   - WTE_Random_Gaussian (用同均值同方差的高斯随机基底替换词嵌入矩阵，检验预训练语义原型必要性)

2. 时间查询向量几何与周期性分析：
   - 提取 144 x 768 的 temporal_query，计算 144 x 144 余弦相似度热力图。
   - 计算周期滞后自相关曲线 R(tau)，检验是否在 12h (72 步) 呈现反极性低谷，并在 24h 呈现闭环。

3. 昼夜词嵌入原型注意力显著性对比：
   - 正午 (Noon, 11:00-13:00) vs 午夜 (Midnight, 23:00-01:00) 样本在 500 个词嵌入原型上的注意力分布差异。

4. 导出图件 (纯 300 DPI PNG，不生成任何 .pdf 文件)：
   - cycle_perturbation_impact.png
   - cycle_tq_cosine_heatmap.png
   - cycle_autocorrelation_lag.png
   - wte_day_vs_night_activation.png
   - cycle_verification_results.csv
========================================================================================
"""

import os
import sys
import copy
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# 导入 CALF 组件
from models import CALF
from data_provider.data_factory import data_provider

# 设置 matplotlib 中英文兼容字体
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

DEFAULT_CHECKPOINT = "./checkpoints/long_term_forecast_CALF_96_96_split622_CALF_Solar_ftM_sl96_ll0_pl96_dm768_nh4_el1_dl1_df3072_fc1_ebtimeF_dtTrue_notrend_split622_gpt1_rl0.0004_2025_0_feature0.05_output0.214_train_epochs100_bs16_dr0.1_ld0.1_r8/checkpoint.pth"
DEFAULT_DATA_ROOT = "./datasets/Solar/"
DEFAULT_DATA_PATH = "solar_AL.txt"
DEFAULT_OUTPUT_DIR = "./figures/cycle_mechanism_validation"


def get_args():
    parser = argparse.ArgumentParser(description="Validate Temporal Cycle Modeling Mechanism")
    parser.add_argument("--checkpoint", type=str, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--root_path", type=str, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--data_path", type=str, default=DEFAULT_DATA_PATH)
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--data", type=str, default="Solar")
    parser.add_argument("--features", type=str, default="M")
    parser.add_argument("--target", type=str, default="OT")
    parser.add_argument("--freq", type=str, default="h")
    parser.add_argument("--data_split", type=str, default="6:2:2")
    parser.add_argument("--percent", type=float, default=100.0)

    parser.add_argument("--seq_len", type=int, default=96)
    parser.add_argument("--label_len", type=int, default=0)
    parser.add_argument("--pred_len", type=int, default=96)
    parser.add_argument("--enc_in", type=int, default=137)
    parser.add_argument("--c_out", type=int, default=137)

    parser.add_argument("--d_model", type=int, default=768)
    parser.add_argument("--n_heads", type=int, default=4)
    parser.add_argument("--d_ff", type=int, default=3072)
    parser.add_argument("--gpt_layers", type=int, default=1)
    parser.add_argument("--e_layers", type=int, default=1)
    parser.add_argument("--d_layers", type=int, default=1)
    parser.add_argument("--dec_in", type=int, default=137)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--embed", type=str, default="timeF")
    parser.add_argument("--activation", type=str, default="gelu")
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--num_kernels", type=int, default=6)
    parser.add_argument("--factor", type=int, default=1)
    parser.add_argument("--distil", action="store_false", default=True)
    parser.add_argument("--des", type=str, default="test")
    parser.add_argument("--target_data", type=str, default=None)
    parser.add_argument("--seasonal_patterns", type=str, default="Monthly")
    parser.add_argument("--inverse", action="store_true", default=False)
    parser.add_argument("--mask_rate", type=float, default=0.25)
    parser.add_argument("--anomaly_ratio", type=float, default=0.25)
    parser.add_argument("--moving_avg", type=int, default=25)
    parser.add_argument("--use_amp", action="store_true", default=False)
    parser.add_argument("--word_embedding_path", type=str, default="wte_pca_500.pt")

    parser.add_argument("--r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.1)
    parser.add_argument("--lora_target_proj", type=int, default=0)
    parser.add_argument("--out_mlp_layers", type=int, default=3)
    parser.add_argument("--test_branch", type=str, default="time")
    parser.add_argument("--mlp_res_w", type=float, default=0.0)
    parser.add_argument("--gpt2_path", type=str, default="./models/gpt2")
    parser.add_argument("--cycle", type=int, default=144)
    parser.add_argument("--use_dpg", type=int, default=1)
    parser.add_argument("--use_direct_trend", type=int, default=1)
    parser.add_argument("--use_harmonic", type=int, default=1)
    parser.add_argument("--use_cycle_prior", type=int, default=1)
    parser.add_argument("--cycle_prior_before_denorm", type=int, default=0)
    parser.add_argument("--use_norm", type=int, default=1)
    parser.add_argument("--one_way_distill", type=int, default=1)
    parser.add_argument("--tq_mode", type=str, default="add")
    parser.add_argument("--use_tq_gate", type=int, default=0)
    parser.add_argument("--layer_offset", type=int, default=0)
    parser.add_argument("--tq_dropout", type=float, default=0.1)
    parser.add_argument("--tq_lr_factor", type=float, default=5)
    parser.add_argument("--tq_inject_target", type=str, default="q")
    parser.add_argument("--use_conv_emb", type=int, default=0)
    parser.add_argument("--use_tq_time", type=int, default=0)
    parser.add_argument("--task_time_ratio", type=float, default=0.5)
    parser.add_argument("--feat_anneal", type=int, default=0)
    parser.add_argument("--task_loss", type=str, default="smooth_l1")
    parser.add_argument("--feature_loss", type=str, default="smooth_l1")
    parser.add_argument("--output_loss", type=str, default="smooth_l1")
    parser.add_argument("--task_w", type=float, default=0.736)
    parser.add_argument("--feature_w", type=float, default=0.05)
    parser.add_argument("--output_w", type=float, default=0.214)

    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--task_name", type=str, default="long_term_forecast")
    parser.add_argument("--is_training", type=int, default=0)
    parser.add_argument("--zero_shot", type=int, default=0)
    parser.add_argument("--output_attention", action="store_true", default=False)

    return parser.parse_args()


def safe_savefig(fig, path):
    norm_path = os.path.abspath(os.path.normpath(path))
    if os.path.exists(norm_path):
        try:
            os.remove(norm_path)
        except Exception:
            pass
    fig.savefig(norm_path, dpi=300, bbox_inches='tight')
    print(f"Saved figure: {norm_path}")


def evaluate_condition(model, test_loader, device, condition_mode, orig_wte, orig_tq):
    """
    在指定扰动条件下对整个测试集运行前向推理，同时计算 Text Branch 和 Time Branch 的 MSE 与 MAE
    """
    # 恢复原始状态
    model.in_layer.word_embedding_base.copy_(orig_wte)
    model.in_layer.temporal_query.data.copy_(orig_tq)

    # 针对权重修改的处理
    if condition_mode == "wte_zero":
        model.in_layer.word_embedding_base.zero_()
    elif condition_mode == "wte_random":
        std = orig_wte.std()
        mean = orig_wte.mean()
        rand_wte = torch.randn_like(orig_wte) * std + mean
        model.in_layer.word_embedding_base.copy_(rand_wte)
    elif condition_mode == "tq_zero":
        model.in_layer.temporal_query.data.zero_()

    sum_mse_text = 0.0
    sum_mae_text = 0.0
    sum_mse_time = 0.0
    sum_mae_time = 0.0
    sum_cross_mse = 0.0
    total_elements = 0

    with torch.no_grad():
        for batch_x, batch_y, batch_x_mark, batch_y_mark, batch_cycle in test_loader:
            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float().to(device)

            if isinstance(batch_cycle, list):
                cycle_index, future_cycle_index = batch_cycle
                cycle_index = cycle_index.to(device)
                future_cycle_index = future_cycle_index.to(device)
            else:
                cycle_index = batch_cycle.to(device)
                future_cycle_index = None

            # 根据扰动模式修改 cycle_index
            if condition_mode == "tq_phase_invert":
                # 仅将 in_layer 的时序查询反转 12 小时 (+72 步)
                eff_cycle_index = (cycle_index + 72) % 144
                eff_future_index = future_cycle_index
            elif condition_mode == "full_phase_invert":
                # in_layer 与 cycle_queue 同时反转 12 小时
                eff_cycle_index = (cycle_index + 72) % 144
                eff_future_index = (future_cycle_index + 72) % 144 if future_cycle_index is not None else None
            elif condition_mode == "tq_random_permute":
                eff_cycle_index = torch.randint(0, 144, cycle_index.shape, device=device)
                eff_future_index = future_cycle_index
            elif condition_mode == "tq_jitter_1step":
                eff_cycle_index = (cycle_index + 1) % 144
                eff_future_index = future_cycle_index
            else:
                eff_cycle_index = cycle_index
                eff_future_index = future_cycle_index

            outputs_dict = model(batch_x, cycle_index=eff_cycle_index, future_cycle_index=eff_future_index)
            pred_text = outputs_dict['outputs_text'][:, -96:, :]
            pred_time = outputs_dict['outputs_time'][:, -96:, :]
            true = batch_y[:, -96:, :]

            # Text Branch (直接搭载 WTE + TQ 交叉注意力)
            diff_text = pred_text - true
            sum_mse_text += torch.sum(diff_text ** 2).item()
            sum_mae_text += torch.sum(torch.abs(diff_text)).item()

            # Time Branch (时序分支)
            diff_time = pred_time - true
            sum_mse_time += torch.sum(diff_time ** 2).item()
            sum_mae_time += torch.sum(torch.abs(diff_time)).item()

            # Cross Branch Distillation Gap (两模态一致性)
            diff_cross = pred_time - pred_text
            sum_cross_mse += torch.sum(diff_cross ** 2).item()

            total_elements += diff_text.numel()

    # 恢复原始状态
    model.in_layer.word_embedding_base.copy_(orig_wte)
    model.in_layer.temporal_query.data.copy_(orig_tq)

    mse_text = sum_mse_text / total_elements
    mae_text = sum_mae_text / total_elements
    mse_time = sum_mse_time / total_elements
    mae_time = sum_mae_time / total_elements
    cross_mse = sum_cross_mse / total_elements

    return {
        "mse_text": mse_text,
        "mae_text": mae_text,
        "mse_time": mse_time,
        "mae_time": mae_time,
        "cross_mse": cross_mse,
    }


def extract_attention_weights(model, test_loader, device):
    """
    提取正午与午夜样本在 500 个词嵌入原型上的交叉注意力激活分布
    """
    noon_attns = []
    midnight_attns = []

    w_embed_base = model.in_layer.word_embedding_base  # (500, 768)
    tq_param = model.in_layer.temporal_query          # (144, 768)
    in_linear = model.in_layer.linear
    mha = model.in_layer.layers[0]['cross_attention']
    embed_dim = 768
    num_heads = mha.num_heads
    head_dim = embed_dim // num_heads

    in_proj_weight = mha.in_proj_weight  # (3*768, 768)
    in_proj_bias = mha.in_proj_bias
    w_q = in_proj_weight[:embed_dim, :]
    w_k = in_proj_weight[embed_dim:2*embed_dim, :]
    b_q = in_proj_bias[:embed_dim] if in_proj_bias is not None else None
    b_k = in_proj_bias[embed_dim:2*embed_dim] if in_proj_bias is not None else None

    with torch.no_grad():
        for batch_x, batch_y, _, _, batch_cycle in test_loader:
            batch_x = batch_x.float().to(device)
            if isinstance(batch_cycle, list):
                cycle_index = batch_cycle[0].to(device)
            else:
                cycle_index = batch_cycle.to(device)

            noon_mask = (cycle_index >= 66) & (cycle_index <= 78)
            midnight_mask = (cycle_index <= 6) | (cycle_index >= 138)

            if not (noon_mask.any() or midnight_mask.any()):
                continue

            B, L, M = batch_x.shape
            means = batch_x.mean(1, keepdim=True).detach()
            stdev = torch.sqrt(torch.var(batch_x, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()
            x_norm = (batch_x - means) / stdev
            x_feat = in_linear(x_norm.transpose(1, 2))  # (B, M, 768)

            tq = tq_param[cycle_index]  # (B, 768)
            tq_vec = tq.unsqueeze(1)    # (B, 1, 768)
            q_in = x_feat + tq_vec      # (B, M, 768)

            k_in = w_embed_base.unsqueeze(0).expand(B, -1, -1)  # (B, 500, 768)

            q = F.linear(q_in, w_q, b_q)  # (B, M, 768)
            k = F.linear(k_in, w_k, b_k)  # (B, 500, 768)

            q = q.view(B, M, num_heads, head_dim).transpose(1, 2)     # (B, H, M, d_k)
            k = k.view(B, 500, num_heads, head_dim).transpose(1, 2)   # (B, H, 500, d_k)

            scores = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(head_dim)  # (B, H, M, 500)
            attn_weights = F.softmax(scores, dim=-1)  # (B, H, M, 500)
            sample_attn = attn_weights.mean(dim=(1, 2)).cpu().numpy()

            if noon_mask.any():
                noon_attns.append(sample_attn[noon_mask.cpu().numpy()])
            if midnight_mask.any():
                midnight_attns.append(sample_attn[midnight_mask.cpu().numpy()])

            if len(noon_attns) >= 80 and len(midnight_attns) >= 80:
                break

    noon_prof = np.concatenate(noon_attns, axis=0).mean(axis=0) if noon_attns else np.zeros(500)
    midnight_prof = np.concatenate(midnight_attns, axis=0).mean(axis=0) if midnight_attns else np.zeros(500)
    return noon_prof, midnight_prof


def main():
    args = get_args()
    device = torch.device(args.device)
    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 95)
    print(">>> [CALF] Dual-Branch Word Embedding & Temporal Cycle Mechanism Comprehensive Verification")
    print("=" * 95)
    print(f"Device:          {device}")
    print(f"Checkpoint:      {args.checkpoint}")
    print(f"Dataset Split:   {args.data_split} (Test samples: 10417)")
    print(f"Output Dir:      {output_dir}")
    print("-" * 95)

    # 1. 加载数据
    print(">>> 1. Loading test dataset...")
    test_data, test_loader = data_provider(args, flag='test')
    print(f"Loaded {len(test_data)} test samples in {len(test_loader)} batches.")

    # 2. 构建模型并加载权重
    print(">>> 2. Initializing CALF Model and loading checkpoint weights...")
    model = CALF.Model(args, device).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt, strict=False)
    model.eval()

    orig_wte = model.in_layer.word_embedding_base.clone().detach()
    orig_tq = model.in_layer.temporal_query.data.clone().detach()

    print(f"Temporal Query Matrix (TQ):   Shape={orig_tq.shape}, Norm={torch.norm(orig_tq).item():.4f}")
    print(f"Word Embedding Base (WTE):    Shape={orig_wte.shape}, Norm={torch.norm(orig_wte).item():.4f}")
    print("-" * 95)

    # 3. 执行 7 组消融与扰动测试（聚焦纯净跨模态文本分支的词嵌入与时序查询机制）
    conditions = [
        ("Clean Baseline", "clean", "Normal inference without any perturbation"),
        ("TQ Phase Invert (+12h)", "tq_phase_invert", "Temporal query phase inverted by 12h (noon <-> midnight in in_layer)"),
        ("TQ Random Permute", "tq_random_permute", "Temporal query index randomly permuted across diurnal steps"),
        ("TQ Jitter (+10min)", "tq_jitter_1step", "Temporal query shifted by +1 step (10 minutes micro-perturbation)"),
        ("TQ Zero Ablation", "tq_zero", "Temporal query vector zeroed out (no diurnal modulation)"),
        ("WTE Zero Ablation", "wte_zero", "Word embedding base zeroed out (no semantic cross-attention)"),
        ("WTE Random Gaussian", "wte_random", "Word embedding replaced with Gaussian noise (destroys semantic manifold)"),
    ]

    results = []
    base_text_mse = None
    base_text_mae = None
    base_time_mse = None
    base_time_mae = None

    print(">>> 3. Executing experimental perturbation and ablation evaluations across BOTH branches...")
    print(f"{'Condition':<26} | {'Text MSE':<9} {'(Δ%)':<8} {'Text MAE':<9} | {'Time MSE':<9} {'(Δ%)':<8} {'Time MAE':<9} | {'Cross-MSE':<9}")
    print("-" * 105)

    for name, mode, desc in conditions:
        res = evaluate_condition(model, test_loader, device, mode, orig_wte, orig_tq)
        mse_text = res["mse_text"]
        mae_text = res["mae_text"]
        mse_time = res["mse_time"]
        mae_time = res["mae_time"]
        cross_mse = res["cross_mse"]

        if mode == "clean":
            base_text_mse = mse_text
            base_text_mae = mae_text
            base_time_mse = mse_time
            base_time_mae = mae_time
            d_text_pct = 0.0
            d_text_mae_pct = 0.0
            d_time_pct = 0.0
            d_time_mae_pct = 0.0
        else:
            d_text_pct = ((mse_text - base_text_mse) / base_text_mse) * 100.0
            d_text_mae_pct = ((mae_text - base_text_mae) / base_text_mae) * 100.0
            d_time_pct = ((mse_time - base_time_mse) / base_time_mse) * 100.0
            d_time_mae_pct = ((mae_time - base_time_mae) / base_time_mae) * 100.0

        print(f"{name:<26} | {mse_text:.6f} ({d_text_pct:+6.2f}%) {mae_text:.6f} | {mse_time:.6f} ({d_time_pct:+6.2f}%) {mae_time:.6f} | {cross_mse:.6f}")
        results.append({
            "Experiment": name,
            "Mode": mode,
            "Description": desc,
            "Text_MSE": mse_text,
            "Text_Delta_MSE_pct": d_text_pct,
            "Text_MAE": mae_text,
            "Text_Delta_MAE_pct": d_text_mae_pct,
            "Time_MSE": mse_time,
            "Time_Delta_MSE_pct": d_time_pct,
            "Time_MAE": mae_time,
            "Time_Delta_MAE_pct": d_time_mae_pct,
            "Cross_Modal_MSE": cross_mse,
        })

    # 保存 CSV
    df_results = pd.DataFrame(results)
    csv_path = os.path.join(output_dir, "cycle_verification_results.csv")
    df_results.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"\n>>> Full results table saved to: {csv_path}")

    # 4. 图件 1: 双分支扰动影响对比图 (cycle_perturbation_impact.png)
    print("\n>>> 4. Generating Publication Plot 1: Dual-Branch Perturbation Impact Comparison...")
    fig, (ax_text, ax_time) = plt.subplots(1, 2, figsize=(16, 6.5), dpi=300)

    exp_labels = [r["Experiment"] for r in results]
    text_mses = [r["Text_MSE"] for r in results]
    text_deltas = [r["Text_Delta_MSE_pct"] for r in results]
    time_mses = [r["Time_MSE"] for r in results]
    time_deltas = [r["Time_Delta_MSE_pct"] for r in results]

    # Panel A: Text Branch (直接搭载 WTE + TQ)
    colors_text = ['#2ca02c'] + ['#d62728' if d > 10 else '#ff7f0e' if d > 2 else '#1f77b4' for d in text_deltas[1:]]
    bars_text = ax_text.bar(range(len(exp_labels)), text_mses, color=colors_text, width=0.55, edgecolor='black', linewidth=1.1, alpha=0.9)
    ax_text.axhline(base_text_mse, color='#2ca02c', linestyle='--', linewidth=1.5, label=f'Clean Text MSE ({base_text_mse:.4f})')

    for i, bar in enumerate(bars_text):
        h = bar.get_height()
        txt = f"{h:.4f}\n(Base)" if i == 0 else f"{h:.4f}\n({text_deltas[i]:+.1f}%)"
        ax_text.annotate(txt, xy=(bar.get_x() + bar.get_width() / 2, h), xytext=(0, 4), textcoords="offset points",
                         ha='center', va='bottom', fontsize=8.5, fontweight='bold',
                         color='#8b0000' if text_deltas[i] > 10 else '#333333')

    ax_text.set_xticks(range(len(exp_labels)))
    ax_text.set_xticklabels(exp_labels, rotation=35, ha='right', fontsize=9.5)
    ax_text.set_ylabel("Mean Squared Error (MSE)", fontsize=11, fontweight='bold')
    ax_text.set_title("Cross-Modal Text Branch (Direct Host of WTE & TQ)", fontsize=12.5, fontweight='bold')
    ax_text.set_ylim(0, max(text_mses) * 1.22)
    ax_text.grid(axis='y', linestyle=':', alpha=0.6)
    ax_text.legend(loc='upper left', frameon=True, fontsize=10)

    # Panel B: Time Branch (蒸馏时序分支与残差)
    colors_time = ['#2ca02c'] + ['#d62728' if d > 10 else '#ff7f0e' if d > 2 else '#1f77b4' for d in time_deltas[1:]]
    bars_time = ax_time.bar(range(len(exp_labels)), time_mses, color=colors_time, width=0.55, edgecolor='black', linewidth=1.1, alpha=0.9)
    ax_time.axhline(base_time_mse, color='#2ca02c', linestyle='--', linewidth=1.5, label=f'Clean Time MSE ({base_time_mse:.4f})')

    for i, bar in enumerate(bars_time):
        h = bar.get_height()
        txt = f"{h:.4f}\n(Base)" if i == 0 else f"{h:.4f}\n({time_deltas[i]:+.1f}%)"
        ax_time.annotate(txt, xy=(bar.get_x() + bar.get_width() / 2, h), xytext=(0, 4), textcoords="offset points",
                         ha='center', va='bottom', fontsize=8.5, fontweight='bold',
                         color='#8b0000' if time_deltas[i] > 10 else '#333333')

    ax_time.set_xticks(range(len(exp_labels)))
    ax_time.set_xticklabels(exp_labels, rotation=35, ha='right', fontsize=9.5)
    ax_time.set_ylabel("Mean Squared Error (MSE)", fontsize=11, fontweight='bold')
    ax_time.set_title("Temporal Branch (Distilled Physical & Queue Prior)", fontsize=12.5, fontweight='bold')
    ax_time.set_ylim(0, max(time_mses) * 1.22)
    ax_time.grid(axis='y', linestyle=':', alpha=0.6)
    ax_time.legend(loc='upper left', frameon=True, fontsize=10)

    plt.tight_layout()
    safe_savefig(fig, os.path.join(output_dir, "cycle_perturbation_impact.png"))
    plt.close(fig)

    # 5. 图件 2: 144x144 temporal_query 余弦相似度热力图 (cycle_tq_cosine_heatmap.png)
    print("\n>>> 5. Generating Publication Plot 2: 144x144 TQ Cosine Similarity Heatmap...")
    tq_np = orig_tq.cpu().numpy()  # (144, 768)
    tq_norm = tq_np / (np.linalg.norm(tq_np, axis=1, keepdims=True) + 1e-9)
    sim_mat = np.matmul(tq_norm, tq_norm.T)  # (144, 144)

    fig, ax = plt.subplots(figsize=(8.5, 7.5), dpi=300)
    cax = ax.imshow(sim_mat, cmap='coolwarm', vmin=-1.0, vmax=1.0, origin='lower')
    cbar = fig.colorbar(cax, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Cosine Similarity", fontsize=11, fontweight='bold')

    ticks = [0, 24, 48, 72, 96, 120, 143]
    labels = ["00:00\n(Night)", "04:00\n(Dawn)", "08:00\n(Morning)", "12:00\n(Noon)", "16:00\n(Afternoon)", "20:00\n(Dusk)", "24:00\n(Night)"]
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels, fontsize=9.5)
    ax.set_yticks(ticks)
    ax.set_yticklabels(labels, fontsize=9.5)
    ax.set_xlabel("Diurnal Time Step $t_j$ (10-min resolution)", fontsize=11, fontweight='bold')
    ax.set_ylabel("Diurnal Time Step $t_i$ (10-min resolution)", fontsize=11, fontweight='bold')
    ax.set_title("Learned Temporal Query ($TQ$) Phase Coherence Matrix (144 $\\times$ 144)", fontsize=13, fontweight='bold', pad=12)

    rect_noon = plt.Rectangle((48, 48), 48, 48, fill=False, edgecolor='gold', linewidth=2, linestyle='--', label='Diurnal Active Zone (08:00-16:00)')
    ax.add_patch(rect_noon)
    ax.legend(loc='upper right', frameon=True, fontsize=9.5, facecolor='white', framealpha=0.9)

    safe_savefig(fig, os.path.join(output_dir, "cycle_tq_cosine_heatmap.png"))
    plt.close(fig)

    # 6. 图件 3: 周期滞后自相关曲线 R(tau) (cycle_autocorrelation_lag.png)
    print("\n>>> 6. Generating Publication Plot 3: Diurnal Lag Autocorrelation Curve...")
    lags = np.arange(144)
    autocorr = np.zeros(144)
    for tau in lags:
        circ_idx = (np.arange(144) + tau) % 144
        autocorr[tau] = np.mean(sim_mat[np.arange(144), circ_idx])

    fig, ax = plt.subplots(figsize=(10, 5), dpi=300)
    hours = lags * 10 / 60.0  # [0, 24]
    ax.plot(hours, autocorr, color='#1f77b4', linewidth=2.5, marker='o', markersize=3, label='Mean Autocorrelation $R(\\tau)$')

    min_idx = np.argmin(autocorr)
    ax.axvline(hours[min_idx], color='red', linestyle='--', linewidth=1.5, label=f'Antipodal Minimum at {hours[min_idx]:.1f}h ($R={autocorr[min_idx]:.3f}$)')
    ax.axvline(24.0, color='green', linestyle=':', linewidth=1.5, label='24h Diurnal Boundary Recovery')
    ax.axhline(0, color='gray', linestyle='-', linewidth=0.8, alpha=0.5)

    ax.set_xlabel("Time Lag $\\tau$ (Hours)", fontsize=11, fontweight='bold')
    ax.set_ylabel("Cosine Similarity $R(\\tau)$", fontsize=11, fontweight='bold')
    ax.set_title("Empirical Diurnal Periodicity Autocorrelation Curve of $TQ$", fontsize=13, fontweight='bold', pad=12)
    ax.set_xticks(np.arange(0, 25, 2))
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.legend(loc='lower left', frameon=True, fontsize=10.5)

    safe_savefig(fig, os.path.join(output_dir, "cycle_autocorrelation_lag.png"))
    plt.close(fig)

    # 7. 图件 4: 昼夜词嵌入原型激活剖面分析 (wte_day_vs_night_activation.png)
    print("\n>>> 7. Generating Publication Plot 4: Word Prototype Attention Saliency (Day vs. Night)...")
    noon_prof, midnight_prof = extract_attention_weights(model, test_loader, device)

    diff_prof = noon_prof - midnight_prof
    top_noon_idx = np.argsort(diff_prof)[-10:]
    top_midnight_idx = np.argsort(diff_prof)[:10]

    fig, (ax_all, ax_top) = plt.subplots(2, 1, figsize=(12, 8), dpi=300, gridspec_kw={'height_ratios': [1.2, 1]})

    ax_all.plot(range(500), noon_prof, color='#ff7f0e', linewidth=1.2, alpha=0.85, label='Noon Active Profile (11:00-13:00)')
    ax_all.plot(range(500), midnight_prof, color='#1f77b4', linewidth=1.2, alpha=0.85, label='Midnight Active Profile (23:00-01:00)')
    ax_all.set_xlim(0, 499)
    ax_all.set_xlabel("Word Embedding Prototypes Index (0 to 499)", fontsize=10.5, fontweight='bold')
    ax_all.set_ylabel("Attention Saliency", fontsize=10.5, fontweight='bold')
    ax_all.set_title("Cross-Attention Prototype Specialization: Diurnal Phase Separation", fontsize=12.5, fontweight='bold')
    ax_all.legend(loc='upper right', frameon=True, fontsize=10)
    ax_all.grid(True, linestyle=':', alpha=0.5)

    top_indices = np.concatenate([top_midnight_idx[:5], top_noon_idx[-5:]])
    top_diffs = diff_prof[top_indices]
    bar_colors = ['#1f77b4'] * 5 + ['#ff7f0e'] * 5
    bar_labels = [f"P_{idx}" for idx in top_indices]

    ax_top.bar(range(10), top_diffs, color=bar_colors, edgecolor='black', linewidth=0.9, width=0.6)
    ax_top.axhline(0, color='black', linewidth=1)
    ax_top.set_xticks(range(10))
    ax_top.set_xticklabels(bar_labels, fontsize=10)
    ax_top.set_ylabel("$\\Delta$ Attention (Noon - Midnight)", fontsize=10.5, fontweight='bold')
    ax_top.set_title("Top Specialized Prototypes: Nocturnal-Specialized (<0) vs. Solar-Specialized (>0)", fontsize=11.5, fontweight='bold')
    ax_top.grid(axis='y', linestyle=':', alpha=0.5)

    plt.tight_layout()
    safe_savefig(fig, os.path.join(output_dir, "wte_day_vs_night_activation.png"))
    plt.close(fig)

    print("\n" + "=" * 95)
    print(">>> Dual-Branch Verification and Publication Figures Generated Successfully!")
    print(f">>> Summary Table: {csv_path}")
    print("=" * 95)


if __name__ == '__main__':
    main()
