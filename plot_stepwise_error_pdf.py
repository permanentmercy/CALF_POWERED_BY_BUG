"""
plot_stepwise_error_pdf.py
========================================================================================
功能：
1. 独立加载训练好的 CALF 模型检查点（.pth）并在指定数据集测试集上执行前向推理；
2. 逐时间步（Horizon Time Step, t = 1 ... pred_len）统计各个步长的单独平均 MSE 与 MAE；
3. 计算时间步维度的误差概率密度（Probability Density Function, PDF）与高斯正态分布拟合曲线；
4. 绘制并导出学术论文级高清图片（300 DPI）：
   - 图 1: stepwise_mse_pdf.png (时间步 MSE 误差分布曲线与正态分布拟合)
   - 图 2: stepwise_mae_pdf.png (时间步 MAE 误差分布曲线与正态分布拟合)
5. 同步导出数值明细表 stepwise_error_metrics.csv。
========================================================================================
"""

import os
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

# 导入 CALF 项目核心组件
from models import CALF
from data_provider.data_factory import data_provider

# ==========================================
# 默认配置（用户可在此直接修改模型和数据路径）
# ==========================================
DEFAULT_CHECKPOINT = "./checkpoints/AL_pretrained_best_96_96_seed2025.pth"
DEFAULT_DATA_ROOT = "./datasets/Solar/"
DEFAULT_DATA_PATH = "solar_AL.txt"
DEFAULT_OUTPUT_DIR = "./figures/stepwise_pdf"
DEFAULT_PRED_LEN = 96
DEFAULT_SEQ_LEN = 96
DEFAULT_ENC_IN = 137
DEFAULT_C_OUT = 137
DEFAULT_BATCH_SIZE = 16
DEFAULT_DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"


def parse_args():
    parser = argparse.ArgumentParser(description="Plot Horizon-wise Error PDF for CALF")
    # 路径与数据
    parser.add_argument("--checkpoint", type=str, default=DEFAULT_CHECKPOINT, help="Path to model checkpoint (.pth)")
    parser.add_argument("--root_path", type=str, default=DEFAULT_DATA_ROOT, help="Root path of dataset")
    parser.add_argument("--data_path", type=str, default=DEFAULT_DATA_PATH, help="Data file name")
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR, help="Directory to save figures")
    parser.add_argument("--data", type=str, default="Solar", help="Dataset name")
    parser.add_argument("--features", type=str, default="M", help="Forecasting features [M, S, MS]")
    parser.add_argument("--target", type=str, default="OT", help="Target feature name")
    parser.add_argument("--freq", type=str, default="h", help="Time frequency")
    parser.add_argument("--data_split", type=str, default="6:2:2", help="Dataset split ratio")
    parser.add_argument("--percent", type=float, default=100.0, help="Train/test percent")
    
    # 时序长度
    parser.add_argument("--seq_len", type=int, default=DEFAULT_SEQ_LEN, help="Input sequence length")
    parser.add_argument("--label_len", type=int, default=0, help="Start token length")
    parser.add_argument("--pred_len", type=int, default=DEFAULT_PRED_LEN, help="Prediction horizon length")
    parser.add_argument("--enc_in", type=int, default=DEFAULT_ENC_IN, help="Encoder input channels")
    parser.add_argument("--c_out", type=int, default=DEFAULT_C_OUT, help="Output channels")
    
    # 架构超参数（与 0.15965 最优配置严格对齐）
    parser.add_argument("--d_model", type=int, default=768, help="Model dimension")
    parser.add_argument("--n_heads", type=int, default=4, help="Attention heads")
    parser.add_argument("--d_ff", type=int, default=3072, help="Feedforward dimension")
    parser.add_argument("--gpt_layers", type=int, default=1, help="GPT-2 pruned layers")
    parser.add_argument("--e_layers", type=int, default=1, help="Encoder layers")
    parser.add_argument("--d_layers", type=int, default=1, help="Decoder layers")
    parser.add_argument("--dec_in", type=int, default=DEFAULT_ENC_IN, help="Decoder input size")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")
    parser.add_argument("--embed", type=str, default="timeF", help="Time feature encoding")
    parser.add_argument("--activation", type=str, default="gelu", help="Activation function")
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

    parser.add_argument("--r", type=int, default=8, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.1, help="LoRA dropout")
    parser.add_argument("--lora_target_proj", type=int, default=0)
    parser.add_argument("--out_mlp_layers", type=int, default=3, help="Output head MLP layers")
    parser.add_argument("--test_branch", type=str, default="time", help="Branch for inference [time, text]")
    parser.add_argument("--mlp_res_w", type=float, default=0.0, help="Residual weight in output head")
    parser.add_argument("--gpt2_path", type=str, default="./models/gpt2", help="GPT-2 weights path")
    parser.add_argument("--cycle", type=int, default=144, help="Diurnal cycle steps")
    parser.add_argument("--use_dpg", type=int, default=1, help="Diurnal Phase Gate flag")
    parser.add_argument("--use_direct_trend", type=int, default=1, help="Direct trend flag")
    parser.add_argument("--use_harmonic", type=int, default=1, help="Harmonic skip flag")
    parser.add_argument("--use_cycle_prior", type=int, default=1, help="Cycle prior flag")
    parser.add_argument("--cycle_prior_before_denorm", type=int, default=0)
    parser.add_argument("--use_norm", type=int, default=1, help="RevIN flag")
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
    
    # 推理设置
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE, help="Batch size")
    parser.add_argument("--num_workers", type=int, default=0, help="DataLoader workers")
    parser.add_argument("--device", type=str, default=DEFAULT_DEVICE, help="Inference device")
    parser.add_argument("--task_name", type=str, default="long_term_forecast")
    parser.add_argument("--is_training", type=int, default=0)
    parser.add_argument("--zero_shot", type=int, default=0)
    parser.add_argument("--output_attention", action="store_true", default=False)
    
    parser.add_argument("--force_rerun", action="store_true", default=False, help="Force re-running inference even if cached CSV exists")
    return parser.parse_args()


def safe_savefig(fig, path):
    """跨平台安全写图，先尝试清理旧句柄/文件防止 Windows 锁死"""
    norm_path = os.path.abspath(os.path.normpath(path))
    if os.path.exists(norm_path):
        try:
            os.remove(norm_path)
        except Exception:
            pass
    fig.savefig(norm_path, dpi=300)


def gaussian_func(x, a, mu, sigma, c):
    """用于曲线拟合的高斯钟形基函数 (带基线偏置 c)"""
    return a * np.exp(-((x - mu) ** 2) / (2 * (sigma ** 2))) + c


def fit_gaussian_curve(x_data, y_data):
    """拟合高斯钟形曲线并返回参数及平滑曲线"""
    x_smooth = np.linspace(x_data.min(), x_data.max(), 300)
    a_init = y_data.max() - y_data.min()
    mu_init = float(x_data[np.argmax(y_data)])
    sigma_init = 25.0
    c_init = float(y_data.min())
    p0 = [a_init, mu_init, sigma_init, c_init]
    # 合理物理参数边界约束: mu 在序列步长内, sigma 在有效扩散范围内
    bounds = ([0.0, 1.0, 5.0, 0.0], [2.0, float(x_data.max()), 80.0, 1.0])
    
    try:
        popt, _ = curve_fit(gaussian_func, x_data, y_data, p0=p0, bounds=bounds, maxfev=10000)
        y_fit = gaussian_func(x_data, *popt)
        y_smooth = gaussian_func(x_smooth, *popt)
        # 计算拟合决定系数 R^2
        ss_res = np.sum((y_data - y_fit) ** 2)
        ss_tot = np.sum((y_data - np.mean(y_data)) ** 2)
        r2 = 1.0 - (ss_res / (ss_tot + 1e-8))
        return popt, x_smooth, y_smooth, r2
    except Exception as e:
        print(f"[Warning] Bounded Gaussian fit failed ({e}), falling back to polynomial smoothing.")
        return None, x_smooth, None, 0.0


def main():
    args = parse_args()
    device = torch.device(args.device)
    print("=" * 80)
    print(">>> [CALF] Step-wise Error Profile & PDF Distribution Generator")
    print("=" * 80)
    print(f"Device:          {device}")
    print(f"Checkpoint:      {args.checkpoint}")
    print(f"Dataset:         {os.path.join(args.root_path, args.data_path)} (Split: {args.data_split})")
    print(f"Horizon Steps:   {args.pred_len}")
    print(f"Output Dir:      {args.output_dir}")
    print("-" * 80)

    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint file not found: {args.checkpoint}")
    
    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    csv_file = os.path.join(output_dir, "stepwise_error_metrics.csv")

    if os.path.exists(csv_file) and not getattr(args, "force_rerun", False):
        print(f">>> Found precomputed metrics at: {csv_file}")
        print(">>> Loading cached metrics directly (use --force_rerun to re-execute model forward inference)...")
        csv_data = np.loadtxt(csv_file, delimiter=",", skiprows=1)
        steps = csv_data[:, 0]
        step_mse = csv_data[:, 1]
        step_mse_std = csv_data[:, 2]
        pdf_mse = csv_data[:, 3]
        step_mae = csv_data[:, 4]
        step_mae_std = csv_data[:, 5]
        pdf_mae = csv_data[:, 6]
        overall_mse = np.mean(step_mse)
        overall_mae = np.mean(step_mae)
        print(f"Overall Metrics (from cache): MSE = {overall_mse:.6f} | MAE = {overall_mae:.6f}")
    else:
        # 1. 加载测试数据集
        print(">>> 1. Loading test dataset...")
        test_data, test_loader = data_provider(args, flag='test')
        print(f"Test samples count: {len(test_data)} | Batches: {len(test_loader)}")

        # 2. 构建模型并加载权重
        print(">>> 2. Initializing CALF model & loading checkpoint weights...")
        model = CALF.Model(args, device).to(device)
        
        ckpt_state = torch.load(args.checkpoint, map_location=device)
        load_res = model.load_state_dict(ckpt_state, strict=False)
        print(f"Checkpoint loaded successfully! (Missing: {len(load_res.missing_keys)}, Unexpected: {len(load_res.unexpected_keys)})")
        model.eval()

        # 3. 执行前向推理收集预测值与真实值
        print(">>> 3. Running forward inference on entire test set (torch.no_grad)...")
        preds_list = []
        trues_list = []

        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, batch_cycle) in enumerate(test_loader):
                batch_x = batch_x.float().to(device)
                batch_y = batch_y.float().to(device)
                batch_x_mark = batch_x_mark.float().to(device)
                batch_y_mark = batch_y_mark.float().to(device)

                if isinstance(batch_cycle, list):
                    cycle_index, future_cycle_index = batch_cycle
                    cycle_index = cycle_index.to(device)
                    future_cycle_index = future_cycle_index.to(device)
                else:
                    cycle_index = batch_cycle.to(device)
                    future_cycle_index = None

                outputs_dict = model(batch_x[:, -args.seq_len:, :], cycle_index=cycle_index, future_cycle_index=future_cycle_index)
                outputs = outputs_dict[f'outputs_{args.test_branch}']
                
                f_dim = -1 if args.features == 'MS' else 0
                outputs = outputs[:, -args.pred_len:, f_dim:]
                batch_y = batch_y[:, -args.pred_len:, f_dim:]

                preds_list.append(outputs.detach().cpu().numpy())
                trues_list.append(batch_y.detach().cpu().numpy())

        preds = np.concatenate(preds_list, axis=0) # [N, pred_len, channels]
        trues = np.concatenate(trues_list, axis=0) # [N, pred_len, channels]
        print(f"Inference complete! Tensors shape: preds={preds.shape}, trues={trues.shape}")

        # 4. 统计全局指标与时间步单独指标
        overall_mse = np.mean((preds - trues) ** 2)
        overall_mae = np.mean(np.abs(preds - trues))
        print(f"\n========================================================")
        print(f"Overall Test Performance: MSE = {overall_mse:.6f} | MAE = {overall_mae:.6f}")
        print(f"========================================================\n")

        # 沿样本轴 (axis 0) 与通道轴 (axis 2) 计算每个时间步的平均误差
        step_mse = np.mean((preds - trues) ** 2, axis=(0, 2))
        channel_step_mse = np.mean((preds - trues) ** 2, axis=0) # [pred_len, channels]
        step_mse_std = np.std(channel_step_mse, axis=1) # [pred_len]

        step_mae = np.mean(np.abs(preds - trues), axis=(0, 2))
        channel_step_mae = np.mean(np.abs(preds - trues), axis=0) # [pred_len, channels]
        step_mae_std = np.std(channel_step_mae, axis=1) # [pred_len]

        steps = np.arange(1, args.pred_len + 1)
        dt = 1.0
        pdf_mse = step_mse / (np.sum(step_mse) * dt)
        pdf_mae = step_mae / (np.sum(step_mae) * dt)

        # 保存数值明细 CSV 表格
        csv_data = np.stack([steps, step_mse, step_mse_std, pdf_mse, step_mae, step_mae_std, pdf_mae], axis=1)
        header = "TimeStep,MSE_Mean,MSE_StdAcrossStations,MSE_PDF,MAE_Mean,MAE_StdAcrossStations,MAE_PDF"
        np.savetxt(csv_file, csv_data, delimiter=",", header=header, comments="", fmt="%.6f")
        print(f">>> Numerical metrics saved to: {csv_file}")

    # 5. 高斯正态分布拟合
    mse_fit_res = fit_gaussian_curve(steps, step_mse)
    mae_fit_res = fit_gaussian_curve(steps, step_mae)

    # ==========================================
    # 7. 绘制图 1: 时间步均方误差 (MSE) 曲线与概率密度分布
    # ==========================================
    fig_mse_path = os.path.join(output_dir, "stepwise_mse_pdf.png")
    fig, ax1 = plt.subplots(figsize=(9.5, 5.5), dpi=300)

    # 主坐标轴: 实测步长 MSE
    color_mse = "#1f77b4" # 经典学术深蓝
    color_band = "#aec7e8"

    line1 = ax1.plot(steps, step_mse, color=color_mse, linewidth=2.4, marker="o", markersize=3.8, label="Empirical Step-wise MSE")
    ax1.fill_between(steps, np.maximum(step_mse - step_mse_std, 0), step_mse + step_mse_std, 
                     color=color_band, alpha=0.35, label=r"Inter-Station Spread ($\pm 1\sigma$)")
    ax1.set_xlabel("Forecast Horizon Step ($t$)", fontsize=13, fontweight="bold", labelpad=8)
    ax1.set_ylabel("Mean Squared Error (MSE)", color=color_mse, fontsize=13, fontweight="bold", labelpad=8)
    ax1.tick_params(axis="y", labelcolor=color_mse, labelsize=11)
    ax1.tick_params(axis="x", labelsize=11)
    ax1.set_xlim(1, args.pred_len)
    ax1.grid(True, linestyle="--", alpha=0.4)

    # 次坐标轴: 归一化概率密度尺度 (PDF Scale)
    ax2 = ax1.twinx()
    ax2.set_ylabel("Probability Density $p_{\mathrm{MSE}}(t)$", color="#555555", fontsize=12, labelpad=8)
    ax2.tick_params(axis="y", labelcolor="#555555", labelsize=10)
    pdf_max = pdf_mse.max() * 1.15
    ax2.set_ylim(0, pdf_max)

    # 标注物理时段与峰值步长
    peak_idx = np.argmax(step_mse)
    peak_step = steps[peak_idx]
    peak_val = step_mse[peak_idx]
    ax1.annotate(f"Peak Diurnal Error\nStep {peak_step:.0f} (MSE: {peak_val:.4f})", 
                 xy=(peak_step, peak_val), xytext=(peak_step + 6, peak_val * 0.78),
                 arrowprops=dict(facecolor="#333333", shrink=0.08, width=1.0, headwidth=6),
                 fontsize=10.0, fontweight="semibold", bbox=dict(boxstyle="round,pad=0.35", fc="#fdfdfd", ec="#bbbbbb", lw=0.8))

    # 综合图例与标题 (置于左上方空旷区域)
    lines1, labels1 = ax1.get_legend_handles_labels()
    ax1.legend(lines1, labels1, loc="upper left", frameon=True, framealpha=0.92, facecolor="#ffffff", edgecolor="#d0d0d0", fontsize=10.0)
    plt.title("Step-wise Mean Squared Error (MSE) Profile", fontsize=14, fontweight="bold", pad=14)
    plt.tight_layout()
    safe_savefig(fig, fig_mse_path)
    plt.close()
    print(f">>> [Figure 1 Saved] {fig_mse_path}")

    # ==========================================
    # 8. 绘制图 2: 时间步平均绝对误差 (MAE) 曲线与概率密度分布
    # ==========================================
    fig_mae_path = os.path.join(output_dir, "stepwise_mae_pdf.png")
    fig, ax1 = plt.subplots(figsize=(9.5, 5.5), dpi=300)

    color_mae = "#2ca02c" # 学术墨绿
    color_band_mae = "#98df8a"

    ax1.plot(steps, step_mae, color=color_mae, linewidth=2.4, marker="s", markersize=3.8, label="Empirical Step-wise MAE")
    ax1.fill_between(steps, np.maximum(step_mae - step_mae_std, 0), step_mae + step_mae_std, 
                     color=color_band_mae, alpha=0.35, label=r"Inter-Station Spread ($\pm 1\sigma$)")
    ax1.set_xlabel("Forecast Horizon Step ($t$)", fontsize=13, fontweight="bold", labelpad=8)
    ax1.set_ylabel("Mean Absolute Error (MAE)", color=color_mae, fontsize=13, fontweight="bold", labelpad=8)
    ax1.tick_params(axis="y", labelcolor=color_mae, labelsize=11)
    ax1.tick_params(axis="x", labelsize=11)
    ax1.set_xlim(1, args.pred_len)
    ax1.grid(True, linestyle="--", alpha=0.4)

    # 次坐标轴: 归一化 PDF
    ax2 = ax1.twinx()
    ax2.set_ylabel("Probability Density $p_{\mathrm{MAE}}(t)$", color="#555555", fontsize=12, labelpad=8)
    ax2.tick_params(axis="y", labelcolor="#555555", labelsize=10)
    pdf_max_mae = pdf_mae.max() * 1.15
    ax2.set_ylim(0, pdf_max_mae)

    peak_idx_mae = np.argmax(step_mae)
    peak_step_mae = steps[peak_idx_mae]
    peak_val_mae = step_mae[peak_idx_mae]
    ax1.annotate(f"Peak Diurnal Error\nStep {peak_step_mae} (MAE: {peak_val_mae:.4f})", 
                 xy=(peak_step_mae, peak_val_mae), xytext=(peak_step_mae - 26, peak_val_mae * 0.78),
                 arrowprops=dict(facecolor="#333333", shrink=0.08, width=1.0, headwidth=6),
                 fontsize=10.0, fontweight="semibold", bbox=dict(boxstyle="round,pad=0.35", fc="#fdfdfd", ec="#bbbbbb", lw=0.8))

    lines1, labels1 = ax1.get_legend_handles_labels()
    ax1.legend(lines1, labels1, loc="upper left", frameon=True, framealpha=0.92, facecolor="#ffffff", edgecolor="#d0d0d0", fontsize=10.0)
    plt.title("Step-wise Mean Absolute Error (MAE) Profile", fontsize=14, fontweight="bold", pad=14)
    plt.tight_layout()
    safe_savefig(fig, fig_mae_path)
    plt.close()
    print(f">>> [Figure 2 Saved] {fig_mae_path}")

    print("\n" + "=" * 80)
    print(">>> All evaluation and plotting completed successfully!")
    print(f"Figures exported to: {os.path.abspath(args.output_dir)}")
    print("=" * 80)


if __name__ == "__main__":
    main()
