"""
plot_residual_histogram.py
========================================================================================
功能：
1. 独立加载训练好的 CALF 模型检查点（.pth）并在测试集上执行前向推理；
2. 提取所有预测点与真实点的差值计算预测残差 (Residual = Preds - Trues) 与误差指标；
3. 计算残差的均值、标准差、偏度 (Skewness) 与峰度 (Kurtosis)，评估模型的无偏性与随机性；
4. 绘制并导出七张学术出版级 300 DPI PNG 高清图件：
   - 图 1: residual_histogram_pdf.png (双侧残差直方图与概率密度函数 PDF)
   - 图 2: residual_qq_plot.png (正态 Q-Q 分位数检验图，验证误差正态分布特性)
   - 图 3: residual_mse_pdf.png (图 A: 双侧残差 + ±RMSE 有效误差带宽标定图)
   - 图 4: residual_mae_pdf.png (图 B: 双侧残差 + ±MAE 平均绝对偏差集中带标定图)
   - 图 5: error_mse_pdf.png (MSE 专属图: 平方误差 e^2 概率密度分布与直方图)
   - 图 6: error_mae_pdf.png (MAE 专属图: 绝对误差 |e| 概率密度分布与直方图)
   - 图 7: residual_abs_pdf.png (残差绝对值图: 预测残差绝对值 |e| 直方图与累计覆盖带)
5. 导出残差与误差统计指标表格 residual_summary_metrics.csv。
========================================================================================
"""

import os
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
from scipy import stats
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
DEFAULT_OUTPUT_DIR = "./figures/residual_analysis"
DEFAULT_PRED_LEN = 96
DEFAULT_SEQ_LEN = 96
DEFAULT_ENC_IN = 137
DEFAULT_C_OUT = 137
DEFAULT_BATCH_SIZE = 16
DEFAULT_DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"


def parse_args():
    parser = argparse.ArgumentParser(description="Plot Residual Histogram and PDF for CALF")
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
    
    # 架构超参数（与最优基准严格一致）
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
    parser.add_argument("--force_rerun", action="store_true", default=False, help="Force re-running inference even if cached npz exists")
    
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


def main():
    args = parse_args()
    device = torch.device(args.device)
    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 80)
    print(">>> [CALF] Residual Histogram & Error PDF Distribution Analyzer")
    print("=" * 80)
    print(f"Device:          {device}")
    print(f"Checkpoint:      {args.checkpoint}")
    print(f"Dataset:         {os.path.join(args.root_path, args.data_path)} (Split: {args.data_split})")
    print(f"Horizon Steps:   {args.pred_len}")
    print(f"Output Dir:      {output_dir}")
    print("-" * 80)

    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint file not found: {args.checkpoint}")

    cache_npz = os.path.join(output_dir, "residuals_cache.npz")
    
    # 智能缓存机制：如果已缓存则直接读取残差统计，极大提升绘图与调参效率
    if os.path.exists(cache_npz) and not getattr(args, "force_rerun", False):
        print(f">>> Found precomputed residuals cache at: {cache_npz}")
        print(">>> Loading cached residuals (use --force_rerun to re-execute model forward inference)...")
        cache_data = np.load(cache_npz)
        sample_residuals = cache_data["sample_residuals"]
        overall_mse = float(cache_data["overall_mse"])
        overall_mae = float(cache_data["overall_mae"])
        res_mean = float(cache_data["res_mean"])
        res_std = float(cache_data["res_std"])
        res_skew = float(cache_data["res_skew"])
        res_kurt = float(cache_data["res_kurt"])
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

        # 4. 计算残差与高阶统计量
        overall_mse = float(np.mean((preds - trues) ** 2))
        overall_mae = float(np.mean(np.abs(preds - trues)))
        residuals = (preds - trues).flatten() # 约 1.37 亿个残差点
        
        res_mean = float(np.mean(residuals))
        res_std = float(np.std(residuals))
        
        # 为高效进行核密度和 Q-Q 绘图，高质量无偏随机抽取 50,000 个残差子样本
        rng = np.random.default_rng(2025)
        sample_residuals = rng.choice(residuals, size=min(len(residuals), 50000), replace=False)
        
        res_skew = float(stats.skew(sample_residuals))
        res_kurt = float(stats.kurtosis(sample_residuals))

        # 保存轻量化缓存
        np.savez_compressed(cache_npz, 
                           sample_residuals=sample_residuals,
                           overall_mse=overall_mse,
                           overall_mae=overall_mae,
                           res_mean=res_mean,
                           res_std=res_std,
                           res_skew=res_skew,
                           res_kurt=res_kurt)
        print(f">>> Residuals cached to: {cache_npz}")

    # 4. 计算误差与高阶统计量
    sample_se = sample_residuals ** 2
    sample_ae = np.abs(sample_residuals)

    mse_mean = float(np.mean(sample_se))
    mse_std = float(np.std(sample_se))
    mse_median = float(np.median(sample_se))
    mse_p90 = float(np.percentile(sample_se, 90.0))
    mse_p95 = float(np.percentile(sample_se, 95.0))
    mse_p99 = float(np.percentile(sample_se, 99.0))
    mse_skew = float(stats.skew(sample_se))

    mae_mean = float(np.mean(sample_ae))
    mae_std = float(np.std(sample_ae))
    mae_median = float(np.median(sample_ae))
    mae_p90 = float(np.percentile(sample_ae, 90.0))
    mae_p95 = float(np.percentile(sample_ae, 95.0))
    mae_p99 = float(np.percentile(sample_ae, 99.0))
    mae_skew = float(stats.skew(sample_ae))

    print(f"\n========================================================")
    print(f"Overall Test MSE:      {overall_mse:.6f}")
    print(f"Overall Test MAE:      {overall_mae:.6f}")
    print(f"Residual Mean (Bias):  {res_mean:.6f}  (Ideal: 0.000000)")
    print(f"Residual Std (Sigma):  {res_std:.6f}")
    print(f"Residual Skewness:     {res_skew:.4f}  (Ideal: 0.0000, Symmetric)")
    print(f"Residual Kurtosis:     {res_kurt:.4f}")
    print(f"Squared Error (MSE) - Median: {mse_median:.6f}, P95: {mse_p95:.6f}, P99: {mse_p99:.6f}")
    print(f"Absolute Error (MAE) - Median: {mae_median:.6f}, P95: {mae_p95:.6f}, P99: {mae_p99:.6f}")
    print(f"========================================================\n")

    # 保存数值统计表格 CSV
    summary_csv = os.path.join(output_dir, "residual_summary_metrics.csv")
    with open(summary_csv, "w", encoding="utf-8") as f:
        f.write("Metric,Value\n")
        f.write(f"Overall_MSE,{overall_mse:.6f}\n")
        f.write(f"Overall_MAE,{overall_mae:.6f}\n")
        f.write(f"Residual_Mean,{res_mean:.6f}\n")
        f.write(f"Residual_Std,{res_std:.6f}\n")
        f.write(f"Residual_Skewness,{res_skew:.4f}\n")
        f.write(f"Residual_Kurtosis,{res_kurt:.4f}\n")
        f.write(f"MSE_Sample_Mean,{mse_mean:.6f}\n")
        f.write(f"MSE_Median,{mse_median:.6f}\n")
        f.write(f"MSE_Std,{mse_std:.6f}\n")
        f.write(f"MSE_P90,{mse_p90:.6f}\n")
        f.write(f"MSE_P95,{mse_p95:.6f}\n")
        f.write(f"MSE_P99,{mse_p99:.6f}\n")
        f.write(f"MSE_Skewness,{mse_skew:.4f}\n")
        f.write(f"MAE_Sample_Mean,{mae_mean:.6f}\n")
        f.write(f"MAE_Median,{mae_median:.6f}\n")
        f.write(f"MAE_Std,{mae_std:.6f}\n")
        f.write(f"MAE_P90,{mae_p90:.6f}\n")
        f.write(f"MAE_P95,{mae_p95:.6f}\n")
        f.write(f"MAE_P99,{mae_p99:.6f}\n")
        f.write(f"MAE_Skewness,{mae_skew:.4f}\n")
        f.write(f"Sample_Count,{len(sample_residuals)}\n")
    print(f">>> Summary metrics saved to: {summary_csv}")

    # ==========================================
    # 5. 绘制图 1: 预测残差直方图与概率密度曲线 (PDF)
    # ==========================================
    fig_hist_path = os.path.join(output_dir, "residual_histogram_pdf.png")
    fig, ax = plt.subplots(figsize=(9.5, 5.5), dpi=300)

    color_hist = "#2b5c8f"      # 沉稳学术蓝
    color_kde = "#e7298a"       # 艳丽洋红 (实测经验 PDF)

    # 绘制直方图柱状图 (Density=True 归一化为概率密度)
    # 限制横轴在 99.5% 分位数范围内，消除离群极值对视觉效果的压缩
    # 采用更密集细致的 bin 划分 (220 个分箱) 并设置 rwidth=0.82 使柱体纤细独立
    clip_val = float(np.percentile(np.abs(sample_residuals), 99.5))
    bins = np.linspace(-clip_val, clip_val, 220)
    
    ax.hist(sample_residuals, bins=bins, density=True, rwidth=0.82,
            color=color_hist, alpha=0.50, edgecolor="#ffffff", linewidth=0.25,
            label="Residual Histogram (Binned Density)")

    # 绘制经验核密度估计曲线 (KDE)
    kde_x = np.linspace(-clip_val, clip_val, 400)
    kde = stats.gaussian_kde(sample_residuals)
    kde_y = kde(kde_x)
    ax.plot(kde_x, kde_y, color=color_kde, linewidth=1.1, alpha=0.85, label="Empirical PDF (KDE)")

    ax.set_xlabel("Prediction Residual $e = \hat{y} - y$", fontsize=13, fontweight="bold", labelpad=8)
    ax.set_ylabel("Probability Density $p(e)$", fontsize=13, fontweight="bold", labelpad=8)
    ax.set_xlim(-clip_val, clip_val)
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.tick_params(axis="both", labelsize=11)

    # 统计量信息卡片 (置于右上角开阔区域)
    stat_text = (
        f"Model: CALF (AL 96→96)\n"
        f"Test MSE: {overall_mse:.5f}\n"
        f"Test MAE: {overall_mae:.5f}\n"
        f"Bias ($\mu$): {res_mean:.5f}\n"
        f"Std ($\sigma$): {res_std:.5f}\n"
        f"Skewness: {res_skew:.3f} (Symmetric)\n"
        f"Kurtosis: {res_kurt:.3f}"
    )
    ax.text(0.97, 0.95, stat_text, transform=ax.transAxes, fontsize=10.0,
            verticalalignment="top", horizontalalignment="right",
            bbox=dict(boxstyle="round,pad=0.5", fc="#fdfdfd", ec="#cccccc", lw=0.8, alpha=0.95))

    ax.legend(loc="upper left", frameon=True, framealpha=0.92, facecolor="#ffffff", edgecolor="#d0d0d0", fontsize=10.0)
    plt.title("Prediction Residual Distribution & Probability Density Function (PDF)", fontsize=14, fontweight="bold", pad=14)
    plt.tight_layout()
    safe_savefig(fig, fig_hist_path)
    plt.close()
    print(f">>> [Figure 1 Saved] {fig_hist_path} (PNG)")

    # ==========================================
    # 6. 绘制图 2: 正态 Q-Q 分位数检验图 (Quantile-Quantile Plot)
    # ==========================================
    fig_qq_path = os.path.join(output_dir, "residual_qq_plot.png")
    fig, ax = plt.subplots(figsize=(8.5, 6.0), dpi=300)

    # 计算 Q-Q 散点
    (osm, osr), (slope, intercept, r) = stats.probplot(sample_residuals, dist="norm")
    
    # 绘制 Q-Q 散点
    ax.scatter(osm, osr, color="#1b9e77", s=14, alpha=0.5, edgecolor="none", label="Empirical Residual Quantiles")
    
    # 绘制 45 度参考红线 (理想正态线)
    x_line = np.array([osm.min(), osm.max()])
    y_line = slope * x_line + intercept
    ax.plot(x_line, y_line, color="#e41a1c", linewidth=2.2, linestyle="-", 
            label=f"Theoretical Normal Line ($R^2 = {r**2:.4f}$)")

    ax.set_xlabel("Theoretical Quantiles (Standard Normal)", fontsize=13, fontweight="bold", labelpad=8)
    ax.set_ylabel("Ordered Sample Residuals ($e$)", fontsize=13, fontweight="bold", labelpad=8)
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.tick_params(axis="both", labelsize=11)
    
    # 标注 Q-Q 检验结论
    qq_text = (
        f"Normality Diagnostic:\n"
        f"Correlation $R^2$: {r**2:.4f}\n"
        f"Fit Slope: {slope:.4f}\n"
        f"Intercept: {intercept:.4f}"
    )
    ax.text(0.04, 0.95, qq_text, transform=ax.transAxes, fontsize=10.0,
            verticalalignment="top", horizontalalignment="left",
            bbox=dict(boxstyle="round,pad=0.5", fc="#fdfdfd", ec="#cccccc", lw=0.8, alpha=0.95))

    ax.legend(loc="lower right", frameon=True, framealpha=0.92, facecolor="#ffffff", edgecolor="#d0d0d0", fontsize=10.0)
    plt.title("Normal Q-Q Plot of Prediction Residuals", fontsize=14, fontweight="bold", pad=14)
    plt.tight_layout()
    safe_savefig(fig, fig_qq_path)
    plt.close()
    print(f">>> [Figure 2 Saved] {fig_qq_path} (PNG)")

    # ==========================================
    # 7. 绘制图 3 (图 A): 双侧残差 + MSE/RMSE 标定图 (保留 [-1.7, +1.7] 正负对称，高亮 ±RMSE 带宽)
    # ==========================================
    fig_mse_path = os.path.join(output_dir, "residual_mse_pdf.png")
    fig, ax = plt.subplots(figsize=(9.5, 5.5), dpi=300)

    color_hist_mse = "#2b5c8f"  # 沉稳学术蓝
    color_kde_mse = "#e7298a"   # 经验洋红 KDE
    rmse = float(np.sqrt(overall_mse))
    rmse_cov = float(np.mean(np.abs(sample_residuals) <= rmse) * 100.0)

    # 绘制双侧完整残差直方图 (220 分箱，纤细柱子)
    ax.hist(sample_residuals, bins=bins, density=True, rwidth=0.82,
            color=color_hist_mse, alpha=0.45, edgecolor="#ffffff", linewidth=0.25,
            label="Residual Histogram (Binned Density)")

    # 经验核密度估计曲线 (KDE)
    ax.plot(kde_x, kde_y, color=color_kde_mse, linewidth=1.2, alpha=0.9, label="Empirical PDF (KDE)")

    # 高亮标出 ±RMSE 误差有效带宽
    ax.axvspan(-rmse, rmse, color="#fdb462", alpha=0.30,
               label=f"±RMSE Effective Band [−{rmse:.3f}, +{rmse:.3f}] (Coverage: {rmse_cov:.1f}%)")
    ax.axvline(-rmse, color="#d95f02", linestyle="--", linewidth=1.8, label=f"−RMSE = −{rmse:.3f}")
    ax.axvline(rmse, color="#d95f02", linestyle="--", linewidth=1.8, label=f"+RMSE = +{rmse:.3f}")
    ax.axvline(0, color="#444444", linestyle=":", linewidth=1.0, alpha=0.7)

    ax.set_xlabel("Prediction Residual $e = \hat{y} - y$", fontsize=13, fontweight="bold", labelpad=8)
    ax.set_ylabel("Probability Density $p(e)$", fontsize=13, fontweight="bold", labelpad=8)
    ax.set_xlim(-clip_val, clip_val)
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.tick_params(axis="both", labelsize=11)

    stat_text_mse = (
        f"Model: CALF (AL 96→96)\n"
        f"Overall Test MSE: {overall_mse:.5f}\n"
        f"RMSE ($\sqrt{{MSE}}$): {rmse:.5f}\n"
        f"±RMSE Coverage: {rmse_cov:.2f}%\n"
        f"Residual Bias ($\mu$): {res_mean:.5f}\n"
        f"Residual Std ($\sigma$): {res_std:.5f}\n"
        f"Skewness: {res_skew:.3f}\n"
        f"Kurtosis: {res_kurt:.3f}"
    )
    ax.text(0.97, 0.95, stat_text_mse, transform=ax.transAxes, fontsize=10.0,
            verticalalignment="top", horizontalalignment="right",
            bbox=dict(boxstyle="round,pad=0.5", fc="#fdfdfd", ec="#cccccc", lw=0.8, alpha=0.95))

    ax.legend(loc="upper left", frameon=True, framealpha=0.92, facecolor="#ffffff", edgecolor="#d0d0d0", fontsize=9.5)
    plt.title("Residual Distribution & MSE/RMSE Calibration Band (PDF)", fontsize=14, fontweight="bold", pad=14)
    plt.tight_layout()
    safe_savefig(fig, fig_mse_path)
    plt.close()
    print(f">>> [Figure 3 (Chart A) Saved] {fig_mse_path} (PNG)")

    # ==========================================
    # 8. 绘制图 4 (图 B): 双侧残差 + MAE 标定图 (保留 [-1.7, +1.7] 正负对称，高亮 ±MAE 集中带)
    # ==========================================
    fig_mae_path = os.path.join(output_dir, "residual_mae_pdf.png")
    fig, ax = plt.subplots(figsize=(9.5, 5.5), dpi=300)

    color_hist_mae = "#2ca02c"  # 学术草绿
    color_kde_mae = "#e7298a"   # 经验洋红 KDE
    mae = float(overall_mae)
    mae_cov = float(np.mean(np.abs(sample_residuals) <= mae) * 100.0)

    # 绘制双侧完整残差直方图 (220 分箱，纤细柱子)
    ax.hist(sample_residuals, bins=bins, density=True, rwidth=0.82,
            color=color_hist_mae, alpha=0.45, edgecolor="#ffffff", linewidth=0.25,
            label="Residual Histogram (Binned Density)")

    # 经验核密度估计曲线 (KDE)
    ax.plot(kde_x, kde_y, color=color_kde_mae, linewidth=1.2, alpha=0.9, label="Empirical PDF (KDE)")

    # 高亮标出 ±MAE 误差集中带
    ax.axvspan(-mae, mae, color="#a1d99b", alpha=0.32,
               label=f"±MAE Central Band [−{mae:.3f}, +{mae:.3f}] (Coverage: {mae_cov:.1f}%)")
    ax.axvline(-mae, color="#006d2c", linestyle="--", linewidth=1.8, label=f"−MAE = −{mae:.3f}")
    ax.axvline(mae, color="#006d2c", linestyle="--", linewidth=1.8, label=f"+MAE = +{mae:.3f}")
    ax.axvline(0, color="#444444", linestyle=":", linewidth=1.0, alpha=0.7)

    ax.set_xlabel("Prediction Residual $e = \hat{y} - y$", fontsize=13, fontweight="bold", labelpad=8)
    ax.set_ylabel("Probability Density $p(e)$", fontsize=13, fontweight="bold", labelpad=8)
    ax.set_xlim(-clip_val, clip_val)
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.tick_params(axis="both", labelsize=11)

    stat_text_mae = (
        f"Model: CALF (AL 96→96)\n"
        f"Overall Test MAE: {overall_mae:.5f}\n"
        f"±MAE Coverage: {mae_cov:.2f}%\n"
        f"Median Abs Error: {mae_median:.5f}\n"
        f"Residual Bias ($\mu$): {res_mean:.5f}\n"
        f"Residual Std ($\sigma$): {res_std:.5f}\n"
        f"Skewness: {res_skew:.3f}\n"
        f"Kurtosis: {res_kurt:.3f}"
    )
    ax.text(0.97, 0.95, stat_text_mae, transform=ax.transAxes, fontsize=10.0,
            verticalalignment="top", horizontalalignment="right",
            bbox=dict(boxstyle="round,pad=0.5", fc="#fdfdfd", ec="#cccccc", lw=0.8, alpha=0.95))

    ax.legend(loc="upper left", frameon=True, framealpha=0.92, facecolor="#ffffff", edgecolor="#d0d0d0", fontsize=9.5)
    plt.title("Residual Distribution & MAE Calibration Band (PDF)", fontsize=14, fontweight="bold", pad=14)
    plt.tight_layout()
    safe_savefig(fig, fig_mae_path)
    plt.close()
    print(f">>> [Figure 4 (Chart B) Saved] {fig_mae_path} (PNG)")

    # ==========================================
    # 9. 绘制图 5: MSE 专属图 (平方误差 e^2 概率密度分布与直方图)
    # ==========================================
    fig_error_mse_path = os.path.join(output_dir, "error_mse_pdf.png")
    fig, ax = plt.subplots(figsize=(9.5, 5.5), dpi=300)

    color_hist_se = "#d95f02"   # 醒目红橙色
    color_kde_se = "#800026"    # 深红 KDE

    # 截断在 99% 分位数避免极度离群值压缩主体视觉
    clip_se = float(np.percentile(sample_se, 99.0))
    bins_se = np.linspace(0, clip_se, 220)

    ax.hist(sample_se, bins=bins_se, density=True, rwidth=0.82,
            color=color_hist_se, alpha=0.48, edgecolor="#ffffff", linewidth=0.25,
            label="Squared Error Histogram (220 Bins)")

    # 经验 KDE 曲线
    kde_x_se = np.linspace(0, clip_se, 400)
    kde_se = stats.gaussian_kde(sample_se)
    ax.plot(kde_x_se, kde_se(kde_x_se), color=color_kde_se, linewidth=1.5, alpha=0.9, label="Empirical PDF (KDE)")

    # 标定实测 MSE (均值) 与中位数
    ax.axvline(overall_mse, color="#e41a1c", linestyle="--", linewidth=2.0, 
               label=f"Overall Test MSE = {overall_mse:.4f}")
    ax.axvline(mse_median, color="#2b5c8f", linestyle=":", linewidth=1.8, 
               label=f"Median Squared Error = {mse_median:.5f}")

    ax.set_xlabel("Squared Error $e^2 = (\hat{y} - y)^2$", fontsize=13, fontweight="bold", labelpad=8)
    ax.set_ylabel("Probability Density $p(e^2)$", fontsize=13, fontweight="bold", labelpad=8)
    ax.set_xlim(0, clip_se)
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.tick_params(axis="both", labelsize=11)

    stat_text_se = (
        f"Model: CALF (AL 96→96)\n"
        f"Overall Test MSE: {overall_mse:.5f}\n"
        f"Sample Mean $e^2$: {mse_mean:.5f}\n"
        f"Median $e^2$: {mse_median:.5f}\n"
        f"Std($e^2$): {mse_std:.5f}\n"
        f"90% Quantile: {mse_p90:.4f}\n"
        f"95% Quantile: {mse_p95:.4f}\n"
        f"99% Quantile: {mse_p99:.4f}\n"
        f"Skewness: {mse_skew:.3f} (Heavy Tail)"
    )
    ax.text(0.97, 0.95, stat_text_se, transform=ax.transAxes, fontsize=10.0,
            verticalalignment="top", horizontalalignment="right",
            bbox=dict(boxstyle="round,pad=0.5", fc="#fdfdfd", ec="#cccccc", lw=0.8, alpha=0.95))

    ax.legend(loc="upper right", bbox_to_anchor=(0.68, 0.96), frameon=True, framealpha=0.92, facecolor="#ffffff", edgecolor="#d0d0d0", fontsize=9.5)
    plt.title("Dedicated Mean Squared Error (MSE, $e^2$) Distribution & Density", fontsize=14, fontweight="bold", pad=14)
    plt.tight_layout()
    safe_savefig(fig, fig_error_mse_path)
    plt.close()
    print(f">>> [Figure 5 (MSE Dedicated) Saved] {fig_error_mse_path} (PNG)")

    # ==========================================
    # 10. 绘制图 6: MAE 专属图 (绝对误差 |e| 概率密度分布与直方图)
    # ==========================================
    fig_error_mae_path = os.path.join(output_dir, "error_mae_pdf.png")
    fig, ax = plt.subplots(figsize=(9.5, 5.5), dpi=300)

    color_hist_ae = "#2ca02c"   # 学术草绿
    color_kde_ae = "#005a32"    # 深墨绿 KDE

    clip_ae = float(np.percentile(sample_ae, 99.0))
    bins_ae = np.linspace(0, clip_ae, 220)

    ax.hist(sample_ae, bins=bins_ae, density=True, rwidth=0.82,
            color=color_hist_ae, alpha=0.48, edgecolor="#ffffff", linewidth=0.25,
            label="Absolute Error Histogram (220 Bins)")

    # 经验 KDE 曲线
    kde_x_ae = np.linspace(0, clip_ae, 400)
    kde_ae = stats.gaussian_kde(sample_ae)
    ax.plot(kde_x_ae, kde_ae(kde_x_ae), color=color_kde_ae, linewidth=1.5, alpha=0.9, label="Empirical PDF (KDE)")

    # 标定实测 MAE (均值) 与中位数
    ax.axvline(overall_mae, color="#e41a1c", linestyle="--", linewidth=2.0, 
               label=f"Overall Test MAE = {overall_mae:.4f}")
    ax.axvline(mae_median, color="#2b5c8f", linestyle=":", linewidth=1.8, 
               label=f"Median Abs Error = {mae_median:.5f}")

    ax.set_xlabel("Absolute Error $|e| = |\hat{y} - y|$", fontsize=13, fontweight="bold", labelpad=8)
    ax.set_ylabel("Probability Density $p(|e|)$", fontsize=13, fontweight="bold", labelpad=8)
    ax.set_xlim(0, clip_ae)
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.tick_params(axis="both", labelsize=11)

    stat_text_ae = (
        f"Model: CALF (AL 96→96)\n"
        f"Overall Test MAE: {overall_mae:.5f}\n"
        f"Sample Mean $|e|$: {mae_mean:.5f}\n"
        f"Median Abs Error: {mae_median:.5f}\n"
        f"Std($|e|$): {mae_std:.5f}\n"
        f"90% Quantile: {mae_p90:.4f}\n"
        f"95% Quantile: {mae_p95:.4f}\n"
        f"99% Quantile: {mae_p99:.4f}\n"
        f"Skewness: {mae_skew:.3f}"
    )
    ax.text(0.97, 0.95, stat_text_ae, transform=ax.transAxes, fontsize=10.0,
            verticalalignment="top", horizontalalignment="right",
            bbox=dict(boxstyle="round,pad=0.5", fc="#fdfdfd", ec="#cccccc", lw=0.8, alpha=0.95))

    ax.legend(loc="upper right", bbox_to_anchor=(0.68, 0.96), frameon=True, framealpha=0.92, facecolor="#ffffff", edgecolor="#d0d0d0", fontsize=9.5)
    plt.title("Dedicated Mean Absolute Error (MAE, $|e|$) Distribution & Density", fontsize=14, fontweight="bold", pad=14)
    plt.tight_layout()
    safe_savefig(fig, fig_error_mae_path)
    plt.close()
    print(f">>> [Figure 6 (MAE Dedicated) Saved] {fig_error_mae_path} (PNG)")

    # ==========================================
    # 11. 绘制图 7: 残差的绝对值的图 (Residual Absolute Value |e| 直方图与累计覆盖)
    # ==========================================
    fig_abs_path = os.path.join(output_dir, "residual_abs_pdf.png")
    fig, ax = plt.subplots(figsize=(9.5, 5.5), dpi=300)

    color_hist_abs = "#7570b3"  # 典雅紫罗兰色
    color_kde_abs = "#381a61"   # 深紫 KDE

    clip_abs = float(np.percentile(sample_ae, 99.0))
    bins_abs = np.linspace(0, clip_abs, 220)

    ax.hist(sample_ae, bins=bins_abs, density=True, rwidth=0.82,
            color=color_hist_abs, alpha=0.48, edgecolor="#ffffff", linewidth=0.25,
            label="Residual Absolute Value Histogram (220 Bins)")

    # 经验 KDE 曲线
    kde_abs = stats.gaussian_kde(sample_ae)
    ax.plot(kde_x_ae, kde_abs(kde_x_ae), color=color_kde_abs, linewidth=1.5, alpha=0.9, label="Empirical PDF (KDE)")

    # 高亮标出在 [0, MAE] 内的高密度集中区间
    ax.axvspan(0, overall_mae, color="#bcbddc", alpha=0.35,
               label=f"Central Concentration Zone [0, MAE] (Coverage: {mae_cov:.1f}%)")
    ax.axvline(overall_mae, color="#542788", linestyle="--", linewidth=1.8, 
               label=f"MAE Mean = {overall_mae:.4f}")
    ax.axvline(mae_median, color="#1b9e77", linestyle=":", linewidth=1.8, 
               label=f"Median (|e|) = {mae_median:.4f}")

    ax.set_xlabel("Prediction Residual Absolute Value $|e| = |\hat{y} - y|$", fontsize=13, fontweight="bold", labelpad=8)
    ax.set_ylabel("Probability Density $p(|e|)$", fontsize=13, fontweight="bold", labelpad=8)
    ax.set_xlim(0, clip_abs)
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.tick_params(axis="both", labelsize=11)

    stat_text_abs = (
        f"Residual Absolute Value ($|e|$):\n"
        f"Mean ($E[|e|]$ = MAE): {overall_mae:.5f}\n"
        f"Median ($Q_2$): {mae_median:.5f}\n"
        f"Std($|e|$): {mae_std:.5f}\n"
        f"75% Quartile ($Q_3$): {np.percentile(sample_ae, 75.0):.4f}\n"
        f"90% Quantile: {mae_p90:.4f}\n"
        f"95% Quantile: {mae_p95:.4f}\n"
        f"Coverage in [0, MAE]: {mae_cov:.2f}%\n"
        f"Coverage in [0, RMSE]: {rmse_cov:.2f}%"
    )
    ax.text(0.97, 0.95, stat_text_abs, transform=ax.transAxes, fontsize=10.0,
            verticalalignment="top", horizontalalignment="right",
            bbox=dict(boxstyle="round,pad=0.5", fc="#fdfdfd", ec="#cccccc", lw=0.8, alpha=0.95))

    ax.legend(loc="upper right", bbox_to_anchor=(0.68, 0.96), frameon=True, framealpha=0.92, facecolor="#ffffff", edgecolor="#d0d0d0", fontsize=9.5)
    plt.title("Prediction Residual Absolute Value ($|e|$) Distribution & Coverage Band", fontsize=14, fontweight="bold", pad=14)
    plt.tight_layout()
    safe_savefig(fig, fig_abs_path)
    plt.close()
    print(f">>> [Figure 7 (Absolute Residual) Saved] {fig_abs_path} (PNG)")

    print("\n" + "=" * 80)
    print(">>> All residual and error analyses completed successfully!")
    print(f"Figures exported to: {output_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
