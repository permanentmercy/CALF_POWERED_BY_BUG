# -*- coding: utf-8 -*-
"""
P1: Main Paired Training Experiment Runner (Solar-AL, 7:1:2 Split)
Compares T0 (use_tq=0) vs T1 (use_tq=1) across seeds [2025, 2026, 2027].
Records dual branch (Time/Text) & diurnal breakdown (Day/Night) metrics.
Automatically purges checkpoints to maintain Zero-Disk Footprint.
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
RESULTS_CSV = ROOT_DIR / "results" / "cycle_mechanism" / "p1_paired_training_results.csv"
LOGS_DIR = ROOT_DIR / "logs" / "cycle_mechanism"
EXECUTION_RESULTS_MD = (
    ROOT_DIR / "public" / "cycle_mechanism" / "v1.0" / "phase2" / "execution_results.md"
)

SEEDS = [2025, 2026, 2027]
CONFIGS = [
    {"group": "T0", "use_tq": 0, "des": "tq0"},
    {"group": "T1", "use_tq": 1, "des": "tq1"},
]

WITHDRAWN_REASON = (
    "P1 toggle training was withdrawn by the user: keep use_tq unchanged. "
    "Existing results are historical; use the fixed-checkpoint TQ diagnostics."
)


def parse_log(log_path: Path) -> dict:
    metrics = {
        "status": "FAILED",
        "time_mse": float("nan"),
        "time_mae": float("nan"),
        "text_mse": float("nan"),
        "text_mae": float("nan"),
        "day_mse": float("nan"),
        "day_mae": float("nan"),
        "night_mse": float("nan"),
        "night_mae": float("nan"),
        "best_epoch": None,
        "train_time_sec": float("nan"),
    }
    if not log_path.exists():
        return metrics

    try:
        content = log_path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        content = log_path.read_text(encoding="latin1", errors="ignore")

    # 1. Dual branch metrics
    dual_match = re.search(
        r">>>\s*\[Dual Branch Metrics\]\s*Time MSE:\s*([0-9\.]+),\s*Time MAE:\s*([0-9\.]+)\s*\|\s*Text MSE:\s*([0-9\.]+),\s*Text MAE:\s*([0-9\.]+)",
        content,
    )
    if dual_match:
        metrics["time_mse"] = float(dual_match.group(1))
        metrics["time_mae"] = float(dual_match.group(2))
        metrics["text_mse"] = float(dual_match.group(3))
        metrics["text_mae"] = float(dual_match.group(4))
        metrics["status"] = "SUCCESS"
    else:
        # Fallback to standard mse/mae
        legacy_matches = re.findall(r"mse:([0-9\.]+),\s*mae:([0-9\.]+)", content)
        if legacy_matches:
            metrics["time_mse"] = float(legacy_matches[-1][0])
            metrics["time_mae"] = float(legacy_matches[-1][1])
            metrics["status"] = "SUCCESS"

    # 2. Diurnal breakdown
    diurnal_match = re.search(
        r">>>\s*\[Diurnal Breakdown \(Time\)\]\s*Day MSE:\s*([0-9\.]+),\s*Day MAE:\s*([0-9\.]+)\s*\|\s*Night MSE:\s*([0-9\.]+),\s*Night MAE:\s*([0-9\.]+)",
        content,
    )
    if diurnal_match:
        metrics["day_mse"] = float(diurnal_match.group(1))
        metrics["day_mae"] = float(diurnal_match.group(2))
        metrics["night_mse"] = float(diurnal_match.group(3))
        metrics["night_mae"] = float(diurnal_match.group(4))

    # 3. Best epoch & training time
    epoch_matches = re.findall(r"Epoch:\s*(\d+)", content)
    if epoch_matches:
        metrics["best_epoch"] = int(epoch_matches[-1])

    train_time_match = re.search(r"average training time:\s*([0-9\.]+)\s*s", content)
    if train_time_match and metrics["best_epoch"]:
        avg_t = float(train_time_match.group(1))
        metrics["train_time_sec"] = avg_t * metrics["best_epoch"]

    return metrics


def check_is_completed(group: str, seed: int) -> bool:
    if not RESULTS_CSV.exists():
        return False
    try:
        df = pd.read_csv(RESULTS_CSV)
        matched = df[
            (df["group"] == group)
            & (df["seed"] == int(seed))
            & (df["status"] == "SUCCESS")
            & (df["time_mse"].notna())
        ]
        return len(matched) > 0
    except Exception:
        return False


def purge_checkpoints():
    ckpt_dir = ROOT_DIR / "checkpoints"
    if not ckpt_dir.exists():
        return
    for item in ckpt_dir.glob("**/*"):
        if item.is_file() and item.suffix in [".pth", ".pt", ".ckpt"]:
            try:
                # Do not delete base wte_pca_500.pt
                if item.name == "wte_pca_500.pt":
                    continue
                item.unlink()
                print(f"[Purge] Deleted weight checkpoint: {item.name}")
            except Exception as e:
                print(f"[Purge Warning] Failed to delete {item.name}: {e}")


def update_results_csv(entry: dict):
    RESULTS_CSV.parent.mkdir(parents=True, exist_ok=True)
    if RESULTS_CSV.exists():
        df = pd.read_csv(RESULTS_CSV)
        # Drop existing entry if present
        df = df[~((df["group"] == entry["group"]) & (df["seed"] == entry["seed"]))]
        df = pd.concat([df, pd.DataFrame([entry])], ignore_index=True)
    else:
        df = pd.DataFrame([entry])
    df.sort_values(by=["seed", "group"], inplace=True)
    df.to_csv(RESULTS_CSV, index=False, encoding="utf-8-sig")
    print(f"[Results] Updated results saved to {RESULTS_CSV}")


def update_markdown_report():
    if not RESULTS_CSV.exists() or not EXECUTION_RESULTS_MD.exists():
        return
    df = pd.read_csv(RESULTS_CSV)

    # 1. Build matrix rows
    matrix_rows = []
    run_idx = 1
    for seed in SEEDS:
        for cfg in CONFIGS:
            grp = cfg["group"]
            use_tq = cfg["use_tq"]
            des = cfg["des"]
            matched = df[(df["group"] == grp) & (df["seed"] == seed)]
            if len(matched) > 0 and matched.iloc[0]["status"] == "SUCCESS":
                row = matched.iloc[0]
                t_sec = f"{row['train_time_sec']:.0f}s" if pd.notna(row['train_time_sec']) else "-"
                b_ep = f"{int(row['best_epoch'])}" if pd.notna(row['best_epoch']) else "-"
                t_mse = f"{row['time_mse']:.6f}" if pd.notna(row['time_mse']) else "-"
                t_mae = f"{row['time_mae']:.6f}" if pd.notna(row['time_mae']) else "-"
                x_mse = f"{row['text_mse']:.6f}" if pd.notna(row['text_mse']) else "-"
                x_mae = f"{row['text_mae']:.6f}" if pd.notna(row['text_mae']) else "-"
                d_mse = f"{row['day_mse']:.6f}" if pd.notna(row['day_mse']) else "-"
                n_mse = f"{row['night_mse']:.6f}" if pd.notna(row['night_mse']) else "-"
                status_str = "**已完成**"
            else:
                t_sec, b_ep, t_mse, t_mae, x_mse, x_mae, d_mse, n_mse = (
                    "-", "-", "-", "-", "-", "-", "-", "-"
                )
                status_str = "待执行"

            matrix_rows.append(
                f"| **Run {run_idx}** | {grp} | {seed} | {use_tq} | `{des}` | {status_str} | {t_sec} | {b_ep} | {t_mse} | {t_mae} | {x_mse} | {x_mae} | {d_mse} | {n_mse} |"
            )
            run_idx += 1

    matrix_table = "\n".join(matrix_rows)

    # 2. Build summary table
    completed_pairs = []
    for seed in SEEDS:
        t0_m = df[(df["group"] == "T0") & (df["seed"] == seed) & (df["status"] == "SUCCESS")]
        t1_m = df[(df["group"] == "T1") & (df["seed"] == seed) & (df["status"] == "SUCCESS")]
        if len(t0_m) > 0 and len(t1_m) > 0:
            completed_pairs.append((t0_m.iloc[0], t1_m.iloc[0]))

    if completed_pairs:
        t0_time_mses = [p[0]["time_mse"] for p in completed_pairs]
        t1_time_mses = [p[1]["time_mse"] for p in completed_pairs]
        diff_mses = [t1 - t0 for t0, t1 in zip(t0_time_mses, t1_time_mses)]

        t0_time_maes = [p[0]["time_mae"] for p in completed_pairs]
        t1_time_maes = [p[1]["time_mae"] for p in completed_pairs]
        diff_maes = [t1 - t0 for t0, t1 in zip(t0_time_maes, t1_time_maes)]

        t0_text_mses = [p[0]["text_mse"] for p in completed_pairs if pd.notna(p[0]["text_mse"])]
        t1_text_mses = [p[1]["text_mse"] for p in completed_pairs if pd.notna(p[1]["text_mse"])]
        diff_text_mses = [t1 - t0 for t0, t1 in zip(t0_text_mses, t1_text_mses)]

        t0_day_mses = [p[0]["day_mse"] for p in completed_pairs if pd.notna(p[0]["day_mse"])]
        t1_day_mses = [p[1]["day_mse"] for p in completed_pairs if pd.notna(p[1]["day_mse"])]
        diff_day_mses = [t1 - t0 for t0, t1 in zip(t0_day_mses, t1_day_mses)]

        def fmt_stat(t0_l, t1_l, d_l):
            if not t0_l or not t1_l:
                return "-", "-", "-", "-", "-"
            m0, s0 = np.mean(t0_l), np.std(t0_l)
            m1, s1 = np.mean(t1_l), np.std(t1_l)
            md = np.mean(d_l)
            rel = (md / m0) * 100.0 if m0 != 0 else 0
            verdict = "显著改善" if md < -0.001 else ("显著退化" if md > 0.001 else "差异无统计学意义")
            return f"{m0:.6f} ± {s0:.6f}", f"{m1:.6f} ± {s1:.6f}", f"{md:+.6f}", f"{rel:+.2f}%", verdict

        s_tmse = fmt_stat(t0_time_mses, t1_time_mses, diff_mses)
        s_tmae = fmt_stat(t0_time_maes, t1_time_maes, diff_maes)
        s_xmse = fmt_stat(t0_text_mses, t1_text_mses, diff_text_mses)
        s_dmse = fmt_stat(t0_day_mses, t1_day_mses, diff_day_mses)

        summary_table = f"""| **主终点：Time MSE (均值±标准差)** | {s_tmse[0]} | {s_tmse[1]} | {s_tmse[2]} | {s_tmse[3]} | **{s_tmse[4]}** |
| **Time MAE (均值±标准差)** | {s_tmae[0]} | {s_tmae[1]} | {s_tmae[2]} | {s_tmae[3]} | **{s_tmae[4]}** |
| **Text MSE (均值±标准差)** | {s_xmse[0]} | {s_xmse[1]} | {s_xmse[2]} | {s_xmse[3]} | **{s_xmse[4]}** |
| **昼间活跃期 Time MSE** | {s_dmse[0]} | {s_dmse[1]} | {s_dmse[2]} | {s_dmse[3]} | **{s_dmse[4]}** |"""
    else:
        summary_table = """| **主终点：Time MSE (均值±标准差)** | 待计算 | 待计算 | 待计算 | 待计算 | 待判定 |
| **Time MAE (均值±标准差)** | 待计算 | 待计算 | 待计算 | 待计算 | 待判定 |
| **Text MSE (均值±标准差)** | 待计算 | 待计算 | 待计算 | 待计算 | 待判定 |
| **昼间活跃期 Time MSE** | 待计算 | 待计算 | 待计算 | 待计算 | 待判定 |"""

    content = EXECUTION_RESULTS_MD.read_text(encoding="utf-8")
    # Replace matrix table
    m_pattern = r"(## 2\. 配对训练执行进度矩阵\s*\n\n\|.*?\|\n\|.*?\|\n)([\s\S]*?)(\n\n---)"
    new_m = f"\\1{matrix_table}\\3"
    content = re.sub(m_pattern, new_m, content)

    # Replace summary table
    s_pattern = r"(## 3\. 配对效应统计汇总表.*?\n\n\|.*?\|\n\|.*?\|\n)([\s\S]*?)(\n\n---)"
    new_s = f"\\1{summary_table}\\3"
    content = re.sub(s_pattern, new_s, content)

    EXECUTION_RESULTS_MD.write_text(content, encoding="utf-8")
    print(f"[Markdown] Updated execution report: {EXECUTION_RESULTS_MD}")


def run_experiment(group: str, use_tq: int, des: str, seed: int):
    # Retain the old protocol below as historical reference; never launch it.
    raise RuntimeError(WITHDRAWN_REASON)

    print("\n" + "=" * 80)
    print(f">>> [P1 RUNNING] Group: {group} (use_tq={use_tq}) | Seed: {seed} | Des: {des}")
    print("=" * 80)

    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    log_file = LOGS_DIR / f"p1_{des}_seed{seed}.log"

    cmd = [
        sys.executable,
        "-u",
        "run.py",
        "--task_name",
        "long_term_forecast",
        "--is_training",
        "1",
        "--model_id",
        "CALF_96_96_p100",
        "--model",
        "CALF",
        "--data",
        "Solar",
        "--root_path",
        "./datasets/Solar/",
        "--data_path",
        "solar_AL.txt",
        "--data_split",
        "7:1:2",
        "--seq_len",
        "96",
        "--label_len",
        "0",
        "--pred_len",
        "96",
        "--percent",
        "100",
        "--batch_size",
        "16",
        "--accumulation_steps",
        "4",
        "--learning_rate",
        "0.0004",
        "--train_epochs",
        "100",
        "--patience",
        "5",
        "--d_model",
        "768",
        "--n_heads",
        "4",
        "--e_layers",
        "1",
        "--d_layers",
        "1",
        "--d_ff",
        "3072",
        "--dropout",
        "0.1",
        "--enc_in",
        "137",
        "--c_out",
        "137",
        "--gpt_layers",
        "1",
        "--itr",
        "1",
        "--cos",
        "1",
        "--tmax",
        "10",
        "--r",
        "8",
        "--lora_alpha",
        "32",
        "--lora_dropout",
        "0.1",
        "--num_workers",
        "0",
        "--task_w",
        "0.736",
        "--feature_w",
        "0.05",
        "--output_w",
        "0.214",
        "--task_loss",
        "smooth_l1",
        "--feature_loss",
        "smooth_l1",
        "--output_loss",
        "smooth_l1",
        "--use_swa",
        "0",
        "--tq_mode",
        "add",
        "--use_tq_gate",
        "0",
        "--layer_offset",
        "0",
        "--test_branch",
        "time",
        "--mlp_res_w",
        "0",
        "--out_mlp_layers",
        "3",
        "--cycle_prior_before_denorm",
        "0",
        "--tq_dropout",
        "0.1",
        "--tq_lr_factor",
        "5",
        "--eval_test_every_epoch",
        "--use_norm",
        "1",
        "--use_direct_trend",
        "1",
        "--tq_inject_target",
        "q",
        "--gpt2_path",
        "./models/gpt2",
        "--cycle",
        "144",
        "--bestmodel",
        "--random_seed",
        str(seed),
        "--des",
        des,
        "--use_tq",
        str(use_tq),
    ]

    t_start = time.time()
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    with log_file.open("w", encoding="utf-8", errors="replace") as lf:
        proc = subprocess.Popen(
            cmd,
            cwd=str(ROOT_DIR),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            env=env,
        )
        for line in proc.stdout:
            try:
                sys.stdout.write(line)
                sys.stdout.flush()
            except Exception:
                pass
            lf.write(line)
            lf.flush()
        proc.wait()

    elapsed = time.time() - t_start
    exit_code = proc.returncode

    if exit_code != 0:
        print(f"[Error] Run failed with exit code {exit_code}")
        entry = {
            "group": group,
            "seed": seed,
            "use_tq": use_tq,
            "des": des,
            "status": "FAILED",
            "time_mse": np.nan,
            "time_mae": np.nan,
            "text_mse": np.nan,
            "text_mae": np.nan,
            "day_mse": np.nan,
            "day_mae": np.nan,
            "night_mse": np.nan,
            "night_mae": np.nan,
            "best_epoch": np.nan,
            "train_time_sec": elapsed,
            "timestamp": datetime.now().isoformat(),
        }
    else:
        metrics = parse_log(log_file)
        if metrics["train_time_sec"] is np.nan or pd.isna(metrics["train_time_sec"]):
            metrics["train_time_sec"] = elapsed
        entry = {
            "group": group,
            "seed": seed,
            "use_tq": use_tq,
            "des": des,
            "status": metrics["status"],
            "time_mse": metrics["time_mse"],
            "time_mae": metrics["time_mae"],
            "text_mse": metrics["text_mse"],
            "text_mae": metrics["text_mae"],
            "day_mse": metrics["day_mse"],
            "day_mae": metrics["day_mae"],
            "night_mse": metrics["night_mse"],
            "night_mae": metrics["night_mae"],
            "best_epoch": metrics["best_epoch"],
            "train_time_sec": metrics["train_time_sec"],
            "timestamp": datetime.now().isoformat(),
        }
        print(
            f"\n>>> [RUN COMPLETED] Status: {metrics['status']} | Time MSE: {metrics['time_mse']:.6f} | "
            f"Time MAE: {metrics['time_mae']:.6f} | Text MSE: {metrics['text_mse']:.6f} | Elapsed: {elapsed:.1f}s"
        )

    update_results_csv(entry)
    purge_checkpoints()
    update_markdown_report()


def main():
    parser = argparse.ArgumentParser(description="P1 Paired Training Suite")
    parser.add_argument("--action", type=str, default="run_all", choices=["run_all", "run_single", "report"])
    parser.add_argument("--group", type=str, default="T0", choices=["T0", "T1"])
    parser.add_argument("--seed", type=int, default=2025)
    args = parser.parse_args()

    parser.error(WITHDRAWN_REASON)

    if args.action == "report":
        update_markdown_report()
        return

    if args.action == "run_single":
        cfg = next(c for c in CONFIGS if c["group"] == args.group)
        run_experiment(args.group, cfg["use_tq"], cfg["des"], args.seed)
        return

    print("==============================================================================")
    print(">>> [P1 Paired Training Suite] Initiating full paired benchmark on Solar-AL...")
    print(f">>> Target Seeds: {SEEDS}")
    print(f">>> Groups: T0 (use_tq=0, baseline) vs T1 (use_tq=1, active TQ)")
    print("==============================================================================")

    for seed in SEEDS:
        for cfg in CONFIGS:
            grp = cfg["group"]
            use_tq = cfg["use_tq"]
            des = cfg["des"]
            if check_is_completed(grp, seed):
                print(f"[Skip] Experiment already completed: Group {grp}, Seed {seed}")
                continue
            run_experiment(grp, use_tq, des, seed)

    print("\n==============================================================================")
    print(">>> [ALL P1 RUNS COMPLETED] Results saved in results/cycle_mechanism/p1_paired_training_results.csv")
    print("==============================================================================")


if __name__ == "__main__":
    main()
