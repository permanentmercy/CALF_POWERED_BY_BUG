import subprocess
import sys
import os
import re
import json

python_exe = r"D:\software\conda\envs\calf\python.exe"
log_dir = os.path.join("logs", "CALF", "Solar")
os.makedirs(log_dir, exist_ok=True)

experiments = [
    # 1. Complete the 3-seed evaluation for pred_len=96
    {"seq_len": 96, "pred_len": 96, "seed": 2026},
    {"seq_len": 96, "pred_len": 96, "seed": 2027},
    # 2. Variable prediction horizons (192, 336, 720) with 1 seed (2025)
    {"seq_len": 96, "pred_len": 192, "seed": 2025},
    {"seq_len": 96, "pred_len": 336, "seed": 2025},
    {"seq_len": 96, "pred_len": 720, "seed": 2025},
]

def run_experiment(exp):
    seq_len = exp["seq_len"]
    pred_len = exp["pred_len"]
    seed = exp["seed"]
    model_id = f"CALF_opt_{seq_len}_{pred_len}"
    log_file = os.path.join(log_dir, f"opt_arch_CALF_{seq_len}_{pred_len}_768_4_0.0004_{seed}.log")

    cmd = [
        python_exe, "-u", "run.py",
        "--root_path", "./datasets/Solar/",
        "--data_path", "solar_AL.txt",
        "--is_training", "1",
        "--task_name", "long_term_forecast",
        "--model_id", model_id,
        "--data", "Solar",
        "--seq_len", str(seq_len),
        "--label_len", "0",
        "--pred_len", str(pred_len),
        "--batch_size", "16",
        "--accumulation_steps", "4",
        "--learning_rate", "0.0004",
        "--train_epochs", "20",
        "--patience", "5",
        "--d_model", "768",
        "--n_heads", "4",
        "--d_ff", "3072",
        "--dropout", "0.1",
        "--enc_in", "137",
        "--c_out", "137",
        "--gpt_layers", "1",
        "--itr", "1",
        "--model", "CALF",
        "--cos", "1",
        "--tmax", "10",
        "--r", "8",
        "--lora_alpha", "32",
        "--lora_dropout", "0.1",
        "--num_workers", "0",
        "--feature_w", "0.05",
        "--output_w", "0.214",
        "--task_w", "0.736",
        "--use_swa", "0",
        "--tq_mode", "add",
        "--use_tq_gate", "0",
        "--layer_offset", "0",
        "--e_layers", "1",
        "--test_branch", "time",
        "--mlp_res_w", "0",
        "--bestmodel",
        "--out_mlp_layers", "3",
        "--cycle_prior_before_denorm", "0",
        "--tq_dropout", "0.1",
        "--tq_lr_factor", "5",
        "--use_norm", "1",
        "--tq_inject_target", "q",
        "--gpt2_path", "./models/gpt2",
        "--task_loss", "smooth_l1",
        "--feature_loss", "smooth_l1",
        "--output_loss", "smooth_l1",
        "--random_seed", str(seed),
        "--cycle", "144",
        "--eval_test_every_epoch",
        "--use_conv_emb", "0",
        "--use_tq_time", "0",
        "--task_time_ratio", "0.5",
        "--lora_target_proj", "0"
    ]

    print("\n" + "="*80)
    print(f">>> [EXPERIMENT] seq_len={seq_len}, pred_len={pred_len}, seed={seed}")
    print(f">>> Log: {log_file}")
    print("="*80)

    last_mse = None
    last_mae = None

    with open(log_file, "w", encoding="utf-8") as f:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )
        for line in process.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            f.write(line)
            f.flush()
            # Match final test metric line: mse:0.1666..., mae:0.2033...
            m = re.search(r"mse:([0-9\.]+),\s*mae:([0-9\.]+)", line)
            if m:
                last_mse = float(m.group(1))
                last_mae = float(m.group(2))

        process.wait()
        if process.returncode != 0:
            print(f">>> [ERROR] Experiment failed with return code {process.returncode}")
        else:
            print(f">>> [SUCCESS] Completed seq_len={seq_len}, pred_len={pred_len}, seed={seed}: MSE={last_mse}, MAE={last_mae}")

    return {"seq_len": seq_len, "pred_len": pred_len, "seed": seed, "mse": last_mse, "mae": last_mae}

results = []
# Pre-populate already finished 96-96 seed 2025
results.append({"seq_len": 96, "pred_len": 96, "seed": 2025, "mse": 0.1666637907963884, "mae": 0.20334090347544098})

for exp in experiments:
    res = run_experiment(exp)
    results.append(res)
    # Persist intermediate results
    with open("experiments_summary.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

print("\n" + "="*80)
print(">>> ALL EXPERIMENTS COMPLETED!")
print(json.dumps(results, indent=2))
print("="*80)
