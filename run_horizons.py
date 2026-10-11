import subprocess
import sys
import os
import shutil

python_exe = sys.executable
random_seed = 2025
seq_len = 96
horizons = [336, 720]

os.makedirs("logs/CALF/Solar", exist_ok=True)
os.makedirs("checkpoints", exist_ok=True)

print("==========================================================================================")
print(f">>> [Multi-Horizon Training Pipeline] Horizons: {horizons} | Seed: {random_seed} | Model: CALF")
print("==========================================================================================\n")

for pred_len in horizons:
    log_file = f"logs/CALF/Solar/opt_CALF_{seq_len}_{pred_len}_768_4_0.0004_{random_seed}.log"
    cmd = [
        python_exe, "-u", "run.py",
        "--root_path", "./datasets/Solar/",
        "--data_path", "solar_AL.txt",
        "--is_training", "1",
        "--task_name", "long_term_forecast",
        "--model_id", f"CALF_opt_{seq_len}_{pred_len}",
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
        "--random_seed", str(random_seed),
        "--cycle", "144",
        "--eval_test_every_epoch",
        "--use_conv_emb", "0",
        "--use_tq_time", "0",
        "--task_time_ratio", "0.5",
        "--lora_target_proj", "0",
        "--use_dpg", "1",
        "--use_direct_trend", "1",
        "--use_harmonic", "1",
        "--use_cycle_prior", "1",
        "--one_way_distill", "1",
        "--data_split", "6:2:2",
        "--percent", "100"
    ]
    
    print(f"\n==========================================================================================")
    print(f">>> [Starting Training] Horizon: {pred_len} | Seed: {random_seed} | Model: CALF | Percent: 100%")
    print(f"==========================================================================================\n")
    
    with open(log_file, "w", encoding="utf-8") as lf:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        for line in proc.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            lf.write(line)
            lf.flush()
        proc.wait()
        if proc.returncode != 0:
            print(f"Error: Training for pred_len={pred_len} failed with exit code {proc.returncode}")
            sys.exit(proc.returncode)

    # 训练完成，复制一份命名友好的专属 checkpoint 备份
    setting_pattern = f"long_term_forecast_CALF_opt_{seq_len}_{pred_len}_CALF_Solar_"
    saved_ckpts = [d for d in os.listdir("checkpoints") if d.startswith(setting_pattern)]
    if saved_ckpts:
        latest_dir = sorted(saved_ckpts)[-1]
        src_ckpt = os.path.join("checkpoints", latest_dir, "checkpoint.pth")
        dst_ckpt = os.path.join("checkpoints", f"AL_best_{seq_len}_{pred_len}_seed{random_seed}.pth")
        if os.path.exists(src_ckpt):
            shutil.copyfile(src_ckpt, dst_ckpt)
            print(f"\n>>> [Checkpoint Saved & Backed Up]: {dst_ckpt} ({os.path.getsize(dst_ckpt):,} bytes)\n")

print("\n>>> All Multi-Horizon Full Training Experiments (336, 720) Completed Successfully!")
