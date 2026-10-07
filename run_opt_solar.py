import subprocess
import sys
import os

python_exe = r"D:\software\conda\envs\calf\python.exe"

cmd = [
    python_exe, "-u", "run.py",
    "--root_path", "./datasets/Solar/",
    "--data_path", "solar_AL.txt",
    "--is_training", "1",
    "--task_name", "long_term_forecast",
    "--model_id", "CALF_opt_96_96",
    "--data", "Solar",
    "--seq_len", "96",
    "--label_len", "0",
    "--pred_len", "96",
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
    "--random_seed", "2025",
    "--cycle", "144",
    "--use_conv_emb", "0",
    "--use_tq_time", "0",
    "--task_time_ratio", "0.5",
    "--lora_target_proj", "0"
]

log_dir = os.path.join("logs", "CALF", "Solar")
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, "opt_arch_CALF_96_96_768_4_0.0004_2025.log")

print(f"Launching training with command:\n{' '.join(cmd)}")
print(f"Logging to: {log_file}")

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

process.wait()
sys.exit(process.returncode)
