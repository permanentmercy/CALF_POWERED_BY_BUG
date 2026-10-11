import subprocess
import sys
import os

python_exe = sys.executable
al_ckpt = "./checkpoints/AL_pretrained_best_96_96_seed2025.pth"

if not os.path.exists(al_ckpt):
    print(f"Error: checkpoint not found: {al_ckpt}")
    sys.exit(1)

os.makedirs("logs/CALF/Solar", exist_ok=True)

ultra_percents = [1, 0.5, 0.1]

print("==========================================================================================")
print(f">>> [IL Ultra-Few-Shot Fine-Tuning Pipeline] Percents: {ultra_percents}% | Seed: 2025 | Model: CALF")
print("==========================================================================================\n")

for percent in ultra_percents:
    p_str = f"{percent:g}"
    log_file = f"logs/CALF/Solar/opt_CALF_IL_ft_p{p_str}_96_96_2025.log"
    cmd = [
        python_exe, "-u", "run.py",
        "--root_path", "./datasets/Solar/",
        "--data_path", "solar_IL.csv",
        "--checkpoint_path", al_ckpt,
        "--is_training", "1",
        "--task_name", "long_term_forecast",
        "--model_id", "CALF_opt_IL_ft_96_96",
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
        "--percent", p_str,
        "--dropout", "0.1",
        "--enc_in", "106",
        "--c_out", "106",
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
        "--no_save_checkpoint",
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
        "--data_split", "6:2:2"
    ]
    
    print(f"\n==========================================================================================")
    print(f">>> [Starting IL Fine-Tuning] Percent: {p_str}% | Seed: 2025 | Model: CALF | Ckpt: {al_ckpt}")
    print(f"==========================================================================================\n")
    
    with open(log_file, "a", encoding="utf-8") as lf:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        for line in proc.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            lf.write(line)
            lf.flush()
        proc.wait()
        if proc.returncode != 0:
            print(f"Error: Process failed with exit code {proc.returncode}")
            sys.exit(proc.returncode)

        # 确保不保留任何临时微调权重文件
        import shutil
        for d in os.listdir("checkpoints"):
            if d.endswith(f"_p{p_str}") and "IL_ft" in d:
                shutil.rmtree(os.path.join("checkpoints", d), ignore_errors=True)

print("\n>>> All IL Ultra-Few-Shot Experiments (1%, 0.5%, 0.1%) Completed Successfully!")
