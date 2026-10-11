import subprocess
import sys
import os

python_exe = sys.executable
al_ckpt = "./checkpoints/AL_pretrained_best_96_96_seed2025.pth"

if not os.path.exists(al_ckpt):
    print(f"Error: checkpoint not found: {al_ckpt}")
    sys.exit(1)

cmd = [
    python_exe, "-u", "run.py",
    "--root_path", "./datasets/Solar/",
    "--data_path", "solar_IL.csv",
    "--checkpoint_path", al_ckpt,
    "--is_training", "0",
    "--task_name", "long_term_forecast",
    "--model_id", "CALF_opt_96_96_zeroshot",
    "--data", "Solar",
    "--seq_len", "96",
    "--label_len", "0",
    "--pred_len", "96",
    "--batch_size", "16",
    "--d_model", "768",
    "--n_heads", "4",
    "--d_ff", "3072",
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
    "--lora_target_proj", "0",
    "--use_dpg", "1",
    "--use_direct_trend", "1",
    "--use_harmonic", "1",
    "--use_cycle_prior", "1",
    "--one_way_distill", "1",
    "--data_split", "6:2:2"
]

print("==========================================================================================")
print(">>> [Starting Zero-Shot Evaluation] Source: AL 96 Model -> Target: solar_IL.csv (106 Channels)")
print("==========================================================================================\n")

proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
for line in proc.stdout:
    sys.stdout.write(line)
    sys.stdout.flush()
proc.wait()
sys.exit(proc.returncode)
