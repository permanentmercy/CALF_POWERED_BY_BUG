import os
import sys
import torch

workspace_dir = os.path.abspath(".")
if workspace_dir not in sys.path:
    sys.path.insert(0, workspace_dir)

from models.CALF import Model

class Configs:
    def __init__(self, enc_in):
        self.task_name = 'long_term_forecast'
        self.seq_len = 96
        self.pred_len = 96
        self.d_model = 768
        self.d_ff = 768 * 4
        self.n_heads = 4
        self.e_layers = 1
        self.gpt_layers = 1
        self.dropout = 0.1
        self.enc_in = enc_in
        self.c_out = enc_in
        self.cycle = 144
        self.use_tq_gate = 0
        self.tq_dropout = 0.1
        self.tq_mode = 'add'
        self.use_tq = 1
        self.tq_inject_target = 'q'
        self.mlp_res_w = 0
        self.out_mlp_layers = 3
        self.use_norm = 1
        self.r = 8
        self.lora_alpha = 32
        self.lora_dropout = 0.1
        self.word_embedding_path = './wte_pca_500.pt'
        self.gpt2_path = './models/gpt2'
        self.cycle_prior_before_denorm = 0

device = torch.device('cpu')

print("=================================================================")
print("1. Creating checkpoint from 137-channel model (e.g. solar_AL)...")
print("=================================================================")
cfg137 = Configs(137)
model_137 = Model(cfg137, device=device)
state_137 = model_137.state_dict()

print("\n=================================================================")
print("2. Test Case A: Target enc_in=27 (smaller than 137) -> Expect Auto-Slice")
print("=================================================================")
cfg27 = Configs(27)
model_27 = Model(cfg27, device=device)
model_27.load_state_dict(state_137) # Should trigger auto-slice

# Verify forward pass
dummy_x27 = torch.randn(16, 96, 27)
res27 = model_27.forecast(dummy_x27, torch.zeros(16, dtype=torch.long), torch.zeros(16, 96, dtype=torch.long))
print(f"[SUCCESS] model_27 forward pass output shape: {res27['outputs_time'].shape} (Expected: [16, 96, 27])")

print("\n=================================================================")
print("3. Test Case B: Target enc_in=29 (smaller than 137) -> Expect Auto-Slice")
print("=================================================================")
cfg29 = Configs(29)
model_29 = Model(cfg29, device=device)
model_29.load_state_dict(state_137) # Should trigger auto-slice

# Verify forward pass
dummy_x29 = torch.randn(16, 96, 29)
res29 = model_29.forecast(dummy_x29, torch.zeros(16, dtype=torch.long), torch.zeros(16, 96, dtype=torch.long))
print(f"[SUCCESS] model_29 forward pass output shape: {res29['outputs_time'].shape} (Expected: [16, 96, 29])")

print("\n=================================================================")
print("4. Test Case C: Target enc_in=137 from 29-channel ckpt -> Expect Error")
print("=================================================================")
state_29 = model_29.state_dict()
model_137_new = Model(cfg137, device=device)
try:
    model_137_new.load_state_dict(state_29)
    print("[FAIL] Should have raised error!")
except ValueError as e:
    print(f"[SUCCESS] Caught expected ValueError:\n{e}")

print("\n=================================================================")
print("ALL ADAPTIVE LOAD TESTS PASSED!")
print("=================================================================")
