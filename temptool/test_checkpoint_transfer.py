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

print("1. Creating model_137 (enc_in=137)...", flush=True)
cfg137 = Configs(137)
model_137 = Model(cfg137, device=device)
state_137 = model_137.state_dict()

# Print which parameters depend on enc_in
print("\nChecking all parameter shapes in state_dict:", flush=True)
enc_in_params = []
for k, v in state_137.items():
    if 137 in v.shape:
        enc_in_params.append((k, v.shape))
        print(f"  -> Param depending on enc_in (137): {k}, shape = {v.shape}", flush=True)

print(f"\nTotal parameters in model: {len(state_137)}, parameters depending on enc_in: {len(enc_in_params)}", flush=True)

print("\n2. Creating model_29 (enc_in=29)...", flush=True)
cfg29 = Configs(29)
model_29 = Model(cfg29, device=device)

print("\n3. Testing direct strict load (expected to fail):", flush=True)
try:
    model_29.load_state_dict(state_137, strict=True)
    print("Direct strict load succeeded!", flush=True)
except Exception as e:
    print(f"Direct strict load failed as expected:\n  {e}", flush=True)

print("\n4. Testing slice cycle_queue load (adapting cycle_queue to 29):", flush=True)
state_adapted = state_137.copy()
state_adapted['cycle_queue'] = state_137['cycle_queue'][:, :29]
try:
    model_29.load_state_dict(state_adapted, strict=True)
    print("Cycle queue sliced load succeeded with strict=True!", flush=True)
    
    # Test forward pass with enc_in=29
    dummy_x = torch.randn(16, 96, 29)
    cycle_idx = torch.zeros(16, dtype=torch.long)
    future_cycle_idx = torch.zeros(16, 96, dtype=torch.long)
    res = model_29.forecast(dummy_x, cycle_idx, future_cycle_idx)
    print(f"Forward pass output shape: {res['outputs_time'].shape}", flush=True)
except Exception as e:
    print(f"Failed: {e}", flush=True)
