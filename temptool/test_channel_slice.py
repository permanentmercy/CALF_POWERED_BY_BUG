import os
import sys
import torch
import numpy as np

workspace_dir = os.path.abspath(".")
if workspace_dir not in sys.path:
    sys.path.insert(0, workspace_dir)

from models.CALF import Model
from types import SimpleNamespace

configs = SimpleNamespace(
    pred_len=96,
    seq_len=96,
    r=8,
    lora_alpha=32,
    lora_dropout=0.1,
    task_name='long_term_forecast',
    gpt2_path='./models/gpt2',
    gpt_layers=1,
    layer_offset=0,
    word_embedding_path='./wte_pca_500.pt',
    d_model=768,
    e_layers=1,
    d_ff=3072,
    cycle=144,
    use_tq_gate=0,
    tq_dropout=0.1,
    tq_mode='add',
    use_tq=1,
    tq_inject_target='q',
    out_mlp_layers=3,
    dropout=0.1,
    enc_in=37,
    c_out=37,
    mlp_res_w=0.0,
    cycle_prior_before_denorm=0,
    use_norm=1
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Model(configs, device=device).to(device)

# Load state dict from a 137 channel checkpoint mock
fake_137_state = {
    'cycle_queue': torch.randn(144, 137)
}
model.load_state_dict(fake_137_state, strict=False)

print("model.cycle_queue shape:", model.cycle_queue.shape)
assert model.cycle_queue.shape == (144, 37)

# Now test forward pass with 137-channel input data (from solar_AL.txt)
batch_x = torch.randn(16, 96, 137).to(device)
cycle_index = torch.zeros(16).long().to(device)
future_cycle_index = torch.zeros(16, 96).long().to(device)

# If model handles slicing internally:
output = model(batch_x[:, :, :configs.enc_in], cycle_index=cycle_index, future_cycle_index=future_cycle_index)
print("Forward with batch_x[:, :, :enc_in] output_time shape:", output['outputs_time'].shape)
assert output['outputs_time'].shape == (16, 96, 37)

print("ALL TESTS PASSED!")
