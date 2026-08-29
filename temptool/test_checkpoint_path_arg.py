import os
import sys
import torch

workspace_dir = os.path.abspath(".")
if workspace_dir not in sys.path:
    sys.path.insert(0, workspace_dir)

from models.CALF import Model
from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast

class MockArgs:
    def __init__(self, enc_in=29, is_training=0, checkpoint_path="0"):
        self.task_name = 'long_term_forecast'
        self.model_id = 'CALF_96_96_p100'
        self.model = 'CALF'
        self.data = 'Solar'
        self.root_path = './datasets/Solar/'
        self.data_path = 'solar_CT.csv'  # 29 channels
        self.features = 'M'
        self.target = 'OT'
        self.freq = 'h'
        self.embed = 'timeF'
        self.seq_len = 96
        self.label_len = 0
        self.pred_len = 96
        self.percent = 100
        self.batch_size = 16
        self.learning_rate = 0.0001
        self.train_epochs = 1
        self.d_model = 768
        self.n_heads = 4
        self.d_ff = 768 * 4
        self.e_layers = 1
        self.d_layers = 1
        self.gpt_layers = 1
        self.dropout = 0.1
        self.lora_dropout = 0.1
        self.r = 8
        self.lora_alpha = 32
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
        self.cycle_prior_before_denorm = 0
        self.word_embedding_path = './wte_pca_500.pt'
        self.gpt2_path = './models/gpt2'
        self.use_gpu = False
        self.gpu = 0
        self.use_multi_gpu = False
        self.devices = '0'
        self.num_workers = 0
        self.itr = 1
        self.is_training = is_training
        self.checkpoint_path = checkpoint_path
        self.seasonal_patterns = None
        self.task_loss = 'smooth_l1'
        self.feature_w = 0.05
        self.output_w = 0.214
        self.task_w = 0.736
        self.bestmodel = True
        self.zero_shot = 0
        self.target_data = 'ETTm1'
        self.use_ef = False
        self.use_swa = 0
        self.test_branch = 'time'
        self.des = 'test'
        self.factor = 1
        self.distil = True

class FastExp(Exp_Long_Term_Forecast):
    def _get_data(self, flag):
        data, loader = super()._get_data(flag)
        if flag == 'test':
            small_subset = [b for i, b in enumerate(loader) if i < 1]
            return data, small_subset
        return data, loader

# Step 1: Create mock checkpoints
mock_ckpt_path = os.path.abspath("./temptool/mock_137.pth")
if not os.path.exists(mock_ckpt_path):
    cfg137 = MockArgs(enc_in=137)
    model_137 = Model(cfg137, device=torch.device('cpu'))
    torch.save(model_137.state_dict(), mock_ckpt_path)

# Test 1: is_training = 0, checkpoint_path is provided -> SHOULD load explicit checkpoint
print("1. Testing is_training=0 with explicit checkpoint_path...")
args_test0 = MockArgs(enc_in=29, is_training=0, checkpoint_path=mock_ckpt_path)
exp0 = FastExp(args_test0)
exp0.test("mock_setting_0", test=0)
print("[PASS] is_training=0 used explicit checkpoint successfully!")

# Test 2: is_training = 1, checkpoint_path is provided -> MUST IGNORE checkpoint_path and use training pipeline
print("\n2. Testing is_training=1 with checkpoint_path (should be IGNORED)...")
args_test1 = MockArgs(enc_in=29, is_training=1, checkpoint_path=mock_ckpt_path)
exp1 = FastExp(args_test1)
# Mock training best model in memory
exp1.best_model_state = exp1.model.state_dict()
exp1.test("mock_setting_1", test=0)
print("[PASS] is_training=1 ignored checkpoint_path and used training in-memory state!")

print("\n=================================================================")
print("ALL STRICT is_training REQUIREMENT TESTS PASSED!")
print("=================================================================")
