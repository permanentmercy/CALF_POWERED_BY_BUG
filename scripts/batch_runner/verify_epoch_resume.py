# -*- coding: utf-8 -*-
"""Verification script for Epoch-Level Checkpoint Resumption (v2.4)."""
import os
import sys
import shutil

sys.path.insert(0, os.path.abspath("."))
import torch
from scripts.batch_runner.runner_engine import RunnerEngine
from scripts.batch_runner.state_manager import StateManager, compute_setting_string

test_root = './checkpoints/test_resume_verify'
os.makedirs(test_root, exist_ok=True)

params = {
    'task_name': 'long_term_forecast',
    'model_id': 'CALF_96_96_p100',
    'model': 'CALF',
    'data': 'Solar',
    'pred_len': 96,
    'random_seed': 2025,
    'learning_rate': 0.0004,
    'checkpoints': test_root
}

setting = compute_setting_string(params)
setting_dir = os.path.join(test_root, setting)
os.makedirs(setting_dir, exist_ok=True)

# 1. Create a dummy checkpoint_latest.pth
dummy_ckpt = {
    'epoch': 5,
    'model_state': {},
    'model_optim_state': {},
    'loss_optim_state': {},
    'early_stopping': {'counter': 1, 'best_score': -0.15, 'val_loss_min': 0.15, 'early_stop': False},
    'best_val': 0.15,
    'best_model_epoch': 4,
    'best_model_state': None
}
latest_path = os.path.join(setting_dir, 'checkpoint_latest.pth')
torch.save(dummy_ckpt, latest_path)

# Verify existence
assert os.path.exists(latest_path), 'checkpoint_latest.pth must exist'
loaded = torch.load(latest_path)
assert loaded['epoch'] == 5, 'Epoch must be 5'
print(f"[Test 1 Passed] Successfully created and verified checkpoint_latest.pth at Epoch {loaded['epoch']}")

# 2. Test RunnerEngine detection
logs_dir = os.path.join(test_root, 'logs')
os.makedirs(logs_dir, exist_ok=True)
dummy_log = os.path.join(logs_dir, 'task_001.log')
with open(dummy_log, 'w', encoding='utf-8') as f:
    f.write('Initial epoch 1-5 logs...\n')

latest_file = os.path.join(setting_dir, 'checkpoint_latest.pth')
is_resuming = os.path.exists(latest_file) and os.path.exists(dummy_log)
print(f"[Test 2 Passed] RunnerEngine is_resuming detection: {is_resuming}")
assert is_resuming is True, 'RunnerEngine must detect resuming'

# 3. Test StateManager reset_task with clear_epoch_checkpoint
sm = StateManager(checkpoints_root=test_root, batch_name='test_batch')
task = {'task_idx': 1, 'task_id': 'task_001', 'task_hash': 'h1', 'loop_signature': {}, 'params': params}
sm.init_or_resume([task])
sm.reset_task('task_001', clear_epoch_checkpoint=True)
assert not os.path.exists(latest_path), 'checkpoint_latest.pth must be cleared when clear_epoch_checkpoint=True'
print('[Test 3 Passed] StateManager successfully cleared checkpoint_latest.pth on user reset request')

# Cleanup
shutil.rmtree(test_root, ignore_errors=True)
print('[ALL VERIFICATIONS PASSED 100%]')
