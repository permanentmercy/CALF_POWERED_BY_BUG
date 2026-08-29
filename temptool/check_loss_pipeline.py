import os
import sys
import torch
import numpy as np

workspace_dir = os.path.abspath(".")
if workspace_dir not in sys.path:
    sys.path.insert(0, workspace_dir)

from data_provider.data_loader import Dataset_Solar
from utils.cmLoss import cmLoss
from models.CALF import Model

print("=================================================================")
print("1. Checking Dataset_Solar slicing for percent=100, percent=10, percent=5")
print("=================================================================")

for p in [100, 10, 5]:
    ds_train = Dataset_Solar(root_path="./datasets/Solar/", flag='train', data_path='solar_AL.txt', percent=p, cycle=288)
    ds_val = Dataset_Solar(root_path="./datasets/Solar/", flag='val', data_path='solar_AL.txt', percent=p, cycle=288)
    ds_test = Dataset_Solar(root_path="./datasets/Solar/", flag='test', data_path='solar_AL.txt', percent=p, cycle=288)
    
    print(f"--- Percent = {p}% ---")
    print(f"Train samples: {len(ds_train)}, data_x shape: {ds_train.data_x.shape}")
    print(f"Val samples:   {len(ds_val)}, data_x shape: {ds_val.data_x.shape}")
    print(f"Test samples:  {len(ds_test)}, data_x shape: {ds_test.data_x.shape}")
    
    # Check sample shape
    x, y, x_mark, y_mark, (c_idx, fut_c_idx) = ds_train[0]
    print(f"Sample 0 -> x: {x.shape}, y: {y.shape}, c_idx: {c_idx}, fut_c_idx: {fut_c_idx.shape}")

print("\n=================================================================")
print("2. Checking Loss Calculation in cmLoss")
print("=================================================================")
criterion = cmLoss(
    feature_loss="smooth_l1",
    output_loss="smooth_l1",
    task_loss="smooth_l1",
    task_name="long_term_forecast",
    feature_w=0.05,
    output_w=0.214,
    task_w=0.736,
    layer_offset=0
)

# Mock model outputs
B, L, M = 16, 96, 137
outputs = {
    "outputs_text": torch.randn(B, L, M),
    "outputs_time": torch.randn(B, L, M),
    "intermidiate_time": (torch.randn(B, M, 768),),
    "intermidiate_text": (torch.randn(B, M, 768),)
}
batch_y = torch.randn(B, L, M)

total_loss, task_loss, output_loss, feature_loss, task_loss_time, task_loss_text = criterion(outputs, batch_y)

print(f"Total Loss:   {total_loss.item():.6f}")
print(f"Task Loss:    {task_loss.item():.6f} (Time: {task_loss_time.item():.6f}, Text: {task_loss_text.item():.6f})")
print(f"Output Loss:  {output_loss.item():.6f}")
print(f"Feature Loss: {feature_loss.item():.6f}")
weighted_sum = 0.736 * task_loss + 0.214 * output_loss + 0.05 * feature_loss
print(f"Manual Weighted Sum: {weighted_sum.item():.6f}")
assert torch.isclose(total_loss, weighted_sum), "Loss weighting mismatch!"
print("[PASS] Loss computation matches exact weights!")
