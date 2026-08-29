import os
import sys
import torch
import numpy as np

# Add workspace root to sys.path
workspace_dir = os.path.abspath(".")
if workspace_dir not in sys.path:
    sys.path.insert(0, workspace_dir)

from data_provider.data_factory import data_provider

class Args:
    def __init__(self, data_path, enc_in=137, batch_size=16):
        self.data = 'Solar'
        self.root_path = './datasets/Solar/'
        self.data_path = data_path
        self.task_name = 'long_term_forecast'
        self.seq_len = 96
        self.label_len = 0
        self.pred_len = 96
        self.percent = 100
        self.batch_size = batch_size
        self.features = 'M'
        self.target = 'OT'
        self.embed = 'timeF'
        self.freq = 'h'
        self.seasonal_patterns = None
        self.cycle = 144
        self.task_loss = 'smooth_l1'
        self.num_workers = 0

def test_dataset(file_name, enc_in):
    print(f"\n=======================================================")
    print(f"Testing Dataset: {file_name} (Expected features: {enc_in})")
    print(f"=======================================================")
    
    args = Args(file_name, enc_in=enc_in)
    
    for split in ['train', 'val', 'test']:
        dataset, loader = data_provider(args, split)
        print(f"[{split.upper()}] Dataset samples: {len(dataset)}, DataLoader batches: {len(loader)}")
        
        # Test first batch
        for batch_x, batch_y, batch_x_mark, batch_y_mark, cycle_tuple in loader:
            print(f"  -> batch_x shape:      {batch_x.shape} (Expected: [B, {args.seq_len}, {enc_in}])")
            print(f"  -> batch_y shape:      {batch_y.shape} (Expected: [B, {args.pred_len}, {enc_in}])")
            print(f"  -> batch_x_mark shape: {batch_x_mark.shape} (Zeros: {torch.all(batch_x_mark == 0).item()})")
            print(f"  -> batch_y_mark shape: {batch_y_mark.shape} (Zeros: {torch.all(batch_y_mark == 0).item()})")
            print(f"  -> cycle_index shape:  {cycle_tuple[0].shape}, future_cycle shape: {cycle_tuple[1].shape}")
            
            # Assertions
            assert batch_x.shape == (args.batch_size if split != 'test' else batch_x.shape[0], args.seq_len, enc_in), f"batch_x shape mismatch: {batch_x.shape}"
            assert batch_y.shape == (args.batch_size if split != 'test' else batch_y.shape[0], args.pred_len, enc_in), f"batch_y shape mismatch: {batch_y.shape}"
            assert torch.all(batch_x_mark == 0), "batch_x_mark should be all zeros (no timestamp input)"
            assert torch.all(batch_y_mark == 0), "batch_y_mark should be all zeros (no timestamp input)"
            assert not torch.isnan(batch_x).any(), "NaN found in batch_x!"
            assert not torch.isnan(batch_y).any(), "NaN found in batch_y!"
            break
            
    print(f"[PASS] {file_name} verified successfully on all splits (train/val/test)!")

if __name__ == '__main__':
    # 1. Test baseline solar_AL.txt
    test_dataset('solar_AL.txt', enc_in=137)
    
    # 2. Test target solar_CT.csv
    test_dataset('solar_CT.csv', enc_in=29)
    
    # 3. Test other solar csv datasets
    all_solar_files = {
        'solar_AL.csv': 137,
        'solar_DE.csv': 16,
        'solar_IL.csv': 106,
        'solar_MA.csv': 50,
        'solar_MT.csv': 4,
        'solar_WY.csv': 7,
    }
    print(f"\n=======================================================")
    print("Testing remaining Solar CSV datasets...")
    print(f"=======================================================")
    for fname, dim in all_solar_files.items():
        test_dataset(fname, enc_in=dim)
        
    print("\n=======================================================")
    print("ALL TESTS PASSED SUCCESSFULLY!")
    print("=======================================================")
