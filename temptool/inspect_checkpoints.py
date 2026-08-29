import os
import torch

ckpt_root = "./checkpoints"
for d in os.listdir(ckpt_root):
    dpath = os.path.join(ckpt_root, d)
    if os.path.isdir(dpath):
        for f in os.listdir(dpath):
            if f.endswith(".pth"):
                fpath = os.path.join(dpath, f)
                sd = torch.load(fpath, map_location='cpu')
                tq_shape = sd.get('in_layer.temporal_query', torch.zeros(0)).shape
                cq_shape = sd.get('cycle_queue', torch.zeros(0)).shape
                print(f"Dir: {d[:40]}.../{f}")
                print(f"   -> temporal_query: {tq_shape}, cycle_queue: {cq_shape}")
