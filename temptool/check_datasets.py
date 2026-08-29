import os
import pandas as pd

solar_dir = r"datasets\Solar"
for fname in os.listdir(solar_dir):
    fpath = os.path.join(solar_dir, fname)
    if fname.endswith(".csv"):
        df = pd.read_csv(fpath)
        print(f"CSV {fname}: shape = {df.shape}, columns = {list(df.columns[:5])}...{list(df.columns[-2:])}")
    elif fname.endswith(".txt"):
        with open(fpath, "r", encoding="utf-8") as f:
            first_line = f.readline().strip().split(",")
            print(f"TXT {fname}: columns count = {len(first_line)}")
