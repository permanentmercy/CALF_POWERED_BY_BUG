import os
import numpy as np
import pandas as pd

def load_solar_raw(root_path, data_path):
    file_path = os.path.join(root_path, data_path)
    if data_path.endswith('.csv'):
        df_raw = pd.read_csv(file_path)
        if 'date' in df_raw.columns:
            df_raw = df_raw.drop(columns=['date'])
        elif not np.issubdtype(df_raw.dtypes.iloc[0], np.number):
            df_raw = df_raw.iloc[:, 1:]
    else:
        df_raw = pd.read_csv(file_path, header=None)
        if not np.issubdtype(df_raw.dtypes.iloc[0], np.number):
            df_raw = df_raw.iloc[:, 1:]
    return df_raw

solar_dir = r"datasets\Solar"
for fname in sorted(os.listdir(solar_dir)):
    fpath = os.path.join(solar_dir, fname)
    if os.path.isfile(fpath) and (fname.endswith('.csv') or fname.endswith('.txt')):
        df = load_solar_raw(solar_dir, fname)
        print(f"File: {fname:15s} | shape: {df.shape} | dtype: {df.dtypes.iloc[0]} | all numeric: {np.all([np.issubdtype(t, np.number) for t in df.dtypes])}")
