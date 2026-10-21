import os
import sys
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

# Import TimeCovariates từ file time_features.py của Google Research TiDE
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

from time_features import TimeCovariates

class TimeSeriesCSVDataset(Dataset):
    """
    Dataset doc du lieu chuoi thoi gian thuc te tu file CSV (chuan GluonTS / TiDE).
    Trích xuat:
      - y_past: Chuoi muc tieu qua khu (Lookback L)
      - y_future: Chuoi nhan thuc te tuong lai (Ground Truth H)
      - x_all: Toan bo bien ngoai sinh thoi gian (L + H buoc)
    """
    def __init__(self, csv_path, target_col='OT', datetime_col='date', L=720, H=96):
        super().__init__()
        self.L = L
        self.H = H
        
        # 1. Doc file CSV bang pandas
        df = pd.read_csv(csv_path)
        
        # 2. Trich xuat chuoi bien muc tieu (Target time-series)
        self.data_values = df[target_col].values.astype(np.float32)
        
        # 3. Trich xuat cac dac trung ngoai sinh thoi gian bang TimeCovariates (chuan Google Research TiDE)
        dti = pd.DatetimeIndex(pd.to_datetime(df[datetime_col]))
        time_cov_extractor = TimeCovariates(dti, normalized=True, holiday=False)
        cov_df = time_cov_extractor.get_covariates()
        
        self.covariates = cov_df.values.astype(np.float32)  # (N, r=7)
        
        self.total_len = len(self.data_values)
        self.num_samples = self.total_len - (L + H) + 1
        
    def __len__(self):
        return self.num_samples
        
    def __getitem__(self, idx):
        # Cat cua so sliding window:
        # Qua khu: [idx : idx + L]
        # Tuong lai: [idx + L : idx + L + H]
        # Ngoai sinh: [idx : idx + L + H]
        y_past = self.data_values[idx : idx + self.L]
        y_future = self.data_values[idx + self.L : idx + self.L + self.H]
        x_all = self.covariates[idx : idx + self.L + self.H]
        
        return (
            torch.from_numpy(y_past),
            torch.from_numpy(x_all),
            torch.from_numpy(y_future)
        )

def get_dataloader(csv_path, target_col='OT', L=720, H=96, batch_size=32, shuffle=False):
    dataset = TimeSeriesCSVDataset(csv_path, target_col=target_col, L=L, H=H)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=True)
    return loader, dataset
