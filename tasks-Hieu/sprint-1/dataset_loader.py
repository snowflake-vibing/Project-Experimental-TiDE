import os
import sys
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

from time_features import TimeCovariates

# Cấu hình danh mục 7 tập dữ liệu benchmark chuẩn trong bài báo TiDE
DATASET_CONFIGS = {
    'electricity': {
        'filename': 'electricity.csv',
        'target_col': 'OT',           # Nếu đa biến, OT là cột mục tiêu đại diện
        'datetime_col': 'date',
        'freq': 'h',
        'desc': 'Electricity Load Forecasting (321 clients)'
    },
    'traffic': {
        'filename': 'traffic.csv',
        'target_col': 'OT',           # Cột occupancy cuối
        'datetime_col': 'date',
        'freq': 'h',
        'desc': 'California PeMS Freeway Occupancy (862 sensors)'
    },
    'weather': {
        'filename': 'weather.csv',
        'target_col': 'OT',           # Khí hậu Max Planck (21 indicators)
        'datetime_col': 'date',
        'freq': '10min',
        'desc': 'Weather Climatological Indicators (21 variables)'
    },
    'ETTh1': {
        'filename': 'ETTh1.csv',
        'target_col': 'OT',           # Oil Temperature (1-hour)
        'datetime_col': 'date',
        'freq': 'h',
        'desc': 'Electricity Transformer Temperature (Hourly 1)'
    },
    'ETTh2': {
        'filename': 'ETTh2.csv',
        'target_col': 'OT',
        'datetime_col': 'date',
        'freq': 'h',
        'desc': 'Electricity Transformer Temperature (Hourly 2)'
    },
    'ETTm1': {
        'filename': 'ETTm1.csv',
        'target_col': 'OT',           # Oil Temperature (15-min)
        'datetime_col': 'date',
        'freq': '15min',
        'desc': 'Electricity Transformer Temperature (15-min 1)'
    },
    'ETTm2': {
        'filename': 'ETTm2.csv',
        'target_col': 'OT',
        'datetime_col': 'date',
        'freq': '15min',
        'desc': 'Electricity Transformer Temperature (15-min 2)'
    }
}

class UniversalLTSFDataset(Dataset):
    """
    Dataset tổng quát cho cả 7 tập benchmark LTTF (Long-Term Time-Series Forecasting).
    Tự động nhận diện cột mục tiêu, trích xuất đặc trưng thời gian và cắt sliding window.
    """
    def __init__(self, dataset_name, data_dir=None, L=720, H=96, target_col=None):
        super().__init__()
        self.L = L
        self.H = H
        
        if dataset_name not in DATASET_CONFIGS:
            raise ValueError(f"Dataset '{dataset_name}' không hợp lệ. Chọn 1 trong 7 tập: {list(DATASET_CONFIGS.keys())}")
            
        cfg = DATASET_CONFIGS[dataset_name]
        if data_dir is None:
            data_dir = os.path.abspath(os.path.join(current_dir, "..", "..", "data"))
            
        csv_path = os.path.join(data_dir, cfg['filename'])
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Không tìm thấy file: {csv_path}")
            
        # 1. Đọc dữ liệu từ file CSV
        df = pd.read_csv(csv_path)
        
        # 2. Xác định cột mục tiêu (target)
        dt_col = cfg['datetime_col']
        if target_col is None:
            target_col = cfg['target_col']
        if target_col not in df.columns:
            # Nếu không có cột OT thì lấy cột số cuối cùng
            numeric_cols = [c for c in df.columns if c != dt_col]
            target_col = numeric_cols[-1]
            
        self.target_name = target_col
        self.data_values = df[target_col].values.astype(np.float32)
        
        # 3. Trích xuất covariates thời gian bằng TimeCovariates (Google Research chuẩn)
        dti = pd.DatetimeIndex(pd.to_datetime(df[dt_col]))
        time_cov_extractor = TimeCovariates(dti, normalized=True, holiday=False)
        cov_df = time_cov_extractor.get_covariates()
        self.covariates = cov_df.values.astype(np.float32)
        
        self.total_len = len(self.data_values)
        self.num_samples = self.total_len - (L + H) + 1
        self.num_features = self.covariates.shape[1]
        self.desc = cfg['desc']
        
    def __len__(self):
        return self.num_samples
        
    def __getitem__(self, idx):
        y_past = self.data_values[idx : idx + self.L]
        y_future = self.data_values[idx + self.L : idx + self.L + self.H]
        x_all = self.covariates[idx : idx + self.L + self.H]
        
        return (
            torch.from_numpy(y_past),
            torch.from_numpy(x_all),
            torch.from_numpy(y_future)
        )

def get_dataloader_for_dataset(dataset_name, L=720, H=96, batch_size=32, shuffle=False):
    dataset = UniversalLTSFDataset(dataset_name, L=L, H=H)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=True)
    return loader, dataset
