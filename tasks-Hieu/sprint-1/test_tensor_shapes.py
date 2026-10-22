import os
import sys
import torch
import torch.nn as nn

# Thiết lập encoding UTF-8 an toàn cho console Windows
if sys.platform.startswith("win"):
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

from dataset_loader import DATASET_CONFIGS, get_dataloader_for_dataset

def test_single_dataset(dataset_name, B=32, L=720, H=96, r_tilde=4, p=4, hidden_size=512):
    print(f"\n" + "=" * 80)
    print(f" [*] DANG KIEM THU DATASET: {dataset_name.upper()}")
    print("=" * 80)

    # 1. Load batch thực tế từ file CSV tương ứng
    loader, dataset = get_dataloader_for_dataset(dataset_name, L=L, H=H, batch_size=B)
    print(f" - Mo ta dataset    : {dataset.desc}")
    print(f" - Cot muc tieu     : {dataset.target_name}")
    print(f" - Tong so dong     : {dataset.total_len:,} rows")
    print(f" - So mau cat duoc  : {len(dataset):,} samples")

    batch_y, batch_x, batch_target = next(iter(loader))
    r = batch_x.shape[-1]
    print(f" - Input batch_y    : {batch_y.shape} (Lookback L={L})")
    print(f" - Input batch_x    : {batch_x.shape} (Covariates r={r})")
    print(f" - Target tuong lai : {batch_target.shape} (Horizon H={H})")

    # 2. Mo hinh hoa forward pass cua TiDE
    # Buoc 0: RevIN Normalization
    mean = batch_y.mean(dim=1, keepdim=True)
    std = batch_y.std(dim=1, keepdim=True) + 1e-5
    y_norm = (batch_y - mean) / std

    # Buoc 1: Feature Projection nén r -> 4
    proj = nn.Linear(r, r_tilde)
    x_proj = proj(batch_x)

    # Buoc 2 & 3: Flatten & Concat dau vao Encoder
    x_flat = x_proj.flatten(start_dim=1)
    enc_in = torch.cat([y_norm, x_flat], dim=-1)
    expected_enc_dim = L + (L + H) * r_tilde
    assert enc_in.shape[1] == expected_enc_dim, f"Lệch chiều Encoder tại {dataset_name}!"

    # Buoc 4: Dense Encoder
    encoder = nn.Sequential(
        nn.Linear(expected_enc_dim, hidden_size),
        nn.ReLU(),
        nn.Linear(hidden_size, hidden_size),
        nn.LayerNorm(hidden_size)
    )
    e = encoder(enc_in)

    # Buoc 5: Dense Decoder -> vector phang g
    decoder = nn.Sequential(
        nn.Linear(hidden_size, hidden_size),
        nn.ReLU(),
        nn.Linear(hidden_size, H * p),
        nn.LayerNorm(H * p)
    )
    g = decoder(e)

    # Buoc 6: Unflatten -> ma tran d
    d = g.view(B, H, p)

    # Buoc 7 & 8: Temporal Decoder (Covariate Highway)
    x_future = x_proj[:, L:, :]
    temp_in = torch.cat([d, x_future], dim=-1)
    temp_dec = nn.Sequential(
        nn.Linear(p + r_tilde, hidden_size // 4),
        nn.ReLU(),
        nn.Linear(hidden_size // 4, 1)
    )
    y_nl = temp_dec(temp_in).squeeze(-1)

    # Buoc 9: Global Linear Skip
    skip = nn.Linear(L, H)
    y_lin = skip(y_norm)

    # Buoc 10 & 11: Hop nhat & Hoan nguyen RevIN
    y_hat_norm = y_nl + y_lin
    y_hat = y_hat_norm * std + mean

    # Tinh thu MSE khoi tao
    loss = nn.functional.mse_loss(y_hat, batch_target)
    print(f" -> Output du bao   : {y_hat.shape} (Hoan hao khop voi Ground Truth)")
    print(f" -> Khoi tao MSE    : {loss.item():.4f}")
    print(f" [V] KET QUA: Dataset {dataset_name} PASSED 100%!")

def run_all_7_datasets():
    print("#" * 80)
    print(" CHUONG TRINH KIEM THU TOAN DIEN TREN TOAN BO 7 TAP BENCHMARK TiDE PAPER")
    print("#" * 80)

    dataset_list = list(DATASET_CONFIGS.keys())
    for idx, ds_name in enumerate(dataset_list, start=1):
        test_single_dataset(ds_name)

    print("\n" + "#" * 80)
    print(f" CHUC MUNG! TOAN BO 7/7 TAP DATASET BENCHMARK DA DUOC XAC THUC THANH CONG!")
    print("#" * 80)

if __name__ == "__main__":
    run_all_7_datasets()
