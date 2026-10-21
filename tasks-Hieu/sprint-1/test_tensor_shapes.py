import os
import sys
import torch
import torch.nn as nn

# Thiết lập encoding UTF-8 an toàn cho console Windows
if sys.platform.startswith("win"):
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

# Import dataset loader thực tế
from dataset_loader import get_dataloader

def run_test_with_real_csv():
    print("=" * 75)
    print(" KIEM TRA LUONG TENSOR TRICH XUAT TRUC TIEP TU FILE CSV BENCHMARK")
    print("=" * 75)

    # 1. Đường dẫn file dữ liệu thực tế
    csv_file = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data", "ETTh1.csv"))
    print(f"[*] File CSV nguon: {csv_file}")
    if not os.path.exists(csv_file):
        raise FileNotFoundError(f"Khong tim thay file {csv_file}")

    # 2. Cau hinh sieu tham so theo dung chuan TiDE paper (Electricity / ETT)
    B = 32          # Batch size
    L = 720         # Lookback window (30 ngay du lieu gio)
    H = 96          # Forecast horizon (4 ngay du bao)
    r_tilde = 4     # temporalWidth = 4 (theo Appendix B.3 TiDE paper)
    p = 4           # decoderOutputDim
    hidden_size = 512

    # 3. Load du lieu that tu file CSV
    loader, dataset = get_dataloader(csv_file, target_col='OT', L=L, H=H, batch_size=B)
    print(f"[*] Tong so dong trong file CSV       : {dataset.total_len}")
    print(f"[*] So luong mau cat duoc (N samples) : {len(dataset)}")
    
    # Lay ra dung 1 batch that tu DataLoader
    batch_y, batch_x, batch_target = next(iter(loader))
    r = batch_x.shape[-1]  # So dac trung ngoai sinh trich xuat tu timestamp
    
    print(f"\n[*] Kich thuoc du lieu thuc te lay tu file CSV:")
    print(f"    [01] batch_y (Lookback qua khu L={L})           : {batch_y.shape}")
    print(f"    [02] batch_x (Ngoai sinh dong toan bo L+H={L+H}): {batch_x.shape} (r={r} dac trung thoi gian)")
    print(f"    [03] batch_target (Ground truth nhan tuong lai) : {batch_target.shape}")
    print("-" * 75)

    # 4. Buoc 0: RevIN Normalization tren batch that
    mean = batch_y.mean(dim=1, keepdim=True)
    std = batch_y.std(dim=1, keepdim=True) + 1e-5
    y_norm = (batch_y - mean) / std
    print(f"[04] y_norm (Sau chuan hoa RevIN)              : {y_norm.shape}")

    # 5. Buoc 1: Feature Projection nén r -> r_tilde = 4
    proj = nn.Linear(r, r_tilde)
    x_proj = proj(batch_x)
    print(f"[05] x_proj (Sau Feature Projection r->4)      : {x_proj.shape}")

    # 6. Buoc 2 & 3: Flatten & Concat dau vao Encoder
    x_flat = x_proj.flatten(start_dim=1)
    enc_in = torch.cat([y_norm, x_flat], dim=-1)
    expected_enc_dim = L + (L + H) * r_tilde
    print(f"[06] x_flat (Ngoai sinh trai phang)            : {x_flat.shape}")
    print(f"[07] enc_in (Dau vao Encoder)                  : {enc_in.shape} (Ky vong: {expected_enc_dim})")
    assert enc_in.shape[1] == expected_enc_dim, "Lệch chiều Encoder!"

    # 7. Buoc 4: Dense Encoder sinh vector boi canh an e
    encoder = nn.Sequential(
        nn.Linear(expected_enc_dim, hidden_size),
        nn.ReLU(),
        nn.Linear(hidden_size, hidden_size),
        nn.LayerNorm(hidden_size)
    )
    e = encoder(enc_in)
    print(f"[08] Vector boi canh an e (Dense Encoder)      : {e.shape}")

    # 8. Buoc 5: Dense Decoder sinh vector giai ma phang g
    decoder = nn.Sequential(
        nn.Linear(hidden_size, hidden_size),
        nn.ReLU(),
        nn.Linear(hidden_size, H * p),
        nn.LayerNorm(H * p)
    )
    g = decoder(e)
    print(f"[09] Vector giai ma phang g (Dense Decoder)    : {g.shape} (H * p = {H * p})")

    # 9. Buoc 6: Unflatten / Reshape vector g thanh ma tran d
    d = g.view(B, H, p)
    print(f"[10] Ma tran tuong lai d (Unflatten)           : {d.shape}")

    # 10. Buoc 7 & 8: Stack ngoai sinh tuong lai & Temporal Decoder
    x_future = x_proj[:, L:, :]
    temp_in = torch.cat([d, x_future], dim=-1)
    temp_dec = nn.Sequential(
        nn.Linear(p + r_tilde, hidden_size // 4),
        nn.ReLU(),
        nn.Linear(hidden_size // 4, 1)
    )
    y_nl = temp_dec(temp_in).squeeze(-1)
    print(f"[11] x_future (Ngoai sinh tuong lai)           : {x_future.shape}")
    print(f"[12] temp_in (Dau vao TempDec)                 : {temp_in.shape}")
    print(f"[13] y_nl (Du bao phi tuyen tu TemporalDecoder): {y_nl.shape}")

    # 11. Buoc 9: Global Linear Skip
    skip = nn.Linear(L, H)
    y_lin = skip(y_norm)
    print(f"[14] y_lin (Du bao tuyen tinh LinearSkip)      : {y_lin.shape}")

    # 12. Buoc 10 & 11: Hop nhat & Hoan nguyen RevIN
    y_hat_norm = y_nl + y_lin
    y_hat = y_hat_norm * std + mean
    print(f"[15] y_hat (Du bao thuc te cuoi cung)          : {y_hat.shape}")

    # 13. Tinh thu MSE Loss giua Du bao (y_hat) va Ground Truth (batch_target)
    loss = nn.functional.mse_loss(y_hat, batch_target)
    print(f"\n[*] Loss MSE khoi tao voi du lieu that tu CSV  : {loss.item():.4f}")

    print("\n" + "=" * 75)
    print(" KET LUAN: DU LIEU THAT TU FILE CSV DA DUOC NAP VA XU LY 100% HOAN HAO!")
    print("=" * 75)

if __name__ == "__main__":
    run_test_with_real_csv()
