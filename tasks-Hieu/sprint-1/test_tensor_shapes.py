import sys
import torch
import torch.nn as nn

# Thiết lập encoding UTF-8 an toàn cho console Windows
if sys.platform.startswith("win"):
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

def run_test():
    print("=" * 70)
    print(" KIEM TRA KICH THUOC TENSOR (TENSOR SHAPES) - MO HINH TiDE")
    print("=" * 70)

    # 1. Cau hinh sieu tham so (Setup Hyperparameters)
    B = 32          # Batch size
    L = 720         # Lookback window
    H = 96          # Forecast horizon
    r = 8           # So bien ngoai sinh ban dau
    r_tilde = 4     # So chieu nen Feature Projection (temporalWidth = 4)
    p = 4           # decoderOutputDim
    hidden_size = 512

    print(f"Cau hinh: Batch={B}, L={L}, H={H}, r={r}, r_tilde={r_tilde}, p={p}\n")

    # 2. Khoi tao du lieu gia lap (Mock Tensors)
    y = torch.randn(B, L)
    x = torch.randn(B, L + H, r)
    print(f"[01] Input y (Qua khu)              : {y.shape}")
    print(f"[02] Input x (Ngoai sinh dong)      : {x.shape}")

    # 3. RevIN Normalization
    mean = y.mean(dim=1, keepdim=True)
    std = y.std(dim=1, keepdim=True) + 1e-5
    y_norm = (y - mean) / std
    print(f"[03] y_norm (Sau chuan hoa RevIN)   : {y_norm.shape}")

    # 4. Feature Projection: r -> r_tilde
    proj = nn.Linear(r, r_tilde)
    x_proj = proj(x)
    print(f"[04] x_proj (Sau Feature Projection): {x_proj.shape}")

    # 5. Flatten & Concat dau vao Encoder
    x_flat = x_proj.flatten(start_dim=1)
    enc_in = torch.cat([y_norm, x_flat], dim=-1)
    expected_enc_dim = L + (L + H) * r_tilde
    print(f"[05] x_flat (Ngoai sinh trai phang) : {x_flat.shape}")
    print(f"[06] enc_in (Dau vao Encoder)       : {enc_in.shape} (Ky vong: {expected_enc_dim})")
    assert enc_in.shape[1] == expected_enc_dim, "Lech chieu dau vao Encoder!"

    # 6. Dense Encoder: enc_in -> e
    encoder = nn.Sequential(nn.Linear(expected_enc_dim, hidden_size), nn.ReLU())
    e = encoder(enc_in)
    print(f"[07] Vector boi canh an e           : {e.shape}")

    # 7. Dense Decoder: e -> g
    decoder = nn.Sequential(nn.Linear(hidden_size, H * p), nn.ReLU())
    g = decoder(e)
    print(f"[08] Vector giai ma phang g         : {g.shape} (H * p = {H * p})")

    # 8. Unflatten: g -> d
    d = g.view(B, H, p)
    print(f"[09] Ma tran tuong lai d (Unflatten): {d.shape}")

    # 9. Stack ngoai sinh tuong lai & Temporal Decoder
    x_future = x_proj[:, L:, :]
    temp_in = torch.cat([d, x_future], dim=-1)
    temp_dec = nn.Linear(p + r_tilde, 1)
    y_nl = temp_dec(temp_in).squeeze(-1)
    print(f"[10] x_future (Ngoai sinh tuong lai): {x_future.shape}")
    print(f"[11] temp_in (Dau vao TempDec)      : {temp_in.shape}")
    print(f"[12] y_nl (Du bao phi tuyen)        : {y_nl.shape}")

    # 10. Global Linear Skip
    skip = nn.Linear(L, H)
    y_lin = skip(y_norm)
    print(f"[13] y_lin (Du bao tuyen tinh)      : {y_lin.shape}")

    # 11. Hop nhat & Denormalize
    y_hat_norm = y_nl + y_lin
    y_hat = y_hat_norm * std + mean
    print(f"[14] y_hat (Du bao thuc te cuoi)    : {y_hat.shape}")

    print("\n" + "=" * 70)
    print(" KET QUA: 100% LUONG TENSOR HOAN HAO, KHONG LOI LECH CHIEU!")
    print("=" * 70)

if __name__ == "__main__":
    run_test()
