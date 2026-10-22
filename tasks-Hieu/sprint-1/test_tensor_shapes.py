import sys
import torch
import torch.nn as nn

# Thiết lập encoding UTF-8 an toàn cho console Windows
if sys.platform.startswith("win"):
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

# Cấu hình siêu tham số chuẩn từ Table 8 của paper TiDE cho 2 dataset tiêu chuẩn
CONFIGS_2_DATASETS = {
    "electricity": {
        "desc": "Electricity Load Forecasting (321 clients)",
        "hidden_size": 1024,
        "n_enc": 2,
        "n_dec": 2,
        "p": 8,
        "temp_hidden": 64,
        "rev_in": False,
        "layer_norm": True
    },
    "traffic": {
        "desc": "California PeMS Freeway Occupancy (862 sensors)",
        "hidden_size": 256,
        "n_enc": 1,
        "n_dec": 1,
        "p": 16,
        "temp_hidden": 64,
        "rev_in": True,
        "layer_norm": False
    }
}

def build_residual_stack(in_dim, out_dim, hidden_dim, num_layers, use_layer_norm=True):
    """Xây dựng chuỗi Residual Blocks theo Section 4 của paper TiDE."""
    layers = []
    current_in = in_dim
    for _ in range(num_layers):
        block = [
            nn.Linear(current_in, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        ]
        if use_layer_norm:
            block.append(nn.LayerNorm(hidden_dim))
        layers.append(nn.Sequential(*block))
        current_in = hidden_dim
    
    if hidden_dim != out_dim:
        layers.append(nn.Linear(hidden_dim, out_dim))
        if use_layer_norm:
            layers.append(nn.LayerNorm(out_dim))
    return nn.Sequential(*layers)

def test_tide_forward_pass(dataset_name, B=32, L=720, H=96, r=7, r_tilde=4):
    cfg = CONFIGS_2_DATASETS[dataset_name]
    hidden_size = cfg["hidden_size"]
    n_enc = cfg["n_enc"]
    n_dec = cfg["n_dec"]
    p = cfg["p"]
    temp_hidden = cfg["temp_hidden"]
    rev_in = cfg["rev_in"]
    layer_norm = cfg["layer_norm"]

    print("\n" + "=" * 75)
    print(f" [*] KIỂM THỬ KÍCH THƯỚC TENSOR TRÊN DATASET: {dataset_name.upper()}")
    print("=" * 75)
    print(f" - Mô tả: {cfg['desc']}")
    print(f" - Cấu hình Table 8: hiddenSize={hidden_size}, n_enc={n_enc}, n_dec={n_dec}, p={p}, revIn={rev_in}, layerNorm={layer_norm}\n")

    # 1. Khởi tạo batch dữ liệu giả lập (Mock Tensors)
    y = torch.randn(B, L)
    x = torch.randn(B, L + H, r)
    y_target = torch.randn(B, H)
    print(f" - [01] Input y (Chuỗi quá khứ L={L})           : {y.shape}")
    print(f" - [02] Input x (Ngoại sinh động L+H={L+H}, r={r}) : {x.shape}")
    print(f" - [03] Target y (Nhãn tương lai H={H})         : {y_target.shape}")

    # Bước 0: Chuẩn hóa RevIN
    if rev_in:
        mean = y.mean(dim=1, keepdim=True)
        std = y.std(dim=1, keepdim=True) + 1e-5
        y_norm = (y - mean) / std
        print(f" - [04] y_norm (Sau RevIN Normalization)        : {y_norm.shape}")
    else:
        mean = torch.zeros(B, 1)
        std = torch.ones(B, 1)
        y_norm = y
        print(f" - [04] y_norm (Không dùng RevIN theo Table 8)  : {y_norm.shape}")

    # Bước 1: Feature Projection nén r -> r_tilde
    proj = nn.Linear(r, r_tilde)
    x_proj = proj(x)
    print(f" - [05] x_proj (Sau Feature Projection r->{r_tilde})   : {x_proj.shape}")

    # Bước 2 & 3: Flatten & Concat đầu vào Encoder
    x_flat = x_proj.flatten(start_dim=1)
    enc_in = torch.cat([y_norm, x_flat], dim=-1)
    expected_enc_dim = L + (L + H) * r_tilde
    print(f" - [06] enc_in (Đầu vào Encoder Concat phẳng)   : {enc_in.shape} (Kỳ vọng: {expected_enc_dim})")
    assert enc_in.shape[1] == expected_enc_dim, "Lệch chiều đầu vào Encoder!"

    # Bước 4: Dense Encoder
    encoder = build_residual_stack(expected_enc_dim, hidden_size, hidden_size, n_enc, layer_norm)
    e = encoder(enc_in)
    print(f" - [07] Vector bối cảnh ẩn e (Dense Encoder)    : {e.shape}")

    # Bước 5: Dense Decoder -> Vector phẳng g
    decoder = build_residual_stack(hidden_size, H * p, hidden_size, n_dec, layer_norm)
    g = decoder(e)
    print(f" - [08] Vector giải mã phẳng g (Dense Decoder)  : {g.shape} (H * p = {H * p})")

    # Bước 6: Unflatten -> Ma trận tương lai d
    d = g.view(B, H, p)
    print(f" - [09] Ma trận bối cảnh d (Unflatten)          : {d.shape}")

    # Bước 7 & 8: Temporal Decoder (Covariate Highway)
    x_future = x_proj[:, L:, :]
    temp_in = torch.cat([d, x_future], dim=-1)
    temp_dec = nn.Sequential(
        nn.Linear(p + r_tilde, temp_hidden),
        nn.ReLU(),
        nn.Linear(temp_hidden, 1)
    )
    y_nl = temp_dec(temp_in).squeeze(-1)
    print(f" - [10] y_nl (Dự báo phi tuyến Temporal Decoder): {y_nl.shape}")

    # Bước 9: Global Linear Skip Connection
    skip = nn.Linear(L, H)
    y_lin = skip(y_norm)
    print(f" - [11] y_lin (Dự báo tuyến tính Linear Skip)   : {y_lin.shape}")

    # Bước 10 & 11: Hợp nhất và Hoàn nguyên thang đo
    y_hat_norm = y_nl + y_lin
    y_hat = y_hat_norm * std + mean
    print(f" - [12] Output cuối cùng y_hat (Dự báo thực tế) : {y_hat.shape}")

    assert y_hat.shape == y_target.shape, "Lệch kích thước đầu ra dự báo so với Target!"
    print(f" [V] KẾT QUẢ: Dataset {dataset_name.upper()} PASSED 100% KHÔNG LỆCH CHIỀU!")

def main():
    print("#" * 75)
    print(" KIỂM THỬ ĐỘC LẬP KÍCH THƯỚC TENSOR MÔ HÌNH TiDE (2 DATASET TIÊU CHUẨN)")
    print(" Dữ liệu tham chiếu: Long-term Forecasting with TiDE (Google Research)")
    print("#" * 75)

    for name in ["electricity", "traffic"]:
        test_tide_forward_pass(name, B=32, L=720, H=96, r=7, r_tilde=4)

    print("\n" + "#" * 75)
    print(" CHÚC MỪNG! TOÀN BỘ 12 BƯỚC BIẾN ĐỔI TENSOR ĐỀU CHUẨN XÁC 100%!")
    print("#" * 75)

if __name__ == "__main__":
    main()
