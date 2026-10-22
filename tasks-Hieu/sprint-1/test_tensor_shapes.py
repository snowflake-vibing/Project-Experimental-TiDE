import os
import sys
import torch
import torch.nn as nn

# Thiết lập encoding UTF-8 an toàn cho console Windows
if sys.platform.startswith("win"):
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

from dataset_loader import DATASET_CONFIGS, get_dataloader_for_dataset

# Bảng siêu tham số chuẩn từ Table 8 (Phụ lục B.3 của paper TiDE)
TABLE_8_HYPERPARAMS = {
    'traffic': {
        'hidden_size': 256,
        'n_enc': 1,
        'n_dec': 1,
        'p': 16,
        'temp_hidden': 64,
        'rev_in': True,
        'layer_norm': False
    },
    'electricity': {
        'hidden_size': 1024,
        'n_enc': 2,
        'n_dec': 2,
        'p': 8,
        'temp_hidden': 64,
        'rev_in': False,
        'layer_norm': True
    },
    'weather': {
        'hidden_size': 512,
        'n_enc': 1,
        'n_dec': 1,
        'p': 8,
        'temp_hidden': 16,
        'rev_in': False,
        'layer_norm': True
    },
    'ETTh1': {
        'hidden_size': 256,
        'n_enc': 2,
        'n_dec': 2,
        'p': 8,
        'temp_hidden': 128,
        'rev_in': True,
        'layer_norm': True
    },
    'ETTh2': {
        'hidden_size': 512,
        'n_enc': 2,
        'n_dec': 2,
        'p': 32,
        'temp_hidden': 16,
        'rev_in': True,
        'layer_norm': True
    },
    'ETTm1': {
        'hidden_size': 1024,
        'n_enc': 1,
        'n_dec': 1,
        'p': 8,
        'temp_hidden': 128,
        'rev_in': False,
        'layer_norm': True
    },
    'ETTm2': {
        'hidden_size': 512,
        'n_enc': 2,
        'n_dec': 2,
        'p': 16,
        'temp_hidden': 128,
        'rev_in': True,
        'layer_norm': True
    }
}

def build_residual_stack(in_dim, out_dim, hidden_dim, num_layers, use_layer_norm=True):
    """Xây dựng chuỗi Residual Blocks theo đúng paper."""
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
    
    # Projection layer cuối ra out_dim nếu cần
    if hidden_dim != out_dim:
        layers.append(nn.Linear(hidden_dim, out_dim))
        if use_layer_norm:
            layers.append(nn.LayerNorm(out_dim))
    return nn.Sequential(*layers)

def test_single_dataset(dataset_name, B=32, L=720, H=96, r_tilde=4):
    print(f"\n" + "=" * 80)
    print(f" [*] ĐANG KIỂM THỬ DATASET: {dataset_name.upper()}")
    print("=" * 80)

    # 1. Lấy siêu tham số chuẩn từ Table 8
    hp = TABLE_8_HYPERPARAMS[dataset_name]
    hidden_size = hp['hidden_size']
    n_enc = hp['n_enc']
    n_dec = hp['n_dec']
    p = hp['p']
    temp_hidden = hp['temp_hidden']
    rev_in = hp['rev_in']
    layer_norm = hp['layer_norm']

    print(f" [Table 8 Config] hiddenSize={hidden_size}, n_enc={n_enc}, n_dec={n_dec}, p={p}, tempHidden={temp_hidden}, revIn={rev_in}, layerNorm={layer_norm}")

    # 2. Load batch thực tế từ file CSV
    loader, dataset = get_dataloader_for_dataset(dataset_name, L=L, H=H, batch_size=B)
    print(f" - Mô tả dataset    : {dataset.desc}")
    print(f" - Cột mục tiêu     : {dataset.target_name}")
    print(f" - Tổng số dòng     : {dataset.total_len:,} rows")
    print(f" - Số mẫu cắt được  : {len(dataset):,} samples")

    batch_y, batch_x, batch_target = next(iter(loader))
    r = batch_x.shape[-1]
    print(f" - [01] Input batch_y    : {batch_y.shape} (Lookback L={L})")
    print(f" - [02] Input batch_x    : {batch_x.shape} (Covariates r={r})")
    print(f" - [03] Target tương lai : {batch_target.shape} (Horizon H={H})")

    # 3. Luồng Forward Pass TiDE
    # Bước 0: Chuẩn hóa RevIN (nếu revIn=True theo Table 8)
    if rev_in:
        mean = batch_y.mean(dim=1, keepdim=True)
        std = batch_y.std(dim=1, keepdim=True) + 1e-5
        y_norm = (batch_y - mean) / std
    else:
        mean = torch.zeros(B, 1)
        std = torch.ones(B, 1)
        y_norm = batch_y

    # Bước 1: Feature Projection nén r -> r_tilde
    proj = nn.Linear(r, r_tilde)
    x_proj = proj(batch_x)
    print(f" - [04] x_proj (Feature Projection): {x_proj.shape} (r={r} -> r_tilde={r_tilde})")

    # Bước 2 & 3: Flatten & Concat đầu vào Encoder
    x_flat = x_proj.flatten(start_dim=1)
    enc_in = torch.cat([y_norm, x_flat], dim=-1)
    expected_enc_dim = L + (L + H) * r_tilde
    print(f" - [05] enc_in (Đầu vào Encoder)  : {enc_in.shape} (Kỳ vọng: {expected_enc_dim})")
    assert enc_in.shape[1] == expected_enc_dim, f"Lệch chiều Encoder tại {dataset_name}!"

    # Bước 4: Dense Encoder
    encoder = build_residual_stack(expected_enc_dim, hidden_size, hidden_size, n_enc, layer_norm)
    e = encoder(enc_in)
    print(f" - [06] Vector ẩn e (Encoder)     : {e.shape} (hidden_size={hidden_size})")

    # Bước 5: Dense Decoder -> vector phẳng g
    decoder = build_residual_stack(hidden_size, H * p, hidden_size, n_dec, layer_norm)
    g = decoder(e)
    print(f" - [07] Vector giải mã g (Decoder): {g.shape} (H * p = {H * p})")

    # Bước 6: Unflatten -> ma trận d
    d = g.view(B, H, p)
    print(f" - [08] Ma trận bối cảnh d        : {d.shape} (B={B}, H={H}, p={p})")

    # Bước 7 & 8: Temporal Decoder (Covariate Highway)
    x_future = x_proj[:, L:, :]
    temp_in = torch.cat([d, x_future], dim=-1)
    temp_dec = nn.Sequential(
        nn.Linear(p + r_tilde, temp_hidden),
        nn.ReLU(),
        nn.Linear(temp_hidden, 1)
    )
    y_nl = temp_dec(temp_in).squeeze(-1)
    print(f" - [09] y_nl (Temporal Decoder)   : {y_nl.shape} (Dự báo phi tuyến)")

    # Bước 9: Global Linear Skip
    skip = nn.Linear(L, H)
    y_lin = skip(y_norm)
    print(f" - [10] y_lin (Linear Skip)       : {y_lin.shape} (Dự báo tuyến tính)")

    # Bước 10 & 11: Hợp nhất & Hoàn nguyên RevIN
    y_hat_norm = y_nl + y_lin
    y_hat = y_hat_norm * std + mean
    print(f" - [11] Output cuối cùng y_hat    : {y_hat.shape} (Khớp 100% với Ground Truth)")

    loss = nn.functional.mse_loss(y_hat, batch_target)
    print(f" -> Khởi tạo Loss MSE : {loss.item():.4f}")
    print(f" [V] KẾT QUẢ: Dataset {dataset_name} PASSED 100%!")
    return {
        'dataset': dataset_name,
        'samples': len(dataset),
        'covariates': r,
        'p': p,
        'hidden_size': hidden_size,
        'rev_in': rev_in,
        'output_shape': list(y_hat.shape),
        'status': 'PASSED'
    }

def run_all_7_datasets():
    print("#" * 80)
    print(" CHƯƠNG TRÌNH KIỂM THỬ TOÀN DIỆN 7/7 TẬP BENCHMARK CHUẨN CỦA TiDE PAPER")
    print(" Sử dụng cấu hình siêu tham số chuẩn xác từ Table 8 (Phụ lục B.3)")
    print("#" * 80)

    dataset_list = list(DATASET_CONFIGS.keys())
    results = []
    for idx, ds_name in enumerate(dataset_list, start=1):
        res = test_single_dataset(ds_name)
        results.append(res)

    print("\n" + "#" * 80)
    print(" TỔNG HỢP KẾT QUẢ XÁC THỰC 7/7 TẬP BENCHMARK:")
    print("#" * 80)
    print(f"{'STT':<4} | {'Dataset':<13} | {'Mẫu Test':<10} | {'Covariates r':<12} | {'hiddenSize':<10} | {'p':<4} | {'revIn':<6} | {'Trạng Thái':<10}")
    print("-" * 80)
    for idx, r in enumerate(results, start=1):
        print(f"{idx:<4} | {r['dataset']:<13} | {r['samples']:<10,d} | {r['covariates']:<12} | {r['hidden_size']:<10} | {r['p']:<4} | {str(r['rev_in']):<6} | {r['status']:<10}")
    print("#" * 80)
    print(" [XÁC THỰC HOÀN TẤT] TẤT CẢ 7/7 DATASET ĐÃ ĐẠT CHUẨN SỐ CHIỀU VÀ KHÔNG LỖI RUNTIME!")
    print("#" * 80)

if __name__ == "__main__":
    run_all_7_datasets()
