# PHẦN 3: THEO DÕI KÍCH THƯỚC TENSOR & SCRIPT KIỂM THỬ CHẠY THẬT

> 👤 **Người thực hiện:** Hiệu  
> 🤖 **Trợ lý kỹ thuật:** Antigravity  
> 📄 **Mục tiêu:** Chứng minh luồng dữ liệu thông suốt qua từng nút mạng, không bị lỗi lệch chiều (*dimension mismatch*).

---

## 1. Bảng Kích Thước Tensor Qua Từng Tầng Mạng

Dưới đây là bảng theo dõi kích thước Tensor chi tiết với cấu hình thực tế chuẩn trên tập **Electricity Benchmark**:
- Kích thước batch: $B = 32$
- Chiều dài quá khứ (Lookback): $L = 720$
- Tầm nhìn tương lai (Horizon): $H = 96$
- Số biến ngoại sinh ban đầu: $r = 8$
- Số chiều nén Feature Projection: $\tilde{r} = 4$
- Số chiều giải mã Decoder: $p = 4$
- Số nơ-ron tầng ẩn: $\text{hiddenSize} = 512$
- Thuộc tính tĩnh: $p_{\text{static}} = 0$ (LTTF setting)

| Bước Xử Lý | Tên Tensor | Ký Hiệu Toán Học | Kích Thước Tensor PyTorch (`Shape`) | Diễn Giải Chi Tiết |
| :---: | :--- | :---: | :---: | :--- |
| **Input 1** | Chuỗi quá khứ | $y_{1:L}$ | `torch.Size([32, 720])` | 32 mẫu, mỗi mẫu có 720 bước lịch sử. |
| **Input 2** | Ngoại sinh động | $x_{1:L+H}$ | `torch.Size([32, 816, 8])` | 32 mẫu, $720 + 96 = 816$ bước, mỗi bước có 8 biến thời gian. |
| **RevIN** | Quá khứ chuẩn hóa | $y_{\text{norm}}$ | `torch.Size([32, 720])` | Chuẩn hóa về mean=0, std=1. |
| **Proj** | Ngoại sinh sau nén | $\tilde{x}_{1:L+H}$ | `torch.Size([32, 816, 4])` | Chiếu từ $r=8$ xuống $\tilde{r}=4$ chiều. |
| **Flatten**| Ngoại sinh trải phẳng | $\text{vec}(\tilde{x})$ | `torch.Size([32, 3264])` | $816 \text{ bước} \times 4 = 3264$ chiều phẳng. |
| **Concat** | Đầu vào Encoder | $u_{\text{enc}}$ | `torch.Size([32, 3984])` | $720 (y) + 3264 (\tilde{x}) = 3984$ chiều. |
| **Encoder**| Vector bối cảnh ẩn | $e$ | `torch.Size([32, 512])` | Nén toàn bộ quá khứ thành 512 đặc trưng. |
| **Decoder**| Vector giải mã phẳng | $g$ | `torch.Size([32, 384])` | $H \times p = 96 \times 4 = 384$ chiều. |
| **Unflatten**| Đặc trưng tương lai | $d_{L+1:L+H}$ | `torch.Size([32, 96, 4])` | Tách thành 96 bước, mỗi bước có $p=4$ đặc trưng. |
| **FutCov** | Ngoại sinh tương lai | $\tilde{x}_{L+1:L+H}$ | `torch.Size([32, 96, 4])` | Cắt lấy 96 bước cuối của tensor ngoại sinh. |
| **Stack**  | Đầu vào TempDec | $[d ; \tilde{x}]$ | `torch.Size([32, 96, 8])` | Ghép $d$ ($p=4$) và $\tilde{x}$ ($\tilde{r}=4$) thành $4+4=8$ chiều. |
| **TempDec**| Dự báo phi tuyến | $\hat{y}_{\text{NL}}$ | `torch.Size([32, 96])` | Sinh con số dự báo phi tuyến cho từng bước. |
| **Skip**   | Dự báo tuyến tính | $\hat{y}_{\text{Lin}}$ | `torch.Size([32, 96])` | Ánh xạ tuyến tính từ 720 bước sang 96 bước. |
| **Sum**    | Dự báo chuẩn hóa | $\hat{y}_{\text{norm}}$ | `torch.Size([32, 96])` | Cộng 2 nhánh: $\hat{y}_{\text{NL}} + \hat{y}_{\text{Lin}}$. |
| **Denorm** | Dự báo thực tế | $\hat{y}_{L+1:L+H}$ | `torch.Size([32, 96])` | Nhân lại std và cộng lại mean $\to$ kết quả cuối cùng! |

---

## 2. Script Python Kiểm Thử Chạy Thật (`test_tensor_shapes.py`)

File script kiểm thử độc lập đã được tạo tại `tasks-Hieu/sprint-1/test_tensor_shapes.py`. Bạn có thể mở terminal và chạy trực tiếp để kiểm chứng:

```bash
python tasks-Hieu/sprint-1/test_tensor_shapes.py
```

### Mã nguồn script kiểm thử:

```python
import torch
import torch.nn as nn

def run_test():
    print("=" * 70)
    print(" KIỂM THỬ KÍCH THƯỚC TENSOR (TENSOR SHAPES) - MÔ HÌNH TiDE")
    print("=" * 70)

    # 1. Cấu hình siêu tham số (Setup Hyperparameters)
    B = 32          # Batch size
    L = 720         # Lookback window
    H = 96          # Forecast horizon
    r = 8           # Số biến ngoại sinh ban đầu
    r_tilde = 4     # Số chiều nén Feature Projection (temporalWidth = 4)
    p = 4           # decoderOutputDim
    hidden_size = 512

    print(f"Cấu hình: Batch={B}, L={L}, H={H}, r={r}, r_tilde={r_tilde}, p={p}\n")

    # 2. Khởi tạo dữ liệu giả lập (Mock Tensors)
    y = torch.randn(B, L)
    x = torch.randn(B, L + H, r)
    print(f"[01] Input y (Quá khứ)              : {y.shape}")
    print(f"[02] Input x (Ngoại sinh động)       : {x.shape}")

    # 3. RevIN Normalization
    mean = y.mean(dim=1, keepdim=True)
    std = y.std(dim=1, keepdim=True) + 1e-5
    y_norm = (y - mean) / std
    print(f"[03] y_norm (Sau chuẩn hóa RevIN)   : {y_norm.shape}")

    # 4. Feature Projection: r -> r_tilde
    proj = nn.Linear(r, r_tilde)
    x_proj = proj(x)
    print(f"[04] x_proj (Sau Feature Projection): {x_proj.shape}")

    # 5. Flatten & Concat đầu vào Encoder
    x_flat = x_proj.flatten(start_dim=1)
    enc_in = torch.cat([y_norm, x_flat], dim=-1)
    expected_enc_dim = L + (L + H) * r_tilde
    print(f"[05] x_flat (Ngoại sinh trải phẳng) : {x_flat.shape}")
    print(f"[06] enc_in (Đầu vào Encoder)       : {enc_in.shape} (Kỳ vọng: {expected_enc_dim})")
    assert enc_in.shape[1] == expected_enc_dim, "Lệch chiều đầu vào Encoder!"

    # 6. Dense Encoder: enc_in -> e
    encoder = nn.Sequential(nn.Linear(expected_enc_dim, hidden_size), nn.ReLU())
    e = encoder(enc_in)
    print(f"[07] Vector bối cảnh ẩn e           : {e.shape}")

    # 7. Dense Decoder: e -> g
    decoder = nn.Sequential(nn.Linear(hidden_size, H * p), nn.ReLU())
    g = decoder(e)
    print(f"[08] Vector giải mã phẳng g         : {g.shape} (H * p = {H * p})")

    # 8. Unflatten: g -> d
    d = g.view(B, H, p)
    print(f"[09] Ma trận tương lai d (Unflatten): {d.shape}")

    # 9. Stack ngoại sinh tương lai & Temporal Decoder
    x_future = x_proj[:, L:, :]
    temp_in = torch.cat([d, x_future], dim=-1)
    temp_dec = nn.Linear(p + r_tilde, 1)
    y_nl = temp_dec(temp_in).squeeze(-1)
    print(f"[10] x_future (Ngoại sinh tương lai): {x_future.shape}")
    print(f"[11] temp_in (Đầu vào TempDec)      : {temp_in.shape}")
    print(f"[12] y_nl (Dự báo phi tuyến)        : {y_nl.shape}")

    # 10. Global Linear Skip
    skip = nn.Linear(L, H)
    y_lin = skip(y_norm)
    print(f"[13] y_lin (Dự báo tuyến tính)      : {y_lin.shape}")

    # 11. Hợp nhất & Denormalize
    y_hat_norm = y_nl + y_lin
    y_hat = y_hat_norm * std + mean
    print(f"[14] y_hat (Dự báo thực tế cuối)    : {y_hat.shape}")

    print("\n" + "=" * 70)
    print(" KẾT QUẢ: 100% LUỒNG TENSOR HOÀN HẢO, KHÔNG LỖI LỆCH CHIỀU!")
    print("=" * 70)

if __name__ == "__main__":
    run_test()
```
