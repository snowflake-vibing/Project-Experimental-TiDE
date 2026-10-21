# PHẦN 2: GIẢI PHẪU LUỒNG FORWARD PASS (PAPER VS CODE PYTORCH)

> 👤 **Người thực hiện:** Hiệu  
> 🤖 **Trợ lý kỹ thuật:** Antigravity  
> 📄 **Tài liệu tham chiếu:** *Long-term Forecasting with TiDE: Time-series Dense Encoder* (Google Research, 2023) - Section 4

---

## 1. Bản Chất Luồng Forward Pass Của TiDE

Hàm `forward(y, x, a)` là trái tim của mô hình TiDE, mô tả chính xác quá trình biến đổi dữ liệu từ 3 nguồn đầu vào cho tới khi sinh ra vector dự báo tương lai $\hat{y}$.

Dưới đây là bảng đối chiếu song song giữa **Công thức toán học trong bài báo (Section 4)** và **Từng dòng code PyTorch thực thi**:

---

## 2. Bảng Đối Chiếu Song Song: Paper Formula vs. PyTorch Code

| Bước | Thành Phần | Công Thức Toán Học Trong Paper | Dòng Code PyTorch Triển Khai | Giải Thích Kỹ Thuật |
| :---: | :--- | :--- | :--- | :--- |
| **0** | **RevIN Normalization** | $y_{\text{norm}} = \frac{y - \mu}{\sigma}$ | `y_norm = self.revin(y, mode='norm')` | Tính trung bình $\mu$ và độ lệch chuẩn $\sigma$ của chuỗi $y_{1:L}$, đưa về $\text{Mean}=0, \text{Std}=1$. |
| **1** | **Feature Projection** | $\tilde{x}_t = \text{ResidualBlock}(x_t) \in \mathbb{R}^{\tilde{r}}$ | `x_proj = self.feature_projection(x)` | Nén số chiều ngoại sinh từ $r$ xuống $\tilde{r}=4$ tại mỗi bước $t \in \{1, \dots, L+H\}$. |
| **2** | **Flatten Covariates** | $\text{vec}(\tilde{x}_{1:L+H}) \in \mathbb{R}^{(L+H)\tilde{r}}$ | `x_flat = x_proj.flatten(start_dim=1)` | Trải phẳng ma trận `(B, L+H, 4)` thành vector 1D `(B, (L+H)*4)`. |
| **3** | **Concat Đầu Vào Encoder** | $u_{\text{enc}} = [ y_{\text{norm}} ; \text{vec}(\tilde{x}) ; a ]$ | `enc_in = torch.cat([y_norm, x_flat, a], dim=-1)` | Ghép 3 nguồn dữ liệu quá khứ thành vector đầu vào Encoder có kích thước $L + (L+H)\tilde{r} + p_{\text{static}}$. |
| **4** | **Dense Encoder** | $e = \text{Encoder}(u_{\text{enc}}) \in \mathbb{R}^{\text{hiddenSize}}$ | `e = self.encoder(enc_in)` | Đi qua $n_e$ khối Dense Residual Blocks để học vector bối cảnh toàn cục $e$. |
| **5** | **Dense Decoder** | $g = \text{Decoder}(e) \in \mathbb{R}^{H \cdot p}$ | `g = self.decoder(e)` | Đi qua $n_d$ khối Dense Residual Blocks, xuất ra vector phẳng $g$ có kích thước $H \cdot p$. |
| **6** | **Unflatten / Reshape** | $D = \text{Unflatten}(g) \in \mathbb{R}^{H \times p}$ | `d = g.view(B, H, p)` | Tái cấu trúc vector $g$ thành tensor 3D `(B, H, p)` gồm $H$ vector lát cắt $d_t \in \mathbb{R}^p$. |
| **7** | **Stack Ngoại Sinh Tương Lai** | $[ d_t ; \tilde{x}_{L+t} ]$ tại mỗi bước $t$ | `x_future = x_proj[:, L:, :]`<br>`temp_in = torch.cat([d, x_future], dim=-1)` | Ghép vector $d_t$ với ngoại sinh tương lai $\tilde{x}_{L+t}$ dọc theo chiều feature $\to$ tạo tensor `(B, H, p + 4)`. |
| **8** | **Temporal Decoder** | $\hat{y}_{\text{NL}, t} = \text{TempDec}([ d_t ; \tilde{x}_{L+t} ])$ | `y_nl = self.temporal_decoder(temp_in).squeeze(-1)` | Khối Residual Block thời gian xuất ra giá trị phi tuyến tại từng bước $\to$ Tensor `(B, H)`. |
| **9** | **Global Linear Skip** | $\hat{y}_{\text{Lin}} = W_{\text{skip}} y_{\text{norm}} + b_{\text{skip}}$ | `y_lin = self.linear_skip(y_norm)` | Tầng tuyến tính nối thẳng từ $y_{1:L}$ sang $H$ bước tương lai $\to$ Tensor `(B, H)`. |
| **10** | **Hợp Nhất Dự Báo** | $\hat{y}_{\text{norm}} = \hat{y}_{\text{NL}} + \hat{y}_{\text{Lin}}$ | `y_hat_norm = y_nl + y_lin` | Phép cộng bảo toàn giữa dự báo phi tuyến và dự báo tuyến tính. |
| **11** | **RevIN Denormalization** | $\hat{y} = \hat{y}_{\text{norm}} \cdot \sigma + \mu$ | `y_hat = self.revin(y_hat_norm, mode='denorm')` | Nhân lại $\sigma$ và cộng lại $\mu$ để đưa dự báo về thang đo thực tế ban đầu. |

---

## 3. Mã Nguồn PyTorch Minh Họa Hàm `forward()`

```python
import torch
import torch.nn as nn

class TiDEForwardPass(nn.Module):
    """
    Minh họa trực tiếp hàm forward() chuẩn xác theo Section 4 của Paper TiDE.
    """
    def __init__(self, L, H, r, r_tilde=4, p=4, hidden_size=512, p_static=0):
        super().__init__()
        self.L = L
        self.H = H
        self.p = p
        
        # 1. Feature Projection: Nén ngoại sinh r -> r_tilde
        self.feature_projection = nn.Sequential(
            nn.Linear(r, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, r_tilde)
        )
        
        # 2. Dense Encoder: Nhận [y; x_flat; a] -> hidden_size
        enc_in_dim = L + (L + H) * r_tilde + p_static
        self.encoder = nn.Sequential(
            nn.Linear(enc_in_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size)
        )
        
        # 3. Dense Decoder: Nhận hidden_size -> H * p (vector g)
        self.decoder = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, H * p),
            nn.LayerNorm(H * p)
        )
        
        # 4. Temporal Decoder: Nhận [d_t ; x_future_t] (chiều p + r_tilde) -> 1
        self.temporal_decoder = nn.Sequential(
            nn.Linear(p + r_tilde, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )
        
        # 5. Global Linear Skip Connection: Nhận L -> H
        self.linear_skip = nn.Linear(L, H)

    def forward(self, y, x, a=None):
        B = y.shape[0]
        
        # Bước 0: Chuẩn hóa RevIN cục bộ (Tính mean & std)
        mean = y.mean(dim=1, keepdim=True)
        std = y.std(dim=1, keepdim=True) + 1e-5
        y_norm = (y - mean) / std                               # (B, L)
        
        # Bước 1: Feature Projection (per time-step)
        x_proj = self.feature_projection(x)                     # (B, L+H, r_tilde)
        
        # Bước 2 & 3: Flatten ngoại sinh & Ghép nối đầu vào Encoder
        x_flat = x_proj.flatten(start_dim=1)                    # (B, (L+H) * r_tilde)
        if a is not None:
            enc_in = torch.cat([y_norm, x_flat, a], dim=-1)     # (B, L + (L+H)*r_tilde + p_static)
        else:
            enc_in = torch.cat([y_norm, x_flat], dim=-1)        # (B, L + (L+H)*r_tilde)
            
        # Bước 4: Dense Encoder sinh vector bối cảnh e
        e = self.encoder(enc_in)                                # (B, hidden_size)
        
        # Bước 5: Dense Decoder sinh vector phẳng g
        g = self.decoder(e)                                     # (B, H * p)
        
        # Bước 6: Unflatten / Reshape vector g thành ma trận bối cảnh tương lai d
        d = g.view(B, self.H, self.p)                           # (B, H, p)
        
        # Bước 7 & 8: Ghép với ngoại sinh tương lai & đưa qua Temporal Decoder
        x_future = x_proj[:, self.L:, :]                        # (B, H, r_tilde)
        temp_in = torch.cat([d, x_future], dim=-1)              # (B, H, p + r_tilde)
        y_nl = self.temporal_decoder(temp_in).squeeze(-1)       # (B, H)
        
        # Bước 9: Nhánh tắt tuyến tính toàn cục (Global Linear Skip)
        y_lin = self.linear_skip(y_norm)                        # (B, H)
        
        # Bước 10: Hợp nhất phi tuyến và tuyến tính
        y_hat_norm = y_nl + y_lin                               # (B, H)
        
        # Bước 11: Hoàn nguyên thang đo RevIN
        y_hat = y_hat_norm * std + mean                         # (B, H)
        
        return y_hat
```
