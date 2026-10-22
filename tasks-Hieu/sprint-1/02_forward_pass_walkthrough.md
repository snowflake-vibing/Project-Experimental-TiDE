# PHẦN 2: GIẢI PHẪU TOÀN DIỆN LUỒNG FORWARD PASS (PAPER VS CODE PYTORCH)

> [!NOTE]
> **Điều hướng tài liệu**: [🏠 Trang Chủ](../../README.md) • [📋 Sprint 1 README](./README.md) • [📜 Yêu Cầu Sprint](./requirement.md) • [📊 Checklist Tiến Độ](./checklist.md) • [🛡️ Tự Kiểm Chứng](./05_self_verification.md)  
> 👤 **Người thực hiện:** Hiệu  
> 🤖 **Trợ lý kỹ thuật:** Antigravity  
> 📄 **Tài liệu tham chiếu:** *Long-term Forecasting with TiDE: Time-series Dense Encoder* (Google Research, 2023) - Section 4

---

## 1. Bản Chất Luồng Forward Pass Của TiDE Là Gì?

Hàm `forward(y, x, a)` là trái tim của mô hình TiDE. Nó mô tả chính xác quá trình **dữ liệu được nạp vào, biến đổi toán học qua từng khối nơ-ron, và sinh ra dự báo tương lai**.

Toàn bộ quy trình forward pass gồm **6 giai đoạn lớn (tương ứng 11 thao tác nguyên tử)**:
1. **Giai đoạn 0:** Chuẩn hóa đầu vào RevIN (chống trôi phân phối).
2. **Giai đoạn 1:** Chiếu giảm chiều ngoại sinh (Feature Projection).
3. **Giai đoạn 2:** Trải phẳng (Flatten) và Mã hóa toàn cục (Dense Encoder).
4. **Giai đoạn 3:** Giải mã bối cảnh (Dense Decoder) và Tái cấu trúc (Unflatten).
5. **Giai đoạn 4:** Làn đường cao tốc ngoại sinh (Temporal Decoder) song song với Nhánh tắt tuyến tính (Global Linear Skip).
6. **Giai đoạn 5:** Hợp nhất dự báo và Hoàn nguyên thang đo (RevIN Denormalization).

> [!TIP]
> **Nguyên tắc đối chiếu học thuật:** Trong bài báo gốc của Google Research, tác giả mô tả một số bước dưới dạng **văn xuôi (prose)** hoặc **sơ đồ khối (Figure 1)** mà không đánh số phương trình riêng. Bảng đối chiếu dưới đây đã **toán học hóa tường minh** từng câu chữ của bài báo để bạn có thể ánh xạ 1-1 trực tiếp sang từng dòng code PyTorch.

---

## 2. Bảng Đối Chiếu Song Song Từng Bước: Paper vs PyTorch Code

| Bước | Phân Khu | Vị Trí Trong Paper | Công Thức Toán Học / Diễn Giải | Dòng Code PyTorch Tương Ứng | Giải Thích Kỹ Thuật |
| :---: | :--- | :--- | :--- | :--- | :--- |
| **0** | **RevIN Norm** | Bảng 8 (Table 8) & Phụ lục B.1 | $y_{\text{norm}} = (y - \mu) / \sigma$ | `y_norm = (y - mean) / std` | Chuẩn hóa chuỗi lịch sử về phân phối chuẩn ($\text{Mean}=0, \text{Std}=1$). |
| **1** | **Feature Proj** | Section 4.1, Công thức (3) | $\tilde{x}_t = \text{Proj}(x_t) \in \mathbb{R}^{\tilde{r}}$ | `x_proj = self.feature_projection(x)` | Nén số chiều ngoại sinh từ $r$ xuống $\tilde{r}=4$ tại mỗi bước thời gian. |
| **2** | **Flatten** | Section 4.1 & Hình 1 (Figure 1) | $\text{vec}(\tilde{x}) \in \mathbb{R}^{(L+H)\tilde{r}}$ | `x_flat = x_proj.flatten(start_dim=1)` | Trải phẳng ma trận ngoại sinh 2D thành một vector 1D duy nhất. |
| **3** | **Concat** | Section 4.1, Công thức (4) | $u_{\text{enc}} = [ y_{\text{norm}} ; \text{vec}(\tilde{x}) ; a ]$ | `enc_in = torch.cat([y_norm, x_flat, a], dim=-1)` | Ghép 3 nguồn dữ liệu làm đầu vào duy nhất cho khối Dense Encoder. |
| **4** | **Dense Enc** | Section 4.1, Công thức (4) | $e = \text{Encoder}(u_{\text{enc}})$ | `e = self.encoder(enc_in)` | Đưa qua $n_e$ khối Residual Blocks tạo vector bối cảnh ẩn $e$. |
| **5** | **Dense Dec** | Section 4.2 (Trang 6) | $g = \text{Decoder}(e) \in \mathbb{R}^{H \cdot p}$ | `g = self.decoder(e)` | Đưa qua $n_d$ khối Residual Blocks tạo vector giải mã phẳng $g$. |
| **6** | **Unflatten** | Section 4.2, Phép Reshape | $D = \text{Reshape}(g) \in \mathbb{R}^{H \times p}$ | `d = g.view(B, H, p)` | Biến đổi vector $g$ thành ma trận gồm $H$ vector lát cắt $d_t$. |
| **7** | **Stack** | Section 4.2 & Hình 1 (Figure 1) | $[ d_t ; \tilde{x}_{L+t} ]$ tại mỗi bước $t$ | `temp_in = torch.cat([d, x_future], dim=-1)` | Ghép vector đặc trưng $d_t$ với ngoại sinh tương lai $\tilde{x}_{L+t}$. |
| **8** | **Temp Dec** | Section 4.2 (Trang 6) | $\hat{y}_{\text{NL}, t} = \text{TempDec}([ d_t ; \tilde{x}_{L+t} ])$ | `y_nl = self.temporal_decoder(temp_in).squeeze(-1)` | Khối nơ-ron theo thời gian sinh ra con số dự báo phi tuyến cho từng bước. |
| **9** | **Linear Skip**| Section 4.2, Đoạn cuối trang 6 | $\hat{y}_{\text{Lin}} = W_{\text{skip}} y_{\text{norm}} + b_{\text{skip}}$ | `y_lin = self.linear_skip(y_norm)` | Tầng tuyến tính ánh xạ trực tiếp từ $L$ bước quá khứ sang $H$ bước tương lai. |
| **10**| **Hợp nhất** | Section 4.2 & Nút (+) Residual | $\hat{y}_{\text{norm}} = \hat{y}_{\text{NL}} + \hat{y}_{\text{Lin}}$ | `y_hat_norm = y_nl + y_lin` | Cộng hai luồng dự báo phi tuyến và tuyến tính. |
| **11**| **RevIN Denorm**| Bảng 8 & Phụ lục B.1 | $\hat{y} = \hat{y}_{\text{norm}} \cdot \sigma + \mu$ | `y_hat = y_hat_norm * std + mean` | Hoàn nguyên thang đo thực tế ban đầu của dữ liệu. |

---

## 3. Giải Phẫu Chi Tiết Từng Bước (Deep-Dive Analysis)

### Bước 0: Chuẩn Hóa Cục Bộ RevIN (Reversible Instance Normalization)

#### 📐 Bản chất toán học:
$$\mu = \frac{1}{L} \sum_{t=1}^L y_t, \quad \sigma = \sqrt{\frac{1}{L} \sum_{t=1}^L (y_t - \mu)^2 + \epsilon}, \quad y_{\text{norm}, t} = \frac{y_t - \mu}{\sigma}$$

#### 💻 Bản chất Code PyTorch:
```python
mean = y.mean(dim=1, keepdim=True)          # Kích thước (B, 1)
std = y.std(dim=1, keepdim=True) + 1e-5     # Kích thước (B, 1)
y_norm = (y - mean) / std                  # Kích thước (B, L)
```

#### 🎯 Tại sao phải làm:
Dữ liệu chuỗi thời gian thực tế thường xuyên bị trôi dạt phân phối theo thời gian (*Distribution Shift* - như lạm phát hoặc tăng trưởng tải theo năm). RevIN đưa dữ liệu của từng chuỗi con về phân phối chuẩn ($\text{Mean}=0, \text{Std}=1$), giúp mô hình không bị bối rối bởi thang đo tuyệt đối mà chỉ tập trung học hình thái dao động và chu kỳ.

---

### Bước 1: Chiếu Giảm Chiều Ngoại Sinh (Feature Projection)

#### 📐 Bản chất toán học:
Áp dụng một khối Residual Block nhỏ độc lập cho từng bước thời gian:
$$\tilde{x}_t = \text{FeatureProjection}(x_t) \in \mathbb{R}^{\tilde{r}} \quad (\text{với } \tilde{r} = 4 \ll r)$$

#### 💻 Bản chất Code PyTorch:
```python
x_proj = self.feature_projection(x)        # Đầu vào (B, L+H, r) -> Đầu ra (B, L+H, 4)
```

#### 🎯 Tại sao phải làm:
Tránh hiện tượng bùng nổ số chiều (*Curse of Dimensionality*). Nếu giữ nguyên số chiều ban đầu $r=20$ với $L+H=1440$, khi trải phẳng sẽ lên tới $28,800$ chiều $\implies$ quá tải bộ nhớ và mạng học vẹt. Nén về $\tilde{r}=4$ giúp giảm hơn 80% gánh nặng tham số mà vẫn giữ trọn thông tin chu kỳ.

---

### Bước 2 & 3: Trải Phẳng (Flatten) & Ghép Nối Đầu Vào Encoder

#### 📐 Bản chất toán học:
$$u_{\text{enc}} = \begin{bmatrix} y_{\text{norm}, 1:L} \\ \text{vec}(\tilde{x}_{1:L+H}) \\ a \end{bmatrix} \in \mathbb{R}^{L + (L+H)\tilde{r} + p_{\text{static}}}$$

#### 💻 Bản chất Code PyTorch:
```python
x_flat = x_proj.flatten(start_dim=1)                # (B, (L+H) * 4)
enc_in = torch.cat([y_norm, x_flat, a], dim=-1)     # (B, L + (L+H)*4 + p_static)
```

#### 🎯 Tại sao phải làm:
Mạng MLP (Multi-Layer Perceptron) cổ điển chỉ nhận vector 1D phẳng. Vì vậy, ta phải trải phẳng ma trận 2D của ngoại sinh và ghép chung với chuỗi quá khứ và thuộc tính tĩnh để đưa vào một cổng duy nhất.

---

### Bước 4: Mã Hóa Dày Đặc (Dense Encoder)

#### 📐 Bản chất toán học:
$$e = \text{DenseEncoder}(u_{\text{enc}}) \in \mathbb{R}^{\text{hiddenSize}}$$

#### 💻 Bản chất Code PyTorch:
```python
e = self.encoder(enc_in)                           # (B, hidden_size) ví dụ hidden_size=512
```

#### 🎯 Tại sao phải làm:
$n_e$ khối Residual Block xếp chồng trong Encoder có nhiệm vụ tổng hợp và chắt lọc toàn bộ tri thức từ quá khứ, biến động ngoại sinh và đặc trưng tĩnh thành một **vector bối cảnh ẩn toàn cục $e$**.

---

### Bước 5: Giải Mã Dày Đặc (Dense Decoder) & Vector $g$

#### 📐 Bản chất toán học:
$$g = \text{DenseDecoder}(e) \in \mathbb{R}^{H \cdot p}$$

#### 💻 Bản chất Code PyTorch:
```python
g = self.decoder(e)                                # (B, H * p) ví dụ 96 * 4 = 384
```

#### 🎯 Tại sao phải làm:
Khối Dense Decoder gồm $n_d$ khối Residual Block nhận vector bối cảnh $e$ và chuyển đổi thành một vector phẳng $g$. Vector $g$ chứa đựng toàn bộ thông tin dự báo bối cảnh cho cả $H$ bước tương lai.

---

### Bước 6: Tái Cấu Trúc (Unflatten) Thành Ma Trận Tương Lai

#### 📐 Bản chất toán học:
$$D = \text{Unflatten}(g) = \begin{bmatrix} d_1, d_2, \dots, d_H \end{bmatrix} \in \mathbb{R}^{H \times p}$$

#### 💻 Bản chất Code PyTorch:
```python
d = g.view(B, self.H, self.p)                     # (B, H, p) ví dụ (B, 96, 4)
```

#### 🎯 Tại sao phải làm:
Vector $g$ là một dải 1D phẳng. Thao tác `view(B, H, p)` chia dải phẳng này thành $H$ phần bằng nhau, mỗi phần là một vector $d_t \in \mathbb{R}^p$ đại diện cho bối cảnh riêng của bước tương lai thứ $t$.

---

### Bước 7 & 8: Đường Cao Tốc Ngoại Sinh & Bộ Giải Mã Thời Gian (Temporal Decoder)

#### 📐 Bản chất toán học:
$$\hat{y}_{\text{NL}, t} = \text{TemporalDecoder} \left( \begin{bmatrix} d_t \\ \tilde{x}_{L+t} \end{bmatrix} \right) \in \mathbb{R} \quad \forall t \in \{1, \dots, H\}$$

#### 💻 Bản chất Code PyTorch:
```python
x_future = x_proj[:, self.L:, :]                   # Cắt lấy H bước tương lai: (B, H, 4)
temp_in = torch.cat([d, x_future], dim=-1)         # Ghép lại: (B, H, p + 4)
y_nl = self.temporal_decoder(temp_in).squeeze(-1)  # Đầu ra: (B, H)
```

#### 🎯 Tại sao phải làm:
Đây là **đột phá lớn nhất của TiDE**. Thay vì để biến tương lai đi xuyên qua Encoder (dễ bị mờ nhạt), TiDE tạo ra một **làn đường cao tốc riêng (Covariate Highway)** đưa thẳng $\tilde{x}_{L+t}$ vào Temporal Decoder. Kết hợp bối cảnh dài hạn $d_t$ với cú sốc tương lai $\tilde{x}_{L+t}$ giúp mô hình phản ứng tức thì với các sự kiện đột xuất (khuyến mãi, bão tuyết).

---

### Bước 9: Nhánh Tắt Tuyến Tính Toàn Cục (Global Linear Skip Connection)

#### 📐 Bản chất toán học:
$$\hat{y}_{\text{Lin}} = W_{\text{skip}} \cdot y_{\text{norm}, 1:L} + b_{\text{skip}} \in \mathbb{R}^H$$

#### 💻 Bản chất Code PyTorch:
```python
y_lin = self.linear_skip(y_norm)                   # Ánh xạ từ (B, L) sang (B, H)
```

#### 🎯 Tại sao phải làm:
Đóng vai trò như một mô hình DLinear thu nhỏ. Nhánh tuyến tính này đảm nhận việc mô hình hóa xu hướng tăng/giảm cơ bản (trend), giải phóng mạng nơ-ron sâu chỉ cần tập trung học các hình thái phi tuyến phức tạp.

---

### Bước 10 & 11: Hợp Nhất Dự Báo & Hoàn Nguyên Thang Đo RevIN

#### 📐 Bản chất toán học:
$$\hat{y}_{\text{norm}} = \hat{y}_{\text{NL}} + \hat{y}_{\text{Lin}}, \quad \hat{y} = \hat{y}_{\text{norm}} \cdot \sigma + \mu$$

#### 💻 Bản chất Code PyTorch:
```python
y_hat_norm = y_nl + y_lin                          # Dự báo chuẩn hóa: (B, H)
y_hat = y_hat_norm * std + mean                    # Dự báo thực tế cuối cùng: (B, H)
```

#### 🎯 Tại sao phải làm:
Cộng kết quả phi tuyến và tuyến tính để ra dự báo chuẩn hóa hoàn chỉnh, sau đó nhân lại độ lệch chuẩn $\sigma$ và cộng lại giá trị trung bình $\mu$ để đưa kết quả về đúng đơn vị thực tế (MW điện năng hoặc tỷ lệ lưu lượng xe).

---

## 4. Mã Nguồn Hoàn Chỉnh Của Lớp `TiDEForwardPass` (Tự Chạy Được)

```python
import torch
import torch.nn as nn

class TiDEForwardPass(nn.Module):
    """
    Triển khai chuẩn xác luồng Forward Pass của mô hình TiDE theo Section 4 bài báo gốc.
    """
    def __init__(self, L, H, r, r_tilde=4, p=4, hidden_size=512, p_static=0):
        super().__init__()
        self.L = L
        self.H = H
        self.p = p
        
        # 1. Khối Feature Projection: Nén số chiều ngoại sinh r -> r_tilde
        self.feature_projection = nn.Sequential(
            nn.Linear(r, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, r_tilde)
        )
        
        # 2. Khối Dense Encoder: Nhận [y_norm; x_flat; a] -> hidden_size
        enc_in_dim = L + (L + H) * r_tilde + p_static
        self.encoder = nn.Sequential(
            nn.Linear(enc_in_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size)
        )
        
        # 3. Khối Dense Decoder: Nhận hidden_size -> H * p (vector phẳng g)
        self.decoder = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, H * p),
            nn.LayerNorm(H * p)
        )
        
        # 4. Khối Temporal Decoder: Nhận [d_t ; x_future_t] (p + r_tilde) -> 1
        self.temporal_decoder = nn.Sequential(
            nn.Linear(p + r_tilde, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )
        
        # 5. Nhánh tắt tuyến tính toàn cục: Nhận L -> H
        self.linear_skip = nn.Linear(L, H)

    def forward(self, y, x, a=None):
        B = y.shape[0]
        
        # [Bước 0] Chuẩn hóa RevIN
        mean = y.mean(dim=1, keepdim=True)
        std = y.std(dim=1, keepdim=True) + 1e-5
        y_norm = (y - mean) / std                               # (B, L)
        
        # [Bước 1] Feature Projection (per time-step)
        x_proj = self.feature_projection(x)                     # (B, L+H, r_tilde)
        
        # [Bước 2 & 3] Trải phẳng Flatten & Ghép nối Concat đầu vào Encoder
        x_flat = x_proj.flatten(start_dim=1)                    # (B, (L+H) * r_tilde)
        if a is not None:
            enc_in = torch.cat([y_norm, x_flat, a], dim=-1)
        else:
            enc_in = torch.cat([y_norm, x_flat], dim=-1)        # (B, L + (L+H)*r_tilde)
            
        # [Bước 4] Dense Encoder sinh vector bối cảnh e
        e = self.encoder(enc_in)                                # (B, hidden_size)
        
        # [Bước 5] Dense Decoder sinh vector giải mã phẳng g
        g = self.decoder(e)                                     # (B, H * p)
        
        # [Bước 6] Tái cấu trúc Unflatten vector g thành ma trận d
        d = g.view(B, self.H, self.p)                           # (B, H, p)
        
        # [Bước 7 & 8] Ghép ngoại sinh tương lai & đưa qua Temporal Decoder
        x_future = x_proj[:, self.L:, :]                        # (B, H, r_tilde)
        temp_in = torch.cat([d, x_future], dim=-1)              # (B, H, p + r_tilde)
        y_nl = self.temporal_decoder(temp_in).squeeze(-1)       # (B, H)
        
        # [Bước 9] Nhánh tắt tuyến tính toàn cục (Global Linear Skip)
        y_lin = self.linear_skip(y_norm)                        # (B, H)
        
        # [Bước 10] Hợp nhất dự báo phi tuyến và tuyến tính
        y_hat_norm = y_nl + y_lin                               # (B, H)
        
        # [Bước 11] Hoàn nguyên thang đo RevIN
        y_hat = y_hat_norm * std + mean                         # (B, H)
        
        return y_hat
```
