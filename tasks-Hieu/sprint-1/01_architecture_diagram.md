# PHẦN 1: SƠ ĐỒ KIẾN TRÚC GỐC & GIẢI PHẪU CHI TIẾT QUY TRÌNH HOẠT ĐỘNG CỦA TiDE

> [!NOTE]
> **Điều hướng tài liệu**: [🏠 Trang Chủ](../../README.md) • [📋 Sprint 1 README](./README.md) • [📜 Yêu Cầu Sprint](./requirement.md) • [📊 Checklist Tiến Độ](./checklist.md) • [🛡️ Tự Kiểm Chứng](./05_self_verification.md)  
> 👤 **Người thực hiện:** Hiệu  
> 🤖 **Trợ lý kỹ thuật:** Antigravity  
> 📄 **Tài liệu tham chiếu:** *Long-term Forecasting with TiDE: Time-series Dense Encoder* (Google Research, 2023) - Trang 5, Figure 1 & Section 4

---

## 1. Sơ Đồ Kiến Trúc Gốc Của TiDE (Figure 1 từ Google Research)

Dưới đây là hình ảnh kiến trúc chính thức trích xuất trực tiếp từ **Trang 5 (Figure 1)** của bài báo gốc:

<p align="center">
  <img src="./images/tide_architecture.png" alt="Kiến trúc TiDE (Figure 1 từ Google Research)" width="780"/>
</p>

*Hình 1: Tổng quan kiến trúc mô hình TiDE (Time-series Dense Encoder) và cấu trúc chi tiết của khối Residual Block cơ sở.*

---

## 2. Trái Tim Của Mô Hình: Cấu Tạo Khối Residual Block (Khung Bên Phải)

Trước khi đi vào các giai đoạn dữ liệu, nhóm tác giả Google Research định nghĩa **Residual Block** là đơn vị nơ-ron cơ sở cấu tạo nên toàn bộ các bộ phận trong TiDE (từ Feature Projection, Dense Encoder, Dense Decoder cho đến Temporal Decoder).

Quan sát khung hình chữ nhật màu xanh ở góc phải của Hình 1:

```
              ┌─────────────────────────────────┐
              │           Layer Norm            │
              └────────────────┬────────────────┘
                               ▲
                              ( + ) <─── [Skip Connection (Linear)] ──┐
                               ▲                                      │
              ┌────────────────┴────────────────┐                     │
              │             Dropout             │                     │
              └────────────────▲────────────────┘                     │
                               │                                      │
              ┌────────────────┴────────────────┐                     │
              │         Dense (Linear)          │                     │
              └────────────────▲────────────────┘                     │
                               │                                      │
              ┌────────────────┴────────────────┐                     │
              │          Dense (ReLU)           │                     │
              └────────────────▲────────────────┘                     │
                               │                                      │
                        [ Đầu vào u ] ────────────────────────────────┘
```

### Cấu tạo 2 nhánh bên trong Residual Block:
1. **Nhánh biến đổi phi tuyến (Tuyến chính):**
   - **Tầng `Dense (ReLU)`:** Chiếu tuyến tính từ chiều đầu vào sang không gian ẩn (`hiddenSize`), đi kèm hàm kích hoạt phi tuyến ReLU để học các quan hệ phi tuyến phức tạp.
   - **Tầng `Dense (Linear)`:** Chiếu tuyến tính từ không gian ẩn về chiều đầu ra mong muốn.
   - **Tầng `Dropout`:** Ngắt kết nối ngẫu nhiên một tỷ lệ nơ-ron (ví dụ $0.3 - 0.5$) để chống hiện tượng học vẹt (overfitting).
2. **Nhánh tắt bảo tồn (Skip Connection):**
   - Ánh xạ trực tiếp từ đầu vào tới điểm cộng thặng dư. Nếu số chiều đầu vào khác đầu ra, nhánh này sử dụng một phép chiếu tuyến tính đơn giản.
3. **Hợp nhất và Chuẩn hóa (Addition & Layer Norm):**
   - **Nút tròn `(+)`:** Thực hiện phép cộng vector: $\text{Output} = \text{Phi tuyến} + \text{Skip}$. Kỹ thuật này giúp gradient truyền thẳng về các tầng trước mà không bị suy hao (*Vanishing Gradient*).
   - **Lớp `Layer Norm`:** Chuẩn hóa trung bình và phương sai ở đầu ra của khối, giữ cho phân phối tín hiệu ổn định khi mạng xếp chồng nhiều tầng sâu.

---

## 3. Giải Phẫu Chi Tiết Quy Trình Hoạt Động Theo Từng Giai Đoạn

Toàn bộ quy trình biến đổi dữ liệu của TiDE từ dưới lên trên được chuẩn hóa thành **8 giai đoạn nối tiếp nhau**. Mỗi giai đoạn đều được bóc tách rõ ràng theo 3 phần: **Input**, **Công thức xử lý**, và **Output**.

*(Các ví dụ số chiều cụ thể dưới đây sử dụng cấu hình tiêu chuẩn: Batch size $B=32$, Lookback $L=720$, Horizon $H=96$, số biến ngoại sinh thô $r=7$, số chiều nén $\tilde{r}=4$, số chiều giải mã $p=8$, kích thước ẩn $\text{hiddenSize}=1024$)*.

---

### 🟢 Giai Đoạn 1: Tiếp Nhận Dữ Liệu & Chuẩn Hóa Cục Bộ (RevIN Normalization)

#### 📥 1. Input (Đầu vào):
Mô hình tiếp nhận 3 nguồn dữ liệu độc lập cho chuỗi thời gian thứ $i$:
* Chuỗi lịch sử quá khứ (Lookback): $y_{1:L}^{(i)} \in \mathbb{R}^L$ $\implies$ Kích thước: `(B, L) = (32, 720)`
* Biến ngoại sinh động toàn thời gian: $x_{1:L+H}^{(i)} \in \mathbb{R}^{(L+H) \times r}$ $\implies$ Kích thước: `(B, L+H, r) = (32, 816, 7)`
* Thuộc tính tĩnh (Attributes): $a^{(i)} \in \mathbb{R}^{p_{\text{static}}}$ $\implies$ Kích thước: `(B, p_static)`

#### ⚙️ 2. Công thức xử lý:
Áp dụng cơ chế Chuẩn hóa Khả nghịch (*Reversible Instance Normalization - RevIN*) trên từng chuỗi quá khứ $y_{1:L}^{(i)}$ để loại bỏ hiện tượng trôi dạt phân phối (*Distribution Shift*):

$$\mu = \frac{1}{L} \sum_{t=1}^L y_t^{(i)}, \quad \sigma = \sqrt{\frac{1}{L} \sum_{t=1}^L \left(y_t^{(i)} - \mu\right)^2 + \epsilon}$$

$$y_{\text{norm}, t}^{(i)} = \frac{y_t^{(i)} - \mu}{\sigma} \quad \forall t \in \{1, \dots, L\}$$

#### 📤 3. Output (Đầu ra):
* Chuỗi quá khứ đã chuẩn hóa: $y_{\text{norm}, 1:L}^{(i)} \in \mathbb{R}^L$ $\implies$ Kích thước: `(B, L) = (32, 720)` (phân phối chuẩn $\text{Mean}=0, \text{Std}=1$).
* Hai tham số thống kê $\mu, \sigma \in \mathbb{R}^1$ được lưu lại vào bộ nhớ để phục vụ giải chuẩn hóa ở khâu cuối cùng.

---

### 🟢 Giai Đoạn 2: Chiếu Giảm Chiều Ngoại Sinh (Feature Projection per time-step)

#### 📥 1. Input (Đầu vào):
* Ma trận biến ngoại sinh động toàn thời gian: $x_{1:L+H}^{(i)} \in \mathbb{R}^{(L+H) \times r}$  
  $\implies$ Kích thước: `(B, L+H, r) = (32, 816, 7)`

#### ⚙️ 2. Công thức xử lý:
Áp dụng một khối Residual Block nhỏ độc lập cho từng bước thời gian $t \in \{1, \dots, L+H\}$ (bài báo gọi là *Feature Projection step*):

$$\tilde{x}_t^{(i)} = \text{FeatureProjection}\left(x_t^{(i)}\right) \in \mathbb{R}^{\tilde{r}} \quad (\text{với } \tilde{r} = 4 \ll r)$$

*Bản chất:* Nén số lượng biến ngoại sinh thô từ $r=7$ xuống $\tilde{r}=4$ tại mọi mốc thời gian để ngăn ngừa bùng nổ số chiều khi trải phẳng.

#### 📤 3. Output (Đầu ra):
* Ma trận ngoại sinh sau khi chiếu nén: $\tilde{x}_{1:L+H}^{(i)} \in \mathbb{R}^{(L+H) \times \tilde{r}}$  
  $\implies$ Kích thước: `(B, L+H, r_tilde) = (32, 816, 4)`

---

### 🟢 Giai Đoạn 3: Trải Phẳng (Flatten) & Ghép Nối Không Gian (Concat)

#### 📥 1. Input (Đầu vào):
Gồm 3 thành phần sẵn sàng được hợp nhất:
* Chuỗi quá khứ đã chuẩn hóa: $y_{\text{norm}, 1:L}^{(i)} \in \mathbb{R}^L$ $\implies$ Kích thước: `(32, 720)`
* Ma trận ngoại sinh đã nén: $\tilde{x}_{1:L+H}^{(i)} \in \mathbb{R}^{(L+H) \times \tilde{r}}$ $\implies$ Kích thước: `(32, 816, 4)`
* Thuộc tính tĩnh: $a^{(i)} \in \mathbb{R}^{p_{\text{static}}}$ $\implies$ Kích thước: `(32, p_static)`

#### ⚙️ 2. Công thức xử lý:
1. **Trải phẳng (Flatten):** Biến đổi ma trận ngoại sinh 2D thành một vector 1D duy nhất trên toàn bộ trục thời gian:
   $$\text{vec}\left(\tilde{x}_{1:L+H}^{(i)}\right) \in \mathbb{R}^{(L+H)\tilde{r}}$$
   *(Độ dài vector phẳng: $(720 + 96) \times 4 = 816 \times 4 = 3,264$ phần tử)*.

2. **Ghép nối (Concat):** Nối chuỗi quá khứ, vector ngoại sinh phẳng và thuộc tính tĩnh lại với nhau:
   $$u_{\text{enc}}^{(i)} = \begin{bmatrix} y_{\text{norm}, 1:L}^{(i)} \\ \text{vec}\left(\tilde{x}_{1:L+H}^{(i)}\right) \\ a^{(i)} \end{bmatrix} \in \mathbb{R}^{L + (L+H)\tilde{r} + p_{\text{static}}}$$

#### 📤 3. Output (Đầu ra):
* Vector đầu vào duy nhất cho Dense Encoder: $u_{\text{enc}}^{(i)}$  
  $\implies$ Kích thước: `(B, dim_enc) = (32, 3984)`  
  *(Trong đó: $3984 = 720 + 3264 + 0$)*.

---

### 🟢 Giai Đoạn 4: Mã Hóa Toàn Cục (Dense Encoder)

#### 📥 1. Input (Đầu vào):
* Vector đặc trưng ghép phẳng: $u_{\text{enc}}^{(i)} \in \mathbb{R}^{L + (L+H)\tilde{r} + p_{\text{static}}}$  
  $\implies$ Kích thước: `(B, 3984)`

#### ⚙️ 2. Công thức xử lý:
Cho vector $u_{\text{enc}}^{(i)}$ đi qua khối Dense Encoder gồm $n_e$ khối Residual Block xếp chồng liên tiếp (`numEncoderLayers`):

$$e^{(i)} = \text{DenseEncoder}\left(u_{\text{enc}}^{(i)}\right) \in \mathbb{R}^{\text{hiddenSize}}$$

*Bản chất:* Encoder tổng hợp toàn bộ bối cảnh lịch sử, sự biến thiên của các biến ngoại sinh trong cả quá khứ và tương lai, cùng đặc tính tĩnh của thực thể thành một vector biểu diễn ẩn toàn cục.

#### 📤 3. Output (Đầu ra):
* Vector bối cảnh ẩn toàn cục (Context Embedding): $e^{(i)} \in \mathbb{R}^{\text{hiddenSize}}$  
  $\implies$ Kích thước: `(B, hidden_size) = (32, 1024)`

---

### 🟢 Giai Đoạn 5: Giải Mã Dày Đặc (Dense Decoder) & Tái Cấu Trúc (Unflatten)

#### 📥 1. Input (Đầu vào):
* Vector bối cảnh ẩn toàn cục: $e^{(i)} \in \mathbb{R}^{\text{hiddenSize}}$  
  $\implies$ Kích thước: `(B, 1024)`

#### ⚙️ 2. Công thức xử lý:
1. **Giải mã dày đặc qua $n_d$ tầng Residual Block (`numDecoderLayers`):**
   $$g^{(i)} = \text{DenseDecoder}\left(e^{(i)}\right) \in \mathbb{R}^{H \cdot p}$$
   *(Tầng cuối ánh xạ thành vector phẳng $g^{(i)}$ có độ dài đúng bằng $H \times p = 96 \times 8 = 768$)*.

2. **Tái cấu trúc (Unflatten / Reshape):**
   Chia đều vector phẳng $g^{(i)}$ thành ma trận 2D gồm $H$ hàng, mỗi hàng là một vector lát cắt $d_t^{(i)}$:
   $$D^{(i)} = \text{Unflatten}\left(g^{(i)}\right) = \begin{bmatrix} d_{L+1}^{(i)} \\ d_{L+2}^{(i)} \\ \vdots \\ d_{L+H}^{(i)} \end{bmatrix} \in \mathbb{R}^{H \times p}$$

#### 📤 3. Output (Đầu ra):
* Vector giải mã phẳng: $g^{(i)} \in \mathbb{R}^{H \cdot p}$ $\implies$ Kích thước: `(B, H * p) = (32, 768)`
* Ma trận đặc trưng bối cảnh tương lai: $D^{(i)} \in \mathbb{R}^{H \times p}$ $\implies$ Kích thước: `(B, H, p) = (32, 96, 8)`

---

### 🟢 Giai Đoạn 6: Làn Đường Cao Tốc & Giải Mã Thời Gian (Temporal Decoder)

#### 📥 1. Input (Đầu vào):
Gồm 2 thành phần được ghép nối tại từng bước thời gian tương lai $t \in \{L+1, \dots, L+H\}$:
* Vector bối cảnh tương lai thứ $t$: $d_t^{(i)} \in \mathbb{R}^p$ $\implies$ Kích thước lát cắt: `(B, p) = (32, 8)`
* Biến ngoại sinh tương lai thứ $t$ (cắt từ $\tilde{x}$): $\tilde{x}_t^{(i)} \in \mathbb{R}^{\tilde{r}}$ $\implies$ Kích thước: `(B, r_tilde) = (32, 4)`

#### ⚙️ 2. Công thức xử lý:
1. **Thao tác Xếp chồng (`Stack`):** Ghép nối bối cảnh $d_t^{(i)}$ với biến ngoại sinh tương lai $\tilde{x}_t^{(i)}$ thành vector đầu vào:
   $$\text{Input}_{\text{temp}, t} = \begin{bmatrix} d_t^{(i)} \\ \tilde{x}_t^{(i)} \end{bmatrix} \in \mathbb{R}^{p + \tilde{r}} \implies \text{Kích thước: } (32, 8 + 4) = (32, 12)$$

2. **Dự báo phi tuyến qua Temporal Decoder:** Áp dụng một khối Residual Block độc lập chiếu từ $(p + \tilde{r})$ chiều về $1$ con số duy nhất:
   $$\hat{y}_{\text{NL}, t}^{(i)} = \text{TemporalDecoder}\left( \begin{bmatrix} d_t^{(i)} \\ \tilde{x}_t^{(i)} \end{bmatrix} \right) \in \mathbb{R} \quad \forall t \in \{L+1, \dots, L+H\}$$

*Ý nghĩa "Highway":* Đây là làn đường cao tốc đưa trực tiếp thông tin ngoại sinh tương lai vào ngõ ra dự báo mà không bị mờ nhạt qua các tầng Encoder sâu.

#### 📤 3. Output (Đầu ra):
* Vector dự báo phi tuyến toàn bộ $H$ bước tương lai: $\hat{y}_{\text{NL}}^{(i)} \in \mathbb{R}^H$  
  $\implies$ Kích thước: `(B, H) = (32, 96)`

---

### 🟢 Giai Đoạn 7: Nhánh Tắt Tuyến Tính Toàn Cục (Global Linear Skip Connection)

#### 📥 1. Input (Đầu vào):
* Chuỗi quá khứ đã chuẩn hóa: $y_{\text{norm}, 1:L}^{(i)} \in \mathbb{R}^L$  
  $\implies$ Kích thước: `(B, L) = (32, 720)`

#### ⚙️ 2. Công thức xử lý:
Áp dụng một phép chiếu tuyến tính trực tiếp (tương tự như mô hình DLinear) ánh xạ thẳng từ $L$ bước quá khứ sang $H$ bước tương lai:

$$\hat{y}_{\text{Lin}}^{(i)} = W_{\text{skip}} \cdot y_{\text{norm}, 1:L}^{(i)} + b_{\text{skip}} \in \mathbb{R}^H$$

*(Trong đó ma trận trọng số $W_{\text{skip}} \in \mathbb{R}^{H \times L}$ và vector hệ số tự do $b_{\text{skip}} \in \mathbb{R}^H$)*.

*Bản chất:* Đảm nhận riêng việc mô hình hóa xu hướng tăng/giảm cơ bản (trend), giải phóng mạng sâu chỉ cần học các dao động sóng phi tuyến.

#### 📤 3. Output (Đầu ra):
* Vector dự báo tuyến tính: $\hat{y}_{\text{Lin}}^{(i)} \in \mathbb{R}^H$  
  $\implies$ Kích thước: `(B, H) = (32, 96)`

---

### 🟢 Giai Đoạn 8: Hợp Nhất Dự Báo & Hoàn Nguyên Thang Đo (Residual Addition & RevIN Denorm)

#### 📥 1. Input (Đầu vào):
* Dự báo phi tuyến từ Temporal Decoder: $\hat{y}_{\text{NL}}^{(i)} \in \mathbb{R}^H$ $\implies$ Kích thước: `(32, 96)`
* Dự báo tuyến tính từ Linear Skip: $\hat{y}_{\text{Lin}}^{(i)} \in \mathbb{R}^H$ $\implies$ Kích thước: `(32, 96)`
* Hai tham số RevIN ban đầu: giá trị trung bình $\mu$ và độ lệch chuẩn $\sigma$

#### ⚙️ 2. Công thức xử lý:
1. **Phép cộng thặng dư tại nút tròn `(+) Residual`:**
   $$\hat{y}_{\text{norm}}^{(i)} = \hat{y}_{\text{NL}}^{(i)} + \hat{y}_{\text{Lin}}^{(i)} \in \mathbb{R}^H$$

2. **Hoàn nguyên thang đo RevIN (Denormalization):**
   Nhân lại độ lệch chuẩn $\sigma$ và cộng lại giá trị trung bình $\mu$ để đưa giá trị về đúng đơn vị vật lý thực tế:
   $$\hat{y}_{L+1:L+H}^{(i)} = \hat{y}_{\text{norm}}^{(i)} \cdot \sigma + \mu \in \mathbb{R}^H$$

#### 📤 3. Output (Đầu ra):
* **Kết quả dự báo cuối cùng (Predictions):** $\hat{y}_{L+1:L+H}^{(i)} \in \mathbb{R}^H$  
  $\implies$ Kích thước: `(B, H) = (32, 96)`  
  *(Khớp hoàn hảo 100% với kích thước nhãn thực tế Ground Truth $y_{L+1:L+H}^{(i)}$)*.

---

## 4. Bảng Tổng Hợp Luồng Input - Xử Lý - Output Của Toàn Bộ Kiến Trúc

| Giai đoạn | Tên Giai Đoạn | 📥 Input Chính (Shape) | ⚙️ Phép Xử Lý Cốt Lõi | 📤 Output Sinh Ra (Shape) |
| :---: | :--- | :--- | :--- | :--- |
| **1** | **RevIN Norm** | $y_{1:L}$ `(32, 720)` | $y_{\text{norm}} = (y - \mu) / \sigma$ | $y_{\text{norm}}$ `(32, 720)`, lưu $(\mu, \sigma)$ |
| **2** | **Feature Projection** | $x_{1:L+H}$ `(32, 816, 7)` | $\tilde{x}_t = \text{ResBlock}(x_t)$ | $\tilde{x}_{1:L+H}$ `(32, 816, 4)` |
| **3** | **Flatten & Concat** | $y_{\text{norm}}, \tilde{x}, a$ | $\text{vec}(\tilde{x}) \to \text{Concat}[y; \tilde{x}; a]$ | $u_{\text{enc}}$ `(32, 3984)` |
| **4** | **Dense Encoder** | $u_{\text{enc}}$ `(32, 3984)` | $e = \text{Encoder}(u_{\text{enc}})$ ($n_e$ tầng) | $e$ `(32, 1024)` |
| **5** | **Dense Decoder & Unflatten** | $e$ `(32, 1024)` | $g = \text{Decoder}(e) \to \text{view}(B, H, p)$ | $D$ `(32, 96, 8)` |
| **6** | **Temporal Decoder (Highway)**| $d_t, \tilde{x}_t$ `(32, 12)` | $\hat{y}_{\text{NL}, t} = \text{TempDec}([d_t; \tilde{x}_t])$ | $\hat{y}_{\text{NL}}$ `(32, 96)` |
| **7** | **Global Linear Skip** | $y_{\text{norm}}$ `(32, 720)` | $\hat{y}_{\text{Lin}} = W_{\text{skip}} y_{\text{norm}} + b_{\text{skip}}$ | $\hat{y}_{\text{Lin}}$ `(32, 96)` |
| **8** | **Residual (+) & Denorm** | $\hat{y}_{\text{NL}}, \hat{y}_{\text{Lin}}$ `(32, 96)` | $\hat{y} = (\hat{y}_{\text{NL}} + \hat{y}_{\text{Lin}}) \cdot \sigma + \mu$ | $\hat{y}_{L+1:L+H}$ `(32, 96)` |

---

## 5. Bảng Tra Cứu Toàn Bộ Ký Hiệu Xuất Hiện Trong Hình 1

| Ký Hiệu Trong Hình | Tên Gọi & Vị Trí Trong Mạng | Bản Chất Kỹ Thuật | Kích Thước Toán Học |
| :--- | :--- | :--- | :---: |
| **$y_{1:L}^{(i)}$** | **Lookback** (Chuỗi quá khứ) | Giá trị thực tế của chuỗi thứ $i$ từ bước $1$ đến $L$. | $\mathbb{R}^L$ |
| **$x_{1:L+H}^{(i)}$** | **Dynamic Covariates** (Biến ngoại sinh động) | Các biến biến thiên theo thời gian đã biết cho cả quá khứ lẫn tương lai. | $\mathbb{R}^{(L+H) \times r}$ |
| **$a^{(i)}$** | **Attributes** (Thuộc tính tĩnh) | Các biến cố định của chuỗi (mã cửa hàng, trạm cảm biến). | $\mathbb{R}^{p_{\text{static}}}$ |
| **$\tilde{x}_{1:L+H}^{(i)}$** | **Projected Covariates** (Ngoại sinh đã nén) | Đầu ra sau khi nén qua khối Feature Projection (`temporalWidth = 4`). | $\mathbb{R}^{(L+H) \times \tilde{r}}$ |
| **$e^{(i)}$** | **Context Embedding** (Vector bối cảnh ẩn) | Đầu ra của Dense Encoder, chứa đựng tri thức toàn cục của chuỗi. | $\mathbb{R}^{\text{hiddenSize}}$ |
| **$g^{(i)}$** | **Decoded Vector** (Vector giải mã phẳng) | Đầu ra của Dense Decoder trước khi Unflatten. | $\mathbb{R}^{H \cdot p}$ |
| **$d_{L+1:L+H}^{(i)}$** | **Decoded Feature Matrix** (Đặc trưng bối cảnh tương lai) | Ma trận $H$ hàng sau khi Unflatten vector $g^{(i)}$. | $\mathbb{R}^{H \times p}$ |
| **$\tilde{x}_{L+1:L+H}^{(i)}$** | **Future Covariates** (Ngoại sinh tương lai) | $H$ bước tương lai của biến ngoại sinh đưa thẳng vào Temporal Decoder. | $\mathbb{R}^{H \times \tilde{r}}$ |
| **$\hat{y}_{L+1:L+H}^{(i)}$** | **Predictions** (Kết quả dự báo cuối cùng) | Giá trị dự báo của $H$ bước thời gian tương lai. | $\mathbb{R}^H$ |
| **$n_e$** | Số tầng Dense Encoder | Số khối Residual Block trong Encoder (`numEncoderLayers`). | $1$ hoặc $2$ |
| **$n_d$** | Số tầng Dense Decoder | Số khối Residual Block trong Decoder (`numDecoderLayers`). | $1$ hoặc $2$ |
