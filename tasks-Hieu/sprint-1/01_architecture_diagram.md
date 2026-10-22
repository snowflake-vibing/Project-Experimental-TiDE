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

## 2. Trái Tim Của Mô Hình: Khối Residual Block (Khung Bên Phải)

Trước khi đi vào luồng dữ liệu chính, nhóm tác giả Google Research định nghĩa **Residual Block** là đơn vị nơ-ron cơ sở cấu tạo nên toàn bộ các bộ phận trong TiDE (từ Feature Projection, Dense Encoder, Dense Decoder cho đến Temporal Decoder).

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

## 3. Giải Phẫu Chi Tiết Quy Trình Hoạt Động Của TiDE (Từng Bước Từ Dưới Lên Trên)

Toàn bộ quy trình forward pass của TiDE hoạt động theo dòng chảy dữ liệu khép kín gồm **6 giai đoạn biến đổi**:

### 🟢 Giai đoạn 1: Tiếp nhận 3 nguồn dữ liệu đầu vào độc lập
Tại mỗi chuỗi thời gian thứ $i$, mô hình tiếp nhận 3 nguồn dữ liệu:
1. **Chuỗi quá khứ (Lookback) $y_{1:L}^{(i)} \in \mathbb{R}^L$:** Lịch sử giá trị cần dự báo trong $L$ bước quá khứ (ví dụ $L=720$ giờ = 30 ngày).
2. **Biến ngoại sinh động (Dynamic Covariates) $x_{1:L+H}^{(i)} \in \mathbb{R}^{(L+H) \times r}$:** Các biến biến thiên theo thời gian nhưng **đã biết trước cho cả quá khứ lẫn tương lai** (ví dụ: giờ trong ngày, ngày trong tuần, tháng trong năm, sự kiện khuyến mãi, dự báo thời tiết).
3. **Thuộc tính tĩnh (Attributes) $a^{(i)} \in \mathbb{R}^{p_{\text{static}}}$:** Các đặc trưng cố định không thay đổi theo thời gian của thực thể (ví dụ: mã trạm biến áp, vị trí địa lý của cảm biến, phân loại khách hàng).

---

### 🟢 Giai đoạn 2: Chiếu giảm chiều ngoại sinh (Feature Projection per time-step)
* **Vấn đề bài toán:** Biến ngoại sinh $x$ bao phủ toàn bộ dải $L+H = 720 + 96 = 816$ bước. Nếu mỗi bước có $r=20$ đặc trưng, việc ghép thẳng vào mạng sẽ khiến số chiều bùng nổ lên $(L+H) \times r = 16,320$ chiều, gây tràn RAM và overfitting.
* **Cách TiDE xử lý:**
  - Áp dụng một khối **Residual Block nhỏ độc lập cho từng bước thời gian $t$**:
    $$\tilde{x}_t^{(i)} = \text{FeatureProjection}(x_t^{(i)}) \in \mathbb{R}^{\tilde{r}}$$
  - Nén số chiều từ $r$ xuống một kích thước cố định rất nhỏ: **$\tilde{r} = 4$** (`temporalWidth = 4`).
  - Ma trận ngoại sinh sau nén trở thành: $\tilde{x}_{1:L+H}^{(i)} \in \mathbb{R}^{(L+H) \times 4}$.

---

### 🟢 Giai đoạn 3: Trải phẳng (Flatten) & Ghép nối (Concat)
* **Thao tác `Flatten`:** Trải phẳng ma trận 2D $\tilde{x}_{1:L+H}^{(i)}$ có kích thước $(L+H) \times 4$ thành một vector 1D duy nhất có độ dài $(L+H) \times 4 = 816 \times 4 = 3,264$.
* **Thao tác `Concat`:** Ghép chung 3 nguồn dữ liệu vào một cửa duy nhất:
  $$u_{\text{enc}}^{(i)} = \begin{bmatrix} y_{1:L}^{(i)} \\ \text{vec}(\tilde{x}_{1:L+H}^{(i)}) \\ a^{(i)} \end{bmatrix} \in \mathbb{R}^{L + (L+H)\tilde{r} + p_{\text{static}}}$$
* Với ví dụ $L=720, H=96, \tilde{r}=4$ (không có static attributes):
  $$\text{dim}_{\text{enc}} = 720 + (720 + 96) \times 4 = 720 + 3264 = \mathbf{3984}$$

---

### 🟢 Giai đoạn 4: Mã hóa toàn cục qua $n_e$ tầng Dense Encoder
* Vector phẳng $u_{\text{enc}}^{(i)}$ kích thước $3984$ được nạp vào khối **Dense Encoder** (hộp màu xanh lá cây).
* Khối này gồm **$n_e$ khối Residual Block xếp chồng liên tiếp** (`numEncoderLayers`, thường $n_e \in \{1, 2\}$).
* **Nhiệm vụ:** Trích xuất và cô đọng toàn bộ tương tác phi tuyến giữa quá khứ, biến ngoại sinh toàn thời gian và đặc trưng tĩnh thành một **vector biểu diễn ẩn toàn cục (Context Embedding) $e^{(i)} \in \mathbb{R}^{\text{hiddenSize}}$** (ví dụ $\text{hiddenSize} = 512$ hoặc $1024$).

---

### 🟢 Giai đoạn 5: Giải mã tương lai qua $n_d$ tầng Dense Decoder & Tái cấu trúc (Unflatten)
1. **Giải mã dày đặc (Dense Decoder - Hộp màu vàng):**
   - Vector $e^{(i)}$ đi qua $n_d$ khối Residual Block (`numDecoderLayers`, thường $n_d \in \{1, 2\}$).
   - Tầng cuối cùng của Dense Decoder ánh xạ ra một vector phẳng $g^{(i)}$ mang thông tin bối cảnh cho toàn bộ $H$ bước tương lai:
     $$g^{(i)} = \text{DenseDecoder}(e^{(i)}) \in \mathbb{R}^{H \cdot p}$$
     *(Trong đó $p$ là siêu tham số `decoderOutputDim`, ví dụ $p=8$ hoặc $16$)*.
2. **Tái cấu trúc (Unflatten / Reshape):**
   - Vector $g^{(i)}$ có độ dài $H \cdot p$ được biến đổi ngược thành ma trận 2D gồm $H$ hàng:
     $$D^{(i)} = \text{Unflatten}(g^{(i)}) = \begin{bmatrix} d_{L+1}^{(i)} \\ d_{L+2}^{(i)} \\ \vdots \\ d_{L+H}^{(i)} \end{bmatrix} \in \mathbb{R}^{H \times p}$$
   - Mỗi vector lát cắt $d_t^{(i)} \in \mathbb{R}^p$ đại diện cho đặc trưng bối cảnh đã giải mã tại riêng thời điểm tương lai thứ $t$.

---

### 🟢 Giai đoạn 6: Hợp nhất dự báo qua Làn cao tốc ngoại sinh & Nhánh tắt tuyến tính
Giai đoạn cuối cùng là sự phối hợp ăn ý giữa **2 luồng song song**:

#### 1. Luồng phi tuyến: Làn đường cao tốc ngoại sinh (Covariate Highway) & Temporal Decoder
* **Thao tác `Stack`:** Tại mỗi bước tương lai $t \in \{L+1, \dots, L+H\}$, mô hình lấy vector bối cảnh $d_t^{(i)}$ ghép trực tiếp với vector ngoại sinh tương lai $\tilde{x}_t^{(i)}$:
  $$\text{Đầu vào tại bước } t = \begin{bmatrix} d_t^{(i)} \\ \tilde{x}_t^{(i)} \end{bmatrix} \in \mathbb{R}^{p + \tilde{r}}$$
* **Bộ giải mã thời gian (Temporal Decoder - Khối màu xanh lam):** Là một Residual Block nhỏ độc lập hoạt động theo từng bước thời gian, ánh xạ từ vector $(p+\tilde{r})$ chiều thành con số dự báo phi tuyến duy nhất cho thời điểm đó:
  $$\hat{y}_{\text{NL}, t}^{(i)} = \text{TemporalDecoder}\left( \begin{bmatrix} d_t^{(i)} \\ \tilde{x}_t^{(i)} \end{bmatrix} \right) \in \mathbb{R}$$
* **Ý nghĩa:** Đây là "làn đường cao tốc" giúp các biến tương lai (như sự kiện khuyến mãi, giờ cao điểm tan tầm) tác động tức thì vào kết quả dự báo mà không bị loãng qua nhiều tầng mạng nơ-ron.

#### 2. Luồng tuyến tính: Nhánh tắt toàn cục (Global Linear Skip Connection)
* Một đường mũi tên chạy dài từ chuỗi quá khứ $y_{1:L}^{(i)}$ vòng qua phía bên trái, đi qua một phép chiếu tuyến tính trực tiếp tới đầu ra:
  $$\hat{y}_{\text{Lin}}^{(i)} = W_{\text{skip}} \cdot y_{1:L}^{(i)} + b_{\text{skip}} \in \mathbb{R}^H$$
* **Ý nghĩa:** Đóng vai trò như mô hình DLinear thu nhỏ, trực tiếp gánh vác việc nắm bắt xu hướng tăng/giảm vĩ mô (trend) của chuỗi thời gian.

#### 3. Điểm cộng thặng dư cuối cùng (Residual Addition `(+)`):
* Dự báo cuối cùng là tổng hợp của cả 2 luồng:
  $$\hat{y}_{L+1:L+H}^{(i)} = \hat{y}_{\text{NL}}^{(i)} + \hat{y}_{\text{Lin}}^{(i)}$$
* Sau đó hoàn nguyên thang đo RevIN (nếu có bật `revIn`) để sinh ra giá trị dự báo thực tế cuối cùng $\hat{y}_{L+1:L+H}^{(i)}$.

---

## 4. Bảng Tra Cứu Toàn Bộ Ký Hiệu Xuất Hiện Trong Hình 1

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

---

## 5. Ba Điểm Đột Phá Giúp TiDE Vượt Trội Hơn Transformer

1. **Độ phức tạp tuyến tính $O(L+H)$ thay vì $O(L^2)$:** Thay vì dùng Self-Attention tốn kém để so sánh từng cặp thời gian, TiDE trải phẳng và dùng mạng nơ-ron dày đặc (Dense MLP), giúp tốc độ huấn luyện và suy luận nhanh hơn 5 - 10 lần.
2. **Khắc phục triệt để điểm yếu của DLinear:** DLinear bỏ qua biến ngoại sinh, còn TiDE nén ngoại sinh qua `Feature Projection` và đưa thẳng vào ngõ ra qua `Covariate Highway`.
3. **Phân công nhiệm vụ rõ ràng:** Nhánh tuyến tính `Global Linear Skip` gánh phần xu hướng cơ bản, còn các khối sâu `Encoder-Decoder` chỉ tập trung học các dao động sóng phi tuyến phức tạp.
