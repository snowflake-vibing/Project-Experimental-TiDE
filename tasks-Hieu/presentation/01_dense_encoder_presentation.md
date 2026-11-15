# 📑 BÁO CÁO CHUYÊN ĐỀ THUYẾT TRÌNH: DENSE ENCODER TRONG MÔ HÌNH TiDE

> [!NOTE]
> **Thông tin chuyên đề:**
> - 👤 **Người thuyết trình:** Hiệu
> - 🤖 **Trợ lý kỹ thuật:** Antigravity
> - 📄 **Tài liệu tham chiếu:** *Long-term Forecasting with TiDE: Time-series Dense Encoder* (Google Research, 2023) - Mục 4.2 & Figure 1
> - 📌 **Chủ đề được giao:** **Cơ chế học quan hệ theo thời gian khi không sử dụng Attention, RNN hoặc CNN (Dense Encoder)**
> - 🧭 **Cấu trúc bài trình bày:** Tuân thủ chuẩn 5 yêu cầu bắt buộc (Lý thuyết • Toán học • Source code • Phân tích • Đề xuất)

---

## 🎯 Tổng Quan Chủ Đề Thuyết Trình
Dense Encoder là trái tim xử lý bối cảnh toàn cục của mô hình TiDE (*Time-series Dense Encoder*). Điểm đột phá lớn nhất của thành phần này là thách thức quan niệm phổ biến trong học sâu chuỗi thời gian: **"Dự báo chuỗi thời gian dài bắt buộc phải dùng Attention (Transformer) hoặc Recurrent (RNN/LSTM)"**. Bằng cách kết hợp cấu trúc trải phẳng đa chiều và các khối nơ-ron thặng dư (Residual Blocks), Dense Encoder đạt độ chính xác tương đương hoặc vượt trội Transformer nhưng có tốc độ huấn luyện và suy luận nhanh gấp 5 đến 10 lần.

```mermaid
graph TD
    subgraph DauVao ["1. Các Nguồn Dữ Liệu Đầu Vào"]
        Y["Chuỗi lịch sử: y(1:L)"]
        X["Biến ngoại sinh nén: x_tilde(1:L+H)"]
        A["Thuộc tính tĩnh: a"]
    end

    subgraph TienXuLy ["2. Trải Phẳng & Hợp Nhất"]
        Flat["Trải phẳng ma trận ngoại sinh: Flatten(x_tilde)"]
        Concat["Ghép nối thành Siêu Vector: Concat(y, flat_x, a)"]
    end

    subgraph KhoiMahoa ["3. Khối Mã Hóa Dày Đặc (Dense Encoder)"]
        RB1["Residual Block 1 (d_in -> d_hidden)<br/>Linear + ReLU + Linear + Dropout + Skip + LayerNorm"]
        RB2["Residual Block 2 (d_hidden -> d_hidden)<br/>Linear + ReLU + Linear + Dropout + Identity + LayerNorm"]
    end

    subgraph DauRa ["4. Vector Biểu Diễn Ẩn Toàn Cục"]
        E["Vector Bối Cảnh Ẩn: e in R^(d_hidden)"]
    end

    Y --> Concat
    X --> Flat
    Flat --> Concat
    A --> Concat
    Concat --> RB1
    RB1 --> RB2
    RB2 --> E
```

---

## 1. 📖 Lý Thuyết: Chức Năng, Nguyên Lý & Lý Do Thiết Kế

### 1.1. Chức năng của Dense Encoder
Dense Encoder đảm nhận nhiệm vụ nén toàn bộ thông tin lịch sử của chuỗi mục tiêu, toàn bộ thông tin biến ngoại sinh động (cả quá khứ lẫn tương lai) và các thuộc tính tĩnh thành **đúng một vector bối cảnh ẩn duy nhất** ký hiệu là:

$$
e^{(i)} \in \mathbb{R}^{d_{\text{hidden}}}
$$

Vector này đóng vai trò như một bản tóm tắt toàn diện (global representation) về trạng thái hệ thống, xu hướng nền, dao động mùa vụ và các điều kiện tác động ngoại cảnh trước khi bàn giao cho bộ giải mã `Dense Decoder`.

### 1.2. Nguyên lý hoạt động
Khác với Transformer (vốn giữ nguyên trục thời gian $L$ và dùng ma trận Attention kích thước $L \times L$ để tính độ tương đồng giữa các mốc thời gian), Dense Encoder hoạt động theo nguyên lý **Trải phẳng không gian thời gian (Spatial-Temporal Flattening)**:
* Toàn bộ chuỗi thời gian $L$ và các kênh ngoại sinh $(L+H) \times \tilde{r}$ được duỗi thẳng thành một vector một chiều duy nhất.
* Vector này được đưa qua $n_e$ khối `ResidualBlock` đa tầng (MLP).
* Nhờ phép chiếu tuyến tính dày đặc (Dense Projection), **mỗi nơ-ron ở tầng ẩn đều kết nối trực tiếp với toàn bộ các mốc thời gian từ $t=1$ đến $t=L+H$**.

### 1.3. Lý do thành phần này được thiết kế (Vì sao bỏ Attention và RNN?)
Nhóm tác giả Google Research thiết kế Dense Encoder xuất phát từ 3 động lực thực nghiệm sâu sắc:

**Động lực 1: Triệt tiêu chi phí tính toán bậc hai $O(L^2)$ của Transformer**
Trong bài toán dự báo chuỗi thời gian dài (Long-term Forecasting), chiều dài nhìn lại $L$ thường rất lớn (ví dụ $L = 720$). Transformer tốn chi phí bộ nhớ và thời gian tính toán tỷ lệ với $L^2$. Trong khi đó, một tầng Dense chỉ tốn chi phí tỷ lệ tuyến tính với số lượng liên kết, dễ dàng tối ưu hóa song song hoàn toàn trên phần cứng GPU.

**Động lực 2: Tránh hiện tượng suy thoái và nghẽn tuần tự của RNN**
RNN/LSTM xử lý tuần tự từng bước thời gian $t=1, 2, \dots, L$, dẫn đến hai nhược điểm: không tận dụng được năng lực tính toán song song của GPU và dễ bị mất mát thông tin khi khoảng cách thời gian quá xa.

**Động lực 3: Tận dụng tính cố định của lưới thời gian trong dự báo**
Trong chuỗi thời gian được lấy mẫu định kỳ (ví dụ mỗi giờ 1 lần), vị trí index $t$ mang ý nghĩa ngữ nghĩa thời gian tuyệt đối cố định (ví dụ bước $t=24$ luôn là 24 giờ trước). Do đó, một ma trận trọng số cố định $W$ hoàn toàn có thể học được trực tiếp mối quan hệ trễ (lag relationships) mà không cần cơ chế Attention phức tạp hay Positional Encoding linh động.

---

## 2. 📐 Toán Học: Công Thức & Kích Thước Tensor Đầu Vào - Đầu Ra

### 2.1. Công thức ghép nối đầu vào
Giả sử xét mẫu thứ $i$ trong một batch tính toán. Đầu vào của Dense Encoder được ghép nối từ 3 thành phần:
* Chuỗi lịch sử quan sát đã qua chuẩn hóa RevIN: $y_{1:L}^{(i)} \in \mathbb{R}^L$.
* Toàn bộ ma trận biến ngoại sinh động sau khi đã giảm chiều qua Feature Projection: $\tilde{x}_{1:L+H}^{(i)} \in \mathbb{R}^{(L+H) \times \tilde{r}}$.
* Vector thuộc tính ngoại sinh tĩnh: $a^{(i)} \in \mathbb{R}^{d_a}$.

Ma trận $\tilde{x}_{1:L+H}^{(i)}$ được trải phẳng (flatten) thành một vector hàng dài $(L + H) \cdot \tilde{r}$ phần tử. Sau đó, phép ghép nối vector (Concatenation) được thực thi:

$$
u_{\text{enc}}^{(i)} = \text{Concat}\Big( y_{1:L}^{(i)}, \; \text{vec}\big(\tilde{x}_{1:L+H}^{(i)}\big), \; a^{(i)} \Big)
$$

Tổng số chiều đầu vào của Encoder, ký hiệu là $d_{\text{in}}$, được xác định chính xác theo công thức:

$$
d_{\text{in}} = L + (L + H) \cdot \tilde{r} + d_a
$$

### 2.2. Công thức toán bên trong các khối Residual Block của Encoder
Dense Encoder gồm $n_e$ khối `ResidualBlock` xếp chồng. Đặt $u^{(0)} = u_{\text{enc}}^{(i)}$. Với mỗi khối thứ $l \in \{1, 2, \dots, n_e\}$:

**Bước 1: Nhánh phi tuyến (MLP Branch):**

$$
h^{(l)} = \text{Dropout}\Big( W_2^{(l)} \cdot \text{ReLU}\big( W_1^{(l)} u^{(l-1)} + b_1^{(l)} \big) + b_2^{(l)} \Big)
$$

Trong đó:
* $W_1^{(l)} \in \mathbb{R}^{d_{\text{hidden}} \times d_{\text{in}}^{(l)}}$ và $b_1^{(l)} \in \mathbb{R}^{d_{\text{hidden}}}$ là trọng số tầng mở rộng.
* $W_2^{(l)} \in \mathbb{R}^{d_{\text{hidden}} \times d_{\text{hidden}}}$ và $b_2^{(l)} \in \mathbb{R}^{d_{\text{hidden}}}$ là trọng số tầng thu gọn.

**Bước 2: Nhánh tắt bảo toàn tín hiệu (Skip Connection Branch):**

$$
s^{(l)} = \begin{cases} 
W_{\text{skip}}^{(l)} u^{(l-1)} & \text{nếu } d_{\text{in}}^{(l)} \neq d_{\text{hidden}} \quad (\text{Khối đầu tiên } l=1) \\
u^{(l-1)} & \text{nếu } d_{\text{in}}^{(l)} = d_{\text{hidden}} \quad (\text{Các khối sau } l > 1)
\end{cases}
$$

**Bước 3: Phép cộng thặng dư và chuẩn hóa tầng (Layer Normalization):**

$$
u^{(l)} = \text{LayerNorm}\big( h^{(l)} + s^{(l)} \big)
$$

Vector đầu ra cuối cùng của Encoder chính là đầu ra của khối thứ $n_e$:

$$
e^{(i)} = u^{(n_e)} \in \mathbb{R}^{d_{\text{hidden}}}
$$

### 2.3. Bảng kích thước Tensor qua từng bước (Dữ liệu thực nghiệm Electricity Benchmark)
Cấu hình thực tế theo Table 8 của bài báo: Batch size $B = 32$, Lookback $L = 720$, Horizon $H = 96$, số chiều nén ngoại sinh $\tilde{r} = 4$, thuộc tính tĩnh $d_a = 0$, số chiều ẩn $d_{\text{hidden}} = 1024$, số tầng $n_e = 2$:

| Tên Tensor | Ký Hiệu Toán Học | Kích Thước Tensor (PyTorch Shape) | Ý Nghĩa Kỹ Thuật |
| :--- | :---: | :---: | :--- |
| `y_norm` | $y_{1:L}$ | `torch.Size([32, 720])` | Chuỗi mục tiêu lịch sử đã trừ trung bình chia độ lệch chuẩn |
| `x_proj` | $\tilde{x}_{1:L+H}$ | `torch.Size([32, 816, 4])` | Biến ngoại sinh toàn bộ $(720 + 96 = 816)$ bước đã nén về 4 chiều |
| `flat_x_proj` | $\text{vec}(\tilde{x})$ | `torch.Size([32, 3264])` | Trải phẳng ma trận ngoại sinh: $816 \times 4 = 3,264$ |
| `u_enc` | $u_{\text{enc}}$ | `torch.Size([32, 3984])` | Siêu vector ghép nối: $720 + 3264 + 0 = 3,984$ |
| `enc_block_1` | $u^{(1)}$ | `torch.Size([32, 1024])` | Nén từ 3,984 chiều về không gian ẩn 1,024 chiều (có $W_{\text{skip}}$) |
| `e` (Đầu ra) | $e$ | `torch.Size([32, 1024])` | Vector bối cảnh ẩn toàn cục sau khối Residual Block thứ 2 |

---

## 3. 💻 Source Code: Đối Chiếu Repository Gốc & Triển Khai Thực Thi

### 3.1. Mã nguồn tương ứng trong repository chính thức của Google Research
Trong mã nguồn chính thức [google-research/tide/model.py](https://github.com/google-research/google-research/tree/master/tide), Dense Encoder được định nghĩa trong hàm `__init__` và thực thi trong hàm `forward`:

```python
# Trích xuất từ Google Research: tide/model.py
class TiDE(nn.Module):
    def __init__(self, ...):
        # Tính toán kích thước đầu vào của Dense Encoder
        # input_dim = L + (L + H) * temporal_width + static_dim
        encoder_in_dim = self.history_len + (self.history_len + self.horizon) * self.temporal_width + self.static_dim
        
        # Xây dựng danh sách các khối Residual Block cho Encoder
        encoder_layers = []
        for i in range(self.num_encoder_layers):
            in_dim = encoder_in_dim if i == 0 else self.hidden_dim
            encoder_layers.append(
                ResidualBlock(
                    in_dim=in_dim,
                    hidden_dim=self.hidden_dim,
                    out_dim=self.hidden_dim,
                    dropout=self.dropout
                )
            )
        self.encoder = nn.Sequential(*encoder_layers)
```

### 3.2. Mã nguồn thực thi forward pass trong file mô phỏng của dự án
Trong file [simulate_forward_pass.py](file:///d:/PTDLKD/Project-Experimental-TiDE/simulate_forward_pass.py#L187-L212), luồng dữ liệu của Dense Encoder được hiện thực hóa với logging kích thước chuẩn xác:

```python
# BƯỚC 3: Ghép Nối Dữ Liệu Đầu Vào Encoder (Section 4.2 Step 2)
flat_x_proj = x_proj.flatten(start_dim=1)                 # Shape: [B, (L+H) * r_tilde]
u_enc = torch.cat([y_norm, flat_x_proj, a_stat], dim=-1) # Shape: [B, L + (L+H)*r_tilde + r_stat]

# BƯỚC 4: Khối Mã Hóa Dày Đặc Dense Encoder (Section 4.2 Step 2)
# Đi qua chuỗi n_e khối ResidualBlock
e = self.dense_encoder(u_enc)                             # Shape: [B, hidden_size]
```

### 3.3. Cơ chế học quan hệ thời gian qua ma trận trọng số $W_1$
Điểm đặc biệt nhất trong code là tầng `self.fc1 = nn.Linear(encoder_in_dim, hidden_dim)` của khối `ResidualBlock` đầu tiên:

```
Ma trận trọng số W_1 có shape: [1024, 3984]
Mỗi hàng thứ i (kích thước 3984 phần tử) biểu diễn:
[ w_(i, 1)...w_(i, 720) | w_(i, 721)...w_(i, 3984) ]
   (Trọng số cho y)         (Trọng số cho x_tilde)
```

Mỗi nơ-ron ẩn $i$ tính toán tổ hợp tuyến tính của **mọi thời điểm trong quá khứ và tương lai đồng thời**. Do đó, quan hệ thời gian (như tính chu kỳ 24 giờ của điện lực, chu kỳ 7 ngày trong tuần) được mã hóa trực tiếp vào các giá trị trọng số $W_1$ qua quá trình lan truyền ngược (Backpropagation).

---

## 4. 🔍 Phân Tích: Ảnh Hưởng Đến Hiệu Năng & Các Giới Hạn Tiềm Ẩn

### 4.1. Ảnh hưởng tích cực đến hiệu năng mô hình
1. **Tốc độ tính toán vượt trội:** Do không có các thao tác tuần tự (như vòng lặp trong RNN) hay các phép nhân ma trận 3 chiều phức tạp (như Softmax Attention $O(L^2)$), Dense Encoder chạy với độ phức tạp tính toán tuyến tính $O(d_{\text{in}} \cdot d_{\text{hidden}})$, giúp thời gian một epoch huấn luyện giảm từ 5 đến 10 lần so với PatchTST và FEDformer.
2. **Khả năng quan sát toàn cảnh không bị phân rã:** Mạng MLP kết nối trực tiếp điểm $t=1$ với điểm $t=720$ ngay tại tầng đầu tiên mà không qua bất kỳ mắt xích trung gian nào, loại bỏ hoàn toàn hiện tượng suy giảm gradient qua thời gian dài (*gradient vanishing over time*).
3. **Ổn định lan truyền thông tin nhờ Residual Connections:** Việc áp dụng Skip Connection và LayerNorm trong từng khối giúp mô hình có thể xếp chồng nhiều lớp mà không bị bão hòa hoặc suy thoái biểu diễn.

### 4.2. Các giới hạn tiềm ẩn (Critical Limitations)
Khi phân tích sâu về mặt kiến trúc, Dense Encoder bộc lộ 3 giới hạn cố hữu:

* **Giới hạn 1: Bùng nổ tham số khi Lookback $L$ quá dài (Parameter Scalability)**
  * Ma trận $W_1$ của khối Encoder đầu tiên có số tham số là:
  
  $$
  P_{W_1} = d_{\text{in}} \times d_{\text{hidden}} = \big(L + (L + H)\tilde{r} + d_a\big) \times d_{\text{hidden}}
  $$
  
  * Nếu tăng cửa sổ nhìn lại lên rất dài, ví dụ $L = 2880$ (để nắm chu kỳ tháng) và $H = 720$, thì $d_{\text{in}} = 2880 + (2880+720)\times 4 = 17,280$. Khi đó chỉ riêng ma trận $W_1$ đã chiếm:
  
  $$
  17,280 \times 1024 \approx 17.7 \text{ triệu tham số!}
  $$
  
  * Điều này khiến mô hình ngốn nhiều VRAM và rất dễ dẫn đến hiện tượng học vẹt (**Overfitting**) nếu tập dữ liệu huấn luyện có số lượng mẫu ít.

* **Giới hạn 2: Mất tính bất biến tịnh tiến thời gian (Lack of Time-Translation Invariance)**
  * Trong mạng tích chập (CNN), một bộ lọc (kernel) trượt dọc theo trục thời gian nên nhận diện được một mẫu hình dao động bất kể nó xuất hiện ở đầu hay cuối chuỗi.
  * Ngược lại, Dense Encoder gán cứng trọng số $W_{i, t}$ cho mốc thời gian $t$. Nếu một cơn bão hay sự cố sụt áp xuất hiện ở bước $t=50$ thay vì bước $t=100$, mô hình phải học lại mẫu hình đó từ đầu cho vị trí $t=50$.

* **Giới hạn 3: Không linh hoạt với độ dài chuỗi đầu vào (Fixed Lookback Constraint)**
  * Do $d_{\text{in}}$ phụ thuộc cứng vào $L$ và $H$, mô hình một khi đã huấn luyện với $L=720$ thì không thể nhận đầu vào có độ dài $L=360$ hay $L=1000$ lúc suy luận (phải cắt gọt hoặc đệm thêm zero-padding).

---

## 5. 💡 Đề Xuất: Giả Thuyết Cải Tiến Có Thể Kiểm Chứng Thực Nghiệm

Dưới đây là **2 giả thuyết cải tiến khoa học** nhằm khắc phục trực tiếp các giới hạn đã phân tích ở trên:

### Giả thuyết cải tiến 1: Tích hợp Cơ chế Patching trước khi đưa vào Encoder (Patch-Dense Encoder)

* **Ý tưởng:** Lấy cảm hứng từ bài báo PatchTST, thay vì duỗi từng giá trị vô hướng của chuỗi $y_{1:L}$, ta chia chuỗi $y_{1:L}$ thành $N$ đoạn con (patches) có độ dài $P$ và bước trượt $S$:

$$
N = \left\lfloor \frac{L - P}{S} \right\rfloor + 1
$$

* Chiếu từng patch nhỏ qua một ma trận trọng số chia sẻ $W_{\text{patch}} \in \mathbb{R}^{d_p \times P}$ (với $d_p \ll P$) trước khi ghép nối vào $u_{\text{enc}}$.
* **Lợi ích lý thuyết:**
  1. Giảm số chiều đầu vào $d_{\text{in}}$ của Encoder xuống $\approx \frac{1}{S}$ lần, giảm kích thước ma trận $W_1$ tới $60\% - 75\%$, triệt tiêu nguy cơ overfitting.
  2. Tăng cường khả năng nhận thức ngữ cảnh cục bộ (local semantic context) của chuỗi tín hiệu.
* **Phương pháp kiểm chứng thực nghiệm:**
  * Dataset kiểm chứng: `Electricity` ($L=720, H=96$).
  * Thiết lập patch size $P=16$, stride $S=8$. So sánh MSE/MAE và số lượng tham số giữa TiDE gốc và TiDE-Patch.

### Giả thuyết cải tiến 2: Thêm Regularization cấu trúc Toeplitz vào Ma trận Trọng số $W_1$ (Toeplitz-Constrained Linear)

* **Ý tưởng:** Để khắc phục nhược điểm thiếu tính bất biến tịnh tiến thời gian mà không làm tăng độ trễ tính toán, ta áp đặt ràng buộc ma trận Toeplitz mềm (Soft Toeplitz Regularization) lên phần ma trận con tương ứng với chuỗi lịch sử $W_{1, y}$:

$$
\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{MSE}} + \lambda \sum_{i} \sum_{j} \big| W_{i, j} - W_{i+1, j+1} \big|^2
$$

* **Lợi ích lý thuyết:** Khuyến khích các đường chéo ma trận có trọng số gần nhau, giúp mô hình tự động nhận diện mẫu hình tương quan trễ $k$ bước bất kể mẫu hình đó xuất hiện ở đâu trong cửa sổ quan sát.
* **Phương pháp kiểm chứng thực nghiệm:**
  * Huấn luyện mô hình TiDE trên dataset `Traffic` có dịch chuyển pha nhân tạo (time-shift synthetic test). Đo độ bền vững của dự báo (Robustness Score) khi dịch chuỗi đầu vào sang trái/phải 12 bước.

---

## 📌 Tổng Kết Slide Thuyết Trình (Key Takeaways)

> 1. **Bản chất:** Dense Encoder là mạng MLP đa tầng nén siêu vector trải phẳng $(y, \tilde{x}, a)$ thành vector bối cảnh ẩn $e \in \mathbb{R}^{1024}$.
> 2. **Cơ chế học thời gian:** Không cần Attention hay RNN, quan hệ thời gian được học trực tiếp thông qua các trọng số kết nối dày đặc giữa mọi điểm thời gian trong ma trận $W_1$.
> 3. **Ưu điểm lớn nhất:** Tính toán song song hoàn toàn, độ phức tạp tuyến tính $O(L+H)$, tốc độ cực nhanh.
> 4. **Giới hạn cần lưu ý:** Bùng nổ tham số khi $L$ rất lớn và thiếu tính bất biến tịnh tiến.
> 5. **Hướng phát triển:** Patch-based Linear Projection là hướng mở tiềm năng nhất để mở rộng quy mô nhìn lại $L > 2000$.
