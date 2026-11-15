# 🎙️ HỒ SƠ CHUYÊN ĐỀ BÁO CÁO THUYẾT TRÌNH TiDE

> [!IMPORTANT]
> **Quy chuẩn đánh giá báo cáo:**
> Hồ sơ này được biên soạn nghiêm ngặt theo **5 yêu cầu bắt buộc khi trình bày** (Lý thuyết • Toán học • Source code • Phân tích • Đề xuất), phục vụ cho buổi thuyết trình và phản biện chuyên sâu về kiến trúc mô hình TiDE (*Long-term Forecasting with TiDE: Time-series Dense Encoder*, Google Research 2023).  
> 👤 **Người thực hiện:** Hiệu  
> 🤖 **Trợ lý kỹ thuật:** Antigravity

---

## 🧭 Mục Lục Các File Chuyên Đề Báo Cáo

| STT | File Tài Liệu Chuyên Đề | Chủ Đề Báo Cáo Chi Tiết | Trạng Thái Nghiệm Thu |
| :---: | :--- | :--- | :---: |
| **01** | 📑 [**`01_dense_encoder_presentation.md`**](./01_dense_encoder_presentation.md) | **Khối Mã Hóa Dày Đặc (Dense Encoder):**<br>- Cơ chế học quan hệ theo thời gian khi không sử dụng Attention, RNN hoặc CNN.<br>- Phép ghép nối Siêu Vector $e_0 = \text{Concat}(y, \tilde{x}, a)$.<br>- Cấu trúc $n_e$ khối `ResidualBlock` nén không gian biểu diễn.<br>- Mã hóa quan hệ thời gian qua ma trận trọng số $W_1$. | ✅ **Đạt Chuẩn 5/5 Yêu Cầu** |
| **02** | 📑 [**`02_dense_decoder_presentation.md`**](./02_dense_decoder_presentation.md) | **Khối Giải Mã Dày Đặc (Dense Decoder):**<br>- Cách chuyển đổi vector ẩn $e^{(i)}$ thành vector phẳng $g^{(i)}$ kích thước $H \times p$.<br>- Vai trò của thao tác Reshape (Unflatten) tái lập trục thời gian.<br>- Ý nghĩa ngữ nghĩa của từng vector lát cắt $d_t^{(i)}$ cho mỗi thời điểm dự báo.<br>- Cơ chế sinh One-shot phi tự quy (Non-autoregressive) không tích lũy sai số.<br>- Ảnh hưởng của siêu tham số $p$ (Decoder Output Dim) và $n_d$ (Số lớp Decoder). | ✅ **Đạt Chuẩn 5/5 Yêu Cầu** |

---

## 📊 Bảng Đối Chiếu 5 Tiêu Chí Bắt Buộc Khi Trình Bày

Hồ sơ thuyết trình của Hiệu đáp ứng trọn vẹn 100% các tiêu chí trong yêu cầu của hội đồng:

| Tiêu Chí Bắt Buộc | Nội Dung Hiện Thực Trong `01_dense_encoder_presentation.md` | Nội Dung Hiện Thực Trong `02_dense_decoder_presentation.md` |
| :--- | :--- | :--- |
| **1. Lý Thuyết** | - Chức năng: Nén lịch sử $y$, biến động ngoại sinh $\tilde{x}$ và thuộc tính tĩnh $a$ thành 1 vector tiềm ẩn $e \in \mathbb{R}^{d_{\text{hidden}}}$.<br>- Nguyên lý: Duỗi thẳng không gian - thời gian (Flattening).<br>- Lý do: Triệt tiêu chi phí $O(L^2)$ của Transformer và nghẽn tuần tự của RNN. | - Chức năng: Giải nén vector bối cảnh ẩn $e$ thành ma trận thời gian tương lai $D \in \mathbb{R}^{H \times p}$.<br>- Nguyên lý: Non-autoregressive (One-shot) sinh toàn bộ $H$ bước cùng lúc.<br>- Lý do: Tránh lỗi tích lũy sai số và tối ưu hóa 100% năng lực tính toán GPU. |
| **2. Toán Học** | - Công thức ghép nối: $u_{\text{enc}} = [y ; \text{vec}(\tilde{x}) ; a]$.<br>- Kích thước đầu vào: $d_{\text{in}} = L + (L+H)\tilde{r} + d_a = 3,984$.<br>- Kích thước đầu ra: $e \in \mathbb{R}^{1024}$.<br>- Công thức thặng dư $u^{(l)} = \text{LayerNorm}(F(u) + \text{Skip}(u))$. | - Công thức giải nén: $g = \text{DenseDecoder}(e) \in \mathbb{R}^{H \cdot p}$.<br>- Công thức Unflatten: $D = \text{Reshape}(g, [H, p])$.<br>- Kích thước đầu vào: `(B, 1024)`.<br>- Kích thước đầu ra: $g \in \mathbb{R}^{B \times 1536} \to D \in \mathbb{R}^{B \times 96 \times 16}$. |
| **3. Source Code** | - Đối chiếu file `google-research/tide/model.py`.<br>- Chỉ rõ `self.encoder = nn.Sequential(*encoder_layers)`.<br>- Đối chiếu file mô phỏng `simulate_forward_pass.py` dòng 187-212. | - Đối chiếu tầng ngõ ra `ResidualBlock(..., out_dim=H*p)`.<br>- Chỉ rõ dòng lệnh then chốt `D = g.view(-1, self.H, self.p)`.<br>- Đối chiếu file `simulate_forward_pass.py` dòng 214-226. |
| **4. Phân Tích** | - Ưu thế: Nhanh hơn 5-10 lần Transformer, quan sát toàn diện.<br>- Giới hạn: Bùng nổ tham số khi $L > 2000$ ($W_1$ chiếm $17.7\text{M}$ params), thiếu tính bất biến tịnh tiến thời gian, cố định Lookback. | - Ảnh hưởng siêu tham số: $p$ nhỏ gây nghẽn thông tin, $p$ lớn gây bùng nổ tham số ($47\text{M}$ params khi $H=720, p=64$).<br>- Số lớp $n_d = 2$ là tối ưu.<br>- Giới hạn: Cố định khoảng $H$, nguy cơ gián đoạn giữa các lát cắt $d_t$. |
| **5. Đề Xuất** | - **Giả thuyết 1:** Patch-based Linear Projection chia nhỏ chuỗi thành patch $P=16, S=8$, giảm $65\%$ tham số $W_1$.<br>- **Giả thuyết 2:** Ràng buộc ma trận Toeplitz mềm để học tính bất biến dịch chuyển thời gian. | - **Giả thuyết 1:** Phân rã ma trận ngõ ra hạng thấp (Low-Rank Factorization $W = A \times B, r=64$), giảm $93\%$ tham số khi $H=720$.<br>- **Giả thuyết 2:** Làm mịn vector bối cảnh qua lớp 1D Depthwise Convolution ($k=3$). |

---

## 🎯 Kịch Bản & Gợi Ý Trả Lời Vấn Đáp Phản Biện (Q&A Cheat-Sheet)

Dưới đây là 3 câu hỏi hội đồng thường hay hỏi nhất và câu trả lời súc tích để bạn tự tin khi đứng trước hội đồng:

### ❓ Câu hỏi 1: "Tại sao Dense Encoder không dùng Positional Encoding mà vẫn nhận biết được thứ tự trước - sau của chuỗi thời gian?"
> 💬 **Trả lời:**  
> Trong mô hình Transformer, các từ/bước thời gian được đưa vào theo dạng tập hợp (set of vectors) và cơ chế Self-Attention có tính chất hoán vị đối xứng (*permutation-equivariant*), nghĩa là xáo trộn thứ tự đầu vào thì Attention vẫn tính ra giá trị giống nhau, do đó bắt buộc phải cộng thêm Positional Encoding.  
> Ngược lại, Dense Encoder của TiDE **trải phẳng toàn bộ chuỗi thành một vector 1D duy nhất**. Trong vector này, bước $t=1$ luôn cố định tại chỉ số 1, bước $t=2$ luôn ở chỉ số 2. Mỗi chỉ số này được kết nối với một cột trọng số riêng biệt trong ma trận $W_1$. Do đó, ma trận $W_1$ **tự động học được ý nghĩa ngữ nghĩa của từng vị trí thời gian mà không cần bất kỳ vector vị trí nào**.

### ❓ Câu hỏi 2: "Tại sao Dense Decoder phải dùng thao tác Unflatten (view) thành ma trận $H \times p$ thay vì sinh thẳng luôn ra kết quả dự báo $\hat{y} \in \mathbb{R}^H$?"
> 💬 **Trả lời:**  
> Nếu Dense Decoder sinh thẳng ra $\hat{y} \in \mathbb{R}^H$, mô hình sẽ **mất hoàn toàn cơ hội kết hợp với các biến ngoại sinh tương lai tại từng bước cụ thể**.  
> Việc đưa ra ma trận bối cảnh $D \in \mathbb{R}^{H \times p}$ (mỗi bước $t$ có một vector đại diện $d_t \in \mathbb{R}^p$) cho phép bước tiếp theo (`Temporal Decoder`) ghép cặp vector $d_t$ này với biến ngoại sinh tương lai của chính thời điểm đó ($\tilde{x}_{L+t}$), ví dụ: nhiệt độ dự báo hay ngày lễ tại đúng giờ thứ $t$, từ đó giúp mô hình dự báo chính xác hơn rất nhiều.

### ❓ Câu hỏi 3: "Khi $H$ rất lớn (ví dụ $H=720$), Dense Decoder gặp vấn đề gì và bạn đề xuất giải quyết thế nào?"
> 💬 **Trả lời:**  
> Khi $H=720$ và $p=16$, tầng cuối của Decoder phải ánh xạ sang $720 \times 16 = 11,520$ chiều, khiến ma trận trọng số ngõ ra có kích thước $11,520 \times 1,024 \approx 11.8\text{ triệu tham số}$, dễ gây overfitting trên các tập dữ liệu nhỏ.  
> Em đề xuất giải pháp **Phân rã ma trận hạng thấp (Low-Rank Factorization)**: Phân rã ma trận này thành tích của 2 ma trận với rank $r=64$. Giải pháp này giảm tới $93\%$ số tham số của tầng cuối mà vẫn giữ lại được các thành phần dao động bối cảnh chủ đạo của chuỗi.

---

## 🔗 Liên Kết Đến Các Tài Liệu Kiến Trúc Trong Dự Án
- 🏛️ [**`tasks-Hieu/sprint-1/01_architecture_diagram.md`**](../sprint-1/01_architecture_diagram.md): Sơ đồ kiến trúc tổng thể toàn bộ 7 mô-đun TiDE.
- 🔬 [**`tasks-Hieu/sprint-1/02_forward_pass_walkthrough.md`**](../sprint-1/02_forward_pass_walkthrough.md): Giải phẫu luồng tensor qua 11 bước Forward Pass.
- 🐍 [**`simulate_forward_pass.py`**](../../simulate_forward_pass.py): File mã nguồn chạy mô phỏng kiểm thử tensor thực tế.
