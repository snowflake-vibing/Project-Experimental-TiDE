# 🎙️ HỒ SƠ CHUYÊN ĐỀ BÁO CÁO THUYẾT TRÌNH: DENSE DECODER (TiDE)

> [!IMPORTANT]
> **Phạm vi phân công (Scope của Hiệu):**  
> **Chủ đề:** `Present - Dense Decoder`  
> **Trọng tâm được giao:** Tập trung vào cách TiDE chuyển biểu diễn ẩn thành các vector dành cho từng bước dự báo.  
> **Yêu cầu bắt buộc:** Hoàn thành tối thiểu 5 tiêu chí (Lý thuyết • Toán học • Source code • Phân tích • Đề xuất) theo đúng hướng dẫn thuyết trình của hội đồng.  
> 👤 **Người thực hiện:** Hiệu  
> 🤖 **Trợ lý kỹ thuật:** Antigravity

---

## 🧭 File Báo Cáo Chuyên Đề & Bộ Slide Chính Thức

👉 [**`present_dense_decoder.md`**](./present_dense_decoder.md): **Báo cáo chuyên sâu giải phẫu toàn diện khối Dense Decoder**

👉 [**`slide/`**](./slide/README.md): **Thư mục 5 Slide nội dung chi tiết** (kèm hình ảnh PNG độ phân giải cao, sơ đồ Mermaid và lời thoại thuyết trình từng slide)

Tài liệu được biên soạn tinh gọn, không lan man, tập trung 100% vào đúng 6 gạch đầu dòng nhiệm vụ của scope:
1. **Kiến trúc Dense Decoder gồm $n_d$ Residual Blocks:** Cấu tạo các khối trung gian và khối ngõ ra đặc biệt.
2. **Cách chuyển $e^{(i)}$ thành vector $g^{(i)}$ có kích thước $H \times p$:** Cơ chế chiếu chiều và công thức ma trận ngõ ra.
3. **Vai trò của thao tác Reshape (Unflatten):** Khôi phục trục thời gian $H$ với chi phí bộ nhớ $O(1)$.
4. **Ý nghĩa của vector $d_t^{(i)}$ cho từng thời điểm dự báo:** Vector bối cảnh tiềm ẩn toàn cục riêng biệt cho từng mốc tương lai.
5. **Cơ chế sinh biểu diễn toàn bộ horizon phi tự quy (Non-autoregressive):** Dự báo song song One-shot, không sinh tuần tự, không tích lũy sai số.
6. **Ảnh hưởng của decoder output dimension $p$ và số lớp decoder $n_d$:** Phân tích điểm cân bằng giữa nghẽn thông tin và bùng nổ tham số.

---

## 📊 Bảng Đối Chiếu 5 Tiêu Chí Bắt Buộc Của Hội Đồng

| STT | 5 Tiêu Chí Bắt Buộc | Nội Dung Hiện Thực Trong File `present_dense_decoder.md` | Đánh Giá |
| :---: | :--- | :--- | :---: |
| **1** | **Lý Thuyết** | - **Chức năng:** Giải nén vector tiềm ẩn toàn cục $e \in \mathbb{R}^{d_{\text{hidden}}}$ thành ma trận đặc trưng $D \in \mathbb{R}^{H \times p}$.<br>- **Nguyên lý:** Non-autoregressive (One-shot) sinh biểu diễn cho cả $H$ bước trong đúng 1 lần forward.<br>- **Lý do thiết kế:** Triệt tiêu hiện tượng tích lũy sai số (Error Accumulation), tối đa hóa song song trên GPU ($O(1)$ sequential step), tách bạch bối cảnh toàn cục và ngoại sinh cục bộ. | ✅ **Đạt Chuẩn** |
| **2** | **Toán Học** | - Công thức giải nén: $g = \text{DenseDecoder}(e) \in \mathbb{R}^{H \cdot p}$.<br>- Công thức Unflatten: $D = \text{Reshape}(g, [H, p]) \in \mathbb{R}^{H \times p}$.<br>- Phân rã thành $H$ vector lát cắt: $d_t \in \mathbb{R}^p \;\forall t \in \{1, \dots, H\}$.<br>- Bảng kích thước tensor mẫu (Electricity): `(32, 1024)` $\to$ `(32, 1536)` $\to$ `(32, 96, 16)`. | ✅ **Đạt Chuẩn** |
| **3** | **Source Code** | - Trích xuất chính xác code từ repo gốc `google-research/tide/models.py` (`#L133-L140`, `#L207-L210`).<br>- Chỉ rõ `_make_dnn_residual(hidden_dims[:-1] + [decoder_output_dim * pred_len])` và `tf.reshape(decoder_out, [bsize, -1, pred_len])`.<br>- Đối chiếu với bản PyTorch `simulate_forward_pass.py` dòng 115–126 và 214–226 (`D = g.view(-1, self.H, self.p)`). | ✅ **Đạt Chuẩn** |
| **4** | **Phân Tích** | - **Ảnh hưởng của $p$:** $p$ nhỏ gây nghẽn cổ chai thông tin (tăng MSE/MAE); $p$ lớn gây bùng nổ tham số ($47\text{M}$ params khi $H=720, p=64$) dẫn đến overfitting.<br>- **Số lớp $n_d$:** $n_d = 2$ là tối ưu, tăng $n_d \ge 3$ không hiệu quả vì đã có Linear Skip.<br>- **Giới hạn tiềm ẩn:** Cố định Horizon $H$ và nguy cơ gián đoạn giữa các lát cắt $d_t, d_{t+1}$. | ✅ **Đạt Chuẩn** |
| **5** | **Đề Xuất** | - **Giả thuyết 1 (Low-Rank Factorization):** Phân rã ma trận tầng cuối $W = A \times B$ với hạng thấp $r_{\text{rank}}=64$, giảm tới **$93\%$ số tham số** khi $H=720$, chống overfitting.<br>- **Giả thuyết 2 (Temporal-Smoothed Decoder):** Tích hợp 1 lớp Depthwise Conv1D nhẹ ($k=3$) sau khi Reshape để làm mịn tính liên tục thời gian giữa các vector $d_t$. | ✅ **Đạt Chuẩn** |

---

## 🎯 Kịch Bản & Gợi Ý Trả Lời Vấn Đáp Phản Biện (Q&A Cheat-Sheet)

Dành cho Hiệu khi thầy cô / hội đồng đặt câu hỏi xoáy sâu vào Dense Decoder:

### ❓ Câu 1: "Tại sao Dense Decoder không sinh thẳng ra giá trị dự báo $\hat{y}$ luôn mà phải sinh qua ma trận $D$ kích thước $H \times p$?"
> 💬 **Trả lời:**  
> Nếu Dense Decoder ánh xạ thẳng ra $\hat{y} \in \mathbb{R}^H$, mô hình sẽ **mất hoàn toàn cơ hội kết hợp với biến ngoại sinh tương lai tại từng bước cụ thể**.  
> Việc sinh ra ma trận $D \in \mathbb{R}^{H \times p}$ (mỗi bước $t$ có một vector $d_t \in \mathbb{R}^p$) cho phép tầng tiếp theo là `Temporal Decoder` ghép cặp $d_t$ với biến ngoại sinh tương lai của chính thời điểm đó ($\tilde{x}_{L+t}$), ví dụ: nhiệt độ dự báo hay ngày nghỉ lễ tại đúng giờ thứ $t$, giúp dự báo chuẩn xác hơn rất nhiều.

### ❓ Câu 2: "Ưu thế lớn nhất của Dense Decoder so với Decoder của Transformer hay LSTM là gì?"
> 💬 **Trả lời:**  
> Đó là **cơ chế sinh phi tự quy (Non-autoregressive / One-shot)**.  
> - Các mô hình Autoregressive (LSTM, Informer) phải dự báo từng bước tuần tự: đoán bước 1, lấy kết quả đoán bước 2... nên bị lỗi tích lũy sai số (sai một li đi một dặm) và tốn $H$ bước tính toán tuần tự.  
> - Dense Decoder của TiDE xuất toàn bộ $H$ bước cùng lúc chỉ trong **1 bước tính toán duy nhất**, triệt tiêu tích lũy sai số và khai thác tối đa năng lực xử lý song song của GPU.

### ❓ Câu 3: "Khi dự báo dài hạn (ví dụ $H=720$), Dense Decoder gặp vấn đề gì và giải quyết thế nào?"
> 💬 **Trả lời:**  
> Tầng cuối của Decoder phải ánh xạ sang số chiều $H \times p = 720 \times 16 = 11,520$ chiều, khiến số tham số ma trận ngõ ra lên tới $\approx 11.8\text{ triệu tham số}$, dễ gây overfitting trên các tập dữ liệu nhỏ.  
> Em đề xuất giải pháp **Phân rã ma trận ngõ ra theo hạng thấp (Low-Rank Factorization)**: biểu diễn ma trận ngõ ra dưới dạng tích của 2 ma trận với rank $r=64$. Giải pháp này giúp giảm hơn $93\%$ số tham số của tầng cuối mà vẫn giữ được thông tin bối cảnh chủ đạo.
