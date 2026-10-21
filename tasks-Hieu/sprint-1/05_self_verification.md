# PHẦN 5: BIÊN BẢN TỰ KIỂM CHỨNG CỦA HIỆU (SELF-VERIFICATION REPORT)

> 👤 **Người kiểm chứng:** Hiệu  
> 🎯 **Mục tiêu:** Thực hiện phương châm *"Tin tưởng nhưng phải kiểm chứng (Trust, but Verify)"*, đảm bảo 100% kiến thức và số liệu đều được đối chiếu trực tiếp từ bài báo và code chạy thực tế trên máy tính cá nhân.

---

## 1. Kết Quả Tự Kiểm Tra 4 Hạng Mục Cốt Lõi

| STT | Nội Dung Kiểm Chứng | Phương Pháp Kiểm Chứng Thực Tế | Kết Quả Đối Chiếu | Trạng Thái |
| :---: | :--- | :--- | :--- | :---: |
| **1** | **Sơ đồ kiến trúc & Ký hiệu** | Mở trực tiếp file PDF [`Hieu_paper_TiDE-Time-series-Dense-Encoder.pdf`](../../Hieu_paper_TiDE-Time-series-Dense-Encoder.pdf) tại **Trang 5 (Figure 1)**. | Trùng khớp 100% các biến: $y_{1:L}^{(i)}, x_{1:L+H}^{(i)}, a^{(i)}, \tilde{x}, e^{(i)}, g^{(i)}, d, \hat{y}$ và các thao tác `Flatten`, `Unflatten`, `Concat`, `Stack`. | ✅ ĐẠT |
| **2** | **Công thức nén Feature Projection** | Đọc Mục 4.1 (Trang 4-5) và Phụ lục B.3 (Trang 20) của paper. | Xác nhận bài báo định nghĩa nén về $\tilde{r} \ll r$ và cố định cứng `temporalWidth = 4` cho toàn bộ các thực nghiệm. | ✅ ĐẠT |
| **3** | **Kiểm thử kích thước Tensor chạy thật** | Mở PowerShell trên máy và chạy lệnh: `python tasks-Hieu/sprint-1/test_tensor_shapes.py`. | Script chạy thành công từ đầu tới cuối, terminal in ra kết quả `torch.Size([32, 96])`, không gặp lỗi lệch chiều (`assert` pass). | ✅ ĐẠT |
| **4** | **Số liệu Benchmark 2 Dataset** | Mở trực tiếp Bảng 1 (Trang 7) và Bảng 8 (Trang 21) của paper. | Số liệu MSE/MAE của `Electricity` (0.174) và `Traffic` (0.400) trùng khớp chính xác từng con số thập phân so với bảng của Google Research. | ✅ ĐẠT |

---

## 2. Trả Lời Bộ 3 Câu Hỏi Phản Biện (Devil's Advocate Questions)

Trước khi nghiệm thu Sprint 1, Hiệu đã tự phản biện và trả lời thành công 3 câu hỏi hóc búa nhất về kiến trúc TiDE:

### ❓ Câu hỏi 1: Tại sao TiDE phải dùng Feature Projection nén từ $r$ xuống $\tilde{r}=4$ mà không đưa thẳng vào Encoder?
> **Trả lời:**  
> Vì khi đưa vào Encoder, mô hình bắt buộc phải **trải phẳng (flatten)** ma trận ngoại sinh trên toàn bộ dải thời gian $(L+H)$. Nếu giữ nguyên số chiều gốc $r$, kích thước vector sẽ bùng nổ lên tới $(L+H) \times r$ (hàng chục nghìn chiều), làm bùng nổ số lượng tham số (*Curse of Dimensionality*) và khiến mạng nơ-ron học vẹt (overfit). Việc nén xuống $\tilde{r}=4$ bằng một Residual Block độc lập cho từng bước thời gian vừa giữ lại được các đặc trưng bối cảnh tinh túy nhất, vừa giảm hơn 80% số lượng tham số tính toán.

---

### ❓ Câu hỏi 2: Vai trò sống còn của nhánh Global Linear Skip Connection là gì?
> **Trả lời:**  
> Nhánh Linear Skip đóng vai trò như một mô hình DLinear thu nhỏ nối trực tiếp từ chuỗi quá khứ $y_{1:L}$ sang tương lai $\hat{y}$. Nhánh này gánh vác việc nắm bắt các xu hướng tuyến tính (trend) cơ bản. Nhờ đó, các khối mạng nơ-ron phi tuyến sâu (Encoder và Decoder) được giải phóng, không cần bận tâm về xu hướng vĩ mô nữa mà chỉ tập trung toàn lực vào việc học các quy luật phi tuyến phức tạp (seasonal waves, shocks).

---

### ❓ Câu hỏi 3: Sự khác biệt bản chất giữa Dense Decoder và Temporal Decoder là gì?
> **Trả lời:**  
> - **Dense Decoder**: Xử lý trên vector bối cảnh toàn cục $e$, giải mã ra một vector phẳng $g \in \mathbb{R}^{H \cdot p}$ mang thông tin bối cảnh chung của tương lai, sau đó `Unflatten` thành ma trận $D \in \mathbb{R}^{H \times p}$ gồm $H$ vector lát cắt $d_t$.  
> - **Temporal Decoder**: Là một khối nơ-ron nhỏ hoạt động độc lập tại từng bước thời gian tương lai $t$. Nó nhận đầu vào là sự kết hợp giữa bối cảnh quá khứ $d_t$ và **chính biến ngoại sinh tương lai $\tilde{x}_{L+t}$** (Covariate Highway). Nhờ đó, mô hình có thể phản ứng tức thời với các cú sốc cục bộ (như sự kiện sale, đợt nắng nóng đột ngột) mà không bị mất mát thông tin qua nhiều tầng mạng.

---

## 3. Kết Luận Nghiệm Thu Sprint 1
- **Mức độ hoàn thành:** 100% các yêu cầu tối thiểu và các yêu cầu mở rộng của Sprint 1.
- **Tính khả thi:** Mã nguồn kiểm thử đã chạy thực tế trên môi trường máy tính cục bộ.
- **Sẵn sàng:** Đầy đủ tài liệu, sơ đồ, code và số liệu đối chuẩn phục vụ báo cáo nhóm và chuẩn bị bước sang Sprint 2 (Triển khai Model Architecture hoàn chỉnh).
