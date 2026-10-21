# BẢNG TIẾN ĐỘ & CHECKLIST CÔNG VIỆC SPRINT 1

> 👤 **Người thực hiện:** Hiệu  
> 🤖 **Trợ lý kỹ thuật:** Antigravity  
> ⏰ **Hạn chót hoàn thành:** 21h00 – 24/10/2026  
> 🎯 **Mục tiêu nghiệm thu:** Hoàn thành đầy đủ 4 hạng mục kỹ thuật cốt lõi theo `requirement.md` và kiểm chứng độc lập.

---

## 📋 1. Bảng Checklist Công Việc Chi Tiết Theo Từng File Báo Cáo

| STT | Hạng Mục Công Việc | Tài Liệu Tương Ứng | Trạng Thái Antigravity Soạn | Trạng Thái Hiệu Tự Check | Ghi Chú & Nội Dung Trọng Tâm |
| :---: | :--- | :--- | :---: | :---: | :--- |
| **1** | **Sơ đồ Kiến trúc TiDE & Giải thích Ký hiệu** | [`01_architecture_diagram.md`](./01_architecture_diagram.md) | ✅ Hoàn thành | ✅ **ĐÃ TỰ CHECK** | Khớp 100% với Figure 1 (Trang 5 PDF), tiếng Việt có dấu, đầy đủ các biến $y, x, a, \tilde{x}, e, g, d, \hat{y}$ và 5 thao tác. |
| **2** | **Giải phẫu Luồng Forward Pass (Paper vs Code)** | [`02_forward_pass_walkthrough.md`](./02_forward_pass_walkthrough.md) | ✅ Hoàn thành | ⏳ **CHƯA CHECK** | Bảng đối chiếu 11 bước, phân tích sâu từng tầng toán học và code PyTorch class `TiDEForwardPass`. |
| **3** | **Theo dõi Kích thước Tensor & Script Kiểm thử** | [`03_tensor_shapes_and_testing.md`](./03_tensor_shapes_and_testing.md)<br>[`test_tensor_shapes.py`](./test_tensor_shapes.py) | ✅ Hoàn thành | ⏳ **CHƯA CHECK** | Bảng shape chi tiết từng nút mạng; script Python chạy thành công trên terminal, kiểm chứng không lệch chiều. |
| **4** | **Cấu hình Siêu tham số & Kết quả 2 Dataset** | [`04_hyperparameters_and_benchmarks.md`](./04_hyperparameters_and_benchmarks.md) | ✅ Hoàn thành | ⏳ **CHƯA CHECK** | 2 tập `Electricity` & `Traffic`; bảng siêu tham số từ Table 8, bảng kết quả MSE/MAE từ Table 1 của paper. |
| **5** | **Biên bản Tự kiểm chứng & Phản biện** | [`05_self_verification.md`](./05_self_verification.md) | ✅ Hoàn thành | ⏳ **CHƯA CHECK** | Ghi nhận bằng chứng đối chiếu thực tế và câu trả lời bộ 3 câu hỏi phản biện hóc búa nhất. |

---

## 🔍 2. Nhật Ký Hiệu Tự Kiểm Chứng Độc Lập (Self-Verification Log)

> Nguyên tắc: *"Tin tưởng nhưng phải kiểm chứng (Trust, but Verify)"* — Chỉ đánh dấu hoàn thành sau khi chính bạn đã trực tiếp đối chiếu với file PDF gốc hoặc chạy code kiểm thử trên máy tính cá nhân.

### [x] Phần 1: Tự Check Sơ Đồ Kiến Trúc & Ký Hiệu Toán Học (ĐÃ HOÀN THÀNH ✅)
- [x] Mở trực tiếp file PDF [`Hieu_paper_TiDE-Time-series-Dense-Encoder.pdf`](../../project/Hieu_paper_TiDE-Time-series-Dense-Encoder.pdf) tại **Trang 5, Hình 1 (Figure 1)**.
- [x] Đối chiếu đầy đủ 3 nguồn đầu vào: $y_{1:L}^{(i)}$ (Lookback), $x_{1:L+H}^{(i)}$ (Dynamic Covariates), $a^{(i)}$ (Attributes).
- [x] Đối chiếu đúng các biến trung gian: $\tilde{x}_{1:L+H}^{(i)}$, vector mã hóa $e^{(i)}$, vector giải mã phẳng $g^{(i)}$, ma trận bối cảnh $d_{L+1:L+H}^{(i)}$, ngoại sinh tương lai $\tilde{x}_{L+1:L+H}^{(i)}$, và dự báo cuối $\hat{y}_{L+1:L+H}^{(i)}$.
- [x] Đối chiếu đúng 5 thao tác nối ghép: `Feature Projection (per time-step)`, `Flatten`, `Concat`, `Unflatten`, `Stack` và nhánh tắt `Residual (+)`.
- [x] Sơ đồ Mermaid viết bằng tiếng Việt có dấu đầy đủ, hiển thị chuẩn xác không lỗi cú pháp trên GitHub.

---

### [ ] Phần 2: Tự Check Giải Phẫu Luồng Forward Pass (CHƯA CHECK ⏳)
- [ ] Đọc file [`02_forward_pass_walkthrough.md`](./02_forward_pass_walkthrough.md) và đối chiếu với Mục 4 của bài báo (Trang 4-6).
- [ ] Kiểm tra xem 11 bước xử lý có ánh xạ đúng từ công thức toán sang code PyTorch không.
- [ ] Kiểm tra xem bản thân đã hiểu rõ bản chất tại sao phải làm từng bước (như tại sao phải nén ngoại sinh, tại sao phải flatten trước Encoder, tại sao phải unflatten sau Decoder).

---

### [ ] Phần 3: Tự Check Kích Thước Tensor & Chạy Script Test (CHƯA CHECK ⏳)
- [ ] Mở PowerShell trên máy tính và tự gõ lệnh chạy script:
  ```powershell
  python tasks-Hieu/sprint-1/test_tensor_shapes.py
  ```
- [ ] Kiểm tra xem terminal có in ra đầy đủ 14 bước kích thước tensor không, có bị lỗi `RuntimeError / AssertionError` nào không.
- [ ] Tự nhẩm lại phép tính số chiều đầu vào Encoder: $720 + (720 + 96) \times 4 = 3984$ xem có khớp với kết quả in ra không.

---

### [ ] Phần 4: Tự Check Cấu Hình & Kết Quả 2 Dataset (CHƯA CHECK ⏳)
- [ ] Mở file PDF tại **Trang 7, Bảng 1 (Table 1)**: Soi số liệu MSE/MAE của `Electricity` (0.174) và `Traffic` (0.400) với các đối thủ PatchTST, DLinear.
- [ ] Mở file PDF tại **Trang 21, Bảng 8 (Table 8)**: Soi các siêu tham số `hiddenSize`, `numEncoderLayers`, `numDecoderLayers`, `decoderOutputDim`, `temporalWidth=4`.

---

### [ ] Phần 5: Tự Trả Lời Bộ 3 Câu Hỏi Phản Biện (CHƯA CHECK ⏳)
- [ ] Đọc và tự trả lời trôi chảy 3 câu hỏi phản biện trong file [`05_self_verification.md`](./05_self_verification.md):
  1. *Tại sao phải dùng Feature Projection nén từ $r$ xuống $\tilde{r}=4$?*
  2. *Vai trò sống còn của nhánh Global Linear Skip Connection là gì?*
  3. *Sự khác biệt bản chất giữa Dense Decoder và Temporal Decoder là gì?*

---

## 📌 3. Tiêu Chí Đánh Giá Hoàn Thành Nghiệm Thu (Definition of Done - DoD)
- [x] Đã có sơ đồ kiến trúc hoàn chỉnh tiếng Việt có dấu, khớp 100% với Figure 1 của paper (Đã nghiệm thu Phần 1).
- [ ] Hiệu đã tự đọc hiểu và duyệt qua Phần 2 (Forward Pass).
- [ ] Hiệu đã tự chạy lệnh kiểm thử Tensor shape trên máy tính thật (Phần 3).
- [ ] Hiệu đã tự đối chiếu bảng số liệu thực nghiệm Table 1 & Table 8 trong PDF (Phần 4).
- [ ] Hiệu đã tự trả lời được 3 câu hỏi phản biện cốt lõi (Phần 5).
