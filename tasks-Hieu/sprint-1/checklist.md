# BẢNG TIẾN ĐỘ & CHECKLIST CÔNG VIỆC SPRINT 1

> [!NOTE]
> **Điều hướng tài liệu**: [🏠 Trang Chủ](../../README.md) • [📋 Sprint 1 README](./README.md) • [📜 Yêu Cầu Sprint](./requirement.md) • [🛡️ Tự Kiểm Chứng](./05_self_verification.md)  
> 👤 **Người thực hiện:** Hiệu  
> 🤖 **Trợ lý kỹ thuật:** Antigravity  
> ⏰ **Hạn chót hoàn thành:** 21h00 – Ngày 24/10/2026  
> 🎯 **Mục tiêu nghiệm thu:** Hoàn thành đầy đủ 4 hạng mục kỹ thuật cốt lõi theo [`requirement.md`](./requirement.md) và tự kiểm chứng độc lập 100%.

---

## 📊 Thước Đo Tiến Độ Tổng Thể (Overall Progress)

| Chỉ số theo dõi | Trạng thái hiện tại | Tỷ lệ hoàn thành |
| :--- | :---: | :---: |
| **Tài liệu Antigravity soạn thảo** | 5/5 tài liệu hoàn tất | **100%** ✅ |
| **Nội dung Hiệu đã tự kiểm chứng** | 3/5 hạng mục đã duyệt | **60%** 🔄 |
| **Mục tiêu thời gian (Deadline)** | 21h00 – 24/10/2026 | *Đúng tiến độ* ⏱️ |

---

## 📋 1. Bảng Checklist Công Việc Chi Tiết Theo Từng File Báo Cáo

| STT | Hạng Mục Công Việc | Tài Liệu Tương Ứng | Trạng Thái Soạn Thảo | Trạng Thái Hiệu TỰ Check | Ghi Chú & Nội Dung Trọng Tâm |
| :---: | :--- | :--- | :---: | :---: | :--- |
| **01** | **Sơ đồ Kiến trúc TiDE & Ký hiệu Toán học** | [`01_architecture_diagram.md`](./01_architecture_diagram.md) | ✅ Hoàn thành | ✅ **ĐÃ TỰ CHECK** | Khớp 100% với Figure 1 (Trang 5 PDF), tiếng Việt có dấu, đầy đủ các biến $y, x, a, \tilde{x}, e, g, d, \hat{y}$ và 5 thao tác. |
| **02** | **Giải phẫu Luồng Forward Pass (Paper vs Code)** | [`02_forward_pass_walkthrough.md`](./02_forward_pass_walkthrough.md) | ✅ Hoàn thành | ✅ **ĐÃ TỰ CHECK** | Bảng đối chiếu 11 bước, phân tích sâu từng tầng toán học Section 4 và code PyTorch class `TiDEForwardPass`. |
| **03** | **Theo dõi Kích thước Tensor & Script Kiểm thử** | [`03_tensor_shapes_and_testing.md`](./03_tensor_shapes_and_testing.md)<br>[`test_tensor_shapes.py`](./test_tensor_shapes.py) | ✅ Hoàn thành | ✅ **ĐÃ TỰ CHECK** | Bảng shape chi tiết từng nút mạng; script Python chạy thành công trên terminal, kiểm chứng không lệch chiều. |
| **04** | **Cấu hình Siêu tham số & Kết quả 7 Datasets** | [`04_hyperparameters_and_benchmarks.md`](./04_hyperparameters_and_benchmarks.md) | ✅ Hoàn thành | ⏳ **CHƯA CHECK** | Cả 7 tập benchmark; bảng siêu tham số từ Table 8, bảng kết quả MSE/MAE từ Table 2 của bài báo. |
| **05** | **Biên bản Tự kiểm chứng & Câu hỏi Phản biện** | [`05_self_verification.md`](./05_self_verification.md) | ✅ Hoàn thành | ⏳ **CHƯA CHECK** | Ghi nhận bằng chứng đối chiếu thực tế từng trang PDF và câu trả lời bộ 3 câu hỏi phản biện hóc búa nhất. |

---

## 🔍 2. Nhật Ký Hiệu Tự Kiểm Chứng Độc Lập (Self-Verification Log)

> [!IMPORTANT]
> **Nguyên tắc hành động**: *"Tin tưởng nhưng phải kiểm chứng (Trust, but Verify)"*  
> Chỉ đánh dấu hoàn thành sau khi chính bạn đã trực tiếp mở file PDF gốc để đối chiếu số liệu hoặc tự gõ lệnh chạy code kiểm thử trên terminal máy tính cá nhân.

### 2.1. Phần 1: Sơ Đồ Kiến Trúc & Ký Hiệu Toán Học (✅ ĐÃ HOÀN THÀNH)

- [x] Mở trực tiếp file PDF [`Hieu_paper_TiDE-Time-series-Dense-Encoder.pdf`](../../Hieu_paper_TiDE-Time-series-Dense-Encoder.pdf) tại **Trang 5, Hình 1 (Figure 1)**.
- [x] Đối chiếu đầy đủ 3 nguồn đầu vào: $y_{1:L}^{(i)}$ (Lookback), $x_{1:L+H}^{(i)}$ (Dynamic Covariates), $a^{(i)}$ (Attributes).
- [x] Đối chiếu đúng các biến trung gian: $\tilde{x}_{1:L+H}^{(i)}$, vector mã hóa $e^{(i)}$, vector giải mã phẳng $g^{(i)}$, ma trận bối cảnh $d_{L+1:L+H}^{(i)}$, ngoại sinh tương lai $\tilde{x}_{L+1:L+H}^{(i)}$, và dự báo cuối $\hat{y}_{L+1:L+H}^{(i)}$.
- [x] Đối chiếu đúng 5 thao tác nối ghép: `Feature Projection (per time-step)`, `Flatten`, `Concat`, `Unflatten`, `Stack` và nhánh tắt `Residual (+)`.
- [x] Sơ đồ Mermaid viết bằng tiếng Việt có dấu đầy đủ, hiển thị chuẩn xác không lỗi cú pháp trên GitHub.

---

### 2.2. Phần 2: Giải Phẫu Luồng Forward Pass (✅ ĐÃ HOÀN THÀNH)

- [x] Đọc file [`02_forward_pass_walkthrough.md`](./02_forward_pass_walkthrough.md) và đối chiếu với Mục 4 của bài báo (Trang 4 - 6).
- [x] Kiểm tra xem 11 bước xử lý có ánh xạ đúng từ công thức toán sang code PyTorch không:
  - Chuẩn hóa RevIN $\to$ Feature Projection $\to$ Flatten $\to$ Concatenate $\to$ Dense Encoder $\to$ Dense Decoder $\to$ Unflatten $\to$ Temporal Decoder $\to$ Global Linear Skip $\to$ Residual Addition $\to$ RevIN Denormalize.
- [x] Tự giải thích được bản chất kỹ thuật:
  - *Tại sao phải nén biến ngoại sinh trước khi đưa vào Encoder?*
  - *Tại sao phải trải phẳng (Flatten) trước Encoder và tái tạo ma trận (Unflatten) sau Decoder?*
  - *Tại sao nhánh Global Linear Skip lại giúp mô hình không bị suy thoái gradient?*

---

### 2.3. Phần 3: Kích Thước Tensor & Chạy Script Test (✅ ĐÃ HOÀN THÀNH)

- [x] Mở PowerShell trên máy tính và tự gõ lệnh chạy script kiểm thử:
  ```powershell
  python tasks-Hieu/sprint-1/test_tensor_shapes.py
  ```
- [x] Kiểm tra xem terminal có in ra đầy đủ 14 bước kích thước tensor không, có bị lỗi `RuntimeError` hay `AssertionError` nào không.
- [x] Tự nhẩm lại phép tính số chiều đầu vào Encoder: $\text{dim}_{\text{enc}} = L + (L + H) \times \tilde{r} = 720 + (720 + 96) \times 4 = 3984$. Kiểm tra xem con số $3984$ có khớp chính xác với kết quả in ra trên terminal hay không.

---

### 2.4. Phần 4: Cấu Hình Siêu Tham Số & Kết Quả 7 Datasets (⏳ CHƯA CHECK)

- [ ] Mở file PDF gốc tại **Trang 8, Bảng 2 (Table 2)**:
  - Soi số liệu MSE/MAE của tập `Electricity` qua 4 horizon ($H \in \{96, 192, 336, 720\}$).
  - Soi số liệu MSE/MAE của tập `Traffic` và đối chiếu mức độ vượt trội so với PatchTST và DLinear.
- [ ] Mở file PDF gốc tại **Trang 21, Bảng 8 (Table 8)**:
  - Soi các siêu tham số của `Traffic`: `hiddenSize=256`, `numEncoderLayers=1`, `numDecoderLayers=1`, `decoderOutputDim=16`, `revIn=True`, `layerNorm=False`.
  - Soi các siêu tham số của `Electricity`: `hiddenSize=1024`, `numEncoderLayers=2`, `numDecoderLayers=2`, `decoderOutputDim=8`, `revIn=False`, `layerNorm=True`.
- [ ] Đọc phần giải thích 5 nguyên nhân vì sao TiDE vượt trội hơn các mô hình Transformer trên 2 tập dữ liệu này trong [`04_hyperparameters_and_benchmarks.md`](./04_hyperparameters_and_benchmarks.md).

---

### 2.5. Phần 5: Tự Trả Lời Bộ 3 Câu Hỏi Phản Biện (⏳ CHƯA CHECK)

- [ ] Mở file [`05_self_verification.md`](./05_self_verification.md) và tự trả lời trôi chảy bộ 3 câu hỏi phản biện then chốt:
  1. **Câu 1**: *Tại sao phải dùng Feature Projection nén biến ngoại sinh từ $r$ xuống $\tilde{r}=4$? Nếu giữ nguyên $r$ đưa thẳng vào Encoder thì điều gì xảy ra?*
  2. **Câu 2**: *Vai trò sống còn của nhánh Global Linear Skip Connection là gì? Nếu bỏ nhánh này thì mô hình bị ảnh hưởng ra sao?*
  3. **Câu 3**: *Sự khác biệt bản chất giữa Dense Decoder và Temporal Decoder là gì? Tại sao cần cả hai mà không dùng một khối duy nhất?*

---

## 📌 3. Tiêu Chí Nghiệm Thu Hoàn Thành (Definition of Done - DoD)

Để chính thức đóng Sprint 1 và chuyển giao kết quả, toàn bộ các tiêu chí sau phải đạt trạng thái `[x]`:

- [x] **DoD 1**: Đã có sơ đồ kiến trúc hoàn chỉnh tiếng Việt có dấu, khớp 100% với Figure 1 của paper (Đã nghiệm thu Phần 1).
- [x] **DoD 2**: Hiệu đã tự đọc hiểu và duyệt qua file Forward Pass Walkthrough (Phần 2).
- [x] **DoD 3**: Hiệu đã trực tiếp chạy lệnh kiểm thử Tensor shape trên terminal máy tính thật và kết quả pass 100% (Phần 3).
- [ ] **DoD 4**: Hiệu đã tự đối chiếu bảng số liệu thực nghiệm Table 2 & Table 8 trong PDF gốc (Phần 4).
- [ ] **DoD 5**: Hiệu đã tự trả lời tự tin, mạch lạc 3 câu hỏi phản biện cốt lõi mà không cần nhìn tài liệu (Phần 5).

---

## 📅 4. Kế Hoạch Tác Chiến Từng Ngày (Action Plan Đến 24/10)

| Ngày | Mục Tiêu Tác Chiến Cụ Thể | Trạng Thái Dự Kiến |
| :---: | :--- | :---: |
| **21/10 (Hôm nay)** | Duyệt xong Phần 1 (Sơ đồ) + Đọc hiểu toàn bộ luồng Forward Pass (Phần 2). | Hoàn thành 40% 🔄 |
| **22/10 (Thứ Năm)** | Bật máy chạy script `test_tensor_shapes.py` trên terminal + Nhẩm lại phép tính số chiều. | Hoàn thành 60% 🔄 |
| **23/10 (Thứ Sáu)** | Mở PDF đối chiếu Table 2 và Table 8 + Đọc hiểu lý do TiDE thắng Transformer trên Traffic & Electricity. | Hoàn thành 80% 🔄 |
| **24/10 (Thứ Bảy)** | Tập trả lời trôi chảy 3 câu hỏi phản biện + Tự tick full checklist nghiệm thu trước 21h00. | **HOÀN THÀNH 100%** 🎯 |

---

> [!TIP]
> **Hỗ trợ khi gặp khúc mắc**: Nếu trong quá trình tự kiểm tra có bất kỳ dòng code, kích thước tensor hay công thức toán nào chưa hiểu rõ, hãy trao đổi ngay với trợ lý **Antigravity** để được giải thích tường tận bằng các ví dụ trực quan.
