# BẢNG TIẾN ĐỘ & CHECKLIST CÔNG VIỆC SPRINT 1

> [!NOTE]
> **Điều hướng tài liệu**: [🏠 Trang Chủ](../../README.md) • [📋 Sprint 1 README](./README.md) • [📜 Yêu Cầu Sprint](./requirement.md) • [🛡️ Tự Kiểm Chứng](./05_self_verification.md)  
> 👤 **Người thực hiện:** Hiệu  
> 🤖 **Trợ lý kỹ thuật:** Antigravity  
> 📌 **Phạm vi nhiệm vụ được giao:** **Kiến trúc TiDE • Đối chiếu Paper với Code • Kích thước Tensor**  
> 🎯 **Trạng thái nghiệm thu:** **HOÀN THÀNH 100% NHIỆM VỤ CỐT LÕI (3/3 HẠNG MỤC)** 🎉

---

## 📊 Thước Đo Tiến Độ Nhiệm Vụ Cốt Lõi (Core Tasks Progress)

| Chỉ số theo dõi | Trạng thái hiện tại | Tỷ lệ hoàn thành | Đánh giá |
| :--- | :---: | :---: | :---: |
| **Phạm vi giao việc của Hiệu (3 nhiệm vụ)** | **3/3 hạng mục đã duyệt** | **100%** 🎯 | **HOÀN THÀNH XUẤT SẮC** ✅ |
| **Mục tiêu thời gian (Deadline: 24/10)** | Hoàn thành trước hạn | *Về đích sớm* ⏱️ | Đạt chuẩn nghiệm thu 🚀 |

---

## 📋 1. Bảng Nghiệm Thu 3 Nhiệm Vụ Cốt Lõi Của Hiệu

| STT | Nhiệm Vụ Cốt Lõi Được Giao | Tài Liệu Tương Ứng | Trạng Thái Hiệu Tự Check | Kết Quả Đạt Được |
| :---: | :--- | :--- | :---: | :--- |
| **01** | **Kiến trúc TiDE & Sơ đồ luồng** | [`01_architecture_diagram.md`](./01_architecture_diagram.md) | ✅ **ĐÃ NGHIỆM THU** | Sơ đồ Mermaid tiếng Việt có dấu, khớp 100% với Figure 1 (Trang 5 PDF), đầy đủ 3 đầu vào, 5 thao tác nối ghép và nhánh skip. |
| **02** | **Đối chiếu Paper với Code PyTorch** | [`02_forward_pass_walkthrough.md`](./02_forward_pass_walkthrough.md) | ✅ **ĐÃ NGHIỆM THU** | Bảng đối chiếu 11 bước song song, đặt từng công thức toán cạnh từng dòng code PyTorch, truy xuất nguồn gốc từng trang paper. |
| **03** | **Kích thước Tensor & Script Test** | [`03_tensor_shapes_and_testing.md`](./03_tensor_shapes_and_testing.md)<br>[`test_tensor_shapes.py`](./test_tensor_shapes.py) | ✅ **ĐÃ NGHIỆM THU** | Giải mã chi tiết số chiều $B, L, H, r, \tilde{r}, p$; script chạy mượt mà trên terminal thật, pass 100% không lệch chiều. |

---

## 📚 2. Tài Liệu Bổ Trợ Dành Cho Nhóm & Huấn Luyện (Bonus Reference)

Các nội dung mở rộng đã được biên soạn sẵn để hỗ trợ nhóm khi bước sang giai đoạn huấn luyện (Sprint 2):

| STT | Tài Liệu Tham Khảo Mở Rộng | Mục Đích Sử Dụng | Trạng Thái Soạn Thảo |
| :---: | :--- | :--- | :---: |
| **04** | [`04_hyperparameters_and_benchmarks.md`](./04_hyperparameters_and_benchmarks.md) | Bảng siêu tham số Table 8 & kết quả Table 2 cho cả 7 datasets (dùng khi nhóm setup file config train). | ✅ Sẵn sàng 100% |
| **05** | [`05_self_verification.md`](./05_self_verification.md) | Biên bản tự kiểm chứng và trả lời bộ 3 câu hỏi phản biện hóc búa nhất (dùng khi báo cáo/vấn đáp). | ✅ Sẵn sàng 100% |

---

## 🔍 3. Nhật Ký Chi Tiết Hiệu Đã Tự Kiểm Chứng

### 3.1. Hạng mục 1: Sơ Đồ Kiến Trúc & Ký Hiệu Toán Học (✅ ĐÃ HOÀN THÀNH)
- [x] Mở trực tiếp file PDF [`Hieu_paper_TiDE-Time-series-Dense-Encoder.pdf`](../../Hieu_paper_TiDE-Time-series-Dense-Encoder.pdf) tại **Trang 5, Hình 1 (Figure 1)**.
- [x] Đối chiếu đầy đủ 3 nguồn đầu vào: $y_{1:L}^{(i)}$ (Lookback), $x_{1:L+H}^{(i)}$ (Dynamic Covariates), $a^{(i)}$ (Attributes).
- [x] Đối chiếu đúng các biến trung gian: $\tilde{x}_{1:L+H}^{(i)}$, vector mã hóa $e^{(i)}$, vector giải mã phẳng $g^{(i)}$, ma trận bối cảnh $d_{L+1:L+H}^{(i)}$, ngoại sinh tương lai $\tilde{x}_{L+1:L+H}^{(i)}$, và dự báo cuối $\hat{y}_{L+1:L+H}^{(i)}$.
- [x] Đối chiếu đúng 5 thao tác nối ghép: `Feature Projection (per time-step)`, `Flatten`, `Concat`, `Unflatten`, `Stack` và nhánh tắt `Residual (+)`.
- [x] Sơ đồ Mermaid viết bằng tiếng Việt có dấu đầy đủ, hiển thị chuẩn xác không lỗi cú pháp trên GitHub.

### 3.2. Hạng mục 2: Giải Phẫu Luồng Forward Pass (✅ ĐÃ HOÀN THÀNH)
- [x] Đọc file [`02_forward_pass_walkthrough.md`](./02_forward_pass_walkthrough.md) và đối chiếu với Mục 4 của bài báo (Trang 4 - 6).
- [x] Kiểm tra xem 11 bước xử lý có ánh xạ đúng từ công thức toán sang code PyTorch không:
  - Chuẩn hóa RevIN $\to$ Feature Projection $\to$ Flatten $\to$ Concatenate $\to$ Dense Encoder $\to$ Dense Decoder $\to$ Unflatten $\to$ Temporal Decoder $\to$ Global Linear Skip $\to$ Residual Addition $\to$ RevIN Denormalize.
- [x] Tự giải thích được bản chất kỹ thuật:
  - *Tại sao phải nén biến ngoại sinh trước khi đưa vào Encoder?*
  - *Tại sao phải trải phẳng (Flatten) trước Encoder và tái tạo ma trận (Unflatten) sau Decoder?*
  - *Tại sao nhánh Global Linear Skip lại giúp mô hình không bị suy thoái gradient?*

### 3.3. Hạng mục 3: Kích Thước Tensor & Chạy Script Test (✅ ĐÃ HOÀN THÀNH)
- [x] Đã hiểu rõ bản chất $L=720$ (nhìn lại 30 ngày) và $H=96$ (dự báo trước 4 ngày) là cửa sổ trượt (Sliding Window), không phải độ dài file CSV.
- [x] Đã chạy script `test_tensor_shapes.py` trên terminal thật thành công 100%.
- [x] Nắm rõ phép tính số chiều đầu vào Encoder: $\text{dim}_{\text{enc}} = L + (L + H) \times \tilde{r} = 720 + (720 + 96) \times 4 = 3984$.
- [x] Tensor đầu ra `torch.Size([32, 96])` khớp hoàn hảo với nhãn mục tiêu Ground Truth.

---

## 📌 4. Tiêu Chí Nghiệm Thu Hoàn Thành (Definition of Done - DoD)

Toàn bộ các tiêu chí nghiệm thu của Hiệu cho Sprint 1:

- [x] **DoD 1**: Đã có sơ đồ kiến trúc hoàn chỉnh tiếng Việt có dấu, khớp 100% với Figure 1 của paper.
- [x] **DoD 2**: Hiệu đã tự đọc hiểu và duyệt qua file Forward Pass Walkthrough đối chiếu Paper vs Code.
- [x] **DoD 3**: Hiệu đã trực tiếp chạy lệnh kiểm thử Tensor shape trên terminal máy tính thật và kết quả pass 100%.
- [x] **DoD 4 (Bonus)**: Đã có sẵn tài liệu siêu tham số và kết quả đối chuẩn sẵn sàng chuyển giao cho nhóm khi làm Sprint 2.

---

> [!TIP]
> **Kết luận Sprint 1:** Hiệu đã hoàn thành **100% khối lượng công việc được phân công** trong Sprint 1, nắm vững trọn vẹn lý thuyết và mã nguồn cốt lõi của TiDE, sẵn sàng báo cáo nhóm!
