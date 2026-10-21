# SPRINT 1: TỔNG QUAN, MỤC TIÊU & DANH MỤC TÀI LIỆU NGHIỆM THU

> 👤 **Người thực hiện (Incharge):** Hiệu  
> 🤖 **Trợ lý kỹ thuật (Pair Assistant):** Antigravity  
> ⏰ **Hạn chót nghiệm thu (Deadline):** 21h00 – Ngày 24/10/2026  
> 🎯 **Trạng thái:** ✅ **HOÀN THÀNH 100% CÁC HẠNG MỤC**

---

## 1. Mục Tiêu & Yêu Cầu Của Sprint 1

Theo phân công công việc từ bảng quản lý dự án:
- **Nhiệm vụ (Task):** Kiến trúc TiDE, đối chiếu paper với code, kích thước tensor.
- **Yêu cầu tối thiểu:** Sơ đồ kiến trúc, giải thích forward pass, cấu hình và kết quả 2 dataset.

Sprint 1 đã được thực hiện trọn vẹn, vượt mức yêu cầu tối thiểu với đầy đủ sơ đồ chuẩn hóa, mã nguồn PyTorch thực thi, script kiểm thử kích thước Tensor và bảng số liệu thực nghiệm trích xuất trực tiếp từ bài báo gốc của Google Research (2023).

---

## 2. Danh Mục Các Phần Tài Liệu Chi Tiết (Sprint 1 Deliverables)

Bộ tài liệu báo cáo của Sprint 1 được phân tách thành **5 file chuyên đề độc lập, rõ ràng và mạch lạc**:

| STT | Tên File Tài Liệu | Nội Dung Trọng Tâm |
| :---: | :--- | :--- |
| **01** | 🏛️ [**`01_architecture_diagram.md`**](./01_architecture_diagram.md) | **Sơ đồ kiến trúc TiDE toàn diện khớp tuyệt đối với Figure 1 (Trang 5)**: Bóc tách từng ký hiệu gốc ($y_{1:L}^{(i)}, x_{1:L+H}^{(i)}, a^{(i)}, \tilde{x}, e^{(i)}, g^{(i)}, d, \hat{y}$), giải thích 5 thao tác `Flatten`, `Concat`, `Unflatten`, `Stack`, `Residual (+)`. |
| **02** | 🔬 [**`02_forward_pass_walkthrough.md`**](./02_forward_pass_walkthrough.md) | **Giải phẫu luồng xử lý Forward Pass (Paper vs Code PyTorch)**: Bảng đối chiếu song song 11 bước từ công thức toán học Section 4 sang từng dòng code PyTorch thực thi kèm class `TiDEForwardPass`. |
| **03** | 📐 [**`03_tensor_shapes_and_testing.md`**](./03_tensor_shapes_and_testing.md) | **Bảng theo dõi kích thước Tensor (`Shape`) & Script kiểm thử**: Bảng kích thước Tensor từ đầu vào đến đầu ra với batch mẫu ($B=32, L=720, H=96, r=8, \tilde{r}=4, p=4$) và hướng dẫn chạy file kiểm thử độc lập. |
| **04** | 📊 [**`04_hyperparameters_and_benchmarks.md`**](./04_hyperparameters_and_benchmarks.md) | **Cấu hình siêu tham số & Kết quả đối chuẩn 2 Dataset (`Electricity` & `Traffic`)**: Bảng siêu tham số tối ưu từ Table 8, bảng kết quả đối chuẩn MSE/MAE với PatchTST, DLinear từ Table 1, phân tích ưu thế tính toán $O(L+H)$. |
| **05** | 🛡️ [**`05_self_verification.md`**](./05_self_verification.md) | **Biên bản tự kiểm chứng của Hiệu (Self-Verification Report)**: Bằng chứng đối chiếu thực tế từng trang PDF, kết quả chạy code terminal thành công, và câu trả lời bộ 3 câu hỏi phản biện hóc búa nhất. |

---

## 3. Các File Quản Trị & Mã Nguồn Đi Kèm

- 📋 [**`requirement.md`**](./requirement.md): Bản đặc tả chi tiết 4 yêu cầu cốt lõi cần đạt được.
- ⏱️ [**`checklist.md`**](./checklist.md): Danh mục công việc chi tiết và kế hoạch tác chiến từng ngày.
- 🧭 [**`progress.md`**](./progress.md): Hướng dẫn cơ chế phối hợp cùng Antigravity và quy trình tự kiểm tra độc lập.
- 🐍 [**`test_tensor_shapes.py`**](./test_tensor_shapes.py): Script Python kiểm thử luồng Tensor tự động chạy thành công trên terminal.

---

## 4. Hướng Dẫn Chạy Kiểm Thử Nhanh (Quick Verification)

Để tự kiểm tra lại kích thước Tensor trực tiếp trên máy tính của bạn:

```powershell
python tasks-Hieu/sprint-1/test_tensor_shapes.py
```

Kết quả in ra sẽ chứng minh luồng Tensor thông suốt 100%, không bị bất kỳ lỗi lệch chiều (*dimension mismatch*) nào!
