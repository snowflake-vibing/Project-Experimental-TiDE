# BẢNG TIẾN ĐỘ & CHECKLIST CÔNG VIỆC SPRINT 1

> 👤 **Người thực hiện:** Hiệu  
> ⏰ **Hạn chót hoàn thành:** 21h00 – 24/10/2026  
> 🎯 **Mục tiêu nghiệm thu:** Hoàn thành đầy đủ 4 hạng mục kỹ thuật cốt lõi theo `requirement.md`.

---

## 📋 Bảng Checklist Công Việc Chi Tiết

| STT | Hạng mục công việc (Task Description) | Mục tiêu cụ thể / Tiêu chí nghiệm thu | Trạng thái | Ghi chú & Liên kết tài liệu |
| :---: | :--- | :--- | :---: | :--- |
| **1** | **Xây dựng Sơ đồ Kiến trúc Toàn diện TiDE** | Vẽ sơ đồ khối hoàn chỉnh (đầy đủ RevIN, Projection, Encoder, Decoder, Temporal Decoder, Global Skip Connection) bằng cú pháp Mermaid tiếng Việt. | 🔄 Đang làm | Đã có phác thảo ban đầu trong `doc/04_model_architecture.md`, cần làm phiên bản trực quan chi tiết cho Sprint 1. |
| **2** | **Giải phẫu Luồng Xử Lý (Forward Pass Walkthrough)** | Viết tài liệu giải thích chi tiết từng bước tính toán khi dữ liệu đi từ đầu vào $y_{1:L}, x_{1:L+H}, a$ tới đầu ra dự báo $\hat{y}_{L+1:L+H}$. | 🔄 Đang làm | Đối chiếu từng công thức trong Section 4 của Paper với từng dòng code PyTorch. |
| **3** | **Theo dõi & Kiểm thử Kích thước Tensor (Tensor Shapes)** | Lập bảng theo dõi kích thước Tensor `(Batch, Steps, Features)` tại từng nút; viết script Python kiểm thử in shape qua từng khối không bị lỗi. | 🔄 Đang làm | Minh họa với kích thước cụ thể: $B=32, L=720, H=96, r=4, p=4$. |
| **4** | **Tổng hợp Cấu hình Siêu tham số 2 Dataset** | Lập bảng cấu hình chi tiết (Lookback $L$, Horizon $H$, hidden size, dropout, learning rate...) cho 2 dataset đại diện từ Phụ lục B. | ⏳ Chờ làm | Chọn 2 dataset: **Electricity** (LTTF tiêu chuẩn) và **Traffic** (hoặc **Weather / M5**). |
| **5** | **Đối chuẩn & Phân tích Kết quả Thực nghiệm** | Trích xuất bảng kết quả MSE/MAE của TiDE và các baseline (PatchTST, DLinear, FEDformer...) trên 2 dataset đã chọn từ Bảng 1 của Paper; viết nhận xét đánh giá. | ⏳ Chờ làm | Lấy số liệu chính xác từ Bảng 1 trang 7 bài báo gốc. |
| **6** | **Rà soát & Đóng gói Báo cáo Sprint 1** | Tổng hợp toàn bộ 5 mục trên vào file báo cáo hoàn chỉnh của Sprint 1, kiểm tra định dạng và sẵn sàng nghiệm thu trước deadline. | ⏳ Chờ làm | Deadline nghiệm thu: **21h00 - 24/10/2026**. |

---

## 🔍 Kế Hoạch Thực Hiện Từng Ngày (Action Plan)

```mermaid
flowchart LR
    D1["Ngày 21/10:\nHoàn thiện Sơ đồ kiến trúc & Bảng luồng Tensor"] --> D2["Ngày 22/10:\nĐối chiếu Forward Pass: Paper vs Code PyTorch"]
    D2 --> D3["Ngày 23/10:\nThu thập Cấu hình & Kết quả 2 Dataset (Electricity & Traffic)"]
    D3 --> D4["Ngày 24/10 (Trước 21h):\nRà soát chất lượng, chạy test code & Nghiệm thu Sprint 1"]
```

- **Ngày 21/10:** Hoàn thiện sơ đồ kiến trúc tổng thể và bảng biến đổi kích thước Tensor qua từng tầng.
- **Ngày 22/10:** Viết phân tích đối chiếu song song giữa công thức toán học và mã nguồn triển khai của hàm `forward()`.
- **Ngày 23/10:** Khai thác Phụ lục B và Bảng thực nghiệm của bài báo để tổng hợp cấu hình siêu tham số và số liệu MSE/MAE của 2 tập dữ liệu.
- **Ngày 24/10:** Tích hợp thành báo cáo hoàn chỉnh, chạy script test tensor shape để kiểm chứng và hoàn thành trước 21h00.

---

## 📌 Tiêu Chí Đánh Giá Hoàn Thành (Definition of Done - DoD)
- [ ] Có sơ đồ kiến trúc rõ ràng, không lỗi render Mermaid.
- [ ] Giải thích mạch lạc luồng forward pass, liên kết chính xác giữa công thức toán và code.
- [ ] Có bảng kích thước tensor và script code mẫu chạy thành công mà không bị lỗi shape mismatch.
- [ ] Đầy đủ thông số cấu hình và bảng kết quả đối chuẩn MSE/MAE cho 2 dataset từ bài báo gốc.
- [ ] Văn bản sử dụng 100% Tiếng Việt có dấu chuẩn xác, định dạng Markdown sạch đẹp.
