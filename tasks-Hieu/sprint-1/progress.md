# HƯỚNG DẪN THỰC THI & THEO DÕI TIẾN ĐỘ SPRINT 1 (PROGRESS GUIDE)

> **Cộng Sự Cùng Tiến (Pair Programming):** Hiệu 🤝 Antigravity  
> **Nhiệm vụ chính:** Hoàn thành toàn diện 4 yêu cầu Sprint 1 trước **21h00 ngày 24/10/2026**.  
> **File báo cáo nghiệm thu đích:** `tasks-Hieu/sprint-1/report_sprint_1.md`

---

## 1. Cơ Chế Phối Hợp: Hiệu & Antigravity Làm Việc Như Thế Nào?

Để bạn **vừa làm chủ kiến thức, vừa hoàn thành công việc nhanh chóng và chuẩn xác nhất**, chúng ta sẽ phối hợp theo mô hình phân công vai trò sau:

```mermaid
flowchart LR
    Hieu["👤 BẠN (HIỆU) - CHỦ DỰ ÁN\n- Định hướng yêu cầu\n- Ra lệnh cho Antigravity\n- Đọc hiểu bản chất\n- Nghiệm thu từng mốc"] 
    <-->
    AGY["🤖 ANTIGRAVITY - TRỢ LÝ KỸ THUẬT\n- Đọc và trích xuất PDF gốc\n- Soạn thảo sơ đồ Mermaid\n- Viết mã nguồn PyTorch minh họa\n- Lập bảng số liệu & Đối chiếu paper-code"]
```

| Vai trò | Hiệu (Bạn) | Antigravity (Tôi) |
| :--- | :--- | :--- |
| **Giai đoạn 1: Sơ đồ kiến trúc** | Ra lệnh yêu cầu vẽ sơ đồ kiến trúc tổng quan. | Tự động sinh mã Mermaid chuẩn tiếng Việt có dấu, mô tả đầy đủ các khối và luồng dữ liệu. |
| **Giai đoạn 2: Forward Pass & Tensor Shape** | Yêu cầu đối chiếu từng dòng code với công thức paper. | Viết phân tích song song giữa công thức toán Section 4 và code PyTorch, kèm script test in `shape` tensor. |
| **Giai đoạn 3: Cấu hình & Kết quả 2 Dataset** | Chọn 2 dataset muốn làm (ví dụ: *Electricity* & *Traffic*). | Trích xuất chính xác siêu tham số từ Phụ lục B và số liệu MSE/MAE từ Bảng 1 của paper gốc. |
| **Giai đoạn 4: Đóng gói báo cáo** | Kiểm tra lần cuối xem đã hiểu hết chưa và duyệt. | Tích hợp thành file báo cáo nghiệm thu hoàn chỉnh, chuẩn bị slide/nội dung trình bày. |

---

## 2. Lộ Trình 4 Chặng Thực Hiện Từng Bước (Step-by-Step)

Dưới đây là 4 chặng công việc bạn sẽ lần lượt ra lệnh cho Antigravity cùng bạn thực hiện:

### 🚩 Chặng 1: Xây Dựng Sơ Đồ Kiến Trúc & Luồng Dữ Liệu Tổng Quan
- **Mục tiêu:** Có một sơ đồ khối hoàn chỉnh, thể hiện rõ cách dữ liệu đi qua từng module của TiDE.
- **Cách bạn ra lệnh cho Antigravity:**
  > *"Antigravity ơi, hãy giúp tôi vẽ sơ đồ kiến trúc TiDE tổng thể chi tiết nhất bằng Mermaid tiếng Việt, thể hiện rõ 3 nguồn đầu vào, khối Feature Projection, Dense Encoder, Dense Decoder, Temporal Decoder, RevIN và Global Skip Connection."*
- **Sản phẩm nhận được:** Sơ đồ khối trực quan kèm phân tích vai trò của từng khối.

---

### 🚩 Chặng 2: Giải Thích Forward Pass & Đo Đạc Kích Thước Tensor
- **Mục tiêu:** Hiểu rõ khi truyền một batch dữ liệu vào hàm `forward(y, x, a)`, kích thước tensor thay đổi thế nào qua từng khối.
- **Cách bạn ra lệnh cho Antigravity:**
  > *"Hãy viết tài liệu đối chiếu chi tiết Forward Pass của TiDE: Đặt công thức trong Section 4 bên cạnh từng dòng code PyTorch tương ứng. Đồng thời viết một script Python giả lập chạy thử để in ra kích thước Tensor qua từng tầng với ví dụ $B=32, L=720, H=96$."*
- **Sản phẩm nhận được:**
  - Bảng đối chiếu Paper vs Code từng bước.
  - Bảng kích thước Tensor: `(Batch, Steps, Channels)`.
  - Script Python chạy độc lập kiểm thử không bị lỗi lệch chiều (*dimension mismatch*).

---

### 🚩 Chặng 3: Thu Thập Cấu Hình & Kết Quả Đối Chuẩn Trên 2 Dataset
- **Mục tiêu:** Nắm được mô hình TiDE thực tế được cấu hình ra sao và đạt hiệu năng thế nào so với các đối thủ (PatchTST, DLinear, FEDformer...).
- **Khuyến nghị 2 dataset chọn lọc:**
  1. **Dataset 1: Electricity (LTTF Benchmark)** – Tập dữ liệu chuỗi thời gian tiêu chuẩn về phụ tải điện 321 khách hàng, thể hiện ưu thế tính toán $O(L+H)$.
  2. **Dataset 2: Traffic (PeMS Cảm biến giao thông)** – Tập dữ liệu đa biến lớn (862 cảm biến), thể hiện tính ưu việt của TiDE khi xử lý dữ liệu phức tạp.
- **Cách bạn ra lệnh cho Antigravity:**
  > *"Hãy trích xuất bảng siêu tham số chuẩn từ Phụ lục B và bảng kết quả MSE/MAE từ Bảng 1 của paper TiDE cho 2 dataset Electricity và Traffic. Hãy giải thích tại sao TiDE lại vượt trội hơn các mô hình Transformer trên 2 tập này."*
- **Sản phẩm nhận được:** Bảng so sánh số liệu thực nghiệm chính xác tuyệt đối từ paper gốc kèm lời bình luận sắc sảo.

---

### 🚩 Chặng 4: Đóng Gói Thành Báo Cáo Nghiệm Thu Sprint 1
- **Mục tiêu:** Gom toàn bộ kết quả của 3 chặng trên vào một file báo cáo duy nhất `tasks-Hieu/sprint-1/report_sprint_1.md`.
- **Cách bạn ra lệnh cho Antigravity:**
  > *"Hãy tổng hợp toàn bộ kết quả từ Chặng 1, 2, 3 thành file báo cáo chính thức report_sprint_1.md theo đúng checklist để tôi nộp trước deadline."*
- **Sản phẩm nhận được:** Một file Markdown chỉn chu, đầy đủ hình vẽ, bảng biểu, code mẫu và phân tích học thuật.

---

## 3. Nhật Ký Theo Dõi Tiến Độ Thực Tế (Live Progress Tracker)

Bảng này sẽ được chúng ta cập nhật liên tục khi hoàn thành từng bước:

| Chặng | Nhiệm vụ | Người thực hiện | Dự kiến hoàn thành | Trạng thái |
| :---: | :--- | :---: | :---: | :---: |
| **Chặng 1** | Vẽ sơ đồ kiến trúc TiDE tổng thể bằng Mermaid | Hiệu & Antigravity | 21/10/2026 | ⏳ **Sẵn sàng bắt đầu** |
| **Chặng 2** | Đối chiếu Forward Pass (Paper vs Code) + Script Tensor Shape | Hiệu & Antigravity | 22/10/2026 | ⏳ Chờ chặng 1 |
| **Chặng 3** | Bảng cấu hình siêu tham số & kết quả đối chuẩn 2 Dataset | Hiệu & Antigravity | 23/10/2026 | ⏳ Chờ chặng 2 |
| **Chặng 4** | Đóng gói file báo cáo nghiệm thu `report_sprint_1.md` | Hiệu & Antigravity | 24/10 (Trước 21h) | ⏳ Chờ chặng 3 |

---

## 💡 Lời Khuyên Dành Riêng Cho Hiệu:
1. Bạn không cần lo lắng về việc nhớ hết các công thức toán phức tạp, Antigravity sẽ bóc tách và giải thích từng bước bằng ngôn ngữ dễ hiểu nhất.
2. Mỗi khi đọc xong một phần do Antigravity viết, nếu có chỗ nào chưa hiểu rõ, hãy hỏi ngay: *"Chỗ này nghĩa là sao?", "Tại sao lại có số chiều này?"* để nắm vững 100% bản chất.
3. Bất cứ lúc nào bạn sẵn sàng, hãy ra lệnh: **"Hãy bắt đầu Chặng 1 đi Antigravity!"** chúng ta sẽ tiến hành ngay lập tức!
