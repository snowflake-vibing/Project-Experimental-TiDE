# YÊU CẦU CÔNG VIỆC SPRINT 1 (REQUIREMENTS)

> **Thông tin công việc từ bảng quản lý:**
> - **Nhiệm vụ (Task):** Kiến trúc TiDE, đối chiếu paper với code, kích thước tensor
> - **Yêu cầu tối thiểu (Requirements tối thiểu):** Sơ đồ kiến trúc, giải thích forward pass, cấu hình và kết quả 2 dataset
> - **Người phụ trách (Incharge):** Hiệu
> - **Hạn chót (Deadline):** 21h00 ngày 24/10/2026
> - **Trạng thái (Progress):** Assigned (Đã giao việc)

---

## 1. Bản Chất Công Việc Là Gì? (Hiểu Đúng Để Làm Đúng)

Sprint 1 không yêu cầu bạn phải tự code lại từ đầu hay chạy các mô hình lớn phức tạp ngay lập tức. Mục tiêu cốt lõi của Sprint 1 là **"Giải mã trọn vẹn mô hình TiDE từ giấy trắng (Paper) sang từng dòng code thực thi (Code Implementation) và hiểu thông suốt luồng đi của dữ liệu"**.

Cụ thể, khi hoàn thành Sprint 1, bạn phải trả lời được rành rọt 4 câu hỏi:
1. **Kiến trúc TiDE gồm những khối nào và liên kết với nhau ra sao?** (Sơ đồ kiến trúc & luồng dữ liệu).
2. **Khi đưa một batch dữ liệu vào hàm `forward()`, dữ liệu biến đổi qua từng dòng code như thế nào?** (Đối chiếu paper - code từng bước).
3. **Tại sao kích thước Tensor (`shape`) lại thay đổi qua từng khối như vậy?** (Tensor dimensions walkthrough).
4. **Mô hình được cấu hình (hyperparameters) như thế nào trên 2 dataset tiêu chuẩn và kết quả thực tế trên paper đạt được là bao nhiêu?**

---

## 2. Chi Tiết 4 Yêu Cầu Cốt Lõi Phải Thực Hiện

### Yêu Cầu 1: Sơ Đồ Kiến Trúc Toàn Diện & Trực Quan (Architecture Diagram)
- **Mục tiêu:** Tạo sơ đồ luồng tổng thể rõ ràng, thể hiện trực quan tất cả các thành phần nơ-ron của TiDE.
- **Nội dung cần làm rõ:**
  - **Dữ liệu đầu vào:** Chuỗi quá khứ $y_{1:L}$, Biến ngoại sinh động toàn thời gian $x_{1:L+H}$, Thuộc tính tĩnh $a$.
  - **Khối Feature Projection:** Nén chiều đặc trưng ngoại sinh từ $r$ xuống $\tilde{r} = 4$ thông qua Residual Block nhỏ.
  - **Khối Dense Encoder:** Ghép (concatenate) các nguồn dữ liệu thành một vector phẳng và đi qua $n_e$ khối Dense Residual Blocks để tạo vector bối cảnh ẩn $e$.
  - **Khối Dense Decoder:** Giải mã vector $e$ qua $n_d$ khối Residual Blocks thành ma trận tương lai $D \in \mathbb{R}^{H \times p}$.
  - **Khối Temporal Decoder:** Lấy từng vector lát cắt $d_t$ ghép với biến tương lai $\tilde{x}_{L+t}$ để sinh ra dự báo phi tuyến $\hat{y}_{\text{NL}}$.
  - **Nhánh tắt toàn cục (Global Linear Skip Connection):** Ánh xạ tuyến tính trực tiếp từ quá khứ $y_{1:L}$ sang tương lai $\hat{y}_{\text{Linear}}$.
  - **Cơ chế Reversible Instance Normalization (RevIN):** Vỏ bọc chuẩn hóa $Z$-score ở đầu vào và hoàn nguyên thang đo ở đầu ra.

### Yêu Cầu 2: Đối Chiếu Chi Tiết Giữa Paper Và Code (Paper vs. Code Walkthrough)
- **Mục tiêu:** Đặt từng công thức toán trong paper (Mục 4 của bài báo) cạnh từng khối code PyTorch tương ứng.
- **Nội dung cần làm rõ:**
  - Hàm `forward(y, x, a)` trong code nhận đối số gì, tương ứng với ký hiệu toán học nào ($y_{1:L}^{(i)}$, $x_{1:L+H}^{(i)}$, $a^{(i)}$).
  - Khối `ResidualBlock`: Công thức $v = \text{LayerNorm}(F(u) + \text{LinearSkip}(u))$ ánh xạ vào dòng code PyTorch nào.
  - Tại sao lại dùng `flatten()` trước khi đưa vào Encoder?
  - Cơ chế lát cắt theo thời gian của Temporal Decoder: Trong code dùng vòng lặp `for t in range(H)` hay dùng phép biến đổi vector hóa `reshape / bmm` để tối ưu tốc độ tính toán?
  - Nhánh cộng tuyến tính cuối cùng $\hat{y} = \hat{y}_{\text{NL}} + \hat{y}_{\text{Linear}}$.

### Yêu Cầu 3: Theo Dõi Kích Thước Tensor Qua Từng Bước (Tensor Shape Flow)
- **Mục tiêu:** Định nghĩa rõ `(Batch_Size, Time_Steps, Channels/Features)` tại từng nút của mạng, giúp phát hiện lỗi lệch chiều (dimension mismatch) ngay lập tức.
- **Ví dụ luồng tensor mẫu với batch size $B$:**
  - Đầu vào quá khứ: `(B, L)`
  - Ngoại sinh động: `(B, L+H, r)`
  - Sau Feature Projection: `(B, L+H, 4)`
  - Vector ghép đưa vào Encoder: `(B, L + (L+H)*4 + p_static)`
  - Đầu ra Encoder: `(B, hiddenSize)`
  - Đầu ra Dense Decoder: `(B, H * decoderOutputDim)` $\to$ chuyển thành `(B, H, decoderOutputDim)`
  - Đầu ra Temporal Decoder: `(B, H, 1)` $\to$ `(B, H)`
  - Kết quả dự báo cuối cùng: `(B, H)`

### Yêu Cầu 4: Cấu Hình Siêu Tham Số & Kết Quả Trên 2 Dataset Tiêu Chuẩn
- **Mục tiêu:** Lựa chọn **2 tập dữ liệu benchmark đại diện** trong bài báo (khuyến nghị chọn 1 tập biến liên tục LTTF như **Electricity** hoặc **Weather**, và 1 tập chuỗi bán lẻ có thuộc tính tĩnh như **M5 Walmart** hoặc **Traffic**).
- **Nội dung cần trình bày:**
  1. **Bảng cấu hình siêu tham số (Hyperparameters):**
     - $L$ (Lookback window) và $H$ (Forecast horizon).
     - Kích thước ẩn `hiddenSize`, số tầng Encoder `numEncoderLayers`, số tầng Decoder `numDecoderLayers`.
     - Kích thước nén ngoại sinh `temporalWidth` ($\tilde{r}$), kích thước `decoderOutputDim`.
     - Tỷ lệ `dropout`, `batchSize`, `learningRate`.
     - Có bật RevIN hay không?
  2. **Bảng kết quả thực nghiệm chuẩn trong bài báo:**
     - Các độ đo chính: **MSE (Mean Squared Error)** và **MAE (Mean Absolute Error)**.
     - So sánh kết quả của TiDE với các baseline mạnh nhất trong paper (PatchTST, DLinear, FEDformer, Autoformer).
     - Rút ra nhận xét vì sao TiDE lại đạt kết quả tốt hơn hoặc tương đương trên 2 dataset này.

---

## 3. Sản Phẩm Bàn Giao Cần Có (Deliverables)

1. Tài liệu báo cáo chi tiết tổng hợp đầy đủ 4 phần trên (được liên kết trực tiếp trong thư mục `tasks-Hieu/sprint-1/`).
2. Sơ đồ kiến trúc Mermaid chuẩn tiếng Việt có dấu.
3. Bảng đối chiếu Paper vs Code chi tiết.
4. Đoạn code mẫu kiểm thử kích thước Tensor chạy mượt mà (chứng minh luồng Tensor thông suốt không báo lỗi).
5. Bảng số liệu đối chuẩn 2 dataset từ bài báo gốc.
