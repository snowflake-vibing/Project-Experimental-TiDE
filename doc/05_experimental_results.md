# Chương 5: Kết Quả Thực Nghiệm & Đánh Giá (Experimental Results)

Chương này trình bày các thực nghiệm toàn diện của TiDE trên 7 tập dữ liệu benchmark dài hạn tiêu chuẩn, bài toán dự báo bán lẻ M5 quy mô lớn, so sánh tốc độ huấn luyện/suy luận và phân tích đóng góp của từng thành phần (Ablation Study).

---

## 5.1. Thiết Lập Thực Nghiệm (Experimental Setup)

### 1. Các Bộ Dữ Liệu Benchmark Dài Hạn
Nhóm tác giả đánh giá trên 7 tập dữ liệu phổ biến nhất trong cộng đồng LTTF:

| Tập Dữ Liệu | Số Lượng Chuỗi ($N$) | Số Bước Thời Gian ($T$) | Tần Suất Ghi Nhận | Lĩnh Vực / Đặc Điểm |
|:---|:---:|:---:|:---:|:---|
| **Traffic** | **862** | 17,544 | 1 giờ | Tỷ lệ chiếm dụng làn đường trên cao tốc California |
| **Electricity** | **321** | 26,304 | 1 giờ | Mức tiêu thụ điện năng hàng giờ của 321 khách hàng |
| **Weather** | 21 | 52,696 | 10 phút | 21 chỉ số khí tượng thời tiết (áp suất, nhiệt độ, độ ẩm...) |
| **ETTh1** | 7 | 17,420 | 1 giờ | Nhiệt độ dầu và phụ tải máy biến áp điện lực (bản 1h) |
| **ETTh2** | 7 | 17,420 | 1 giờ | Bản ghi máy biến áp điện lực (bản 1h số 2) |
| **ETTm1** | 7 | 69,680 | 15 phút | Bản ghi máy biến áp điện lực (bản 15 phút) |
| **ETTm2** | 7 | 69,680 | 15 phút | Bản ghi máy biến áp điện lực (bản 15 phút số 2) |

### 2. Các Mô Hình Đối Chuẩn (Baselines)
- **Nhóm Transformer SOTA**: PatchTST (Nie et al., 2022), FEDformer (Zhou et al., 2022), Autoformer (Wu et al., 2021), Informer (Zhou et al., 2021), Pyraformer (Liu et al., 2021), LogTrans (Li et al., 2019).
- **Nhóm Tuyến tính & MLP**: DLinear (Zeng et al., 2023), N-HiTS (Challu et al., 2023).
- **Mô hình xác suất & Covariates**: DeepAR (Salinas et al., 2020).

### 3. Cấu Hình Dự Báo
- Chiều dài cửa sổ quá khứ cố định: $L = 720$.
- Các khoảng dự báo tương lai: $H \in \{96, 192, 336, 720\}$.
- Thang đo đánh giá: MSE (Mean Squared Error) và MAE (Mean Absolute Error).

---

## 5.2. Kết Quả Trên Các Bộ Benchmark LTTF

Dưới đây là bảng trích xuất kết quả đối chuẩn trên 3 tập dữ liệu lớn và tiêu biểu nhất: **Weather**, **Traffic**, **Electricity** (Giá trị càng nhỏ càng tốt):

| Dataset | $H$ | TiDE (Ours)<br>MSE / MAE | PatchTST<br>MSE / MAE | N-HiTS<br>MSE / MAE | DLinear<br>MSE / MAE | FEDformer<br>MSE / MAE | Autoformer<br>MSE / MAE | Informer<br>MSE / MAE |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Weather** | 96 | 0.166 / 0.222 | **0.149 / 0.198** | 0.158 / 0.195 | 0.176 / 0.237 | 0.238 / 0.314 | 0.249 / 0.329 | 0.354 / 0.405 |
| | 192 | 0.209 / 0.263 | **0.194 / 0.241** | 0.211 / 0.247 | 0.220 / 0.282 | 0.275 / 0.329 | 0.325 / 0.370 | 0.419 / 0.434 |
| | 336 | 0.254 / 0.301 | **0.245 / 0.282** | 0.274 / 0.300 | 0.265 / 0.319 | 0.339 / 0.377 | 0.351 / 0.391 | 0.583 / 0.543 |
| | 720 | **0.313** / 0.340 | 0.314 / **0.334** | 0.401 / 0.413 | 0.323 / 0.362 | 0.389 / 0.409 | 0.415 / 0.426 | 0.916 / 0.705 |
| **Traffic** | 96 | **0.336** / 0.253 | 0.360 / **0.249** | 0.402 / 0.282 | 0.410 / 0.282 | 0.576 / 0.359 | 0.597 / 0.371 | 0.733 / 0.410 |
| | 192 | **0.346 / 0.257** | 0.379 / 0.256 | 0.420 / 0.297 | 0.423 / 0.287 | 0.610 / 0.380 | 0.607 / 0.382 | 0.777 / 0.435 |
| | 336 | **0.355 / 0.260** | 0.392 / 0.264 | 0.448 / 0.313 | 0.436 / 0.296 | 0.608 / 0.375 | 0.623 / 0.387 | 0.776 / 0.434 |
| | 720 | **0.386 / 0.273** | 0.432 / 0.286 | 0.539 / 0.353 | 0.466 / 0.315 | 0.621 / 0.375 | 0.639 / 0.395 | 0.827 / 0.466 |
| **Electricity** | 96 | 0.132 / 0.229 | **0.129 / 0.222** | 0.147 / 0.249 | 0.140 / 0.237 | 0.186 / 0.302 | 0.196 / 0.313 | 0.304 / 0.393 |
| | 192 | **0.147** / 0.243 | **0.147 / 0.240** | 0.167 / 0.269 | 0.153 / 0.249 | 0.197 / 0.311 | 0.211 / 0.324 | 0.327 / 0.417 |
| | 336 | **0.161** / 0.261 | 0.163 / **0.259** | 0.186 / 0.290 | 0.169 / 0.267 | 0.213 / 0.328 | 0.214 / 0.327 | 0.333 / 0.422 |
| | 720 | **0.196** / 0.294 | 0.197 / **0.290** | 0.243 / 0.340 | 0.203 / 0.301 | 0.233 / 0.344 | 0.236 / 0.342 | 0.378 / 0.443 |

### Nhận Xét Quan Trọng:
1. **Trên tập Traffic (862 chuỗi)**: TiDE vượt trội hoàn toàn tất cả các mô hình, bao gồm cả PatchTST (ví dụ tại $H=720$, TiDE đạt MSE $0.386$ so với $0.432$ của PatchTST và $0.466$ của DLinear).
2. **Trên tập Electricity (321 chuỗi)**: TiDE ngang ngửa với PatchTST và vượt trội rõ rệt các mô hình DLinear, FEDformer, Autoformer.
3. So với các Transformer đời đầu (Informer, Autoformer), TiDE cải thiện độ chính xác từ **40% đến 60%** sai số.

---

## 5.3. Bài Toán Dự Báo Nhu Cầu Bán Lẻ Thực Tế (M5 Competition)

Để kiểm chứng năng lực xử lý **covariates phong phú** (cả thuộc tính tĩnh và biến động tương lai), nhóm tác giả thử nghiệm trên tập dữ liệu bán lẻ khổng lồ **M5 Forecasting** của tập đoàn Walmart.

Chỉ số đánh giá là **WRMSSE** (Weighted Root Mean Squared Scaled Error) trên tập Private Test (càng thấp càng tốt):

| Mô Hình | Loại Covariates Sử Dụng | Test WRMSSE |
|:---|:---|:---:|
| **TiDE (Ours)** | **Static + Dynamic (Đầy đủ)** | **0.611 ± 0.009** |
| TiDE (Ablation) | Chỉ sử dụng đặc trưng ngày tháng (Date only) | 0.637 ± 0.005 |
| DeepAR | Static + Dynamic (Đầy đủ) | 0.789 ± 0.025 |
| PatchTST | Không dùng covariates (None) | 0.976 ± 0.014 |

### Kết luận từ thử nghiệm M5:
- TiDE vượt trội mô hình DeepAR chuyên biệt tới **22.5%** ($0.611$ so với $0.789$).
- PatchTST bị tụt hậu rất xa ($0.976$) do thiết kế gốc không hỗ trợ tiếp nhận covariates.
- Thậm chí khi chỉ dùng các biến ngày tháng cơ bản, TiDE ($0.637$) vẫn đánh bại hoàn toàn DeepAR.

---

## 5.4. Hiệu Quả Tính Toán: Tốc Độ & Bộ Nhớ (Efficiency & GPU Memory)

Thực nghiệm đo lường trên một GPU đơn NVIDIA T4 (kèm CPU Intel Xeon 64 lõi) với tập Electricity (batch size = 8, mỗi batch gồm $8 \times 321 \times L$ điểm):

### 1. Thời Gian Suy Luận (Inference Time per Batch)
Khi tăng chiều dài cửa sổ nhìn lại $L$ từ 192 lên 2880:
- TiDE duy trì độ trễ suy luận gần như không đổi ở mức thấp (~ $600 \mu s$ đến $800 \mu s$).
- PatchTST tốn từ $1,500 \mu s$ đến hơn $2,500 \mu s$.
- **TiDE nhanh hơn từ 5 đến 10 lần** so với PatchTST.

### 2. Thời Gian Huấn Luyện (Training Time per Epoch)
- TiDE mất khoảng 50 - 100 giây cho một epoch.
- PatchTST mất từ 500 đến hơn 1000 giây cho một epoch (chậm hơn gấp 10 lần).

### 3. Khả Năng Mở Rộng Bộ Nhớ (Memory Scalability)
- **PatchTST gặp lỗi tràn bộ nhớ (Out-of-Memory - OOM)** ngay khi $L \ge 1440$ trên GPU T4.
- **TiDE chạy mượt mà ngay cả khi $L = 2880$** nhờ độ phức tạp bộ nhớ tuyến tính $O(L + H)$.

---

## 5.5. Phân Tích Đóng Góp Thành Phần (Ablation Study)

### 1. Vai Trò Của Temporal Decoder
Để chứng minh tác dụng của Temporal Decoder đối với biến động tương lai, nhóm tác giả tạo tập dữ liệu bán-tổng-hợp từ Electricity bằng cách chèn sự kiện giả định (Type A Event - ví dụ đợt nắng nóng đột ngột).
- **Kết quả**:
  - Mô hình có Temporal Decoder phản ứng chính xác ngay trong đợt sự kiện và lập tức hồi phục quỹ đạo bình thường sau sự kiện chỉ sau 1 epoch huấn luyện.
  - Mô hình không có Temporal Decoder bị mất phương hướng kéo dài nhiều bước sau sự kiện do quá khứ bị xáo trộn mà không có đường dẫn tắt tương lai để cân chỉnh.

### 2. Vai Trò Của Residual Connection
Thực nghiệm loại bỏ toàn bộ nhánh tắt residual trên tập Electricity:
- Khi có Residual Connections: MSE đạt $0.132$ ($H=96$) và $0.196$ ($H=720$).
- Khi bỏ Residual Connections: MSE tăng lên $0.136$ ($H=96$) và mất mát hội tụ trên các horizon dài hơn.
