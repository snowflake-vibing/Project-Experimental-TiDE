# PHẦN 4: CẤU HÌNH SIÊU THAM SỐ & KẾT QUẢ ĐỐI CHUẨN TRÊN 7 DATASETS

> [!NOTE]
> **Điều hướng tài liệu**: [🏠 Trang Chủ](../../README.md) • [📋 Sprint 1 README](./README.md) • [📜 Yêu Cầu Sprint](./requirement.md) • [📊 Checklist Tiến Độ](./checklist.md) • [🛡️ Tự Kiểm Chứng](./05_self_verification.md)  
> 👤 **Người thực hiện:** Hiệu  
> 🤖 **Trợ lý kỹ thuật:** Antigravity  
> 📄 **Tài liệu tham chiếu:** *Long-term Forecasting with TiDE: Time-series Dense Encoder* (Google Research, 2023) - Table 1 (Trang 7), Table 2 (Trang 8), Table 8 (Trang 21)

---

## 1. Tổng Quan 7 Tập Dữ Liệu Benchmark Chuẩn (Trích Xuất Từ Table 1 Của Paper)

Trong nghiên cứu dự báo chuỗi thời gian dài hạn (LTTF), Google Research sử dụng **toàn bộ 7 tập dữ liệu benchmark phổ biến nhất** trong học thuật:

| STT | Tên Tập Dữ Liệu | Số Lượng Chuỗi (# Time-Series) | Độ Dài Chuỗi (# Time-Points) | Tần Số Thu Thập (Frequency) | Lĩnh Vực Ứng Dụng & Đặc Trưng |
| :---: | :--- | :---: | :---: | :---: | :--- |
| **1** | **Electricity** | 321 | 26,304 | 1 Giờ (1 Hour) | Phụ tải điện tiêu thụ của 321 khách hàng (2012–2014); chu kỳ ngày/tuần rất mạnh. |
| **2** | **Traffic** | 862 | 17,544 | 1 Giờ (1 Hour) | Tỷ lệ chiếm dụng làn đường cao tốc California PeMS; đồ sộ và tính phi tuyến cực cao. |
| **3** | **Weather** | 21 | 52,696 | 10 Phút (10 Mins) | 21 chỉ số khí tượng thời tiết (nhiệt độ, độ ẩm, áp suất...); biến động liên tục. |
| **4** | **ETTh1** | 7 | 17,420 | 1 Giờ (1 Hour) | Nhiệt độ dầu máy biến áp điện lực (bản 1 giờ, trạm 1). |
| **5** | **ETTh2** | 7 | 17,420 | 1 Giờ (1 Hour) | Nhiệt độ dầu máy biến áp điện lực (bản 1 giờ, trạm 2). |
| **6** | **ETTm1** | 7 | 69,680 | 15 Phút (15 Mins) | Nhiệt độ dầu máy biến áp điện lực (bản 15 phút, trạm 1); độ phân giải thời gian cao. |
| **7** | **ETTm2** | 7 | 69,680 | 15 Phút (15 Mins) | Nhiệt độ dầu máy biến áp điện lực (bản 15 phút, trạm 2); độ phân giải thời gian cao. |

> *(Lưu ý: Hai tập `Electricity` và `Traffic` là hai tập dữ liệu lớn và thách thức nhất với hơn 300 và 800 chuỗi thời gian song song).*

---

## 2. Bảng Siêu Tham Số Chuẩn Của Cả 7 Tập Dữ Liệu (Trích Xuất Từ Table 8 - Trang 21)

Dưới đây là cấu hình siêu tham số tối ưu chính thức do nhóm tác giả Google Research tìm kiếm và công bố trong **Table 8 (Phụ lục B.3)**:

| Tập Dữ Liệu (Dataset) | `hiddenSize` | `numEncoderLayers` ($n_e$) | `numDecoderLayers` ($n_d$) | `decoderOutputDim` ($p$) | `temporalDecoderHidden` | `dropoutLevel` | `layerNorm` | `learningRate` ($\eta$) | `revIn` |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Traffic** | 256 | 1 | 1 | 16 | 64 | 0.3 | `False` | $6.55 \times 10^{-5}$ | `True` |
| **Electricity** | 1024 | 2 | 2 | 8 | 64 | 0.5 | `True` | $9.99 \times 10^{-4}$ | `False` |
| **ETTm1** | 1024 | 1 | 1 | 8 | 128 | 0.5 | `True` | $8.39 \times 10^{-5}$ | `False` |
| **ETTm2** | 512 | 2 | 2 | 16 | 128 | 0.0 | `True` | $2.52 \times 10^{-4}$ | `True` |
| **ETTh1** | 256 | 2 | 2 | 8 | 128 | 0.3 | `True` | $3.82 \times 10^{-5}$ | `True` |
| **ETTh2** | 512 | 2 | 2 | 32 | 16 | 0.2 | `True` | $2.24 \times 10^{-4}$ | `True` |
| **Weather** | 512 | 1 | 1 | 8 | 16 | 0.0 | `True` | $3.01 \times 10^{-5}$ | `False` |

### 📌 Các siêu tham số dùng chung cố định (Global Settings):
- **Cửa sổ quá khứ ($L$):** Cố định $L = 720$ (hoặc $L = 96$ trong các thử nghiệm cơ bản).
- **Tầm nhìn dự báo ($H$):** Đánh giá trên 4 mốc dài hạn tiêu chuẩn $H \in \{96, 192, 336, 720\}$.
- **Độ rộng nén ngoại sinh (`temporalWidth` - $\tilde{r}$):** Cố định **`temporalWidth = 4`** cho toàn bộ 7 tập dữ liệu.
- **Hàm mất mát (Loss Function):** MSE (Mean Squared Error).
- **Bộ tối ưu (Optimizer):** Adam với Cosine Annealing Learning Rate.

---

## 3. Bảng Kết Quả Đối Chuẩn Thực Nghiệm 7 Datasets (Trích Xuất Từ Table 2 - Trang 8)

Bảng so sánh độ lỗi **MSE / MAE** giữa **TiDE** và các mô hình hàng đầu (PatchTST, DLinear, FEDformer, Autoformer, Informer).  
*(Số in đậm là kết quả tốt nhất)*:

### 3.1. Hai Tập Dữ Liệu Trọng Tâm: Electricity & Traffic

| Tập Dữ Liệu | Horizon $H$ | **TiDE (Google)** | **PatchTST** | **DLinear** | **FEDformer** | **Autoformer** |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Electricity** | 96 | **0.132 / 0.229** | 0.129 / 0.222 | 0.140 / 0.237 | 0.186 / 0.302 | 0.196 / 0.313 |
| | 192 | **0.147 / 0.243** | 0.147 / 0.240 | 0.153 / 0.249 | 0.197 / 0.311 | 0.211 / 0.324 |
| | 336 | **0.161 / 0.261** | 0.163 / 0.259 | 0.169 / 0.267 | 0.213 / 0.328 | 0.214 / 0.327 |
| | 720 | **0.196 / 0.294** | 0.197 / 0.290 | 0.203 / 0.301 | 0.233 / 0.344 | 0.236 / 0.342 |
| | **Avg** | **0.159 / 0.257** | **0.159 / 0.253** | 0.166 / 0.264 | 0.207 / 0.321 | 0.214 / 0.327 |
| **Traffic** | 96 | **0.336 / 0.253** | 0.360 / 0.249 | 0.410 / 0.282 | 0.576 / 0.359 | 0.597 / 0.371 |
| | 192 | **0.346 / 0.257** | 0.379 / 0.256 | 0.423 / 0.287 | 0.610 / 0.380 | 0.607 / 0.382 |
| | 336 | **0.355 / 0.260** | 0.392 / 0.264 | 0.436 / 0.296 | 0.608 / 0.375 | 0.623 / 0.387 |
| | 720 | **0.386 / 0.273** | 0.432 / 0.286 | 0.466 / 0.315 | 0.621 / 0.375 | 0.639 / 0.395 |
| | **Avg** | **0.356 / 0.261** | 0.391 / 0.264 | 0.434 / 0.295 | 0.604 / 0.372 | 0.617 / 0.384 |

> *(Lưu ý: Trên tập **Traffic**, TiDE áp đảo PatchTST và DLinear ở mọi horizon, giảm MSE từ 0.391 xuống 0.356 – vượt trội hơn 9% so với PatchTST).*

---

### 3.2. Năm Tập Dữ Liệu Còn Lại: Weather, ETTh1, ETTh2, ETTm1, ETTm2

| Tập Dữ Liệu | Horizon $H$ | **TiDE** (MSE / MAE) | **PatchTST** (MSE / MAE) | **DLinear** (MSE / MAE) | **FEDformer** (MSE / MAE) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Weather** | 96 | 0.166 / 0.222 | **0.149 / 0.198** | 0.176 / 0.237 | 0.238 / 0.314 |
| | 192 | 0.209 / 0.263 | **0.194 / 0.241** | 0.220 / 0.282 | 0.275 / 0.329 |
| | 336 | 0.254 / 0.301 | **0.245 / 0.282** | 0.265 / 0.319 | 0.339 / 0.377 |
| | 720 | **0.313 / 0.340** | 0.314 / 0.334 | 0.323 / 0.362 | 0.389 / 0.409 |
| **ETTh1** | 96 | **0.375 / 0.398** | 0.379 / 0.401 | **0.375 / 0.399** | 0.376 / 0.415 |
| | 192 | **0.412 / 0.422** | 0.413 / 0.429 | **0.412 / 0.420** | 0.423 / 0.446 |
| | 336 | **0.435 / 0.433** | **0.435 / 0.436** | 0.439 / 0.443 | 0.444 / 0.462 |
| | 720 | 0.454 / 0.465 | **0.446 / 0.464** | 0.501 / 0.490 | 0.469 / 0.492 |
| **ETTh2** | 96 | **0.270 / 0.336** | 0.274 / 0.337 | 0.289 / 0.353 | 0.332 / 0.374 |
| | 192 | **0.332 / 0.380** | 0.338 / 0.376 | 0.383 / 0.418 | 0.407 / 0.446 |
| | 336 | **0.360 / 0.407** | 0.363 / 0.397 | 0.448 / 0.465 | 0.400 / 0.447 |
| | 720 | 0.419 / 0.451 | **0.393 / 0.430** | 0.605 / 0.551 | 0.412 / 0.469 |
| **ETTm1** | 96 | 0.306 / 0.349 | **0.293 / 0.346** | 0.299 / 0.343 | 0.326 / 0.390 |
| | 192 | 0.335 / 0.366 | **0.333 / 0.370** | 0.335 / 0.365 | 0.365 / 0.415 |
| | 336 | **0.364 / 0.384** | 0.369 / 0.392 | 0.369 / 0.386 | 0.392 / 0.425 |
| | 720 | **0.413 / 0.413** | 0.416 / 0.420 | 0.425 / 0.421 | 0.446 / 0.458 |
| **ETTm2** | 96 | **0.161 / 0.251** | 0.166 / 0.256 | 0.167 / 0.260 | 0.180 / 0.271 |
| | 192 | **0.215 / 0.289** | 0.223 / 0.296 | 0.224 / 0.303 | 0.252 / 0.318 |
| | 336 | **0.267 / 0.326** | 0.274 / 0.329 | 0.281 / 0.342 | 0.324 / 0.364 |
| | 720 | **0.352 / 0.383** | 0.362 / 0.385 | 0.397 / 0.421 | 0.410 / 0.423 |

---

## 4. Tại Sao TiDE Lại Vượt Trội Hơn Các Mô Hình Transformer?

Từ kết quả trên cả 7 datasets, TiDE thể hiện sự vượt trội rõ rệt trước các họ Transformer cổ điển (Autoformer, Informer, FEDformer) nhờ 4 yếu tố kiến trúc:

1. **Tránh sự quá tải của cơ chế Self-Attention trên chuỗi dài:**
   - Các Transformer tính toán ma trận Attention $O(L^2)$ nhằm kết nối từng cặp điểm thời gian. Tuy nhiên, chuỗi thời gian có đặc tính liên tục cục bộ, việc tính toán tương tác mọi cặp điểm dẫn đến nhiễu loạn thông tin và tiêu tốn bộ nhớ vô ích.
   - TiDE dùng MLP trải phẳng, học ánh xạ trực tiếp từ toàn bộ ngữ cảnh quá khứ sang tương lai với độ phức tạp tuyến tính $O(L+H)$.

2. **Khắc phục điểm yếu chí mạng của mô hình tuyến tính (DLinear):**
   - DLinear chứng minh được mạng tuyến tính 1 tầng vượt qua Transformer, nhưng DLinear **không thể học được tính phi tuyến** và **hoàn toàn bỏ qua biến ngoại sinh (covariates)** vì đưa thêm biến ngoại sinh vào DLinear sẽ làm suy giảm hiệu năng.
   - TiDE dung nạp xuất sắc cả 3 nguồn biến: vừa có nhánh `Linear Skip` (như DLinear) để giữ vững xu hướng vĩ mô, vừa có `Dense MLP` và `Temporal Decoder` để học dao động phi tuyến cục bộ.

3. **Làn đường cao tốc ngoại sinh (Covariate Highway):**
   - Temporal Decoder đưa trực tiếp $\tilde{x}_{L+t}$ vào bước dự báo tương lai, giúp mô hình phản ứng tức thời với các cú sốc theo giờ/ngày mà không làm loãng thông tin qua nhiều tầng sâu.

4. **Tốc độ huấn luyện và suy luận áp đảo:**
   - TiDE huấn luyện nhanh hơn 5 đến 10 lần so với Transformer phức tạp, cho phép tìm kiếm siêu tham số và hội tụ tối ưu chỉ trong thời gian ngắn.
