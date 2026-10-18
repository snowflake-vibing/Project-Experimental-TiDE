# Phụ Lục B: Chi Tiết Thực Nghiệm & Siêu Tham Số (More Experimental Details & Hyperparameters)

Phụ lục này cung cấp toàn bộ chi tiết triển khai kỹ thuật, so sánh bổ sung với mô hình Không gian trạng thái S4, cấu trúc bộ nạp dữ liệu (Data Loader), và bảng tra cứu siêu tham số đầy đủ của mô hình TiDE trên từng bộ dữ liệu benchmark.

---

## B.1. So Sánh Bổ Sung Với Mô Hình S4 (Structured State Spaces)

Mô hình S4 (Gu et al., 2021) là một kiến trúc nổi tiếng dựa trên không gian trạng thái liên tục, được thiết kế chuyên biệt để xử lý các phụ thuộc cực dài.

Bảng so sánh đối đầu giữa TiDE và S4 trên các tập dữ liệu tại các khoảng dự báo dài ($H=336$ và $H=720$):

| Dataset | Horizon ($H$) | TiDE (Ours)<br>MSE / MAE | S4 Model<br>MSE / MAE |
|:---|:---:|:---:|:---:|
| **Weather** | 336 | **0.254 / 0.301** | 0.531 / 0.539 |
| | 720 | **0.313 / 0.340** | 0.578 / 0.578 |
| **ETTh1** | 336 | **0.435 / 0.433** | 1.407 / 0.910 |
| | 720 | **0.454 / 0.465** | 1.162 / 0.842 |
| **ETTh2** | 336 | **0.360 / 0.407** | 0.531 / 0.539 |
| | 720 | **0.419 / 0.451** | 2.650 / 1.340 |
| **Electricity** | 336 | **0.161 / 0.261** | 0.531 / 0.539 |
| | 720 | **0.196 / 0.294** | 0.578 / 0.578 |

**Kết luận**: TiDE vượt trội hoàn toàn so với mô hình S4 trên tất cả các tiêu chí và khoảng thời gian dự báo, với mức giảm sai số MSE từ **50% đến hơn 80%**.

---

## B.2. Thiết Kế Bộ Nạp Dữ Liệu (Data Loader)

Quy trình chuẩn bị dữ liệu trong TiDE tuân thủ các tiêu chuẩn nghiêm ngặt của cộng đồng nghiên cứu chuỗi thời gian:

### 1. Phân Chia Tập Dữ Liệu
- Tập Train, Validation và Test được chia tuần tự theo trục thời gian (Time-based split, thường theo tỷ lệ 7:1:2 hoặc 7:1:1 tùy dataset) để đảm bảo không bị rò rỉ dữ liệu tương lai (data leakage).

### 2. Trích Xuất Covariates Thời Gian (Calendar Features)
Dấu mốc thời gian (timestamp) được trích xuất thành các đặc trưng tuần hoàn và chuẩn hóa về đoạn $[-0.5, 0.5]$:
- Phút trong giờ: $\frac{\text{minute}}{59} - 0.5$
- Giờ trong ngày: $\frac{\text{hour}}{23} - 0.5$
- Ngày trong tuần: $\frac{\text{dayofweek}}{6} - 0.5$
- Ngày trong tháng: $\frac{\text{day} - 1}{30} - 0.5$
- Tháng trong năm: $\frac{\text{month} - 1}{11} - 0.5$

### 3. Cơ Chế Lấy Mẫu Mini-Batch
- Tại mỗi bước huấn luyện, một batch gồm `batchSize` chuỗi được chọn ngẫu nhiên.
- Với mỗi chuỗi, một mốc thời gian $t$ ngẫu nhiên được chọn làm điểm cắt để lấy ra cặp dữ liệu:
  - Đầu vào quá khứ: $y_{t-L:t-1}^{(i)}$
  - Covariates: $x_{t-L:t+H-1}^{(i)}$
  - Mục tiêu tương lai: $y_{t:t+H-1}^{(i)}$

---

## B.3. Bảng Tra Cứu Siêu Tham Số Chi Tiết (Hyperparameters Table)

Dưới đây là bảng thông số cấu hình tối ưu được sử dụng trong các thực nghiệm chính thức của bài báo (Table 8 trong paper):

| Siêu Tham Số (Hyperparameter) | Traffic | Electricity | ETTm1 | ETTm2 | ETTh1 | ETTh2 | Weather |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| `hiddenSize` | 256 | 1024 | 1024 | 512 | 256 | 512 | 512 |
| `numEncoderLayers` ($n_e$) | 1 | 2 | 1 | 2 | 2 | 2 | 1 |
| `numDecoderLayers` ($n_d$) | 1 | 2 | 1 | 2 | 2 | 2 | 1 |
| `decoderOutputDim` ($p$) | 16 | 8 | 8 | 16 | 8 | 32 | 8 |
| `temporalDecoderHidden` | 64 | 64 | 128 | 128 | 128 | 16 | 16 |
| `temporalWidth` ($\tilde{r}$) | 4 | 4 | 4 | 4 | 4 | 4 | 4 |
| `dropoutLevel` | 0.3 | 0.5 | 0.5 | 0.0 | 0.3 | 0.2 | 0.0 |
| `layerNorm` | False | True | True | True | True | True | True |
| `learningRate` | 6.55e-5 | 9.99e-4 | 8.39e-5 | 2.52e-4 | 3.82e-5 | 2.24e-4 | 3.01e-5 |
| `revIn` (Chuẩn hóa đảo nghịch) | **True** | False | False | **True** | **True** | **True** | False |
| `batchSize` | 8 | 8 | 8 | 8 | 8 | 8 | 8 |
| `lossFunction` | MSE | MSE | MSE | MSE | MSE | MSE | MSE |
| `lookbackWindow` ($L$) | 720 | 720 | 720 | 720 | 720 | 720 | 720 |

### Hướng Dẫn Tinh Chỉnh Nhanh (Practical Tuning Tips):
1. **`revIn`**: Rất quan trọng đối với các bộ dữ liệu có xu hướng phi dừng mạnh như Traffic và ETT (bật `revIn = True` giúp giảm lỗi đáng kể).
2. **`hiddenSize`**: Các tập dữ liệu lớn như Electricity và ETTm1 hưởng lợi từ kích thước ẩn lớn ($1024$), trong khi Traffic chỉ cần $256$.
3. **`dropoutLevel`**: Nên đặt từ $0.3$ đến $0.5$ trên các tập nhỏ để hạn chế overfitting, và có thể để $0.0$ trên Weather.
