# PHẦN 3: THEO DÕI KÍCH THƯỚC TENSOR & KIỂM THỬ THỰC TẾ TRÊN TOÀN BỘ 7 TẬP BENCHMARK TiDE

> 👤 **Người thực hiện:** Hiệu  
> 🤖 **Trợ lý kỹ thuật:** Antigravity  
> 📄 **Tài liệu tham chiếu:** *Long-term Forecasting with TiDE: Time-series Dense Encoder* (Google Research, 2023) - Table 1

---

## 1. Dữ Liệu Nguồn Thực Tế: Đầy Đủ Toàn Bộ 7 Tập Benchmark LTTF

Toàn bộ **7/7 tập dữ liệu benchmark chuẩn** được nhóm tác giả Google Research sử dụng trong bài báo đều đã được tải về thực tế trong thư mục [`data/`](../../data/):

| STT | Tập Dữ Liệu | File CSV Nguồn | Tổng Số Dòng (Rows) | Tần Số (Frequency) | Ý Nghĩa Thực Tế |
| :---: | :--- | :--- | :---: | :---: | :--- |
| **1** | **Electricity** | [`data/electricity.csv`](../../data/electricity.csv) | **26,304** | 1 Giờ | Phụ tải điện lực của 321 khách hàng (2012 - 2014). |
| **2** | **Traffic** | [`data/traffic.csv`](../../data/traffic.csv) | **17,544** | 1 Giờ | Tỷ lệ chiếm dụng làn đường từ 862 trạm cảm biến freeway California. |
| **3** | **Weather** | [`data/weather.csv`](../../data/weather.csv) | **52,696** | 10 Phút | 21 chỉ số thời tiết khí tượng từ trạm Max Planck (áp suất, nhiệt độ, độ ẩm...). |
| **4** | **ETTh1** | [`data/ETTh1.csv`](../../data/ETTh1.csv) | **17,420** | 1 Giờ | Nhiệt độ dầu máy biến áp điện lực (Trạm 1 - Theo giờ). |
| **5** | **ETTh2** | [`data/ETTh2.csv`](../../data/ETTh2.csv) | **17,420** | 1 Giờ | Nhiệt độ dầu máy biến áp điện lực (Trạm 2 - Theo giờ). |
| **6** | **ETTm1** | [`data/ETTm1.csv`](../../data/ETTm1.csv) | **69,680** | 15 Phút | Nhiệt độ dầu máy biến áp điện lực (Trạm 1 - Độ mịn 15 phút). |
| **7** | **ETTm2** | [`data/ETTm2.csv`](../../data/ETTm2.csv) | **69,680** | 15 Phút | Nhiệt độ dầu máy biến áp điện lực (Trạm 2 - Độ mịn 15 phút). |

---

## 2. Bảng Theo Dõi Kích Thước Tensor Qua Từng Tầng Mạng Trên Batch Thật

Với mỗi tập dữ liệu trong số 7 tập trên, khi trích xuất một batch thực tế ($B=32, L=720, H=96, \tilde{r}=4, p=4$):

| Bước Xử Lý | Tên Tensor | Ký Hiệu Toán Học | Kích Thước Tensor PyTorch | Diễn Giải Chi Tiết |
| :---: | :--- | :---: | :---: | :--- |
| **Input 1** | Chuỗi quá khứ | $y_{1:L}$ | `torch.Size([32, 720])` | 32 mẫu thực tế cắt từ file CSV, mỗi mẫu dài 720 bước. |
| **Input 2** | Ngoại sinh động | $x_{1:L+H}$ | `torch.Size([32, 816, 7])` | $720 + 96 = 816$ bước, $r=7$ đặc trưng thời gian chuẩn GluonTS. |
| **Ground Truth**| Nhãn thật tương lai | $y_{L+1:L+H}$ | `torch.Size([32, 96])` | 96 bước giá trị thật tương lai dùng để tính hàm mất mát (loss). |
| **RevIN** | Quá khứ chuẩn hóa | $y_{\text{norm}}$ | `torch.Size([32, 720])` | Chuẩn hóa cục bộ về $\text{Mean}=0, \text{Std}=1$. |
| **Feature Proj**| Ngoại sinh sau nén | $\tilde{x}_{1:L+H}$ | `torch.Size([32, 816, 4])` | Chiếu từ $r=7$ xuống $\tilde{r}=4$ chiều tinh gọn (`temporalWidth=4`). |
| **Flatten** | Trải phẳng ngoại sinh | $\text{vec}(\tilde{x})$ | `torch.Size([32, 3264])` | $816 \times 4 = 3264$ chiều phẳng. |
| **Concat** | Đầu vào Encoder | $u_{\text{enc}}$ | `torch.Size([32, 3984])` | $720 (y) + 3264 (\tilde{x}) = 3984$ chiều. |
| **Dense Enc** | Vector bối cảnh ẩn | $e$ | `torch.Size([32, 512])` | Nén toàn bộ quá khứ thành vector 512 chiều. |
| **Dense Dec** | Vector giải mã phẳng | $g$ | `torch.Size([32, 384])` | $H \times p = 96 \times 4 = 384$ chiều. |
| **Unflatten** | Ma trận tương lai | $d_{L+1:L+H}$ | `torch.Size([32, 96, 4])` | Tái cấu trúc $g$ thành 96 bước, mỗi bước $p=4$ đặc trưng. |
| **FutCov** | Ngoại sinh tương lai | $\tilde{x}_{L+1:L+H}$ | `torch.Size([32, 96, 4])` | Cắt lấy 96 bước cuối của tensor ngoại sinh đã nén. |
| **Stack** | Đầu vào TempDec | $[d ; \tilde{x}]$ | `torch.Size([32, 96, 8])` | Ghép $d$ ($p=4$) và $\tilde{x}$ ($\tilde{r}=4$) thành $4+4=8$ chiều. |
| **Temp Dec** | Dự báo phi tuyến | $\hat{y}_{\text{NL}}$ | `torch.Size([32, 96])` | Sinh con số dự báo phi tuyến cho từng bước. |
| **Linear Skip** | Dự báo tuyến tính | $\hat{y}_{\text{Lin}}$ | `torch.Size([32, 96])` | Ánh xạ tuyến tính trực tiếp từ 720 bước sang 96 bước. |
| **Hợp nhất** | Dự báo chuẩn hóa | $\hat{y}_{\text{norm}}$ | `torch.Size([32, 96])` | Cộng 2 luồng: $\hat{y}_{\text{NL}} + \hat{y}_{\text{Lin}}$. |
| **RevIN Denorm**| Dự báo thực tế cuối | $\hat{y}_{L+1:L+H}$ | `torch.Size([32, 96])` | Nhân lại $\sigma$ và cộng lại $\mu$ $\to$ hoàn nguyên thang đo thực tế! |

---

## 3. Kết Quả Kiểm Thử Tự Động Toàn Diện (Toàn Bộ 7/7 Tập Đều PASSED)

Bạn có thể tự chạy lại script kiểm thử trên cả 7 tập bất cứ lúc nào bằng lệnh:

```powershell
python tasks-Hieu/sprint-1/test_tensor_shapes.py
```

### Log ghi nhận kết quả thực tế trên terminal:

```
################################################################################
 CHUONG TRINH KIEM THU TOAN DIEN TREN TOAN BO 7 TAP BENCHMARK TiDE PAPER
################################################################################

[*] DANG KIEM THU DATASET: ELECTRICITY (26,304 rows, 25,489 samples)
    -> Output du bao: torch.Size([32, 96]) (Hoan hao khop voi Ground Truth)
    [V] KET QUA: Dataset electricity PASSED 100%!

[*] DANG KIEM THU DATASET: TRAFFIC (17,544 rows, 16,729 samples)
    -> Output du bao: torch.Size([32, 96]) (Hoan hao khop voi Ground Truth)
    [V] KET QUA: Dataset traffic PASSED 100%!

[*] DANG KIEM THU DATASET: WEATHER (52,696 rows, 51,881 samples)
    -> Output du bao: torch.Size([32, 96]) (Hoan hao khop voi Ground Truth)
    [V] KET QUA: Dataset weather PASSED 100%!

[*] DANG KIEM THU DATASET: ETTH1 (17,420 rows, 16,605 samples)
    -> Output du bao: torch.Size([32, 96]) (Hoan hao khop voi Ground Truth)
    [V] KET QUA: Dataset ETTh1 PASSED 100%!

[*] DANG KIEM THU DATASET: ETTH2 (17,420 rows, 16,605 samples)
    -> Output du bao: torch.Size([32, 96]) (Hoan hao khop voi Ground Truth)
    [V] KET QUA: Dataset ETTh2 PASSED 100%!

[*] DANG KIEM THU DATASET: ETTM1 (69,680 rows, 68,865 samples)
    -> Output du bao: torch.Size([32, 96]) (Hoan hao khop voi Ground Truth)
    [V] KET QUA: Dataset ETTm1 PASSED 100%!

[*] DANG KIEM THU DATASET: ETTM2 (69,680 rows, 68,865 samples)
    -> Output du bao: torch.Size([32, 96]) (Hoan hao khop voi Ground Truth)
    [V] KET QUA: Dataset ETTm2 PASSED 100%!

################################################################################
 CHUC MUNG! TOAN BO 7/7 TAP DATASET BENCHMARK DA DUOC XAC THUC THANH CONG!
################################################################################
```
