# PHẦN 3: THEO DÕI KÍCH THƯỚC TENSOR & KIỂM THỬ THỰC TẾ TRÊN FILE CSV BENCHMARK

> 👤 **Người thực hiện:** Hiệu  
> 🤖 **Trợ lý kỹ thuật:** Antigravity  
> 📄 **Mục tiêu:** Chứng minh luồng dữ liệu thông suốt qua từng nút mạng, trích xuất trực tiếp từ file dữ liệu CSV thật (`ETTh1.csv`) và kiểm tra không bị lỗi lệch chiều (*dimension mismatch*).

---

## 1. Dữ Liệu Nguồn Thực Tế (Real Benchmark Dataset)

Thay vì dùng dữ liệu giả lập ngẫu nhiên (`torch.randn`), mô hình đã được nạp dữ liệu thực tế từ tập benchmark tiêu chuẩn:
- **Tập dữ liệu:** [`data/ETTh1.csv`](../../data/ETTh1.csv) (Electricity Transformer Temperature - 1 giờ).
- **Tổng số dòng dữ liệu:** 17,420 mốc thời gian.
- **Biến mục tiêu ($y$):** Cột nhiệt độ dầu máy biến áp `OT` (Oil Temperature).
- **Biến ngoại sinh động ($x$):** Trích xuất tự động 7 đặc trưng thời gian chuẩn GluonTS từ cột `date` thông qua module [`time_features.py`](./time_features.py) của Google Research:
  1. `moh`: Phút trong giờ (Minute of Hour)
  2. `hod`: Giờ trong ngày (Hour of Day)
  3. `dom`: Ngày trong tháng (Day of Month)
  4. `dow`: Ngày trong tuần (Day of Week)
  5. `doy`: Ngày trong năm (Day of Year)
  6. `moy`: Tháng trong năm (Month of Year)
  7. `woy`: Tuần trong năm (Week of Year)

---

## 2. Bảng Kích Thước Tensor Qua Từng Tầng Mạng (Chạy Thực Tế Trên Batch Thật)

Cấu hình thử nghiệm chuẩn theo bài báo TiDE:
- Batch size: $B = 32$
- Lookback window: $L = 720$ (30 ngày lịch sử)
- Forecast horizon: $H = 96$ (4 ngày dự báo)
- Số biến ngoại sinh trích xuất từ CSV: $r = 7$
- Số chiều nén Feature Projection: $\tilde{r} = 4$ (`temporalWidth = 4`)
- Số chiều giải mã Decoder: $p = 4$ (`decoderOutputDim = 4`)
- Tầng ẩn: $\text{hiddenSize} = 512$

| Bước Xử Lý | Tên Tensor | Ký Hiệu Toán Học | Kích Thước Tensor Thực Tế | Diễn Giải Chi Tiết |
| :---: | :--- | :---: | :---: | :--- |
| **Input 1** | Cắt từ CSV (`OT`) | $y_{1:L}$ | `torch.Size([32, 720])` | 32 mẫu, mỗi mẫu có 720 bước lịch sử thực tế. |
| **Input 2** | Ngoại sinh từ `date` | $x_{1:L+H}$ | `torch.Size([32, 816, 7])` | $720 + 96 = 816$ bước thời gian, mỗi bước có $r=7$ đặc trưng. |
| **Ground Truth**| Nhãn thật tương lai | $y_{L+1:L+H}$ | `torch.Size([32, 96])` | 96 bước nhiệt độ thực tế dùng để tính loss. |
| **RevIN** | Quá khứ chuẩn hóa | $y_{\text{norm}}$ | `torch.Size([32, 720])` | Đưa về phân phối chuẩn $\text{Mean}=0, \text{Std}=1$. |
| **Feature Proj** | Ngoại sinh sau nén | $\tilde{x}_{1:L+H}$ | `torch.Size([32, 816, 4])` | Chiếu từ $r=7$ chiều xuống $\tilde{r}=4$ chiều tinh gọn. |
| **Flatten** | Trải phẳng ngoại sinh | $\text{vec}(\tilde{x})$ | `torch.Size([32, 3264])` | $816 \times 4 = 3264$ chiều phẳng. |
| **Concat** | Đầu vào Encoder | $u_{\text{enc}}$ | `torch.Size([32, 3984])` | $720 (y) + 3264 (\tilde{x}) = 3984$ chiều. |
| **Dense Enc** | Vector bối cảnh ẩn | $e$ | `torch.Size([32, 512])` | Nén tri thức quá khứ thành vector 512 chiều. |
| **Dense Dec** | Vector giải mã phẳng | $g$ | `torch.Size([32, 384])` | $H \times p = 96 \times 4 = 384$ chiều phẳng. |
| **Unflatten** | Ma trận tương lai | $d_{L+1:L+H}$ | `torch.Size([32, 96, 4])` | Tái định hình $g$ thành 96 bước, mỗi bước $p=4$ đặc trưng. |
| **FutCov** | Ngoại sinh tương lai | $\tilde{x}_{L+1:L+H}$ | `torch.Size([32, 96, 4])` | Cắt lấy 96 bước cuối của tensor ngoại sinh đã nén. |
| **Stack** | Đầu vào TempDec | $[d ; \tilde{x}]$ | `torch.Size([32, 96, 8])` | Ghép $d$ ($p=4$) và $\tilde{x}$ ($\tilde{r}=4$) thành $4+4=8$ chiều. |
| **Temp Dec** | Dự báo phi tuyến | $\hat{y}_{\text{NL}}$ | `torch.Size([32, 96])` | Sinh con số dự báo phi tuyến cho từng bước. |
| **Linear Skip** | Dự báo tuyến tính | $\hat{y}_{\text{Lin}}$ | `torch.Size([32, 96])` | Ánh xạ tuyến tính trực tiếp từ 720 bước sang 96 bước. |
| **Hợp nhất** | Dự báo chuẩn hóa | $\hat{y}_{\text{norm}}$ | `torch.Size([32, 96])` | Cộng 2 nhánh: $\hat{y}_{\text{NL}} + \hat{y}_{\text{Lin}}$. |
| **RevIN Denorm**| Dự báo thực tế cuối | $\hat{y}_{L+1:L+H}$ | `torch.Size([32, 96])` | Nhân lại $\sigma$ và cộng lại $\mu$ $\to$ đơn vị nhiệt độ thực tế! |

---

## 3. Hướng Dẫn Tự Chạy Script Kiểm Thử Trên Máy Tính Của Bạn

File script hoàn chỉnh kết nối file CSV thực tế và mô hình TiDE nằm tại [`test_tensor_shapes.py`](./test_tensor_shapes.py).

Bạn mở terminal PowerShell và gõ:

```powershell
python tasks-Hieu/sprint-1/test_tensor_shapes.py
```

### Kết quả chạy thực tế:

```
===========================================================================
 KIEM TRA LUONG TENSOR TRICH XUAT TRUC TIEP TU FILE CSV BENCHMARK
===========================================================================
[*] File CSV nguon: d:\PTDLKD\Project-Experimental-TiDE\data\ETTh1.csv
[*] Tong so dong trong file CSV       : 17420
[*] So luong mau cat duoc (N samples) : 16605

[*] Kich thuoc du lieu thuc te lay tu file CSV:
    [01] batch_y (Lookback qua khu L=720)           : torch.Size([32, 720])
    [02] batch_x (Ngoai sinh dong toan bo L+H=816): torch.Size([32, 816, 7]) (r=7 dac trung thoi gian)
    [03] batch_target (Ground truth nhan tuong lai) : torch.Size([32, 96])
---------------------------------------------------------------------------
[04] y_norm (Sau chuan hoa RevIN)              : torch.Size([32, 720])
[05] x_proj (Sau Feature Projection r->4)      : torch.Size([32, 816, 4])
[06] x_flat (Ngoai sinh trai phang)            : torch.Size([32, 3264])
[07] enc_in (Dau vao Encoder)                  : torch.Size([32, 3984]) (Ky vong: 3984)
[08] Vector boi canh an e (Dense Encoder)      : torch.Size([32, 512])
[09] Vector giai ma phang g (Dense Decoder)    : torch.Size([32, 384]) (H * p = 384)
[10] Ma tran tuong lai d (Unflatten)           : torch.Size([32, 96, 4])
[11] x_future (Ngoai sinh tuong lai)           : torch.Size([32, 96, 4])
[12] temp_in (Dau vao TempDec)                 : torch.Size([32, 96, 8])
[13] y_nl (Du bao phi tuyen tu TemporalDecoder): torch.Size([32, 96])
[14] y_lin (Du bao tuyen tinh LinearSkip)      : torch.Size([32, 96])
[15] y_hat (Du bao thuc te cuoi cung)          : torch.Size([32, 96])

[*] Loss MSE khoi tao voi du lieu that tu CSV  : 20.2063

===========================================================================
 KET LUAN: DU LIEU THAT TU FILE CSV DA DUOC NAP VA XU LY 100% HOAN HAO!
===========================================================================
```
