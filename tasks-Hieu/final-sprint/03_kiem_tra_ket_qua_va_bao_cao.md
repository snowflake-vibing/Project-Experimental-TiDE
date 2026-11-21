# 📊 BƯỚC 3: KIỂM TRA `test_metrics.json`, TÍNH % CẢI THIỆN & BÁO CÁO (`M07` & `M08`)

> [!NOTE]
> **Mục tiêu bước này:** Mở kết quả đầu ra trên Google Drive sau khi 4 notebook Kaggle hoàn thành, tính phần trăm cải thiện `MSE` và `MAE` so với Baseline gốc và báo cáo ngay cho nhóm trưởng.

---

## 1. Vị trí file kết quả trên Google Drive

Sau khi mỗi notebook chạy xong, mở Google Drive và đi tới thư mục `REMOTE_BASE` tương ứng:
* Cho **M07:** `<REMOTE_BASE>/M07/<dataset>/seed_1/H96/results/test_metrics.json`
* Cho **M08:** `<REMOTE_BASE>/M08/<dataset>/seed_1/H96/results/test_metrics.json`

Mở file **`./results/test_metrics.json`** của từng lượt chạy để lấy hai chỉ số đối chiếu với kết quả Baseline cũ của nhóm:
* **`MAE`:** Đọc trực tiếp từ khóa `"mae"` trong `test_metrics.json`.
* **`MSE`:** Lưu ý trong `models.py#L355-L362`, `METRICS` chỉ xuất 6 khóa là `mape`, `wape`, `smape`, `nrmse`, **`rmse`**, và **`mae`** (không có khóa `"mse"` trực tiếp). Vì vậy bạn tính lại **`MSE = (rmse)^2`** từ giá trị `"rmse"` trong `test_metrics.json` (hoặc đối chiếu thêm dòng `Test Loss` trong log train).

---

## 2. Công thức tính mức độ cải thiện (%) so với Baseline

Đối chiếu kết quả mới (`new`) của `M07` / `M08` với kết quả Baseline gốc (`base`) của nhóm theo đúng công thức chuẩn:

$$
\mathrm{Improvement}_{MSE}(\%) = \frac{MSE_{\mathrm{base}} - MSE_{\mathrm{new}}}{MSE_{\mathrm{base}}} \times 100
$$

$$
\mathrm{Improvement}_{MAE}(\%) = \frac{MAE_{\mathrm{base}} - MAE_{\mathrm{new}}}{MAE_{\mathrm{base}}} \times 100
$$

### Ví dụ minh họa cách tính:

| Chỉ số | TiDE gốc (Baseline) | TiDE cải tiến (New) | Mức giảm (Improvement) |
| :--- | :---: | :---: | :---: |
| **MSE** | `0.200` | `0.192` | **`4.00%`** |
| **MAE** | `0.300` | `0.294` | **`2.00%`** |

*(Giá trị dương nghĩa là sai số giảm / tốt hơn Baseline; giá trị âm nghĩa là sai số tăng / kém hơn Baseline).*

---

## 3. Bảng tổng hợp kết quả thực nghiệm 4 Run (`M07` & `M08` — 1 Seed, `H=96`)

Điền đầy đủ số liệu từ `./results/test_metrics.json` vào bảng dưới đây trước khi gửi báo cáo cho nhóm trưởng:

| Đề xuất (Branch) | Dataset | Horizon | Cấu hình riêng | $MSE_{\mathrm{base}}$ | $MSE_{\mathrm{new}}$ | Cải thiện MSE (%) | $MAE_{\mathrm{base}}$ | $MAE_{\mathrm{new}}$ | Cải thiện MAE (%) | Ghi chú (Params / Train time) |
| :--- | :--- | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **`M07` (`feat/m07`)** | `ETTh1` | `96` | `decoder_rank=64` | `...` | `...` | `...%` | `...` | `...` | `...%` | Giảm ~85.4% params khối cuối |
| **`M07` (`feat/m07`)** | `ETTh2` | `96` | `decoder_rank=64` | `...` | `...` | `...%` | `...` | `...` | `...%` | Giảm ~85.4% params khối cuối |
| **`M08` (`feat/m08`)** | `ETTh1` | `96` | `smooth_kernel_size=3` | `...` | `...` | `...%` | `...` | `...` | `...%` | +128 params sau Unflatten |
| **`M08` (`feat/m08`)** | `ETTh2` | `96` | `smooth_kernel_size=3` | `...` | `...` | `...%` | `...` | `...` | `...%` | +128 params sau Unflatten |

---

## 4. Quy tắc báo cáo nhóm trưởng sau khi có kết quả

> [!IMPORTANT]
> **Nhắn ngay cho nhóm trưởng ngay sau khi đọc được `test_metrics.json`!**
> * **Dù MSE/MAE có cải thiện hay KHÔNG cải thiện**, đều phải gửi ngay số liệu cho nhóm trưởng để kiểm tra và phân tích nguyên nhân kịp thời.
> * **Đặc biệt lưu ý:** Nếu phương pháp nào **giảm khoảng 5% trở lên** so với kết quả Baseline cũ thì liên hệ ngay lập tức với nhóm trưởng để chốt mở rộng chạy đủ bộ Horizon (`192, 336, 720`) và nhiều seed/dataset hơn (như `Electricity`).
