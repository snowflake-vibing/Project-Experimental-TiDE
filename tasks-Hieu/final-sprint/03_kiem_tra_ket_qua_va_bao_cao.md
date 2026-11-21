# 📊 BƯỚC 3: KIỂM TRA KẾT QUẢ `test_metrics.json`, TÍNH % CẢI THIỆN & BÁO CÁO

> [!NOTE]
> **Mục tiêu bước này:** Đọc kết quả từ Google Drive sau khi notebook Kaggle hoàn thành, tính phần trăm cải thiện MSE/MAE so với Baseline và báo cáo ngay cho nhóm trưởng.

---

## 1. Kiểm tra file kết quả trên Google Drive

Sau khi notebook chạy thành công:
1. Mở Google Drive và đi tới đúng thư mục đầu ra của bạn (trong `REMOTE_BASE` thuộc `M0x`).
2. Tìm và mở file **`./results/test_metrics.json`** tương ứng của từng dataset (`ETTh1 H96` và `ETTh2 H96`).
3. Ghi lại hai chỉ số chính là **`MSE`** và **`MAE`**.

---

## 2. Công thức tính mức độ cải thiện (%) so với Baseline

Đối chiếu kết quả mới (`new`) với kết quả Baseline cũ (`base`) của nhóm bằng công thức sau:

$$
\mathrm{Improvement}_{MSE}(\%) = \frac{MSE_{\mathrm{base}} - MSE_{\mathrm{new}}}{MSE_{\mathrm{base}}} \times 100
$$

$$
\mathrm{Improvement}_{MAE}(\%) = \frac{MAE_{\mathrm{base}} - MAE_{\mathrm{new}}}{MAE_{\mathrm{base}}} \times 100
$$

### Ví dụ minh họa cách tính:

| Chỉ số | TiDE gốc (Baseline) | TiDE cải tiến (Low-Rank) | Mức giảm (Improvement) |
| :--- | :---: | :---: | :---: |
| **MSE** | `0.200` | `0.192` | **`4.00%`** |
| **MAE** | `0.300` | `0.294` | **`2.00%`** |

*(Nếu kết quả dương nghĩa là sai số đã giảm so với Baseline; nếu âm nghĩa là sai số cao hơn Baseline).*

---

## 3. Bảng tổng hợp kết quả thực nghiệm (1 Seed — H96)

Điền số liệu thực tế của bạn vào bảng này trước khi nhắn báo cáo nhóm trưởng:

| Dataset | Horizon | Seed | $MSE_{\text{base}}$ | $MSE_{\text{new}}$ | Cải thiện MSE (%) | $MAE_{\text{base}}$ | $MAE_{\text{new}}$ | Cải thiện MAE (%) | Ghi chú (Params / Thời gian) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **`ETTh1`** | `96` | `...` | `...` | `...` | `...%` | `...` | `...` | `...%` | `...` |
| **`ETTh2`** | `96` | `...` | `...` | `...` | `...%` | `...` | `...` | `...%` | `...` |

---

## 4. Quy tắc báo cáo nhóm trưởng

> [!IMPORTANT]
> **Báo ngay cho nhóm trưởng ngay khi có kết quả!**
> * **Dù MSE/MAE có cải thiện hay KHÔNG cải thiện**, đều phải gửi ngay kết quả `./results/test_metrics.json` cho nhóm trưởng để kiểm tra và phân tích nguyên nhân kịp thời.
> * **Đặc biệt lưu ý:** Nếu phương pháp mới **giảm khoảng 5% trở lên** so với Baseline cũ thì liên hệ ngay lập tức với nhóm trưởng để chốt cấu hình mở rộng sang các horizon/seed tiếp theo!
