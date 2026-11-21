# 🧪 BƯỚC 2: CẤU HÌNH NOTEBOOK KAGGLE, rclone KEY & CHẠY 4 THỰC NGHIỆM (`M07` & `M08`)

> [!IMPORTANT]
> **Điều kiện tiên quyết:** Hai branch `feat/m07` và `feat/m08` đã được push lên `AIVIETNAM-AIO-dthphuc/TiDE` và **được nhóm trưởng duyệt**.
> **Nguyên tắc vàng:** Trên notebook Kaggle **chỉ thay đổi siêu tham số, branch Git cần clone và đường dẫn `REMOTE_BASE`**, tuyệt đối **không cài đặt code cải tiến trực tiếp trong notebook**.

---

## 1. Duplicate & Đặt tên chuẩn 4 Notebook Kaggle

Duplicate notebook tái lập TiDE gốc của nhóm thành **4 notebook riêng biệt** theo đúng quy ước `IS403_TiDE-fm0x_[dataset]_[horizon]`:

1. **`IS403_TiDE-fm07_ETTh1_H96`** *(Nhánh `feat/m07` — Low-Rank trên ETTh1, H=96)*
2. **`IS403_TiDE-fm07_ETTh2_H96`** *(Nhánh `feat/m07` — Low-Rank trên ETTh2, H=96)*
3. **`IS403_TiDE-fm08_ETTh1_H96`** *(Nhánh `feat/m08` — Temporal-Smoothed trên ETTh1, H=96)*
4. **`IS403_TiDE-fm08_ETTh2_H96`** *(Nhánh `feat/m08` — Temporal-Smoothed trên ETTh2, H=96)*

---

## 2. Cấu hình Secret rclone mới (`RCLONE_CONFIG_B64`)

Trong mỗi notebook, vào **Add-ons $\to$ Secrets**, kiểm tra/cập nhật Secret `RCLONE_CONFIG_B64` (và nhớ **bật nút Attach** cho notebook mới):

* **Name:** `RCLONE_CONFIG_B64`
* **Value:**

```text
W2dkcml2ZV0NCnR5cGUgPSBkcml2ZQ0KY2xpZW50X2lkID0gMTQ2OTI2NTk1MTY5LWFzbTdrdWM0MTRydjMyMDlhZWRjNWM5b3ExY2gzczJsLmFwcHMuZ29vZ2xldXNlcmNvbnRlbnQuY29tDQpjbGllbnRfc2VjcmV0ID0gR09DU1BYLVpLSE1KY3hTQmU2Qm9HUUJSMGZmcV9vc1Y1NEcNCnNjb3BlID0gZHJpdmUNCnRva2VuID0geyJhY2Nlc3NfdG9rZW4iOiJ5YTI5LmEwQWVoZlRPblZRb19mbDRzVmVZaWl5NEtxbkhQZXhianpUVHRvOFdja2NkdXlwdmZRVkxsNjE0cHlfczNtdzRSbklBWEFGWHo2Q1F5M21kLUVDOGQtcjVUcXo0RFFqdkRkNnU5VXNnREdXWDF0Ymp3WURfXzY3MEdsVHl2enVhaXpVdV8tZDIwR01wVFdvdXllSENQR05yUXhUSVgtQmdEOEJuSlpXNGR2UTNyMjFZMGtNWGNUUTFYdDRLWmR5OTBtUzJLLVNwc2FDZ1lLQWFjU0FSQVNGUUhHWDJNaXY0U1RZREFhMm0zaXdfOXZsM0xFWmcwMjA2IiwidG9rZW5fdHlwZSI6IkJlYXJlciIsInJlZnJlc2hfdG9rZW4iOiIxLy8wZzZpLVNzMUtaR19WQ2dZSUFSQUFHQkFTU1FGLUw5SXJTdGV0dDhYN3ZVWkFUb2RBVVpFM1ZibjhNLWJ2al85cHJyRDQtdUR0SVhfZWJBd3JpTF84ODdDaXdlM2tMUDlwcEVidmJjMWJVcERiU1psNjJvS3JOWkpRX0RnIiwiZXhwaXJ5IjoiMjAyNi0xMS0xMlQxNToxMToxNi4zMDE1NDgxKzA3OjAwIiwiZXhwaXJlc19pbiI6MzU5OX0NCnRlYW1fZHJpdmUgPSANCg0K
```

---

## 3. Cập nhật cell Google Drive (`REMOTE_BASE`) & Branch Git

### 3.1. Sửa cell Git Clone cho đúng từng phương pháp
* Với 2 notebook **`fm07`**:
  ```bash
  git clone -b feat/m07 https://github.com/AIVIETNAM-AIO-dthphuc/TiDE.git
  ```
* Với 2 notebook **`fm08`**:
  ```bash
  git clone -b feat/m08 https://github.com/AIVIETNAM-AIO-dthphuc/TiDE.git
  ```

### 3.2. Sửa biến `REMOTE_BASE` trong cell Google Drive
* **Cho 2 notebook `M07` (`ETTh1` và `ETTh2`):**
  ```python
  # Google drive — M07 (Low-Rank Dense Decoder)
  REMOTE_BASE = (
      "gdrive:[54] IS403 - BA/"
      "[1] PROJECT/"
      "[IS403]_Experiment/"
      "improvements/"
      "improvements/"
      "M07"
  )
  ```
* **Cho 2 notebook `M08` (`ETTh1` và `ETTh2`):**
  ```python
  # Google drive — M08 (Temporal-Smoothed Dense Decoder)
  REMOTE_BASE = (
      "gdrive:[54] IS403 - BA/"
      "[1] PROJECT/"
      "[IS403]_Experiment/"
      "improvements/"
      "improvements/"
      "M08"
  )
  ```

> [!WARNING]
> **Kiểm tra thực tế cấu trúc thư mục trên Drive:** Trong tài liệu hướng dẫn gốc của nhóm có viết lặp `"improvements/"` hai lần. Trước khi bấm Run, bạn hãy mở thư mục `[IS403]_Experiment` trên Google Drive xem thư mục `M07` / `M08` nằm dưới 1 lớp `improvements/M07` hay 2 lớp `improvements/improvements/M07` để điền cho khớp tuyệt đối!

---

## 4. Lệnh chạy chuẩn (`scripts/etth1.sh` & `scripts/etth2.sh` cho `H = 96`, 1 Seed)

Theo đúng bộ siêu tham số tái lập gốc trong `scripts/etth1.sh` và `scripts/etth2.sh` (kết hợp cờ `--seed=1 --drive_backup=true --remote_base=...` của `train.py`):

### 🔹 Notebook 1: `IS403_TiDE-fm07_ETTh1_H96` (`M07`, `r = 64`)
```bash
python train.py \
  --transform=true \
  --layer_norm=true \
  --holiday=true \
  --dropout_rate=0.5 \
  --batch_size=512 \
  --hidden_size=512 \
  --num_layers=2 \
  --hist_len=720 \
  --pred_len=96 \
  --dataset=etth1 \
  --decoder_output_dim=32 \
  --final_decoder_hidden=16 \
  --num_split=1 \
  --learning_rate=0.000984894211777642 \
  --min_num_epochs=20 \
  --decoder_rank=64 \
  --seed=1 \
  --drive_backup=true \
  --remote_base="$REMOTE_BASE"
```

### 🔹 Notebook 2: `IS403_TiDE-fm07_ETTh2_H96` (`M07`, `r = 64`)
```bash
python train.py \
  --transform=true \
  --layer_norm=true \
  --holiday=true \
  --dropout_rate=0.2 \
  --batch_size=512 \
  --hidden_size=512 \
  --num_layers=2 \
  --hist_len=720 \
  --pred_len=96 \
  --dataset=etth2 \
  --decoder_output_dim=32 \
  --final_decoder_hidden=16 \
  --num_split=1 \
  --learning_rate=0.00022400011085303776 \
  --min_num_epochs=0 \
  --decoder_rank=64 \
  --seed=1 \
  --drive_backup=true \
  --remote_base="$REMOTE_BASE"
```

### 🔹 Notebook 3: `IS403_TiDE-fm08_ETTh1_H96` (`M08`, `k = 3`)
```bash
python train.py \
  --transform=true \
  --layer_norm=true \
  --holiday=true \
  --dropout_rate=0.5 \
  --batch_size=512 \
  --hidden_size=512 \
  --num_layers=2 \
  --hist_len=720 \
  --pred_len=96 \
  --dataset=etth1 \
  --decoder_output_dim=32 \
  --final_decoder_hidden=16 \
  --num_split=1 \
  --learning_rate=0.000984894211777642 \
  --min_num_epochs=20 \
  --smooth_kernel_size=3 \
  --seed=1 \
  --drive_backup=true \
  --remote_base="$REMOTE_BASE"
```

### 🔹 Notebook 4: `IS403_TiDE-fm08_ETTh2_H96` (`M08`, `k = 3`)
```bash
python train.py \
  --transform=true \
  --layer_norm=true \
  --holiday=true \
  --dropout_rate=0.2 \
  --batch_size=512 \
  --hidden_size=512 \
  --num_layers=2 \
  --hist_len=720 \
  --pred_len=96 \
  --dataset=etth2 \
  --decoder_output_dim=32 \
  --final_decoder_hidden=16 \
  --num_split=1 \
  --learning_rate=0.00022400011085303776 \
  --min_num_epochs=0 \
  --smooth_kernel_size=3 \
  --seed=1 \
  --drive_backup=true \
  --remote_base="$REMOTE_BASE"
```

> [!NOTE]
> **Lưu ý bẫy `--resume=true` trong `train.py#L183-L188`:**
> Nếu sau này bạn muốn chạy thử giá trị khác (ví dụ đổi `--decoder_rank=32` hay `128`, hoặc `--smooth_kernel_size=5`) trên cùng `dataset` / `seed` / `horizon` đã có checkpoint trên Drive, hệ thống sẽ báo lỗi lệch `run_config.json`. Lúc đó hãy thêm **`--run_id=r32 --resume=false`** để lưu vào thư mục chạy độc lập mà không đụng checkpoint cũ!

---

## 5. Checklist 6 bước trước khi bấm Run & Thao tác sau khi xong

### Trước khi bấm Run:
1. [ ] Đã clone đúng branch **`feat/m07`** (cho 2 notebook `fm07`) hoặc **`feat/m08`** (cho 2 notebook `fm08`).
2. [ ] `REMOTE_BASE` đã trỏ đúng vào thư mục **`M07`** hoặc **`M08`**.
3. [ ] Đã bật (Attach) Secret **`RCLONE_CONFIG_B64`** mới nhất.
4. [ ] **Internet = ON** trong phần cài đặt notebook Kaggle.
5. [ ] **Accelerator = GPU** (T4 hoặc P100).
6. [ ] Kiểm tra kỹ `dataset` (`etth1` / `etth2`), `pred_len=96`, `seed=1` (hoặc seed nhóm quy định).

### Sau khi Notebook chạy thành công:
* Thêm tất cả các notebook đã chạy xong vào **group `IS403_Project`** trên Kaggle theo đúng quy định của nhóm.
