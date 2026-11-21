# 🧪 BƯỚC 2: THIẾT LẬP NOTEBOOK KAGGLE, rclone KEY & CHẠY THỰC NGHIỆM

> [!IMPORTANT]
> **Điều kiện tiên quyết:** Branch `feat/m0x` của bạn đã được push lên GitHub và **được nhóm trưởng duyệt**.
> **Nguyên tắc vàng:** Trên notebook Kaggle **chỉ thay đổi siêu tham số và đường dẫn**, tuyệt đối **không viết/sửa code cải tiến đề xuất trực tiếp trong notebook**.

---

## 1. Tạo & Đặt tên Notebook Kaggle

* Duplicate lại notebook tái lập TiDE gốc cũ của nhóm.
* Đặt tên notebook theo đúng quy ước chuẩn:
  `IS403_TiDE-fm0x_[dataset]_[horizon]`
* Với đợt chạy nhanh đầu tiên (**1 seed** trên `ETTh1` và `ETTh2` ở `H=96`), bạn tạo 2 notebook:
  1. `IS403_TiDE-fm0x_ETTh1_H96`
  2. `IS403_TiDE-fm0x_ETTh2_H96`
  *(Thay `x` bằng đúng số thứ tự đề xuất của bạn trong sheet `improvement_proposals`, ví dụ `fm01`, `fm02`...)*.

---

## 2. Cập nhật Key rclone mới (`RCLONE_CONFIG_B64`)

Trong phần **Add-ons $\to$ Secrets** của Kaggle Notebook, kiểm tra và cập nhật (hoặc thêm mới) Secret dùng để đồng bộ checkpoint lên Google Drive:

* **Name:** `RCLONE_CONFIG_B64`
* **Value:** (Copy nguyên văn chuỗi dưới đây)

```text
W2dkcml2ZV0NCnR5cGUgPSBkcml2ZQ0KY2xpZW50X2lkID0gMTQ2OTI2NTk1MTY5LWFzbTdrdWM0MTRydjMyMDlhZWRjNWM5b3ExY2gzczJsLmFwcHMuZ29vZ2xldXNlcmNvbnRlbnQuY29tDQpjbGllbnRfc2VjcmV0ID0gR09DU1BYLVpLSE1KY3hTQmU2Qm9HUUJSMGZmcV9vc1Y1NEcNCnNjb3BlID0gZHJpdmUNCnRva2VuID0geyJhY2Nlc3NfdG9rZW4iOiJ5YTI5LmEwQWVoZlRPblZRb19mbDRzVmVZaWl5NEtxbkhQZXhianpUVHRvOFdja2NkdXlwdmZRVkxsNjE0cHlfczNtdzRSbklBWEFGWHo2Q1F5M21kLUVDOGQtcjVUcXo0RFFqdkRkNnU5VXNnREdXWDF0Ymp3WURfXzY3MEdsVHl2enVhaXpVdV8tZDIwR01wVFdvdXllSENQR05yUXhUSVgtQmdEOEJuSlpXNGR2UTNyMjFZMGtNWGNUUTFYdDRLWmR5OTBtUzJLLVNwc2FDZ1lLQWFjU0FSQVNGUUhHWDJNaXY0U1RZREFhMm0zaXdfOXZsM0xFWmcwMjA2IiwidG9rZW5fdHlwZSI6IkJlYXJlciIsInJlZnJlc2hfdG9rZW4iOiIxLy8wZzZpLVNzMUtaR19WQ2dZSUFSQUFHQkFTU1FGLUw5SXJTdGV0dDhYN3ZVWkFUb2RBVVpFM1ZibjhNLWJ2al85cHJyRDQtdUR0SVhfZWJBd3JpTF84ODdDaXdlM2tMUDlwcEVidmJjMWJVcERiU1psNjJvS3JOWkpRX0RnIiwiZXhwaXJ5IjoiMjAyNi0xMS0xMlQxNToxMToxNi4zMDE1NDgxKzA3OjAwIiwiZXhwaXJlc19pbiI6MzU5OX0NCnRlYW1fZHJpdmUgPSANCg0K
```

> [!TIP]
> Nhớ gạt nút **bật (Attach)** Secret `RCLONE_CONFIG_B64` vào từng notebook mới duplicate!

---

## 3. Cập nhật đường dẫn lưu Checkpoint (`REMOTE_BASE`)

Tìm tới cell cấu hình Google Drive trong notebook cũ và thay `REMOTE_BASE` thành đường dẫn dành riêng cho thư mục cải tiến **`M0x`**:

```python
# Google drive
REMOTE_BASE = (
    "gdrive:[54] IS403 - BA/"
    "[1] PROJECT/"
    "[IS403]_Experiment/"
    "improvements/"
    "improvements/"
    "M0x"
)
```

> [!WARNING]
> * Thay **`M0x`** thành đúng mã của bạn (VD: `M01`, `M02`...).
> * Hãy mở trực tiếp thư mục `[IS403]_Experiment` trên Google Drive của nhóm để đối chiếu xem cấu trúc thư mục thực tế là 1 cấp `improvements/` hay 2 cấp `improvements/improvements/` như hướng dẫn gốc, đảm bảo lưu đúng nơi quy định.

---

## 4. Checklist 6 bước bắt buộc trước khi bấm "Run All" / "Save & Run All"

Kiểm tra kỹ toàn bộ 6 mục dưới đây trước mỗi lần chạy:

1. **Git Branch:** Cell `git clone` trong notebook đã trỏ đúng vào repo `https://github.com/AIVIETNAM-AIO-dthphuc/TiDE` và nhánh **`feat/m0x`** (không phải `main`).
2. **rclone Key:** Đã gắn Secret `RCLONE_CONFIG_B64` mới nhất.
3. **Đường dẫn Drive:** `REMOTE_BASE` đã trỏ đúng vào thư mục `M0x`.
4. **Internet:** Trong **Notebook Options $\to$ Internet** đã chuyển sang **ON**.
5. **Accelerator (GPU):** Đã chọn **GPU** (T4 x2 hoặc P100).
6. **Siêu tham số:** Kiểm tra đúng `dataset` (`ETTh1` hoặc `ETTh2`), `horizon = 96`, `seed` (1 seed trước mắt) và các siêu tham số của đề xuất (như `r = 64`).

---

## 5. Thao tác sau khi Notebook chạy thành công

* Sau khi notebook chạy xong hoàn toàn và lưu kết quả lên Drive, **bắt buộc thêm notebook đó vào group `IS403_Project`** trên Kaggle để cả nhóm quản lý tập trung.
