# 💻 BƯỚC 1: CLONE REPO GITHUB, CÀI ĐẶT CODE ĐỀ XUẤT & PUSH BRANCH `feat/m0x`

> [!NOTE]
> **Mục tiêu bước này:** Đưa code cài đặt phương pháp đề xuất (**Low-Rank Dense Decoder Projection**) lên một nhánh riêng `feat/m0x` của repo nhóm trưởng để Kaggle có thể pull về chạy thực nghiệm.

---

## 1. Clone repo GitHub của nhóm về máy Local

* **Link repo nhóm trưởng:** `https://github.com/AIVIETNAM-AIO-dthphuc/TiDE`
* **Lưu ý:** Clone nhánh **`main`** về máy local (có thể dùng lệnh Terminal hoặc **GitHub Desktop**).

```bash
# 1. Clone nhánh main về máy
git clone -b main https://github.com/AIVIETNAM-AIO-dthphuc/TiDE.git
cd TiDE

# 2. Kiểm tra sheet improvement_proposals để lấy đúng mã x của bạn (VD: m01, m02...)
# Tạo và chuyển ngay sang branch mới theo đúng quy ước feat/m0x
git checkout -b feat/m0x
```

> [!CAUTION]
> **Tuyệt đối không** push trực tiếp lên nhánh `main`! Mọi thay đổi bắt buộc phải nằm trên nhánh `feat/m0x`.

---

## 2. Yêu cầu bắt buộc: Tạo/Cập nhật `README.md` cho phương pháp

Trong branch `feat/m0x`, bạn **bắt buộc phải có 1 file `README.md`** trình bày:
1. **Title phương pháp cải tiến**
2. **Sơ lược ý tưởng cải tiến**

Bạn có thể tham khảo mẫu dưới đây để điền vào `README.md`:

```markdown
# M0x — Low-Rank Dense Decoder Projection for TiDE

## Sơ lược ý tưởng cải tiến
- **Vấn đề của TiDE gốc:** Tầng chiếu cuối của khối `Dense Decoder` sử dụng hai ma trận dày đặc (`lin_b` và `lin_res`) chiếu trực tiếp từ không gian ẩn `d_h` sang kích thước phẳng `H * p`. Khi dự báo tầm xa (`H = 720`) hoặc `p` lớn, số lượng tham số tầng cuối bùng nổ mạnh (VD: ~11.8M params trên Electricity).
- **Giải pháp đề xuất:** Kế thừa kỹ thuật phân rã ma trận hạng thấp (Sainath et al., 2013 & LoRA, Hu et al., 2022), thay thế phép chiếu trực tiếp `d_h -> H * p` bằng hai phép biến đổi liên tiếp đi qua nút thắt hạng thấp `r`: `d_h -> r -> H * p` (với `r << min(d_h, H * p)`).
- **Kỳ vọng:** Giảm mạnh lượng tham số tầng cuối (~85%–92%), giảm FLOPs, giữ nguyên tốc độ suy luận One-Shot O(1) và đóng vai trò điều chuẩn ngầm giúp cải thiện hoặc duy trì MSE/MAE.
```

---

## 3. Cài đặt code đề xuất vào nhánh `feat/m0x`

* Tìm tới phần định nghĩa khối `Dense Decoder` (hoặc `MLPResidual` tầng cuối) trong codebase của repo `AIVIETNAM-AIO-dthphuc/TiDE`.
* Tái cấu trúc tầng chiếu cuối (`d_h -> H * p`) thành chuỗi 2 tầng Linear không hàm kích hoạt ở giữa (hoặc qua nút thắt `rank = r`):
  * Lớp 1: `d_h -> r` (nén chiều xuống nút thắt hạng thấp `r`, VD: `r = 64`).
  * Lớp 2: `r -> H * p` (mở rộng ra toàn bộ horizon).
  * Áp dụng đồng bộ cho cả nhánh chính và nhánh đường tắt (`skip` / `lin_res`) ở khối cuối cùng của `Dense Decoder` để cắt giảm triệt để tham số.
* Nên thêm siêu tham số `decoder_rank` (hoặc `r`) vào phần config để trên notebook Kaggle có thể dễ dàng điều chỉnh (VD: `r = 64`).
* Chạy test nhanh trên local để đảm bảo không lỗi shape tensor (đầu ra `Dense Decoder` sau reshape vẫn phải khớp chuẩn `(B, p, H)` hoặc shape quy ước của codebase).

---

## 4. Push lên nhánh `feat/m0x` & Báo nhóm trưởng duyệt

Sau khi hoàn tất code và `README.md`, thực hiện push lên remote nhánh `feat/m0x`:

```bash
git status
git add .
git commit -m "feat(m0x): implement Low-Rank Dense Decoder Projection"
git push -u origin feat/m0x
```

*(Nếu dùng GitHub Desktop: Publish branch `feat/m0x` lên GitHub).*

> [!IMPORTANT]
> **Thao tác bắt buộc ngay sau khi Push:**
> **Nhắn tin ngay cho nhóm trưởng để duyệt branch `feat/m0x`!**
> Phải được duyệt và push thành công lên GitHub thì mới tiến hành chạy thực nghiệm trên Kaggle được.
