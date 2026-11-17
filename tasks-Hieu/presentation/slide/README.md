# 🎞️ BỘ SLIDE THUYẾT TRÌNH 3 PHẦN: TEMPORAL DECODER & GLOBAL RESIDUAL CONNECTION

> **Phạm vi & Bố cục (Không tính slide Intro/Outro):** Tập trung vào **hai cơ chế trực tiếp tạo ra dự báo cuối cùng** trong TiDE, chia thành **3 phần rõ ràng** theo đúng bố cục bạn yêu cầu — đáp ứng trọn vẹn **5 tiêu chuẩn bắt buộc** và **7 nội dung cốt lõi**. Mỗi slide đều có sẵn hình ảnh PNG độ phân giải cao (180 DPI), sơ đồ Mermaid, nội dung súc tích chiếu trên slide, vị trí dòng code chính xác và kịch bản lời nói (Speaker Notes).

> [!TIP]
> 🎙️ **KỊCH BẢN THUYẾT TRÌNH TỪNG SLIDE (WORD-FOR-WORD SCRIPT):**  
> Đã có file kịch bản lời nói chi tiết 1-đối-1 cho cả 11 Slide (kèm thời lượng, phân đoạn mở - thân - kết, điểm nhấn laser và dự phòng câu hỏi phản biện):  
> 👉 [**`script_thuyet_trinh_tung_slide.md`**](./script_thuyet_trinh_tung_slide.md)
>
> 📌 **Bộ quy tắc thiết kế slide chuẩn mực:** Xem tại [**`rule_slide.md`**](./rule_slide.md) *(bắt buộc áp dụng cho toàn bộ slide mới hoặc cập nhật)*.

---

## 📑 Danh Sách 3 Phần Slide Chính Thức (Chuẩn Seminar 16:9 — 11 Slides)

| Phần | File Nội Dung Chi Tiết | Nội Dung & Yêu Cầu Đáp Ứng | Slides Đính Kèm (`images/`) |
| :---: | :--- | :--- | :--- |
| **Phần 1** | 📖 [**`slide_01_phan_1_ly_thuyet_va_toan_hoc.md`**](./slide_01_phan_1_ly_thuyet_va_toan_hoc.md) | **Thuật ngữ, Lý thuyết & Toán học (Slides 01–05):**<br>• **Slide 01 mở đầu:** 5 thuật ngữ (`MLPResidual`, `Dense Encoder`, `Dense Decoder`, `Temporal Decoder`, `Covariate Highway`).<br>• Định vị Figure 1 (Trang 5).<br>• So sánh 2 cột: KHÔNG CÓ vs CÓ Temporal Decoder.<br>• Hai nhánh song song (ghi rõ `Shape:`) & DLinear $\subset$ TiDE + Shape tổng quát. | • [`slide_01_bang_thuat_ngu_cot_loi.png`](./images/slide_01_bang_thuat_ngu_cot_loi.png)<br>• [`slide_02_vi_tri_kien_truc.png`](./images/slide_02_vi_tri_kien_truc.png)<br>• [`slide_03_phan_biet_3_khoi_va_highway.png`](./images/slide_03_phan_biet_3_khoi_va_highway.png)<br>• [`slide_04_hai_nhanh_song_song.png`](./images/slide_04_hai_nhanh_song_song.png)<br>• [`slide_05_lien_he_dlinear_va_tensor.png`](./images/slide_05_lien_he_dlinear_va_tensor.png) |
| **Phần 2** | 💻 [**`slide_02_phan_2_source_code_va_phan_tich.md`**](./slide_02_phan_2_source_code_va_phan_tich.md) | **Source code, Hyperparameter & Ablation (Slides 06–09):**<br>• Code chính thức Google Research (`models.py#L128-L215`) ánh xạ thẳng ra Equations.<br>• **Hyperparameters (Table 7 & Table 8, Trang 20–21):** cấu hình 7 datasets.<br>• Ablation Study 1 (Figure 3 Trang 10–11): Sự kiện Type A sau 1 Epoch.<br>• Ablation Study 2 (Table 5 Trang 11): Kết quả gốc trên `Electricity` & lý do chỉ show 1/7 tập. | • [`slide_06_source_code_va_equation.png`](./images/slide_06_source_code_va_equation.png)<br>• [`slide_07_cau_hinh_hyperparameter.png`](./images/slide_07_cau_hinh_hyperparameter.png)<br>• [`slide_08_ablation_temporal_decoder.png`](./images/slide_08_ablation_temporal_decoder.png)<br>• [`slide_09_ablation_global_residual.png`](./images/slide_09_ablation_global_residual.png) |
| **Phần 3** | 💡 [**`slide_03_phan_3_de_xuat_cai_tien.md`**](./slide_03_phan_3_de_xuat_cai_tien.md) | **Đề xuất tốt nhất & Điểm cải thiện (Slides 10–11):**<br>• **Slide 10:** *Adaptive Gated Fusion* — Kế thừa từ TFT GRN (Lim et al., 2021) & Highway Networks; vì sao tối ưu nhất (+21 params, $O(L+H)$, giải thích được).<br>• **Slide 11:** Cải thiện 3 điểm cụ thể (`models.py#L214`, `Table 2 Weather / Fig. 3 Type A`, hoàn thiện `Mục 5.4`) & 3 bước kiểm chứng. | • [`slide_10_de_xuat_adaptive_gated_fusion.png`](./images/slide_10_de_xuat_adaptive_gated_fusion.png)<br>• [`slide_11_diem_cai_thien_va_roadmap.png`](./images/slide_11_diem_cai_thien_va_roadmap.png) |

---

## 🖼️ Thư Viện 11 Slide Trình Chiếu Chuẩn 16:9 (`images/` & [`slide_images_gallery.md`](./slide_images_gallery.md))
👉 **Xem toàn bộ 11 slide đánh số (`Slide 01 / 11` → `Slide 11 / 11`) trong 1 file duy nhất:** [**`slide_images_gallery.md`**](./slide_images_gallery.md)

Tất cả 11 slide PNG đồng bộ chuẩn Seminar 16:9 (180 DPI, thống nhất typography lớn rõ, khoảng cách Title-to-Body gọn chặt, Footer nguồn gốc & số trang) đã nằm sẵn trong [`images/`](./images/) để trình chiếu trực tiếp:
* `Slide 01 / 11`: [`slide_01_bang_thuat_ngu_cot_loi.png`](./images/slide_01_bang_thuat_ngu_cot_loi.png) — Bảng 5 thuật ngữ cốt lõi (`MLPResidual`, `Dense Encoder`, `Dense Decoder`, `Temporal Decoder`, `Covariate Highway`)
* `Slide 02 / 11`: [`slide_02_vi_tri_kien_truc.png`](./images/slide_02_vi_tri_kien_truc.png) — Vị trí khoanh vùng Temporal Decoder & Global Residual trên Figure 1 gốc
* `Slide 03 / 11`: [`slide_03_phan_biet_3_khoi_va_highway.png`](./images/slide_03_phan_biet_3_khoi_va_highway.png) — So sánh 2 cột: KHÔNG CÓ vs CÓ Temporal Decoder ('Covariate Highway')
* `Slide 04 / 11`: [`slide_04_hai_nhanh_song_song.png`](./images/slide_04_hai_nhanh_song_song.png) — Sơ đồ 2 nhánh song song (ghi rõ `Shape: (...)`)
* `Slide 05 / 11`: [`slide_05_lien_he_dlinear_va_tensor.png`](./images/slide_05_lien_he_dlinear_va_tensor.png) — Chứng minh DLinear là lớp con của TiDE & Luồng Shape tổng quát
* `Slide 06 / 11`: [`slide_06_source_code_va_equation.png`](./images/slide_06_source_code_va_equation.png) — Show nguyên đoạn Code gốc (`models.py#L128-L215`) ánh xạ thẳng ra Equations
* `Slide 07 / 11`: [`slide_07_cau_hinh_hyperparameter.png`](./images/slide_07_cau_hinh_hyperparameter.png) — Cấu hình Siêu tham số chính thức trên 7 Benchmark Datasets (Table 7 & Table 8, Trang 20–21)
* `Slide 08 / 11`: [`slide_08_ablation_temporal_decoder.png`](./images/slide_08_ablation_temporal_decoder.png) — Ablation Study 1 (Figure 3 Trang 10–11): Temporal Decoder sau 1 Epoch
* `Slide 09 / 11`: [`slide_09_ablation_global_residual.png`](./images/slide_09_ablation_global_residual.png) — Ablation Study 2 (Ảnh cắt gốc Table 5 Trang 11): Vai trò Residual Connections (`Electricity`)
* `Slide 10 / 11`: [`slide_10_de_xuat_adaptive_gated_fusion.png`](./images/slide_10_de_xuat_adaptive_gated_fusion.png) — Đề xuất tốt nhất: Adaptive Gated Fusion (Lim et al., 2021 TFT GRN)
* `Slide 11 / 11`: [`slide_11_diem_cai_thien_va_roadmap.png`](./images/slide_11_diem_cai_thien_va_roadmap.png) — Điểm cải thiện cụ thể trong TiDE & Lộ trình kiểm chứng thực nghiệm
