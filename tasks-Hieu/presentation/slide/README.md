# Bộ Slide Thuyết Trình Seminar: Khối Dense Decoder trong TiDE

Thư mục này chứa toàn bộ tài liệu trình chiếu Seminar tập trung **100% đúng phạm vi phân công về khối Dense Decoder** (*"Cách TiDE chuyển biểu diễn ẩn thành các vector dành cho từng bước dự báo"*), tuân thủ chặt chẽ quy chuẩn thiết kế tại [`rule_slide.md`](./rule_slide.md).

## Phạm vi 6 Nội Dung Trọng Tâm

1. **Kiến trúc Dense Decoder gồm** $n_d$ **Residual Blocks:** Vai trò tầng trung gian và tầng chiếu cuối.
2. **Cách chuyển vector ẩn** $\mathbf{e}^{(i)}$ **thành vector phẳng** $\mathbf{g}^{(i)}$: Kích thước mở rộng $H \times p$ và ma trận đường tắt $\mathbf{W}_{\mathrm{skip}}^{(n_d)}$.
3. **Vai trò thao tác `Reshape (Unflatten)`:** Khôi phục trục thời gian $H$ với độ phức tạp $O(1)$.
4. **Ý nghĩa vector** $\mathbf{d}_t^{(i)}$ **cho từng thời điểm dự báo:** Đại diện bối cảnh quá khứ cho bước tương lai $t \in [H]$.
5. **Giải mã One-Shot (Non-autoregressive):** Tạo biểu diễn toàn horizon $O(1)$, triệt tiêu tích lũy sai số so với cách sinh tuần tự $O(H)$.
6. **Ảnh hưởng của `decoderOutputDim`** ($p$) **và số lớp decoder** ($n_d$): Nghẽn thông tin vs bùng nổ tham số ($H = 720$) và đề xuất cải thiện bằng **Low-Rank Projection**.

---

## Cấu Trúc 13 Slide Chính Thức (3 Phần)

| Phần | File Markdown | Nội dung & Slide thuộc phần |
| :--- | :--- | :--- |
| **Intro & Outro** | [`slide_images_gallery.md`](./slide_images_gallery.md) | **Slide 01 / 13 (Intro)** & **Slide 13 / 13 (Outro)** (`TiDE: Dense Decoder` — `Thanks for listening!`) |
| **Phần 1** | [`slide_01_phan_1_ly_thuyet_va_toan_hoc.md`](./slide_01_phan_1_ly_thuyet_va_toan_hoc.md) | **Slide 02 – 07:** Thuật ngữ cốt lõi, Vị trí trên Figure 1, So sánh One-Shot vs Tuần tự, $n_d$ Residual Blocks ($\mathbf{e}^{(i)} \rightarrow \mathbf{g}^{(i)}$), Reshape & $\mathbf{d}_t^{(i)}$ |
| **Phần 2** | [`slide_02_phan_2_source_code_va_phan_tich.md`](./slide_02_phan_2_source_code_va_phan_tich.md) | **Slide 08 – 10:** Source code chính thức `models.py` & PyTorch, Bảng siêu tham số Table 7 & Table 8 (7 datasets), Phân tích $p, n_d$ & Giới hạn $H$ dài |
| **Phần 3** | [`slide_03_phan_3_de_xuat_cai_tien.md`](./slide_03_phan_3_de_xuat_cai_tien.md) | **Slide 11 – 12:** Đề xuất tốt nhất (Low-Rank Dense Decoder Projection — Hu et al. 2022 LoRA & Sainath et al. 2013) & 3 Điểm cải thiện cụ thể trong TiDE |
| **Gallery** | [`slide_images_gallery.md`](./slide_images_gallery.md) | Trang xem nhanh toàn bộ **13 ảnh PNG** (`13.333 × 7.5 in`, 180 DPI) |
| **Kịch bản** | [`script_thuyet_trinh_tung_slide.md`](./script_thuyet_trinh_tung_slide.md) | Lời nói chi tiết (từng giây, chuyển ý, Q&A) cho toàn bộ 13 Slide |
| **Quy chuẩn** | [`rule_slide.md`](./rule_slide.md) | Bộ 5 nhóm quy tắc thiết kế slide Seminar & Zero-Dead-Space Auto-Fit |
