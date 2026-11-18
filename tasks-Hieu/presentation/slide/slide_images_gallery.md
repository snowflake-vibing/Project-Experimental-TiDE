# Gallery Toàn Bộ 10 Slide Hình Ảnh Thuyết Trình — Khối Dense Decoder

Mọi slide bên dưới được xuất tự động ở chuẩn màn hình rộng 16:9 (**`13.333 × 7.5 in`, 180 DPI**), tuân thủ cơ chế **Zero-Dead-Space Auto-Fit** trong [`rule_slide.md`](./rule_slide.md) và tập trung **100% đúng scope Dense Decoder**.

---

## Phần 1: Thuật Ngữ, Kiến Trúc & Toán Học (Slide 01 – 05)

### Slide 01 / 10: Thuật ngữ cốt lõi cho Dense Decoder (Nền tảng)
![Slide 01](./images/slide_01_bang_thuat_ngu_dense_decoder.png)

### Slide 02 / 10: Vị trí khối Dense Decoder & Unflatten trên Figure 1
![Slide 02](./images/slide_02_vi_tri_dense_decoder.png)

### Slide 03 / 10: Cách Dense Decoder tạo biểu diễn toàn horizon (One-Shot vs Tuần tự)
![Slide 03](./images/slide_03_one_shot_vs_tuan_tu.png)

### Slide 04 / 10: Kiến trúc $n_d$ Residual Blocks & Chuyển $\mathbf{e}^{(i)}$ thành $\mathbf{g}^{(i)}$
![Slide 04](./images/slide_04_kien_truc_nd_residual_blocks.png)

### Slide 05 / 10: Vai trò thao tác Reshape (Unflatten) & Ý nghĩa vector $\mathbf{d}_t^{(i)}$
![Slide 05](./images/slide_05_reshape_va_y_nghia_dt.png)

---

## Phần 2: Source Code, Hyperparameter & Phân Tích (Slide 06 – 08)

### Slide 06 / 10: Source Code chính thức (`models.py`) & PyTorch
![Slide 06](./images/slide_06_source_code_dense_decoder.png)

### Slide 07 / 10: Cấu hình Siêu tham số Dense Decoder trên 7 Benchmark Datasets (Table 7 & 8)
![Slide 07](./images/slide_07_hyperparameters_dense_decoder.png)

### Slide 08 / 10: Phân tích ảnh hưởng của $p$, số lớp $n_d$ & Giới hạn khi $H$ dài
![Slide 08](./images/slide_08_phan_tich_p_nd_va_gioi_han.png)

---

## Phần 3: Đề Xuất Tốt Nhất & Lộ Trình Kiểm Chứng (Slide 09 – 10)

### Slide 09 / 10: Đề xuất tốt nhất — Low-Rank Dense Decoder Projection
![Slide 09](./images/slide_09_de_xuat_low_rank_decoder.png)

### Slide 10 / 10: Điểm cải thiện cụ thể trong TiDE & Validation Protocol
![Slide 10](./images/slide_10_diem_cai_thien_va_roadmap.png)
