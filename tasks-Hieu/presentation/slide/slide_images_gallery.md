# Gallery Toàn Bộ Slide Hình Ảnh Thuyết Trình — Khối Dense Decoder (13 Slide: 01 – 13)

Mọi slide bên dưới được xuất tự động ở chuẩn màn hình rộng 16:9 (**`13.333 × 7.5 in`, 180 DPI**), tuân thủ cơ chế **Zero-Dead-Space Auto-Fit** trong [`rule_slide.md`](./rule_slide.md) và tập trung **100% đúng scope Dense Decoder**.

---

## Slide 01 / 13: Intro — TiDE: Dense Decoder (Presenter: Le Thanh Hieu)
![Slide Intro](./images/slide_00_intro.png)

---

## Phần 1: Thuật Ngữ, Kiến Trúc & Toán Học (Slide 02 – 07)

### Slide 02 / 13: Vai trò 4 thuật ngữ nền tảng & Ký hiệu kích thước (`B`, `H`, `p`, `hiddenSize`)
![Slide 02](./images/slide_01_bang_thuat_ngu_dense_decoder.png)

### Slide 03 / 13: Vị trí khối Dense Decoder & Unflatten trên Figure 1
![Slide 03](./images/slide_02_vi_tri_dense_decoder.png)

### Slide 04 / 13: Cách Dense Decoder tạo biểu diễn toàn horizon (One-Shot vs Tuần tự)
![Slide 04](./images/slide_03_one_shot_vs_tuan_tu.png)

### Slide 05 / 13: Từ Công thức Toán học suy ra Code (`models.py`) — Độc lập tính toán vs Tương quan biểu diễn
![Slide 05](./images/slide_03b_ban_chat_toan_hoc.png)

### Slide 06 / 13: Kiến trúc $n_d$ Residual Blocks & Chuyển $\mathbf{e}^{(i)}$ thành $\mathbf{g}^{(i)}$
![Slide 06](./images/slide_04_kien_truc_nd_residual_blocks.png)

### Slide 07 / 13: Vai trò thao tác Reshape (Unflatten) & Ý nghĩa vector $\mathbf{d}_t^{(i)}$
![Slide 07](./images/slide_05_reshape_va_y_nghia_dt.png)

---

## Phần 2: Source Code, Hyperparameter & Phân Tích (Slide 08 – 10)

### Slide 08 / 13: Source Code chính thức (`models.py`) & PyTorch
![Slide 08](./images/slide_06_source_code_dense_decoder.png)

### Slide 09 / 13: Cấu hình Siêu tham số Dense Decoder trên 7 Benchmark Datasets (Table 7 & 8)
![Slide 09](./images/slide_07_hyperparameters_dense_decoder.png)

### Slide 10 / 13: Phân tích ảnh hưởng của $p$, số lớp $n_d$ & Giới hạn khi $H$ dài
![Slide 10](./images/slide_08_phan_tich_p_nd_va_gioi_han.png)

---

## Phần 3: Đề Xuất Tốt Nhất & Lộ Trình Kiểm Chứng (Slide 11 – 12)

### Slide 11 / 13: Đề xuất tốt nhất — Low-Rank Dense Decoder Projection
![Slide 11](./images/slide_09_de_xuat_low_rank_decoder.png)

### Slide 12 / 13: Điểm cải thiện cụ thể trong TiDE
![Slide 12](./images/slide_10_diem_cai_thien_va_roadmap.png)

---

## Slide 13 / 13: Outro — Thanks for listening!
![Slide Outro](./images/slide_11_outro.png)
