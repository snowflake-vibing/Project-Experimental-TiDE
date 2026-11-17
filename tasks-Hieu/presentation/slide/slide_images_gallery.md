# 🖼️ THƯ VIỆN ẢNH TRÌNH CHIẾU THEO TỪNG NỘI DUNG SLIDE

> [!NOTE]
> File này tổng hợp **11 Slide trình chiếu chuẩn Seminar 16:9 đồng bộ** (180 DPI, đánh số từ **Slide 01 / 11** đến **Slide 11 / 11**, khoảng cách Title-to-Body chặt chẽ, cỡ chữ lớn 10.2–11.8 pt rõ nét, Footer trích nguồn chuẩn) xếp đúng **1-đối-1 theo từng mục nội dung**.
> 
> 🎙️ **Kịch bản lời nói thuyết trình tương ứng:** Xem tại [**`script_thuyet_trinh_tung_slide.md`**](./script_thuyet_trinh_tung_slide.md).

---

## 📖 PHẦN 1: LÝ THUYẾT & TOÁN HỌC ([`slide_01_phan_1_ly_thuyet_va_toan_hoc.md`](./slide_01_phan_1_ly_thuyet_va_toan_hoc.md))

### Slide 01 / 11 (1.0) — Bảng thuật ngữ cốt lõi trong TiDE (`MLPResidual`, `Dense Encoder`, `Dense Decoder`, `Temporal Decoder`, `Covariate Highway`)
* **Nguồn gốc:** Trích **Mục 4 Model & Figure 1 (Trang 4–5)**, **Mục 4.1–4.2 (Trang 5–6)** và [`models.py#L29-L61`](https://github.com/google-research/google-research/blob/master/tide/models.py#L29-L61). Giải thích 5 thuật ngữ nền tảng dạng 5 thẻ độc lập.

![Slide 01: Bảng thuật ngữ cốt lõi trong TiDE](./images/slide_01_bang_thuat_ngu_cot_loi.png)

---

### Slide 02 / 11 (1.1) — Định vị hai cơ chế trên Figure 1 gốc (Trang 5)
* **Nguồn gốc:** Trích trực tiếp **Figure 1 (Trang 5)** bài báo gốc. Khoanh rõ đúng 2 khối trình bày: `Global Residual Connection` (xanh lá) và `Temporal Decoder` (xanh dương).

![Slide 02: Vị trí Temporal Decoder và Global Residual trên Figure 1](./images/slide_02_vi_tri_kien_truc.png)

---

### Slide 03 / 11 (1.2) — So sánh 2 cột: KHÔNG CÓ vs CÓ Temporal Decoder ("Covariate Highway")
* **Nguồn gốc:** Diễn giải **Mục 4.2 (Trang 6)** & **Figure 3 (Trang 10)** theo 2 cột đối chiếu trực diện để thấy rõ vì sao bỏ `Temporal Decoder` sẽ làm tín hiệu ngoại sinh tương lai bị pha loãng qua nút thắt chung.

![Slide 03: Phân biệt KHÔNG CÓ vs CÓ Temporal Decoder](./images/slide_03_phan_biet_3_khoi_va_highway.png)

---

### Slide 04 / 11 (1.3) — Luồng song song hai cơ chế & Phương trình tổng hợp cuối cùng (kèm Shape rõ ràng)
* **Nguồn gốc:** Trực quan hóa **Mục 4.2 Decoding (Trang 6)** với phương trình chuẩn $\hat{\mathbf{y}}_{L+1:L+H}^{(i)} = \mathrm{TemporalDecoder}(\mathbf{d}_t^{(i)}; \tilde{\mathbf{x}}_{L+t}^{(i)}) + \mathbf{W}_{\mathrm{res}}\mathbf{y}_{1:L}^{(i)}$ và đầy đủ ký hiệu `Shape: (...)` trên mọi khối.

![Slide 04: Hai cơ chế song song tạo dự báo cuối cùng](./images/slide_04_hai_nhanh_song_song.png)

---

### Slide 05 / 11 (1.4) — Chứng minh toán học mối liên hệ với DLinear & Luồng kích thước Tensor tổng quát
* **Nguồn gốc:** Diễn giải nhận định *"a purely linear model like (Zeng et al., 2023) is always a subclass of our model"* tại **Mục 4.2 (Trang 6)** kèm Shape tổng quát `(B, H, p)`, `(B, L)`, `(B, H)`.

![Slide 05: Mối liên hệ DLinear và Kích thước Tensor tổng quát](./images/slide_05_lien_he_dlinear_va_tensor.png)

---

## 💻 PHẦN 2: SOURCE CODE, HYPERPARAMETERS & ABLATION STUDY ([`slide_02_phan_2_source_code_va_phan_tich.md`](./slide_02_phan_2_source_code_va_phan_tich.md))

### Slide 06 / 11 (2.1) — Source code chính thức (`models.py`) & Ánh xạ thẳng ra Phương trình Toán học
* **Nguồn gốc:** Repo chính thức Google Research [`tide/models.py#L141-L214`](https://github.com/google-research/google-research/blob/master/tide/models.py#L141-L214) & [`simulate_forward_pass.py#L228-L266`](../../../simulate_forward_pass.py#L228-L266).

![Slide 06: Source code models.py và ánh xạ phương trình](./images/slide_06_source_code_va_equation.png)

---

### Slide 07 / 11 (2.2) — Cấu hình Siêu tham số (Hyperparameters) trên 7 Benchmark Datasets
* **Nguồn gốc:** Trích trực tiếp **Table 8 (Trang 21)** & **Phụ lục B.3 / Table 7 (Trang 20)** — làm rõ `decoderOutputDim` ($p$), `temporalDecoderHidden`, `temporalWidth` ($\tilde{r}=4$), `batchSize=512` và quy luật chọn cho 7 tập dữ liệu.

![Slide 07: Cấu hình Siêu tham số Hyperparameters trên 7 Datasets](./images/slide_07_cau_hinh_hyperparameter.png)

---

### Slide 08 / 11 (2.3) — Ablation Study 1: Vai trò Temporal Decoder (Ảnh gốc từ bài báo)
* **Nguồn gốc:** Trích trực tiếp **Figure 3 — Mục 5.4 Ablation Study (Trang 10–11)** bài báo gốc kèm 3 thẻ phân tích ngắn gọn.

![Slide 08: Ablation Study 1 Temporal Decoder Figure 3](./images/slide_08_ablation_temporal_decoder.png)

---

### Slide 09 / 11 (2.4) — Ablation Study 2: Vai trò Residual Connections (Ảnh gốc từ bài báo — `Electricity`)
* **Nguồn gốc:** Trích trực tiếp **Figure 5 / Table 5 — Mục 5.4 Ablation Study (Trang 11)** bài báo gốc (Tác giả chỉ thực nghiệm bỏ Residual trên 1/7 tập đại diện là `Electricity`).

![Slide 09: Ablation Study 2 Residual Connections Electricity](./images/slide_09_ablation_global_residual.png)

---

## 💡 PHẦN 3: ĐỀ XUẤT TỐT NHẤT & ĐIỂM CẢI THIỆN TRONG TiDE ([`slide_03_phan_3_de_xuat_cai_tien.md`](./slide_03_phan_3_de_xuat_cai_tien.md))

### Slide 10 / 11 (3.1) — Đề xuất tốt nhất: Cổng Hợp Nhất Thích Nghi (*Adaptive Gated Fusion*) — Cơ sở bài báo & Lý do lựa chọn
* **Nguồn gốc:** Kế thừa cơ chế **Gated Residual Network (GRN)** từ **Lim et al. (2021) TFT (*IJF*)** & **Srivastava et al. (2015) Highway Networks**, tích hợp trực tiếp tại `models.py#L214`.

![Slide 10: Đề xuất Adaptive Gated Fusion và cơ sở bài báo](./images/slide_10_de_xuat_adaptive_gated_fusion.png)

---

### Slide 11 / 11 (3.2) — Adaptive Gated Fusion cải thiện ở điểm nào trong TiDE? & Lộ trình kiểm chứng
* **Nguồn gốc:** Tổng hợp cải thiện tại [`models.py#L214`](https://github.com/google-research/google-research/blob/master/tide/models.py#L214), **Table 2 (Trang 8)** (`Weather`), **Figure 3 (Trang 10)**, **Mục 5.4 (Trang 11)** & **Phụ lục A** kèm 3 bước kiểm chứng thực nghiệm.

![Slide 11: Điểm cải thiện cụ thể trong TiDE và lộ trình kiểm chứng](./images/slide_11_diem_cai_thien_va_roadmap.png)
