# PHẦN 2: Source Code Chính Thức, Hyperparameter & Phân Tích Chiều $p, n_d$ (Slide 08 – 10)

> **Phạm vi:** Đối chiếu trực tiếp từ codebase TensorFlow chính thức (`google-research/tide/models.py`) sang PyTorch (`simulate_forward_pass.py`), trích xuất bảng siêu tham số chuẩn trên 7 benchmark datasets (Table 7 & Table 8 — Phụ lục B.3), và phân tích sâu vai trò của `decoderOutputDim` ($p$), số tầng $n_d$, cũng như nút thắt tham số khi $H$ lớn.

---

## Slide 08 / 13 — Source Code Chính Thức (`models.py`) & Đối Chiếu PyTorch

![Slide 08: Source Code Dense Decoder](./images/slide_06_source_code_dense_decoder.png)

### Ánh xạ từng dòng code sang phương trình toán học & Shape tổng quát

| Khối | Vị trí Code chính thức | Ý nghĩa triển khai & Shape tổng quát |
| :--- | :--- | :--- |
| **[1] Chuỗi** $n_d$ **khối `MLPResidual`** | `models.py#L64-L81` (`_make_dnn_residual`) | Vòng lặp tạo từng khối `MLPResidual(hdim, hidden_dims[i+1], ...)` gồm `Dense(ReLU)` $\rightarrow$ `Dense(Linear)` + Dropout + nhánh tắt `lin_res` + `LayerNorm`. Các tầng ẩn đầu giữ Shape: $(B, d_h)$. |
| **[2] Khối cuối chiếu sang** $H \times p$ | `models.py#L133-L140` (`self.decoder`) | Thay phần tử cuối của danh sách `hidden_dims` bằng `decoder_output_dim * self.pred_len` ($p \cdot H$). Thực thi phương trình $\mathbf{g}^{(i)} = \mathrm{Decoder}(\mathbf{e}^{(i)})$ với Shape: $(B, H \cdot p)$. |
| **[3] Forward & Reshape / Unflatten** | `models.py#L207-L210` & `simulate_forward_pass.py#L217-L218` | TensorFlow dùng `tf.reshape(decoder_out, [bsize, -1, self.pred_len])` tạo Shape: $(B, p, H)$; PyTorch dùng `g.view(-1, self.H, self.p)` tạo Shape: $(B, H, p)$. Mỗi bước $t$ nhận $\mathbf{d}_t^{(i)}$ có Shape: $(B, p)$. |

---

## Slide 09 / 13 — Cấu Hình Siêu Tham Số Dense Decoder Trên 7 Benchmark Datasets

![Slide 09: Hyperparameters Dense Decoder](./images/slide_07_hyperparameters_dense_decoder.png)

### Phân tích từ Bảng Cấu Hình Chính Thức (Table 7 Trang 20 & Table 8 Trang 21)

1. **Không gian tìm kiếm (Table 7):**
   - `hiddenSize` ($d_h$): $[256, 512, 1024]$.
   - `numDecoderLayers` ($n_d$): $[1, 2, 3]$.
   - `decoderOutputDim` ($p$): $[4, 8, 16, 32]$.
   - `batchSize` $= 512$, tối ưu hóa hàm mất mát MSE bằng Adam.
2. **Giá trị tối ưu chọn trên 7 datasets (Table 8):**
   - **Số lớp** $n_d$: Toàn bộ 7/7 tập chuẩn chỉ dừng ở $n_d = 1$ (Traffic, ETTm1, Weather) hoặc $n_d = 2$ (Electricity, ETTm2, ETTh1, ETTh2). Không một tập dữ liệu nào cần tới $n_d = 3$.
   - **Chiều** $p$: Phổ biến nhất là $p = 8$ (Electricity, ETTm1, ETTh1, Weather) và $p = 16$ (Traffic, ETTm2). Riêng **ETTh2 chọn** $p = 32$, làm kích thước tầng chiếu cuối $(H \cdot p)$ tăng vọt ở horizon dài.

---

## Slide 10 / 13 — Phân Tích Ảnh Hưởng Của $p, n_d$ & Giới Hạn Khi Horizon $H$ Dài

![Slide 10: Phân tích p, n_d và Giới hạn](./images/slide_08_phan_tich_p_nd_va_gioi_han.png)

### Tổng kết kỹ thuật & Hai điểm nghẽn thực nghiệm

- **Ảnh hưởng của `decoderOutputDim`** ($p$):
  - *Khi $p$ quá nhỏ ($p = 1, 2$):* Gây **nghẽn cổ chai thông tin (Information Bottleneck)** — vector $\mathbf{d}_t^{(i)}$ quá hẹp, thiếu dung lượng mã hóa đồng thời pha, xu hướng và dao động đa biến tại bước $t$.
  - *Khi $p$ quá lớn ($p = 32, 64$):* Kích thước ngõ ra $H \cdot p$ phình to khiến tầng chiếu cuối cực nặng và dễ **Overfitting** trên các tập dữ liệu nhỏ (như ETT). Khoảng cân bằng tối ưu theo Table 8 là $p \in [8, 16]$.
- **Ảnh hưởng của số lớp Dense Decoder** ($n_d$):
  - $n_d = 1$ giúp tốc độ nhanh gọn (1 khối vừa lọc phi tuyến vừa mở rộng $d_h \rightarrow H \cdot p$).
  - $n_d = 2$ thêm 1 tầng tinh chỉnh ở không gian ẩn $(B, d_h)$ trước khi chiếu.
  - $n_d \geq 3$ bị bão hòa (0/7 datasets chọn) vì nhánh cộng tuyến tính toàn cục (Global Linear) đã gánh xu hướng chính.
- **Hai giới hạn cốt lõi của Dense Decoder gốc:**
  1. **Cố định cứng Horizon** $H$: Ma trận ngõ ra khóa cứng số hàng bằng $H \cdot p$ lúc khởi tạo (`models.py#L136`).
  2. **Bùng nổ tham số tầng chiếu cuối:** Trong `MLPResidual` (`models.py#L40-L47`), cả `lin_b` và đường tắt `lin_res` đều có kích thước $(d_h \times H \cdot p)$. Trên **Electricity** ở $H = 720$ ($d_h = 1024, p = 8 \Rightarrow H \cdot p = 5760$), riêng 2 ma trận cuối đã chiếm:

$$
\text{Params}_{\text{last}} \approx 2 \times (5{,}760 \times 1{,}024) \approx 11.8\text{ triệu tham số}
$$

Điều này lý giải trực tiếp nhận định của tác giả ở **Mục 6 Conclusion (Trang 11)**: *"Transformers are generally more parameter efficient than MLPs"*, mở đường trực tiếp cho đề xuất duy nhất tốt nhất ở Phần 3.
