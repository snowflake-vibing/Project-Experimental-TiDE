# PHẦN 1: Thuật Ngữ, Kiến Trúc & Toán Học Khối Dense Decoder (Slide 02 – 07)

> **Phạm vi:** 5 Slide đầu tiên xây dựng nền tảng từ điển thuật ngữ, định vị khối Dense Decoder trên kiến trúc tổng thể TiDE, làm rõ cơ chế One-Shot Non-autoregressive, công thức toán học qua $n_d$ khối Residual Block và phép `Reshape (Unflatten)` sinh ra các vector bối cảnh từng bước $\mathbf{d}_t^{(i)}$.

---

## Slide 02 / 13 — Vai trò 4 thuật ngữ nền tảng & Ký hiệu kích thước ($B, H, p$, hiddenSize)

![Slide 02: Vai trò 4 thuật ngữ nền tảng & Ký hiệu kích thước](./images/slide_01_bang_thuat_ngu_dense_decoder.png)

### Giải thích ngắn gọn vai trò từng thuật ngữ & ký hiệu trong bài báo TiDE

| STT | Thuật ngữ / Ký hiệu | Vai trò trong bài báo TiDE |
| :--- | :--- | :--- |
| **1** | **MLPResidual (Residual Block)** | Khối xử lý phi tuyến nòng cốt được xếp chồng nhiều tầng trong TiDE để trích xuất đặc trưng; dùng thêm đường tắt tuyến tính giúp đổi số chiều và truyền gradient ổn định. |
| **2** | **Unflatten (Thao tác Reshape)** | Tách vector phẳng đầu ra thành các vector riêng cho từng mốc thời gian tương lai $t = 1 \dots H$, khôi phục trục thời gian $H$ để tầng tiếp theo ghép tương ứng với biến ngoại sinh. |
| **3** | **LayerNorm (Chuẩn hóa theo tầng)** | Chuẩn hóa đặc trưng của từng mẫu ở đầu ra mỗi khối `MLPResidual` (sau cộng nhánh tắt), giúp ổn định tín hiệu và quá trình huấn luyện khi xếp chồng nhiều tầng hoặc horizon dài. |
| **4** | **Dropout (Điều chuẩn ngẫu nhiên)** | Tắt ngẫu nhiên một tỷ lệ kết nối ở nhánh phi tuyến của khối `MLPResidual` lúc huấn luyện nhằm giảm Overfitting của các tầng Dense lớn (tự động tắt khi suy luận). |
| **5** | **Ký hiệu ($B, H, p$, `hiddenSize`)** | $B$: `Batch Size` ($B=512$); $H$: `Horizon` ($96, 192, 336, 720$); $p$: `decoderOutputDim` ($8, 16, 32$); `hiddenSize` ($d_h$): độ rộng tầng ẩn ($256, 512, 1024$). |

---

## Slide 03 / 13 — Vị trí khối Dense Decoder & Unflatten trên Kiến trúc TiDE

![Slide 03: Vị trí Dense Decoder](./images/slide_02_vi_tri_dense_decoder.png)

### Điểm nhấn trên Figure 1 (Trang 5 Bài báo gốc)

- **Đầu vào:** Vector ẩn $\mathbf{e}^{(i)}$ có Shape: $(B, d_h)$ do Dense Encoder truyền lên.
- **Xử lý bên trong Dense Decoder:** Đi qua $n_d$ khối Residual Block để giải nén toàn bộ horizon cùng lúc thành $\mathbf{g}^{(i)}$ có Shape: $(B, H \cdot p)$.
- **Đầu ra sau Unflatten:** Vector phẳng $\mathbf{g}^{(i)}$ được reshape thành ma trận $\mathbf{D}^{(i)}$ và tách ra $H$ lát cắt $\mathbf{d}_t^{(i)}$ để xếp chồng cùng ngoại sinh $\tilde{\mathbf{x}}_{L+t}^{(i)}$ ở tầng kế tiếp.

---

## Slide 04 / 13 — Cách Dense Decoder tạo biểu diễn toàn horizon (One-Shot vs Tuần tự)

![Slide 04: One-Shot vs Tuần tự](./images/slide_03_one_shot_vs_tuan_tu.png)

### So sánh trực diện hai cơ chế giải mã

1. **Autoregressive truyền thống (RNN / Seq2Seq / Decoder tự hồi quy):** Sinh $\hat{y}_1 \rightarrow$ nạp lại dự đoán đó làm đầu vào đoán $\hat{y}_2 \rightarrow \dots \rightarrow \hat{y}_H$. Độ phức tạp tuần tự $O(H)$ trên GPU và làm **tích lũy sai số (Error Accumulation)** nghiêm trọng khi $H = 336, 720$.

$$
P(y_1, y_2, \dots, y_H \mid y_{1:L}) = \prod_{t=1}^H P(y_t \mid y_{1:L}, y_1, \dots, y_{t-1})
$$

2. **Dense Decoder (TiDE — Non-autoregressive / One-Shot):** Từ vector toàn cục $\mathbf{e}^{(i)}$, khối cuối $n_d$ chiếu 1 lần duy nhất ra vector phẳng $\mathbf{g}^{(i)}$ có Shape: $(B, H \cdot p)$ chứa trọn vẹn biểu diễn của toàn bộ $H$ bước tương lai. Không phụ thuộc kết quả dự đoán bước trước, triệt tiêu hoàn toàn tích lũy sai số giữa các bước và đạt độ phức tạp $O(1)$ song song hóa triệt để trên GPU (nhanh hơn Transformer 5–10 lần theo Mục 5.3).

---

## Slide 05 / 13 — Từ Công thức Toán học suy ra Code (`models.py`): Độc lập tính toán vs Tương quan biểu diễn

![Slide 05: Từ công thức toán học suy ra code](./images/slide_03b_ban_chat_toan_hoc.png)

### Liên kết từ Công thức Toán học sang cài đặt nguyên văn trong `google-research/tide/models.py`

#### A. Độc lập tính toán ($O(1)$ GPU Parallelism)

* **1. Công thức Toán học:**

$$
\mathbf{W} = [\mathbf{W}_1; \dots; \mathbf{W}_H] \in \mathbb{R}^{(H\cdot p)\times d_h}, \quad \mathbf{d}_t^{(i)} = \mathbf{W}_t \mathbf{v}^{(n_d-1)} + \mathbf{b}_t
$$

$$
\frac{\partial \mathbf{d}_t^{(i)}}{\partial \mathbf{d}_s^{(i)}} = \mathbf{0} \quad (\forall t \neq s)
$$

* **2. Suy ra cài đặt nguyên văn trong Code (`models.py#L133-L140, #L207-L210`):**

```python
# models.py#L133-L137: 1 tầng Dense xuất p * pred_len
self.decoder = _make_dnn_residual(
    model_config.get('hidden_dims')[:-1]
    + [model_config.get('decoder_output_dim') * self.pred_len],
    layer_norm=layer_norm, dropout_rate=dropout_rate,
)
# models.py#L207-L210: Forward song song & Reshape ra H bước
decoder_out = self.decoder(encoding)
decoder_out = tf.reshape(decoder_out, [bsize, -1, self.pred_len])
```

* **3. Nhận xét:**
  * Không có cạnh nối giữa $\mathbf{d}_t$ và $\mathbf{d}_s$: toàn bộ $H$ bước được tính trong 1 GEMM Kernel ($O(1)$ song song GPU), triệt tiêu lỗi tích lũy sai số.
  * Không dùng vòng lặp `for t in range(H)` vì toàn bộ đã gộp thành 1 tầng `Dense(p * H)` + `tf.reshape` chi phí $O(1)$.

#### B. Tương quan biểu diễn (Representational Correlation)

* **1. Công thức Toán học:**

$$
\mathrm{Cov}(\mathbf{d}_t^{(i)}, \mathbf{d}_s^{(i)}) = \mathbf{W}_t \, \mathbf{\Sigma}_{\mathbf{e}} \, \mathbf{W}_s^\top \neq \mathbf{0} \quad \left(\mathrm{Cov}(\mathbf{d}_t, \mathbf{d}_s \mid \mathbf{e}) = 0\right)
$$

$$
\nabla_{\mathbf{e}^{(i)}} \mathcal{L} = \sum_{t=1}^H \mathbf{W}_t^\top \nabla_{\mathbf{d}_t^{(i)}} \mathcal{L}, \qquad \hat{y}_{\mathrm{res}, t}^{(i)} = \sum_{l=1}^L W_{\mathrm{res}, t, l} \cdot y_l^{(i)}
$$

* **2. Suy ra cài đặt nguyên văn trong Code (`models.py#L203-L214, #L225`):**

```python
# models.py#L203-L214: Dùng chung encoding & Linear Skip past_ts
residual_out = self.linear(past_ts)
encoding = self.encoder(encoder_input)
final_in = tf.concat([decoder_out, enc_fut], axis=1)
out += residual_out
# models.py#L225: Tối ưu chung 1 hàm Loss cho cả H bước
loss = train_loss(ytrue, all_preds)
```

* **3. Nhận xét:**
  * Dù độc lập tính toán, cả $H$ lát cắt vẫn liên kết chặt chẽ qua chung $\mathbf{e}^{(i)}$, chung hàm Loss $\mathcal{L}$ và nhánh `self.linear(past_ts)` giữ chu kỳ.
  * Đổi lại, lớp chiếu cuối có kích thước cứng $(H \cdot p) \times d_h$ gây phình tham số ở $H=720$ — cơ sở trực tiếp cho giải pháp Low-Rank ở Phần 3.

---

## Slide 06 / 13 — Kiến trúc $n_d$ Residual Blocks & Chuyển $\mathbf{e}^{(i)}$ thành $\mathbf{g}^{(i)}$

![Slide 06: Kiến trúc n_d Residual Blocks](./images/slide_04_kien_truc_nd_residual_blocks.png)

### Hệ phương trình biến đổi qua từng tầng

Đặt đầu vào $\mathbf{v}^{(0)} = \mathbf{e}^{(i)} \in \mathbb{R}^{d_h}$ (Shape: $(B, d_h)$):

**1. Với các khối trung gian** $l \in \{1, \dots, n_d - 1\}$ (khi $n_d = 1$ thì bỏ qua tầng này):

$$
\mathbf{v}^{(l)} = \mathrm{LayerNorm}\!\left(\mathrm{Dropout}\!\left(\mathbf{W}_2^{(l)}\,\mathrm{ReLU}\!\left(\mathbf{W}_1^{(l)}\mathbf{v}^{(l-1)} + \mathbf{b}_1^{(l)}\right) + \mathbf{b}_2^{(l)}\right) + \mathbf{W}_{\mathrm{res}}^{(l)}\mathbf{v}^{(l-1)}\right)
$$

Mỗi khối trung gian duy trì kích thước Shape: $(B, d_h)$.

**2. Với khối cuối cùng** $l = n_d$ (Chiếu từ không gian ẩn $d_h$ sang không gian phẳng $H \cdot p$):

$$
\mathbf{g}^{(i)} = \mathrm{LayerNorm}\!\left(\mathrm{Dropout}\!\left(\mathbf{W}_2^{(n_d)}\,\mathrm{ReLU}\!\left(\mathbf{W}_1^{(n_d)}\mathbf{v}^{(n_d-1)} + \mathbf{b}_1^{(n_d)}\right) + \mathbf{b}_2^{(n_d)}\right) + \mathbf{W}_{\mathrm{skip}}^{(n_d)}\mathbf{v}^{(n_d-1)}\right)
$$

Vì chiều đầu ra $(H \cdot p) \neq d_h$, khối cuối **bắt buộc** dùng ma trận chiếu đường tắt $\mathbf{W}_{\mathrm{skip}}^{(n_d)} \in \mathbb{R}^{(H \cdot p) \times d_h}$ để đồng bộ kích thước cộng residual. Đầu ra $\mathbf{g}^{(i)}$ có Shape: $(B, H \cdot p)$.

```mermaid
flowchart LR
    E["e^(i)<br/>Shape: (B, d_h)"] -->|"n_d - 1 khối trung gian"| V["v^(n_d - 1)<br/>Shape: (B, d_h)"]
    V -->|"Khối cuối n_d + W_skip"| G["g^(i)<br/>Shape: (B, H * p)"]
```

---

## Slide 07 / 13 — Vai trò Reshape (Unflatten) & Ý nghĩa vector $\mathbf{d}_t^{(i)}$

![Slide 07: Reshape và Ý nghĩa d_t](./images/slide_05_reshape_va_y_nghia_dt.png)

### Giải phẫu thao tác tách chiều thời gian

Từ vector phẳng $\mathbf{g}^{(i)}$ (Shape: $(B, H \cdot p)$), phép Unflatten định hình lại ma trận 2 chiều:

$$
\mathbf{D}^{(i)} = \mathrm{Reshape}\!\left(\mathbf{g}^{(i)}\right) = \begin{bmatrix} \mathbf{d}_1^{(i)} & \mathbf{d}_2^{(i)} & \dots & \mathbf{d}_H^{(i)} \end{bmatrix} \in \mathbb{R}^{p \times H}
$$

- **Ký hiệu Shape tổng quát:** PyTorch dùng `g.view(-1, H, p)` cho Shape: $(B, H, p)$ (còn TensorFlow trong bài báo gốc reshape thành Shape: $(B, p, H)$). Khi lấy ví dụ thực tế trên Electricity (Table 8): $B = 512, d_h = 1024, H = 96, p = 8 \Rightarrow H \cdot p = 768$.
- **Chi phí bằng 0** ($O(1)$): `view` / `reshape` chỉ thay đổi thông số bước nhảy (stride metadata) trên vùng nhớ liên tục, không cấp phát hay sao chép bộ nhớ mới.
- **Ý nghĩa từng lát cắt** $\mathbf{d}_t^{(i)}$ (Shape mỗi bước: $(B, p)$): Là vector tóm tắt bối cảnh quá khứ **dành riêng cho thời điểm tương lai** $t$, mang đúng pha chu kỳ và xu hướng tương ứng với độ trễ $t$ để đưa sang tầng kế tiếp kết hợp ngoại sinh $\tilde{\mathbf{x}}_{L+t}^{(i)}$.
