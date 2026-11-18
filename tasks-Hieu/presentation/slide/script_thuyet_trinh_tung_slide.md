# 🎙️ KỊCH BẢN THUYẾT TRÌNH NỘI BỘ TỪNG SLIDE (INTERNAL TEAM SHARING SCRIPT)
## CHUYÊN ĐỀ: KHỐI DENSE DECODER TRONG KIẾN TRÚC MÔ HÌNH TiDE

> [!NOTE]
> **Bối cảnh sử dụng (Context):**  
> * **Đối tượng người nghe:** Toàn bộ thành viên trong nhóm đồ án / nhóm nghiên cứu TiDE.
> * **Mục tiêu bài nói:** **Giúp tất cả thành viên trong nhóm hiểu cặn kẽ, sâu sắc từng con số, từng ký hiệu, cơ chế ghép ngoại sinh, quy trình luồng code, và bản chất tại sao chọn One-Shot thay vì tuần tự**, giải quyết dứt điểm các câu hỏi phản biện kỹ thuật để nhóm tự tin code và bảo vệ đồ án.
> * **Phong cách trình bày:** Gần gũi, cởi mở, đậm chất trao đổi kỹ thuật (Tech-talk / Code-walkthrough), xưng hô "mình" và "anh em / các bạn trong nhóm". Đi thẳng vào bản chất, bóc tách từng dòng code và giải thích chi tiết các bẫy shape tensor.
> * **Tài liệu tham chiếu:** Khớp chính xác 1-đối-1 theo đúng thứ tự **10 Slide** trong [`slide_images_gallery.md`](./slide_images_gallery.md) (từ **Slide 01 / 10** đến **Slide 10 / 10**) và tuân thủ chuẩn mực [`rule_slide.md`](./rule_slide.md) cùng [`RULE.md`](../../../RULE.md).

---

## ⏱️ PHÂN BỔ THỜI GIAN TOÀN BỘ BÀI NÓI (10 SLIDES)

```mermaid
flowchart LR
    P1["Phần 1: Thuật Ngữ, Kiến Trúc & Toán<br/>(Slides 01 - 05)<br/>⏱️ ~4.5 Phút<br/>Hiểu rõ B, H, p, d_h & Bản chất One-Shot"] --> P2["Phần 2: Quy Trình Code & Phân Tích Giới Hạn<br/>(Slides 06 - 08)<br/>⏱️ ~4.0 Phút<br/>Luồng chạy 6 bước & Bùng nổ 11.8M params"]
    P2 --> P3["Phần 3: Đề Xuất Low-Rank & Roadmap<br/>(Slides 09 - 10)<br/>⏱️ ~3.0 Phút<br/>Phân rã r=64 giảm 92.6% & Kế hoạch Kaggle"]
```

---

## 📖 PHẦN 1: THUẬT NGỮ, KIẾN TRÚC & TOÁN HỌC (SLIDES 01 – 05)

---

### 🟢 SLIDE 01 / 10 — THUẬT NGỮ NỀN TẢNG CỐT LÕI (`MLPResidual`, `Unflatten`, `Information Bottleneck`)

* **Ảnh slide:** `slide_01_bang_thuat_ngu_dense_decoder.png` ([Xem ảnh trong gallery](./slide_images_gallery.md#slide-01--10-thuật-ngữ-nền-tảng-cốt-lõi-mlpresidual-unflatten-information-bottleneck))
* **Thời lượng dự kiến:** 55 – 65 giây.
* **Mục tiêu slide:** Thống nhất với cả nhóm các khái niệm nền tảng cùng các ký hiệu $B, H, p, d_h$ trước khi bước vào sơ đồ kiến trúc ở Slide 02.

![Slide 01: Thuật ngữ nền tảng cốt lõi](./images/slide_01_bang_thuat_ngu_dense_decoder.png)

#### 🎙️ Lời thoại trình bày (Team Tech-talk):

Chào mọi người! Trong kiến trúc TiDE, sau khi `Dense Encoder` nén toàn bộ dữ liệu lịch sử thì trọng trách giải mã tương lai thuộc về khối **`Dense Decoder`**. Trước khi đi sâu vào sơ đồ từ Slide 02 trở đi, anh em nhìn nhanh qua Slide 01 để mình thống nhất rõ ràng từng ký hiệu và thuật ngữ:

**1. Về bộ ký hiệu $B, H, p$ và $d_h$:**
* **$B$ là `Batch Size`:** Số lượng mẫu chuỗi nạp vào cùng lúc. Tác giả **cố định $B = 512$** cho toàn bộ 7 benchmark datasets.
* **$H$ là `Horizon`:** Tầm dự báo tương lai cần đoán trước ($96, 192, 336$ và dài nhất là $720$ bước).
* **$p$ là `decoderOutputDim`:** Số chiều đặc trưng của vector bối cảnh cho mỗi bước thời gian (thường chọn $p = 8, 16$ hoặc $32$). Nghĩa là tại mỗi thời điểm $t$, ta giữ lại một vector gồm $p$ thuộc tính để chứa bối cảnh lịch sử.
* **$d_h = \text{hiddenSize}$ là bao nhiêu?** Trong Bảng 7 Phụ lục, tác giả quét trong tập $\{256, 512, 1024\}$. Với các tập dữ liệu lớn như `Electricity` và `Traffic`, **$d_h$ được chọn tối đa là $1024$**; còn các tập `ETT` nhỏ hơn thì chọn $d_h = 256$ hoặc $512$.

**2. Ba thuật ngữ cốt lõi:**
* **`MLPResidual` (Residual Block):** Đơn vị xử lý cơ sở lặp $n_d$ lần, gồm nhánh phi tuyến Linear $\to$ ReLU $\to$ Linear $\to$ Dropout, cộng song song với nhánh tắt tuyến tính `lin_res` và `LayerNorm`.
* **`Unflatten` (Thao tác Reshape):** Phép biến đổi hình học khôi phục lại trục thời gian $H$ từ vector phẳng `Shape: (B, H * p)` thành ma trận 2 chiều `Shape: (B, p, H)` trong TensorFlow (hoặc `(B, H, p)` trong PyTorch), chỉ cập nhật metadata stride bộ nhớ với chi phí $O(1)$ không tốn copy dữ liệu.
* **`Information Bottleneck` (Nghẽn cổ chai thông tin):** Hiện tượng mất mát thông tin khi ép biểu diễn qua không gian chiều $p$ quá nhỏ (như $p = 1, 2$) — thiếu dung lượng giữ cả pha chu kỳ lẫn xu hướng; ngược lại $p$ quá lớn ($32, 64$) lại gây bùng nổ tham số và Overfitting.

Nắm chắc các con số này rồi, giờ chúng ta đi vào vị trí kiến trúc trên Figure 1 ở Slide 02 nhé!

* **🎬 Tương tác & Điểm nhấn màn hình (Visual Cues):**
  * Chỉ trực tiếp vào thẻ số 2: *"Nhớ nhé anh em: $B=512$, $d_h = 1024$, $H=96 \to 720$, $p=8 \to 32$"*.
  * Khoanh vào Thẻ 2 và 3: *"Phép Unflatten giải nén bối cảnh, còn Information Bottleneck là lý do ta phải chọn $p$ cẩn thận"*.

---

### 🟢 SLIDE 02 / 10 — VỊ TRÍ KHỐI DENSE DECODER & UNFLATTEN TRÊN FIGURE 1

* **Ảnh slide:** `slide_02_vi_tri_dense_decoder.png` ([Xem ảnh trong gallery](./slide_images_gallery.md#slide-02--10-vị-trí-khối-dense-decoder--unflatten-trên-figure-1))
* **Thời lượng dự kiến:** 50 – 60 giây.
* **Mục tiêu slide:** Định vị chính xác vị trí trên Figure 1; **giải thích cụ thể, chi tiết cơ chế ghép nối với ngoại sinh ở tầng kế tiếp** diễn ra như thế nào.

![Slide 02: Vị trí Dense Decoder](./images/slide_02_vi_tri_dense_decoder.png)

#### 🎙️ Lời thoại trình bày (Team Tech-talk):

Nhìn lên Figure 1 trích từ trang 5 của bài báo gốc ở Slide 02:

Khối `Dense Decoder` (được khoanh vàng ở giữa) nhận vector ẩn toàn cục $\mathbf{e}^{(i)}$ từ Encoder truyền lên, xử lý qua $n_d$ khối để bung ra vector phẳng $\mathbf{g}^{(i)}$ `Shape: (B, H * p)`. Ngay sau đó, phép `Unflatten` bổ đôi kích thước này ra thành $H$ lát cắt vector $\mathbf{d}_t^{(i)}$ (mỗi lát cắt có chiều dài $p$).

**Ở đây, anh em chắc chắn sẽ thắc mắc: 'Tầng kế tiếp ghép ngoại sinh thì cụ thể là ghép như thế nào?'**

Mình xin giải thích cặn kẽ luồng ghép nối này:

**Bước 1: Trích xuất vector bối cảnh:** Tại mỗi bước thời gian tương lai $t \in \{1, \dots, H\}$, mô hình trích lấy đúng lát cắt $\mathbf{d}_t^{(i)}$ (chiều dài $p$).

**Bước 2: Nối mảng với biến ngoại sinh tương lai:** Đem ghép nối tiếp (nối mảng `concatenate`) với vector biến ngoại sinh tương lai tại đúng bước $t$ đó — ký hiệu là $\tilde{\mathbf{x}}_{L+t}^{(i)}$ (ngoại sinh này đã được chiếu nén về số chiều $\tilde{r} = 4$).

Kết quả tạo thành một vector tổng hợp có kích thước:

$$
\mathbf{u}_t = \big[\mathbf{d}_t^{(i)} \; ; \; \tilde{\mathbf{x}}_{L+t}^{(i)}\big] \in \mathbb{R}^{p + \tilde{r}}
$$

*(Ví dụ: $8 + 4 = 12$, hoặc $16 + 4 = 20$).*

**Bước 3: Dự báo phi tuyến:** Vector ghép $\mathbf{u}_t$ này được nạp vào một khối `MLPResidual` nhỏ ở cổng ra có `output_dim = 1` để xuất ra đúng 1 con số dự báo phi tuyến tại ngày $t$.

Như vậy, vai trò sống còn của Dense Decoder là **tạo sẵn $H$ bộ khung bối cảnh quá khứ $\mathbf{d}_t$ cho từng ngày**, để sang tầng sau, ngày nào có sự kiện ngoại sinh gì (như khuyến mãi, thời tiết xấu) thì chỉ việc ráp thông tin ngoại sinh của ngày đó vào khung $\mathbf{d}_t$ tương ứng để dự báo!

* **🎬 Tương tác & Điểm nhấn màn hình (Visual Cues):**
  * Dùng tay chỉ luồng: $\mathbf{e}$ đi vào Decoder $\to$ Unflatten ra $H$ nhánh $\mathbf{d}_t$ $\to$ mỗi nhánh $t$ ghép với ngoại sinh tương lai $\tilde{\mathbf{x}}_{L+t}$ tương ứng $\to$ tạo thành vector $p + \tilde{r}$.

---

### 🟢 SLIDE 03 / 10 — CÁCH DENSE DECODER TẠO BIỂU DIỄN TOÀN HORIZON (ONE-SHOT VS TUẦN TỰ)

* **Ảnh slide:** `slide_03_one_shot_vs_tuan_tu.png` ([Xem ảnh trong gallery](./slide_images_gallery.md#slide-03--10-cách-dense-decoder-tạo-biểu-diễn-toàn-horizon-one-shot-vs-tuần-tự))  
  *(Tùy chọn mở rộng: Anh em có thể tách thành 2 slide **Slide 03A: So sánh One-Shot vs Tuần tự** và **Slide 03B: Hệ thống công thức toán học**, hoặc trình bày liền mạch trên Slide 03)*.
* **Thời lượng dự kiến:** 85 – 95 giây.
* **Mục tiêu slide:** **Giải quyết triệt để 2 vấn đề cốt lõi bằng hệ thống công thức toán học tường minh**: (1) Vì sao mô hình cũ phải chạy nhiều bước tuần tự mà TiDE lại làm được One-Shot? (2) Công thức nào chứng minh Dense Decoder **vừa độc lập tính toán ($O(1)$) vừa tương quan biểu diễn sâu sắc (Hiệp phương sai toàn phần)**?

![Slide 03: One-Shot vs Tuần tự](./images/slide_03_one_shot_vs_tuan_tu.png)

#### 🎙️ Lời thoại trình bày (Team Tech-talk):

Slide 03 này làm sáng tỏ triết lý giải mã của TiDE và giải quyết câu hỏi học thuật bản chất nhất: **"Tại sao Dense Decoder vừa độc lập tính toán trên GPU nhưng các bước thời gian vẫn tương quan mật thiết với nhau?"**

Để cả nhóm nắm trọn vẹn, mình xin trình bày thành hai luận điểm toán học rõ ràng:

**Phần 1: Vì sao mô hình cũ phải chạy nhiều bước tuần tự (Autoregressive)?**

Trong các kiến trúc cũ như RNN, LSTM, Seq2Seq hay Transformer Decoder cổ điển (cột bên trái):
Họ xuất phát từ giả định chuỗi Markov: muốn đoán ngày mai ($y_2$), bắt buộc phải biết kết quả ngày hôm nay ($y_1$). Tức là xác suất đồng thời bị phân rã thành tích chuỗi điều kiện nối đuôi nhau:

$$
P(y_1, y_2, \dots, y_H \mid y_{1:L}) = \prod_{t=1}^H P(y_t \mid y_{1:L}, y_1, \dots, y_{t-1})
$$

Họ bắt buộc phải chạy vòng lặp tuần tự $H$ bước: đoán $\hat{y}_1 \to$ nạp lại đoán $\hat{y}_2 \to \dots \to \hat{y}_H$. Hậu quả là GPU bị trói buộc ở độ phức tạp tuần tự $O(H)$, và nguy hiểm nhất là **tích lũy sai số (Error Accumulation)** — bước đầu lệch 1 ly thì đến bước $H=720$ kết quả bị trôi dạt hoàn toàn!

---

**Phần 2: Hệ thống công thức chứng minh Dense Decoder vừa Độc lập tính toán vừa Tương quan biểu diễn**

Anh em sẽ phản biện ngay: *"Nếu TiDE tính các bước độc lập thì có làm mất mối liên hệ thời gian giữa các ngày không?"*

Câu trả lời được chứng minh chính xác qua 2 nhóm công thức sau:

**A. Công thức chứng minh: Độc lập tính toán triệt để (Computational Independence - $O(1)$ GPU):**

Xét ma trận chiếu ngõ ra tầng cuối cùng $\mathbf{W} \in \mathbb{R}^{(H \cdot p) \times d_h}$ và vector bias $\mathbf{b} \in \mathbb{R}^{H \cdot p}$. Ta bóc tách thành $H$ ma trận con $\mathbf{W}_t \in \mathbb{R}^{p \times d_h}$ xếp chồng theo từng bước thời gian:

$$
\mathbf{W} = \begin{bmatrix} \mathbf{W}_1 \\ \mathbf{W}_2 \\ \vdots \\ \mathbf{W}_H \end{bmatrix}, \quad \mathbf{b} = \begin{bmatrix} \mathbf{b}_1 \\ \mathbf{b}_2 \\ \vdots \\ \mathbf{b}_H \end{bmatrix}
$$

Khi đó, lát cắt vector bối cảnh $\mathbf{d}_t^{(i)} \in \mathbb{R}^p$ tại mỗi thời điểm $t$ được tính trực tiếp từ tầng ẩn $\mathbf{v}^{(n_d-1)}$:

$$
\mathbf{d}_t^{(i)} = \mathbf{W}_t \mathbf{v}^{(n_d-1)} + \mathbf{b}_t
$$

Đạo hàm riêng chéo giữa hai thời điểm bất kỳ $t$ và $s$ ($t \neq s$) triệt tiêu hoàn toàn:

$$
\frac{\partial \mathbf{d}_t^{(i)}}{\partial \mathbf{d}_s^{(i)}} = \mathbf{0}, \quad \forall t \neq s
$$

Điều này chứng minh trên đồ thị tính toán: **không hề có cạnh nối nào giữa bước $t$ và bước $s$**! Bước $t$ không chờ, không nhận đầu vào từ bước $s$. Phép tính $\mathbf{g}^{(i)} = \mathbf{W}\mathbf{v} + \mathbf{b}$ là đúng 1 toán tử GEMM (nhân ma trận) duy nhất, phân bổ đồng loạt lên hàng nghìn nhân CUDA chạy song song trong thời gian $O(1)$!

**B. Công thức chứng minh: Tương quan biểu diễn sâu sắc (Representational Correlation):**

Vậy nếu tính toán độc lập thì tương quan giữa các ngày đến từ đâu? Có 3 công thức trả lời cho điều này:

**1. Luật Hiệp phương sai toàn phần (Law of Total Covariance):**

Về mặt xác suất, TiDE giả định các bước tương lai độc lập có điều kiện khi đã biết trạng thái toàn cục $\mathbf{e}^{(i)}$:

$$
P(y_1, y_2, \dots, y_H \mid \mathbf{e}^{(i)}) = \prod_{t=1}^H P(y_t \mid \mathbf{e}^{(i)})
$$

Điều này có nghĩa là $\mathrm{Cov}(\mathbf{d}_t^{(i)}, \mathbf{d}_s^{(i)} \mid \mathbf{e}^{(i)}) = \mathbf{0}$. Nhưng hiệp phương sai biên giữa hai bước thời gian $t$ và $s$ bất kỳ thì lại khác không:

$$
\mathrm{Cov}(\mathbf{d}_t^{(i)}, \mathbf{d}_s^{(i)}) = \mathbb{E}_{\mathbf{e}}\!\left[\mathrm{Cov}(\mathbf{d}_t^{(i)}, \mathbf{d}_s^{(i)} \mid \mathbf{e}^{(i)})\right] + \mathrm{Cov}_{\mathbf{e}}\!\left(\mathbb{E}[\mathbf{d}_t^{(i)} \mid \mathbf{e}^{(i)}], \, \mathbb{E}[\mathbf{d}_s^{(i)} \mid \mathbf{e}^{(i)}]\right)
$$

$$
\mathrm{Cov}(\mathbf{d}_t^{(i)}, \mathbf{d}_s^{(i)}) = \mathbf{W}_t \, \mathbf{\Sigma}_{\mathbf{e}} \, \mathbf{W}_s^\top \neq \mathbf{0}
$$

Với $\mathbf{\Sigma}_{\mathbf{e}} = \mathrm{Cov}(\mathbf{e}^{(i)}) \in \mathbb{R}^{d_h \times d_h}$ là ma trận hiệp phương sai của vector bối cảnh lịch sử. Vì $\mathbf{\Sigma}_{\mathbf{e}}$ chứa đầy đủ thông tin xu hướng và chu kỳ của quá khứ, biểu thức $\mathbf{W}_t \mathbf{\Sigma}_{\mathbf{e}} \mathbf{W}_s^\top$ chứng minh toán học rằng **hai bước $t$ và $s$ luôn tương quan mật thiết nhờ cùng chia sẻ nguồn gốc vector ẩn $\mathbf{e}^{(i)}$**!

**2. Động lực học liên kết Gradient khi Backpropagation:**

Khi huấn luyện với hàm mất mát toàn dải $\mathcal{L} = \sum_{t=1}^H \ell(\hat{y}_t^{(i)}, y_t^{(i)})$, gradient lan truyền ngược về vector $\mathbf{e}^{(i)}$ là tổng đóng góp đồng thời của tất cả các bước:

$$
\nabla_{\mathbf{e}^{(i)}} \mathcal{L} = \sum_{t=1}^H \mathbf{W}_t^\top \nabla_{\mathbf{d}_t^{(i)}} \mathcal{L}
$$

Tất cả các bước $t$ cùng đồng thời kéo không gian ẩn $\mathbf{e}^{(i)}$ về nghiệm tối ưu chung, ép các ma trận $\mathbf{W}_t$ phải học được tính trơn tru thời gian (temporal smoothness) và quan hệ chu kỳ liên tục.

**3. Khóa tương quan trễ tuyến tính qua Global Residual:**

$$
\hat{\mathbf{y}}_{\mathrm{res}}^{(i)} = \mathbf{W}_{\mathrm{res}} \mathbf{y}_{1:L}^{(i)} \implies \hat{y}_{\mathrm{res}, t}^{(i)} = \sum_{l=1}^L W_{\mathrm{res}, t, l} \cdot y_l^{(i)}
$$

Nhánh đường tắt tuyến tính $\mathbf{W}_{\mathrm{res}} \in \mathbb{R}^{H \times L}$ liên kết trực tiếp từng điểm quá khứ $l$ đến từng điểm tương lai $t$, giữ vững nguyên vẹn các tương quan tự hồi quy (như chu kỳ 24h hay 168h).

Nhờ kiến trúc này, TiDE vừa triệt tiêu tích lũy sai số, vừa giữ trọn vẹn tương quan thời gian, và chạy nhanh hơn Transformer từ 5 đến 10 lần!

* **🎬 Tương tác & Điểm nhấn màn hình (Visual Cues):**
  * Chỉ vào công thức $\frac{\partial \mathbf{d}_t}{\partial \mathbf{d}_s} = \mathbf{0}$: nhấn mạnh *"Đồ thị tính toán không có cạnh nối giữa các bước, GPU tính $O(1)$"*.
  * Chỉ vào công thức hiệp phương sai $\mathbf{W}_t \mathbf{\Sigma}_{\mathbf{e}} \mathbf{W}_s^\top$: nhấn mạnh *"Tương quan biên khác không nhờ chia sẻ chung vector ẩn $\mathbf{e}$"*.

---

### 🟢 SLIDE 04 / 10 — KIẾN TRÚC $n_d$ RESIDUAL BLOCKS & CHUYỂN $\mathbf{e}^{(i)}$ THÀNH $\mathbf{g}^{(i)}$

* **Ảnh slide:** `slide_04_kien_truc_nd_residual_blocks.png` ([Xem ảnh trong gallery](./slide_images_gallery.md#slide-04--10-kiến-trúc-n_d-residual-blocks--chuyển-mathbfei-thành-mathbfgi))
* **Thời lượng dự kiến:** 60 – 70 giây.
* **Mục tiêu slide:** Bóc tách hệ phương trình toán học qua từng tầng; giải thích vai trò bắt buộc của ma trận chiếu đường tắt $\mathbf{W}_{\text{skip}}^{(n_d)}$ ở khối cuối cùng.

![Slide 04: Kiến trúc n_d Residual Blocks](./images/slide_04_kien_truc_nd_residual_blocks.png)

#### 🎙️ Lời thoại trình bày (Team Tech-talk):

Slide 04 đi sâu vào cấu trúc toán học bên trong Dense Decoder khi nhận đầu vào $\mathbf{v}^{(0)} = \mathbf{e}^{(i)} \in \mathbb{R}^{d_h}$:

Dense Decoder chia làm 2 giai đoạn biến đổi rất rõ ràng:

**Giai đoạn 1: Các khối trung gian từ tầng $1$ đến tầng $n_d - 1$ (nếu chọn $n_d \geq 2$):**

Ở các tầng này, chiều vào và chiều ra đều giữ cố định là `Shape: (B, d_h)`:

$$
\mathbf{v}^{(l)} = \mathrm{LayerNorm}\!\left(\mathrm{Dropout}\!\left(\mathbf{W}_2^{(l)}\,\mathrm{ReLU}\!\left(\mathbf{W}_1^{(l)}\mathbf{v}^{(l-1)} + \mathbf{b}_1^{(l)}\right) + \mathbf{b}_2^{(l)}\right) + \mathbf{W}_{\mathrm{res}}^{(l)}\mathbf{v}^{(l-1)}\right)
$$

**Giai đoạn 2: Khối cuối cùng $l = n_d$ — Tầng chiếu mở rộng:**

Đây là tầng đặc biệt nhất: Chiều đầu ra không còn là $d_h$, mà được mở rộng ra kích thước khổng lồ: $H \cdot p$!

$$
\mathbf{g}^{(i)} = \mathrm{LayerNorm}\!\left(\mathrm{Dropout}\!\left(\mathbf{W}_2^{(n_d)}\,\mathrm{ReLU}\!\left(\mathbf{W}_1^{(n_d)}\mathbf{v}^{(n_d-1)} + \mathbf{b}_1^{(n_d)}\right) + \mathbf{b}_2^{(n_d)}\right) + \mathbf{W}_{\mathrm{skip}}^{(n_d)}\mathbf{v}^{(n_d-1)}\right)
$$

Anh em lưu ý chi tiết kỹ thuật này:
Vì số chiều đầu vào là $d_h$ còn đầu ra là $H \cdot p$ (hai số này lệch nhau rất lớn, ví dụ $1024$ vs $5760$), nhánh tắt Skip Connection không thể cộng trực tiếp phần tử với phần tử được. Do đó, tác giả **bắt buộc phải đặt một ma trận chiếu đường tắt $\mathbf{W}_{\text{skip}}^{(n_d)}$ có kích thước $(H \cdot p) \times d_h$** để nhân với vector đầu vào, đưa nó lên cùng số chiều $H \cdot p$ rồi mới cộng residual!

Đầu ra thu được là vector phẳng $\mathbf{g}^{(i)}$ có `Shape: (B, H * p)`. Chính sự xuất hiện của ma trận $\mathbf{W}_{\text{skip}}$ này cộng với ma trận của nhánh chính sẽ dẫn đến bài toán bùng nổ tham số mà chúng ta sẽ mổ xẻ ở Slide 08!

* **🎬 Tương tác & Điểm nhấn màn hình (Visual Cues):**
  * Chỉ vào phương trình khối cuối: khoanh tròn vào số hạng $\mathbf{W}_{\text{skip}}^{(n_d)} \mathbf{v}^{(n_d-1)}$ để anh em thấy rõ sự xuất hiện của ma trận chiếu đường tắt.

---

### 🟢 SLIDE 05 / 10 — VAI TRÒ RESHAPE (UNFLATTEN) & Ý NGHĨA VECTOR $\mathbf{d}_t^{(i)}$

* **Ảnh slide:** `slide_05_reshape_va_y_nghia_dt.png` ([Xem ảnh trong gallery](./slide_images_gallery.md#slide-05--10-vai-trò-thao-tác-reshape-unflatten--ý-nghĩa-vector-mathbfd_ti))
* **Thời lượng dự kiến:** 60 – 70 giây.
* **Mục tiêu slide:** Giải thích chuẩn xác theo đúng công thức toán học và mã nguồn của bài báo gốc: Phép Reshape tạo ra ma trận $\mathbf{D}^{(i)} \in \mathbb{R}^{p \times H}$ với Shape: $(B, p, H)$, bản chất chi phí $O(1)$ và ý nghĩa vật lý của từng vector cột $\mathbf{d}_t^{(i)}$.

![Slide 05: Reshape và Ý nghĩa d_t](./images/slide_05_reshape_va_y_nghia_dt.png)

#### 🎙️ Lời thoại trình bày (Team Tech-talk):

Sau khi tầng cuối cho ra vector phẳng $\mathbf{g}^{(i)}$ có kích thước $H \cdot p$, Slide 05 giải phẫu thao tác toán học tiếp theo: **`Reshape` (hay `Unflatten`)** để bóc tách thông tin cho từng bước tương lai.

Nhóm mình bám sát 100% theo đúng cách tác giả trình bày trong bài báo gốc và mã nguồn TensorFlow:

**1. Về mặt toán học (Mục 4.2, Trang 6 bài báo gốc):**

Tác giả định nghĩa phương trình:

$$
\mathbf{D}^{(i)} = \mathrm{Reshape}\big(\mathbf{g}^{(i)}\big) \in \mathbb{R}^{p \times H}
$$

Trong đại số tuyến tính, một chuỗi thời gian gồm $H$ bước được biểu diễn chuẩn mực dưới dạng một ma trận gồm **$H$ cột**:

$$
\mathbf{D}^{(i)} = \begin{bmatrix} \mathbf{d}_1^{(i)} & \mathbf{d}_2^{(i)} & \dots & \mathbf{d}_H^{(i)} \end{bmatrix} \in \mathbb{R}^{p \times H}
$$

Trong đó, **vector cột thứ $t$ chính là $\mathbf{d}_t^{(i)} \in \mathbb{R}^p$**.

**2. Trong mã nguồn chính thức (`models.py#L210`):**

Tác giả thực hiện đúng y như công thức toán:
`decoder_out = tf.reshape(decoder_out, [bsize, -1, self.pred_len])` $\to$ cho ra tensor có `Shape: (B, p, H)`.
Chiều thời gian $H$ nằm ở trục cuối cùng. Sau đó ở dòng 211, tác giả ghép với ngoại sinh tương lai `enc_fut` dọc theo trục đặc trưng `axis=1` để tạo thành tensor `(B, p + r_tilde, H)`, rồi mới chuyển trục đưa vào `Temporal Decoder`.

**3. Về chi phí và ý nghĩa của vector $\mathbf{d}_t^{(i)}$:**
* Phép `reshape` này có **chi phí tính toán và bộ nhớ bằng 0 ($O(1)$)**, nó không copy hay cấp phát thêm ô nhớ nào mà chỉ thay đổi metadata bước nhảy (stride) của mảng liên tục.
* Ý nghĩa của mỗi vector cột $\mathbf{d}_t^{(i)}$ (có chiều dài $p$): Nó không phải là con số dự báo thô, mà là một **vector bối cảnh mang đặc trưng quá khứ dành riêng cho thời điểm tương lai $t$**, sẵn sàng ghép nối với ngoại sinh ở bước tiếp theo!

* **🎬 Tương tác & Điểm nhấn màn hình (Visual Cues):**
  * Chỉ vào công thức ma trận $\mathbf{D}^{(i)} \in \mathbb{R}^{p \times H}$: nhấn mạnh đây là ma trận gồm $H$ cột, mỗi cột là vector $\mathbf{d}_t^{(i)}$ dài $p$.
  * Nhấn mạnh: *"Làm đúng chuẩn bài báo: Reshape ra `(B, p, H)` với chi phí $O(1)$"*.

---

## 💻 PHẦN 2: QUY TRÌNH CODE, HYPERPARAMETERS & GIỚI HẠN (SLIDES 06 – 08)

---

### 🟢 SLIDE 06 / 10 — SOURCE CODE CHÍNH THỨC (`models.py`) & ĐỐI CHIẾU PYTORCH

* **Ảnh slide:** `slide_06_source_code_dense_decoder.png` ([Xem ảnh trong gallery](./slide_images_gallery.md#slide-06--10-source-code-chính-thức-modelspy--pytorch))
* **Thời lượng dự kiến:** 70 – 80 giây.
* **Mục tiêu slide:** **Giải thích cặn kẽ, rõ ràng quy trình chạy code 6 bước từ đầu đến cuối**, chỉ rõ luồng dữ liệu biến đổi qua từng dòng code trong `models.py` và `simulate_forward_pass.py`.

![Slide 06: Source Code Dense Decoder](./images/slide_06_source_code_dense_decoder.png)

#### 🎙️ Lời thoại trình bày (Team Tech-talk):

Bây giờ chúng ta cùng soi mã nguồn thực tế ở Slide 06. Để anh em nắm chắc quy trình, mình sẽ đi theo **luồng chạy 6 bước thực tế khi forward pass qua Dense Decoder**:

* **Bước 1 — Khởi tạo danh sách tầng (`__init__` - Dòng 64–81 trong `models.py`):**
  Hàm `_make_dnn_residual` dùng vòng lặp tạo ra một danh sách gồm $n_d$ khối `MLPResidual`. Mọi tầng ẩn trung gian đều có kích thước $d_h$.
* **Bước 2 — Cài đặt tầng chiếu mở rộng (Dòng 136 trong `models.py`):**
  Tác giả thay thế phần tử cuối cùng: `hidden_dims[-1] = decoder_output_dim * self.pred_len`. Phép gán này biến khối cuối cùng thành tầng mở rộng từ $d_h$ sang $p \cdot H$.
* **Bước 3 — Tiếp nhận tensor đầu vào (Dòng 207):**
  Khối Dense Decoder nhận đầu vào là vector ẩn $\mathbf{e}$ từ Encoder truyền lên, với shape chuẩn `(B, d_h)`.
* **Bước 4 — Chạy qua các tầng MLPResidual:**
  Tensor $\mathbf{e}$ đi tuần tự qua các khối trung gian (giữ nguyên $d_h$), rồi tới khối cuối cùng, nhân với ma trận chiếu chính và ma trận đường tắt $\mathbf{W}_{\text{skip}}$ để xuất ra vector phẳng `decoder_out` có shape `(B, H * p)`.
* **Bước 5 — Tái cấu trúc không gian (Unflatten - Dòng 210 TF / Dòng 217 PyTorch):**
  Trong code PyTorch của nhóm mình: `D = g.view(-1, self.H, self.p)`. Tensor lập tức được chuyển thành ma trận 3 chiều có shape `(B, H, p)`.
* **Bước 6 — Bàn giao cho Temporal Decoder:**
  Tensor `D` sau khi tách ra $H$ bước sẽ được ghép nối tiếp với ngoại sinh tương lai `enc_fut` ở chiều cuối cùng `torch.cat([D, enc_fut], dim=-1)` để tiến hành dự báo phi tuyến.

Như vậy toàn bộ quy trình từ khởi tạo, chiếu phẳng, reshape đến ghép nối ngoại sinh diễn ra hoàn toàn liền mạch!

* **🎬 Tương tác & Điểm nhấn màn hình (Visual Cues):**
  * Dùng chuột trỏ lần lượt qua 3 khối code trên slide tương ứng với 6 bước: Khởi tạo $\to$ Đổi chiều ngõ ra $\to$ Thực thi forward $\to$ Reshape.

---

### 🟢 SLIDE 07 / 10 — CẤU HÌNH SIÊU THAM SỐ DENSE DECODER TRÊN 7 BENCHMARK DATASETS

* **Ảnh slide:** `slide_07_hyperparameters_dense_decoder.png` ([Xem ảnh trong gallery](./slide_images_gallery.md#slide-07--10-cấu-hình-siêu-tham-số-dense-decoder-trên-7-benchmark-datasets-table-7--8))
* **Thời lượng dự kiến:** 75 – 85 giây.
* **Mục tiêu slide:** Phân tích bảng cấu hình chính thức từ Bảng 7 và 8 Phụ lục B.3; chỉ ra không gian tìm kiếm siêu tham số; **giải thích cặn kẽ cơ sở thực nghiệm vì sao tác giả chọn $n_d = 1$ hoặc $n_d = 2$ cho từng nhóm dữ liệu và lý do tuyệt đối $0/7$ tập chọn $n_d = 3$**.

![Slide 07: Hyperparameters Dense Decoder](./images/slide_07_hyperparameters_dense_decoder.png)

#### 🎙️ Lời thoại trình bày (Team Tech-talk):

Slide 07 là bảng cấu hình siêu tham số chính thức của Dense Decoder trên toàn bộ 7 bộ dữ liệu chuẩn, trích từ Bảng 7 (trang 20) và Bảng 8 (trang 21) ở Phụ lục bài báo.

Khi nhìn vào bảng này, anh em sẽ thấy tác giả đưa ra không gian tìm kiếm (Table 7):
* Kích thước ẩn `hiddenSize` ($d_h$): Quét trong $\{256, 512, 1024\}$.
* Số lớp Dense Decoder `numDecoderLayers` ($n_d$): Quét trong $\{1, 2, 3\}$.
* Số chiều bối cảnh `decoderOutputDim` ($p$): Quét trong $\{4, 8, 16, 32\}$.

Nhưng điều đáng bàn nhất nằm ở kết quả lựa chọn tối ưu trên validation set ở Table 8: **Tại sao tác giả lại chọn số tầng giải mã $n_d$ như vậy?**

Nhóm mình hãy bóc tách bản chất thành 3 điểm cốt lõi:

**1. Tại sao nhóm `Traffic`, `ETTm1`, `Weather` chỉ cần chọn đúng $n_d = 1$?**
* **Với tập `Traffic`:** Bộ dữ liệu này có tới **862 chuỗi cảm biến đa biến** chạy song song. Kích thước mô hình trên tập này vốn dĩ đã rất lớn. Việc giữ $n_d = 1$ giúp khống chế lượng tham số, tránh hiện tượng bùng nổ tham số (Parameter Explosion) và tránh tràn bộ nhớ GPU khi Encoder đã làm rất tốt việc trích xuất biểu diễn.
* **Với `Weather` và `ETTm1` (chu kỳ lấy mẫu ngắn 15 phút):** Đây là các hiện tượng vật lý có chu kỳ tự nhiên lặp đi lặp lại cực kỳ đều đặn (nhịp ngày/đêm của nhiệt độ, áp suất, điện thế biến áp). Tính chất chuỗi thời gian ở đây mang tính tuyến tính và mùa vụ rất mạnh. Vì thế, chỉ cần đúng $1$ tầng giải mã phi tuyến là đã đủ để ánh xạ bối cảnh, bởi vì toàn bộ xu hướng chính đã được đường tắt `Global Residual` gánh trọn vẹn!

**2. Tại sao nhóm `Electricity`, `ETTm2`, `ETTh1`, `ETTh2` lại bắt buộc phải chọn $n_d = 2$?**
* Các tập dữ liệu này (phụ tải điện lưới và nhiệt độ biến áp theo giờ) bị chi phối nặng nề bởi **hành vi sinh hoạt phức tạp của con người**: phụ tải biến động mạnh giữa giờ cao điểm và thấp điểm, giữa ngày làm việc và cuối tuần, kèm các đợt sốc nhiệt đột ngột.
* Với $n_d = 2$, khối đầu tiên đóng vai trò xử lý trung gian trong không gian ẩn $(d_h \to d_h)$ để bóc tách các tương tác phi tuyến sâu hơn, sau đó khối thứ hai mới bung ra không gian $H \cdot p$. Hai tầng là cần thiết để mô hình đủ dung lượng học được các mẫu hình phi tuyến gãy khúc phức tạp này.

**3. Phát hiện đặc biệt: Tại sao TUYỆT ĐỐI $0/7$ tập dữ liệu nào chọn $n_d = 3$?**
* **Bão hòa năng lực phi tuyến (Diminishing Returns):** Trong TiDE, Dense Decoder không hoạt động đơn độc: bên dưới nó đã có `Global Residual` gánh thành phần tuyến tính, phía sau nó còn có `Temporal Decoder` để ráp nối ngoại sinh và tinh chỉnh chi tiết. Việc nhồi thêm tầng thứ 3 trong Dense Decoder không cung cấp thêm thông tin hữu ích nào cho mô hình.
* **Nguy cơ Overfitting nghiêm trọng:** Các tập dữ liệu chuẩn như ETT có kích thước mẫu khá hạn chế (chỉ vài nghìn đến chục nghìn điểm thời gian). Khi tăng lên $n_d = 3$, số lượng tham số dư thừa sẽ khiến mô hình học vẹt các nhiễu ngẫu nhiên, làm điểm kiểm định (validation loss) kém đi và lập tức bị thuật toán chọn siêu tham số loại bỏ!

Bây giờ, chúng ta cùng đào sâu vào việc chọn $p$ và $n_d$ sẽ gây ra hiện tượng nghẽn tham số cụ thể như thế nào ở Slide 08 nhé!

* **🎬 Tương tác & Điểm nhấn màn hình (Visual Cues):**
  * Chỉ vào cột $n_d$: nhấn mạnh sự phân hóa rõ rệt: nhóm vật lý/đa biến lớn chọn 1, nhóm hành vi phức tạp chọn 2.
  * Gạch chân điểm đặc biệt: **$0/7$ tập chọn $n_d = 3$** do bão hòa phi tuyến và nguy cơ Overfitting.
  * Khoanh vào hàng `ETTh2`: *"Tập duy nhất chọn $p = 32$, làm kích thước tầng cuối phình to gấp 4 lần"*.

---

### 🟢 SLIDE 08 / 10 — PHÂN TÍCH ẢNH HƯỞNG CỦA $p, n_d$ & GIỚI HẠN KHI HORIZON $H$ DÀI

* **Ảnh slide:** `slide_08_phan_tich_p_nd_va_gioi_han.png` ([Xem ảnh trong gallery](./slide_images_gallery.md#slide-08--10-phân-tích-ảnh-hưởng-của-p-n_d--giới-hạn-khi-horizon-h-dài))
* **Thời lượng dự kiến:** 75 – 85 giây.
* **Mục tiêu slide:** Chỉ rõ vị trí xuất hiện của $p$ trong bài báo (Mục 4.2 Trang 6, Table 7 Trang 20, Table 8 Trang 21); khẳng định việc tác giả chưa làm ablation riêng cho $p$; giải thích cơ sở toán học và mã nguồn thực tế dẫn đến con số 11.8M $\to$ 23.6M tham số.

![Slide 08: Phân tích p, n_d và Giới hạn](./images/slide_08_phan_tich_p_nd_va_gioi_han.png)

#### 🎙️ Lời thoại trình bày (Team Tech-talk):

Slide 08 này giải quyết một câu hỏi học thuật cực kỳ sắc bén mà anh em khi đọc bài báo chắc chắn sẽ thắc mắc:
**'Tác giả có nói về vấn đề chọn $p$ không, vị trí nó nằm ở đâu trong bài báo, và nếu không thì cơ sở nào để nhóm mình đưa ra các con số này?'**

Mình xin làm rõ tường minh 3 luận điểm:

**1. Vị trí tác giả đề cập đến $p$ trong bài báo gốc:**
* **Mục 4.2 (Trang 6):** Tác giả định nghĩa $p$ là siêu tham số `decoderOutputDim`, quyết định số chiều của vector bối cảnh $\mathbf{d}_t \in \mathbb{R}^p$ và quy định chiều ngõ ra của Dense Decoder là $H \cdot p$.
* **Phụ lục B.3 & Table 7 (Trang 20):** Tác giả đưa ra dải quét siêu tham số: `decoderOutputDim` $\in [4, 8, 16, 32]$. Tác giả ghi rõ dùng validation loss để chọn giá trị tối ưu cho từng tập.
* **Table 8 (Trang 21):** Công bố kết quả chọn thực tế: 4 tập chọn $p = 8$ (`Electricity`, `ETTm1`, `ETTh1`, `Weather`); 2 tập chọn $p = 16$ (`Traffic`, `ETTm2`); và chỉ đúng 1 tập duy nhất chọn $p = 32$ (`ETTh2`). **Không có bất kỳ tập nào chọn $p = 4$**.

**2. Tác giả có làm Ablation Study phân tích riêng cho $p$ không?**
* **Sự thật là KHÔNG!** Trong bài báo gốc, tác giả chỉ làm thí nghiệm Ablation cho Context Size (Hình 4), Temporal Decoder (Hình 3) và Residual (Bảng 5). Tác giả **hoàn toàn không công bố bài phân tích thực nghiệm hay biểu đồ nào về việc thay đổi $p$ sẽ ảnh hưởng ra sao**. Đây là một khoảng trống thực nghiệm rất lớn!

**3. Vậy cơ sở nào để nhóm mình đưa ra nhận định và các con số trên slide?**
* **Cơ sở 1 — Về hiện tượng nghẽn thông tin khi $p$ nhỏ:** Chuỗi thời gian thực tế có đa dạng động lực học: xu hướng dài hạn, chu kỳ ngày/đêm, chu kỳ tuần và tương quan chéo đa biến. Việc cả 7/7 tập benchmark đều 'chê' $p=4$ và chọn $p \geq 8$ chứng minh rằng nếu chọn $p$ quá nhỏ, vector $\mathbf{d}_t$ bị nghẽn cổ chai thông tin (Information Bottleneck), không đủ sức chứa bối cảnh quá khứ phức tạp.
* **Cơ sở 2 — Phép tính tham số chuẩn xác 100% từ code (`models.py#L40-L47` & dòng 136):**
  Tầng cuối cùng mở rộng ra $H \cdot p$. Ở tầm xa $H = 720$:
  * Nếu $p = 4$: $H \cdot p = 720 \times 4 = 2{,}880$.
  * Nếu $p = 8$ (chuẩn Electricity): $H \cdot p = 720 \times 8 = 5{,}760$.
  * Nếu $p = 32$ (như Table 8 chọn cho ETTh2): $H \cdot p = 720 \times 32 = \mathbf{23{,}040}$ — tức **số chiều ngõ ra nhảy vọt gấp 8 lần**!
  * Tầng cuối gồm 2 ma trận: ma trận Linear chính `lin_b` và ma trận đường tắt `lin_res`. Mỗi ma trận có kích thước $d_h \times (H \cdot p)$.
  * Với $d_h = 1024, p = 8$ trên `Electricity`, tổng tham số của riêng 2 ma trận này là:

$$
\text{Params}_{\text{last}} = 2 \times (1{,}024 \times 5{,}760) = 11{,}796{,}480 \approx 11.8\text{ triệu tham số!}
$$

  * Còn nếu lấy $p = 32$ với $d_h = 512$ trên `ETTh2`: $2 \times (512 \times 23{,}040) \approx 23.6\text{ triệu tham số}$! (Nếu $d_h = 1024$ thì lên tới **47.2 triệu**).
* **Cơ sở 3 — Chính lời thừa nhận của tác giả tại Mục 6 Conclusion (Trang 11):**
  Tác giả viết nguyên văn: *"transformers are generally more parameter efficient than MLPs while being much more memory and compute intensive. This could be a limitation..."* — tức tác giả chính thức thừa nhận nhược điểm tốn tham số của kiến trúc MLP của mình khi mô hình mở rộng!

Chính vì tác giả bỏ ngỏ việc khảo sát độ nhạy của $p$, nhóm mình mới đưa bài toán quét nhạy cảm $p \in \{2, 8, 16, 32\}$ vào Lộ trình thực nghiệm ở Slide 10!

* **🎬 Tương tác & Điểm nhấn màn hình (Visual Cues):**
  * Chỉ rõ vị trí xuất xứ: Mục 4.2 Trang 6, Table 7 Trang 20, Table 8 Trang 21 và Mục 6 Trang 11.
  * Chỉ vào phép nhân ma trận $2 \times (d_h \times H \cdot p)$ chứng minh con số 11.8M tham số là phép tính số học chính xác từ code.

---

## 💡 PHẦN 3: ĐỀ XUẤT TỐI ƯU & LỘ TRÌNH THỰC NGHIỆM (SLIDES 09 – 10)

---

### 🟢 SLIDE 09 / 10 — ĐỀ XUẤT TỐT NHẤT: LOW-RANK DENSE DECODER PROJECTION

* **Ảnh slide:** `slide_09_de_xuat_low_rank_decoder.png` ([Xem ảnh trong gallery](./slide_images_gallery.md#slide-09--10-đề-xuất-tốt-nhất--low-rank-dense-decoder-projection))
* **Thời lượng dự kiến:** 75 – 85 giây.
* **Mục tiêu slide:** Trình bày giải pháp Low-Rank Factorized Projection; giải thích nguồn gốc (LoRA & Sainath et al.), chứng minh giảm 92.6% tham số, giảm FLOPs hơn 13 lần và giữ nguyên tính chất One-Shot $O(1)$.

![Slide 09: Đề xuất Low-Rank Projection](./images/slide_09_de_xuat_low_rank_decoder.png)

#### 🎙️ Lời thoại trình bày (Team Tech-talk):

Để giải quyết triệt để nút thắt 11.8 triệu tham số mà tác giả bài báo để lại, ở Slide 09, mình xin đề xuất phương án tối ưu nhất cho nhóm mình: **Low-Rank Factorized Dense Decoder Projection**.

**1. Nguồn gốc học thuật vững chắc:**

Ý tưởng này kế thừa trực tiếp từ bài báo kinh điển **LoRA của Hu et al. (ICLR 2022)** và nghiên cứu nền tảng về **Low-Rank Factorization cho mạng sâu của Sainath et al. (ICASSP 2013)**. 
Thay vì học một ma trận trọng số khổng lồ $\mathbf{W} \in \mathbb{R}^{(H \cdot p) \times d_h}$ nối trực tiếp từ $d_h$ sang $H \cdot p$, ta **phân rã nó thành tích của hai ma trận hạng thấp qua một nút thắt $r$**:

$$
\mathbf{W} \approx \mathbf{A} \cdot \mathbf{B}, \quad \text{với } \mathbf{B} \in \mathbb{R}^{r \times d_h},\; \mathbf{A} \in \mathbb{R}^{(H \cdot p) \times r} \quad (r \ll \min(d_h, H \cdot p))
$$

**2. Vì sao đây là phương án TỐT NHẤT cho khối Dense Decoder của nhóm mình?**

* **Giảm sốc tới 92.6% tham số:** Nếu ta chọn hạng $r = 64$ trên cấu hình `Electricity` ($d_h = 1024, H \cdot p = 5760$), số tham số của 2 ma trận chiếu cuối giảm từ 11.8 triệu xuống chỉ còn:

$$
\text{Params}_{\text{Low-Rank}} = 2 \times (1{,}024 \times 64 + 64 \times 5{,}760) \approx 0.87\text{ triệu tham số!}
$$

* **Giảm FLOPs tính toán hơn 13 lần:** Thay vì $5760 \times 1024 \approx 5.9 \times 10^6$ phép nhân, ta chỉ tốn $(64 \times 1024) + (5760 \times 64) \approx 4.3 \times 10^5$ phép tính.
* **Bảo toàn 100% tính chất One-Shot $O(1)$:** Mô hình vẫn chiếu toàn bộ horizon trong 1 lần forward duy nhất, không thêm vòng lặp tuần tự nào, tốc độ GPU giữ nguyên tuyệt đối.
* **Đóng vai trò điều chuẩn chống Overfitting (Implicit Regularization):** Việc ép thông tin qua cổ chai hạng $r$ buộc mạng chỉ giữ lại các thành phần dao động chính, loại bỏ việc ghi nhớ các điểm nhiễu ngẫu nhiên trong chuỗi dài.

* **🎬 Tương tác & Điểm nhấn màn hình (Visual Cues):**
  * Chỉ vào sơ đồ phân rã: từ $d_h \to$ qua nút thắt $r=64 \to$ bung ra $H \cdot p$.
  * Nhấn mạnh phép so sánh: **11.8M $\to$ 0.87M** (giảm 92.6%).

---

### 🟢 SLIDE 10 / 10 — ĐIỂM CẢI THIỆN CỤ THỂ TRONG TiDE & LỘ TRÌNH THỰC NGHIỆM

* **Ảnh slide:** `slide_10_diem_cai_thien_va_roadmap.png` ([Xem ảnh trong gallery](./slide_images_gallery.md#slide-10--10-điểm-cải-thiện-cụ-thể-trong-tide--validation-protocol))
* **Thời lượng dự kiến:** 65 – 75 giây.
* **Mục tiêu slide:** Chỉ rõ 3 điểm yếu cụ thể trong TiDE mà giải pháp này cải thiện; trình bày lộ trình kiểm chứng thực nghiệm 3 bước và kêu gọi nhóm phân công chạy trên Kaggle.

![Slide 10: Điểm cải thiện cụ thể trong TiDE và lộ trình kiểm chứng](./images/slide_10_diem_cai_thien_va_roadmap.png)

#### 🎙️ Lời thoại trình bày (Team Tech-talk):

Để khép lại bài thuyết trình, Slide 10 chỉ rõ **đề xuất này cải thiện trực tiếp ở 3 điểm nào trong TiDE** và lộ trình thực nghiệm cụ thể của nhóm mình:

**3 Điểm cải thiện cụ thể:**
1. **Khắc phục tầng chiếu cuối tại `models.py#L136`:** Thay vì mở rộng đột ngột $d_h \to H \cdot p$, ta tách thành chuỗi $d_h \to r \to H \cdot p$, giảm hơn 85%–92% tham số nhưng giữ nguyên luồng forward của `MLPResidual`.
2. **Giải quyết triệt để nút thắt khi $H=720$ và cho phép tăng $p$:** Nhờ có nút thắt hạng $r$, nhóm mình có thể tự tin tăng kích thước vector bối cảnh lên $p = 16$ hoặc $32$ trên các tập dữ liệu phức tạp mà không sợ mô hình bị phình to hay tràn VRAM.
3. **Lấp đầy khoảng trống Ablation của bài báo:** Nhóm tác giả Google chưa từng công bố bảng quét độ nhạy tách riêng cho $p$ và $n_d$ trên 7 datasets. Nhóm mình sẽ bổ sung trọn vẹn thí nghiệm quét nhạy cảm giữa $(p, n_d)$ và các mức hạng $r \in \{32, 64, 128\}$.

**Lộ trình thực nghiệm 3 bước rất rõ ràng:**
* **Bước 1:** Chạy đối chứng TiDE gốc vs TiDE Low-Rank ($r \in \{32, 64, 128\}$) trên 3 tập đại diện: `Electricity`, `ETTh2` và `ETTm1` ở các horizon dài $H = 336$ và $720$ với 5 seed độc lập.
* **Bước 2:** Quét độ nhạy siêu tham số $p \in \{2, 8, 16, 32\}$ và $n_d \in \{1, 2, 3\}$.
* **Bước 3:** Nghiệm thu tiêu chí: Giảm $\geq 85\%$ tham số tầng cuối, tốc độ suy luận One-Shot $O(1)$ không đổi, và MSE/MAE bằng hoặc tốt hơn TiDE gốc.

Mình đã chuẩn bị sẵn tài liệu hướng dẫn dùng trình duyệt Brave chạy song song 2 tài khoản Kaggle trong thư mục `kaggle/` để nhóm có 60 giờ GPU/tuần. Giờ anh em mình cùng bắt tay vào thảo luận và phân chia nhiệm vụ chạy thực nghiệm nhé!

* **🎬 Tương tác & Điểm nhấn màn hình (Visual Cues):**
  * Chỉ vào bảng 3 điểm cải thiện và 3 bước trong Validation Protocol.
  * Mở nụ cười, chuyển sang màn hình code / thảo luận để cả nhóm cùng bắt đầu.

---

## 💬 GỢI Ý GIẢI ĐÁP THẮC MẮC NỘI BỘ (INTERNAL TEAM Q&A & GOTCHAS)

Dưới đây là các câu hỏi mà thành viên trong nhóm rất dễ thắc mắc khi cùng đọc code Dense Decoder và chuẩn bị chạy thực nghiệm:

### ❓ Câu hỏi 1: *"Ủa tại sao trong code PyTorch dùng `g.view(-1, H, p)` mà trong code TensorFlow của Google lại là `[bsize, -1, pred_len]`?"*

**Giải thích cho bạn:**
Đó là do sự khác biệt về quy ước thứ tự trục giữa hai framework:
* Trong PyTorch, quy ước phổ biến cho dữ liệu chuỗi là `(Batch, Time, Feature)`, tức `(B, H, p)`. Nhóm mình để chiều thời gian $H$ ở giữa để các bước sau ghép với ngoại sinh tương lai `enc_fut` rất tự nhiên.
* Còn trong TensorFlow của bài báo gốc, tác giả lại gom chiều đặc trưng ra trước và để chiều thời gian ở cuối, tức `(B, p, H)`. Cả hai bản chất đều lưu đúng $H \cdot p$ phần tử liên tục trong bộ nhớ, chỉ khác nhau góc nhìn trục khi lập trình thôi!

### ❓ Câu hỏi 2: *"Phép Low-Rank Factorization $\mathbf{W} \approx \mathbf{A} \cdot \mathbf{B}$ có làm chậm tốc độ suy luận của mô hình không?"*

**Giải thích cho bạn:**
Hoàn toàn không làm chậm, thậm chí còn có thể tính toán nhanh hơn! Vì thay vì phải nhân một ma trận khổng lồ $(5760 \times 1024) \approx 5.9 \times 10^6$ phép nhân cộng, ta chỉ cần nhân hai ma trận nhỏ: $(64 \times 1024) + (5760 \times 64) \approx 4.3 \times 10^5$ phép tính — tức số phép tính nhân ma trận (FLOPs) giảm hơn **13 lần**! Trên GPU, cả hai phép nhân này đều được thực hiện song song cực nhanh.

### ❓ Câu hỏi 3: *"Nếu chọn hạng $r$ quá nhỏ (ví dụ $r=8$ hay $16$) thì mô hình có bị mất thông tin (Underfitting) không?"*

**Giải thích cho bạn:**
Chắc chắn có rủi ro đó. Nếu $r$ quá nhỏ thì không gian cổ chai không đủ dung lượng để biểu diễn sự biến thiên của 96 hay 720 bước tương lai. Đó là lý do trong lộ trình thực nghiệm Bước 1, nhóm mình sẽ quét các mức hạng $r \in \{32, 64, 128\}$: mức $r=64$ là điểm cân bằng hoàn hảo (Sweet Spot) vừa giảm 92.6% tham số vừa giữ trọn vẹn năng lực biểu diễn của không gian ẩn.
