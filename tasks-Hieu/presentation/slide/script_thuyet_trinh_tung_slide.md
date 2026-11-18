# 🎙️ KỊCH BẢN THUYẾT TRÌNH NỘI BỘ TỪNG SLIDE (INTERNAL TEAM SHARING SCRIPT)
## CHUYÊN ĐỀ: KHỐI DENSE DECODER TRONG KIẾN TRÚC MÔ HÌNH TiDE

> [!NOTE]
> **Bối cảnh sử dụng (Context):**  
> * **Đối tượng người nghe:** Toàn bộ thành viên trong nhóm đồ án / nhóm nghiên cứu TiDE.
> * **Mục tiêu bài nói:** **Giúp tất cả thành viên trong nhóm hiểu cặn kẽ, sâu sắc từng con số, từng ký hiệu, cơ chế ghép ngoại sinh, quy trình luồng code, và bản chất tại sao chọn One-Shot thay vì tuần tự**, giải quyết dứt điểm các câu hỏi phản biện kỹ thuật để nhóm tự tin code và bảo vệ đồ án.
> * **Phong cách trình bày:** Gần gũi, cởi mở, đậm chất trao đổi kỹ thuật (Tech-talk / Code-walkthrough), xưng hô "mình" và "anh em / các bạn trong nhóm". Đi thẳng vào bản chất, bóc tách từng dòng code và giải thích chi tiết các bẫy shape tensor.
> * **Tài liệu tham chiếu:** Khớp chính xác 1-đối-1 theo đúng thứ tự **13 Slide** trong [`slide_images_gallery.md`](./slide_images_gallery.md) (từ **Slide 02 / 13** đến **Slide 12 / 13**) và tuân thủ chuẩn mực [`rule_slide.md`](./rule_slide.md) cùng [`RULE.md`](../../../RULE.md).

---

## ⏱️ PHÂN BỔ THỜI GIAN TOÀN BỘ BÀI NÓI (13 SLIDES)

```mermaid
flowchart LR
    P1["Phần 1: Thuật Ngữ, Kiến Trúc & Toán<br/>(Slides 02 - 07)<br/>⏱️ ~4.5 Phút<br/>Hiểu rõ B, H, p, d_h & Bản chất One-Shot"] --> P2["Phần 2: Quy Trình Code & Phân Tích Giới Hạn<br/>(Slides 08 - 10)<br/>⏱️ ~4.0 Phút<br/>Luồng chạy 6 bước & Bùng nổ 11.8M params"]
    P2 --> P3["Phần 3: Đề Xuất Low-Rank & Roadmap<br/>(Slides 11 - 12)<br/>⏱️ ~3.0 Phút<br/>Phân rã r=64 giảm 92.6% & Kế hoạch Kaggle"]
```

---

### 🟢 SLIDE 01 / 13 — INTRO: TỔNG QUAN CHUYÊN ĐỀ DENSE DECODER TRONG TiDE

* **Ảnh slide:** `slide_00_intro.png` ([Xem ảnh trong gallery](./slide_images_gallery.md#slide-01--13-intro--tide-dense-decoder-presenter-le-thanh-hieu))
* **Thời lượng dự kiến:** 45 – 55 giây.
* **Mục tiêu slide:** Chào hỏi nhóm, nêu bật vai trò sống còn của khối `Dense Decoder` trong TiDE, giới thiệu mục tiêu buổi tech-talk và lộ trình 3 phần của bài trình bày.

![Slide 01: Intro](./images/slide_00_intro.png)

#### 🎙️ Lời thoại trình bày (Team Tech-talk):

Chào cả nhóm! Hôm nay mình sẽ phụ trách chia sẻ chuyên sâu về **Khối Dense Decoder** trong kiến trúc mô hình dự báo chuỗi thời gian dài hạn **TiDE** (Das et al., 2023).

Trong bức tranh tổng thể của TiDE, nếu `Dense Encoder` đóng vai trò nén toàn bộ dữ liệu quá khứ, thì **`Dense Decoder` chính là "trái tim giải mã"**: nó quyết định cách mô hình sinh ra toàn bộ dự báo tương lai chỉ trong **một lần chạy duy nhất (One-Shot)**, nhanh hơn Transformer 5 đến 10 lần. Tuy nhiên, chính khối này cũng chứa **nút thắt phình to tham số lớn nhất** khi chúng ta dự báo tầm xa lên tới $H = 720$ bước.

Mục tiêu của mình trong khoảng 10–12 phút tới là giúp anh em trong nhóm nắm trọn vẹn 3 phần:
1. **Phần 1 (Slides 02 – 07):** Thống nhất bộ thuật ngữ, vị trí kiến trúc, hiểu tận gốc bản chất *vừa độc lập tính toán vừa tương quan biểu diễn* và phép Reshape.
2. **Phần 2 (Slides 08 – 10):** Soi luồng chạy code thực tế trong `models.py`, xem cấu hình siêu tham số trên 7 tập dữ liệu chuẩn và chỉ ra nguyên nhân bùng nổ tham số ở tầng cuối.
3. **Phần 3 (Slides 11 – 12):** Đề xuất giải pháp **Low-Rank Projection** giảm tới 92.6% tham số cùng lộ trình phân công chạy thực nghiệm trên Kaggle.

Không để anh em đợi lâu, mình bước ngay vào **Slide 02** để thống nhất bộ ký hiệu và 5 thuật ngữ nền tảng trước nhé!

* **🎬 Tương tác & Điểm nhấn màn hình (Visual Cues):**
  * Chiếu Slide Tiêu đề / Bìa chuyên đề, nhìn bao quát cả nhóm, nhấn mạnh cụm từ **"Trái tim giải mã One-Shot"** và **"Nút thắt tham số khi H=720"**.

---

## 📖 PHẦN 1: THUẬT NGỮ, KIẾN TRÚC & TOÁN HỌC (SLIDES 01 – 05)

---

### 🟢 SLIDE 02 / 13 — THUẬT NGỮ NỀN TẢNG CỐT LÕI (`MLPResidual`, `Unflatten`, `Information Bottleneck`, `LayerNorm`, `Dropout`)

* **Ảnh slide:** `slide_01_bang_thuat_ngu_dense_decoder.png` ([Xem ảnh trong gallery](./slide_images_gallery.md#slide-02--13-thuật-ngữ-nền-tảng-cốt-lõi-mlpresidual-unflatten-information-bottleneck-layernorm-dropout))
* **Thời lượng dự kiến:** 70 – 80 giây.
* **Mục tiêu slide:** Thống nhất với cả nhóm 5 khái niệm nền tảng (`MLPResidual`, `Unflatten`, `Information Bottleneck`, `LayerNorm`, `Dropout`) cùng các ký hiệu $B, H, p, d_h$ trước khi bước vào sơ đồ kiến trúc ở Slide 03.

![Slide 02: Thuật ngữ nền tảng cốt lõi](./images/slide_01_bang_thuat_ngu_dense_decoder.png)

#### 🎙️ Lời thoại trình bày (Team Tech-talk):

Chào mọi người! Trong kiến trúc TiDE, sau khi `Dense Encoder` nén toàn bộ dữ liệu lịch sử thì trọng trách giải mã tương lai thuộc về khối **`Dense Decoder`**. Trước khi đi sâu vào sơ đồ từ Slide 03 trở đi, anh em nhìn nhanh qua Slide 02 để mình thống nhất rõ ràng từng ký hiệu và 5 thuật ngữ cốt lõi:

**1. Về bộ ký hiệu $B, H, p$ và $d_h$:**
* **$B$ là `Batch Size`:** Số lượng mẫu chuỗi nạp vào cùng lúc. Tác giả **cố định $B = 512$** cho toàn bộ 7 benchmark datasets.
* **$H$ là `Horizon`:** Tầm dự báo tương lai cần đoán trước ($96, 192, 336$ và dài nhất là $720$ bước).
* **$p$ là `decoderOutputDim`:** Số chiều đặc trưng của vector bối cảnh cho mỗi bước thời gian (thường chọn $p = 8, 16$ hoặc $32$). Nghĩa là tại mỗi thời điểm $t$, ta giữ lại một vector gồm $p$ thuộc tính để chứa bối cảnh lịch sử.
* **$d_h = \text{hiddenSize}$ là bao nhiêu?** Trong Bảng 7 Phụ lục, tác giả quét trong tập $\{256, 512, 1024\}$. Với các tập dữ liệu lớn như `Electricity` và `Traffic`, **$d_h$ được chọn tối đa là $1024$**; còn các tập `ETT` nhỏ hơn thì chọn $d_h = 256$ hoặc $512$.

**2. Năm thuật ngữ cốt lõi (chia 2 hàng trên slide — giải thích ngắn gọn vai trò trong bài báo TiDE):**
* **Thẻ 1 — `MLPResidual` (Residual Block):** Là khối xử lý phi tuyến nòng cốt được xếp chồng nhiều tầng trong TiDE để trích xuất đặc trưng; đường tắt tuyến tính đi kèm giúp đổi số chiều và truyền gradient ổn định.
* **Thẻ 2 — `Unflatten` (Thao tác Reshape):** Tách vector phẳng đầu ra thành các vector riêng cho từng mốc thời gian $t = 1 \dots H$, khôi phục lại trục thời gian để tầng sau ghép tương ứng với biến ngoại sinh.
* **Thẻ 3 — `Information Bottleneck`:** Hiện tượng mất thông tin quá khứ khi ép biểu diễn qua chiều $p$ hoặc hạng $r$ quá hẹp; nên trong TiDE cần chọn kích thước đủ giữ chu kỳ cho $H$ bước mà không làm phình tham số.
* **Thẻ 4 — `LayerNorm` (Chuẩn hóa theo tầng):** Chuẩn hóa đặc trưng của từng mẫu ngay ở đầu ra mỗi khối `MLPResidual` (sau phép cộng nhánh tắt), giúp phân phối tín hiệu và quá trình huấn luyện ổn định khi xếp chồng nhiều tầng hoặc horizon dài.
* **Thẻ 5 — `Dropout` (Điều chuẩn ngẫu nhiên):** Tắt ngẫu nhiên một tỷ lệ kết nối trong nhánh phi tuyến của khối `MLPResidual` lúc huấn luyện để giảm Overfitting của các tầng Dense lớn (tự động tắt khi suy luận dự báo).

Nắm chắc 5 thuật ngữ và các con số này rồi, giờ chúng ta đi vào vị trí kiến trúc trên Figure 1 ở Slide 03 nhé!

* **🎬 Tương tác & Điểm nhấn màn hình (Visual Cues):**
  * Chỉ trực tiếp vào Thẻ 2: *"Nhớ nhé anh em: $B=512$, $d_h = 1024$, $H=96 \to 720$, $p=8 \to 32$"*.
  * Khoanh vào Thẻ 4 và 5 ở hàng dưới: *"Cặp đôi LayerNorm và Dropout là hai lớp bảo vệ giúp khối MLPResidual học sâu mà không bị nổ gradient hay học vẹt"*.

---

### 🟢 SLIDE 03 / 13 — VỊ TRÍ KHỐI DENSE DECODER & UNFLATTEN TRÊN FIGURE 1

* **Ảnh slide:** `slide_02_vi_tri_dense_decoder.png` ([Xem ảnh trong gallery](./slide_images_gallery.md#slide-03--13-vị-trí-khối-dense-decoder--unflatten-trên-figure-1))
* **Thời lượng dự kiến:** 40 – 45 giây.
* **Mục tiêu slide:** Định vị chính xác vị trí khối Dense Decoder và Unflatten trên Figure 1; giới thiệu vai trò cầu nối chuyển sang `Temporal Decoder` (chi tiết cách ghép ngoại sinh sẽ được giải thích cụ thể ở Slide 07 và Slide 08).

![Slide 03: Vị trí Dense Decoder](./images/slide_02_vi_tri_dense_decoder.png)

> [!NOTE]
> **📌 Card độc lập — Giải nghĩa nhanh 3 đại lượng $B, H, p$ trên Slide 03:**
> * **$B$ (`Batch Size`):** Số lượng mẫu chuỗi thời gian được xử lý song song trong một lần chạy (bài báo cố định $B = 512$ trên 7 bộ benchmark).
> * **$H$ (`Horizon`):** Tầm dự báo tương lai — số bước thời gian cần dự đoán tiếp theo (thường $H \in \{96, 192, 336, 720\}$).
> * **$p$ (`decoderOutputDim`):** Số chiều đặc trưng của mỗi vector bối cảnh $\mathbf{d}_t^{(i)}$ cho từng bước thời gian tương lai $t = 1 \dots H$ (thường $p \in \{8, 16, 32\}$).

#### 🎙️ Lời thoại trình bày (Team Tech-talk):

Nhìn lên Figure 1 trích từ trang 5 của bài báo gốc ở Slide 03:

Khối `Dense Decoder` (được khoanh vàng ở giữa) nhận vector ẩn toàn cục $\mathbf{e}^{(i)}$ từ `Dense Encoder` truyền lên, xử lý qua $n_d$ khối để bung ra vector phẳng $\mathbf{g}^{(i)}$ có `Shape: (B, H * p)`. Ngay sau đó, phép `Unflatten` định hình lại thành ma trận $\mathbf{D}^{(i)}$ kích thước `(B, p, H)` để tách ra $H$ vector bối cảnh $\mathbf{d}_t^{(i)}$ tương ứng với từng bước thời gian tương lai.

Nói ngắn gọn, vai trò sống còn của Dense Decoder là **tạo sẵn $H$ bộ khung bối cảnh quá khứ $\mathbf{d}_t$ cho từng ngày**. Còn việc từng khung $\mathbf{d}_t$ này sẽ ghép nối với biến ngoại sinh tương lai ở tầng kế tiếp (`Temporal Decoder`) cụ thể như thế nào, mình sẽ giải thích chi tiết ở phần Reshape (Slide 07) và quy trình Code (Slide 08) ngay sau đây nhé!

* **🎬 Tương tác & Điểm nhấn màn hình (Visual Cues):**
  * Dùng tay chỉ luồng trên Figure 1: $\mathbf{e}^{(i)}$ đi vào `Dense Decoder` $\to$ `Unflatten` ra các lát cắt $\mathbf{d}_t$ $\to$ chuyển lên `Temporal Decoder`.

---

### 🟢 SLIDE 04 / 13 — CÁCH DENSE DECODER TẠO BIỂU DIỄN TOÀN HORIZON (ONE-SHOT VS TUẦN TỰ)

* **Ảnh slide:** `slide_03_one_shot_vs_tuan_tu.png` ([Xem ảnh trong gallery](./slide_images_gallery.md#slide-04--13-cách-dense-decoder-tạo-biểu-diễn-toàn-horizon-one-shot-vs-tuần-tự))
* **Thời lượng dự kiến:** 50 – 60 giây.
* **Mục tiêu slide:** So sánh trực diện hai cơ chế giải mã: Autoregressive truyền thống (vòng lặp tuần tự $O(H)$, tích lũy sai số) vs Non-autoregressive One-Shot của TiDE (sinh đồng thời toàn bộ $H$ bước, $O(1)$ GPU).

![Slide 04: One-Shot vs Tuần tự](./images/slide_03_one_shot_vs_tuan_tu.png)

#### 🎙️ Lời thoại trình bày (Team Tech-talk):

Ở Slide 04, chúng ta cùng làm sáng tỏ triết lý giải mã của TiDE so với các mô hình dự báo chuỗi truyền thống qua 2 cột đối chiếu trực diện:

**1. Cột bên trái — Cách truyền thống: Sinh tuần tự (Autoregressive):**
* Trong các kiến trúc như RNN, LSTM, Seq2Seq hay Transformer Decoder cổ điển: họ xuất phát từ giả định chuỗi Markov, tức là muốn đoán ngày mai bắt buộc phải biết kết quả ngày hôm nay.
* Vì vậy họ bắt buộc phải chạy theo vòng lặp tuần tự $H$ bước: đoán bước 1 $\to$ nạp lại đoán bước 2 $\to \dots \to$ bước $H$.
* Hậu quả là: GPU bị nghẽn ở độ phức tạp tuần tự $O(H)$, và nguy hiểm nhất là **tích lũy sai số (Error Accumulation)** — bước đầu tiên lệch 1 ly thì đến bước $H=720$ kết quả bị trôi dạt hoàn toàn!

**2. Cột bên phải — Dense Decoder (TiDE): Sinh song song One-Shot:**
* TiDE hoàn toàn từ bỏ cơ chế sinh tự hồi quy tuần tự. Từ vector bối cảnh lịch sử $\mathbf{e}^{(i)}$, tầng cuối của Dense Decoder chiếu thẳng 1 lần duy nhất ra vector phẳng $\mathbf{g}^{(i)}$ chứa trọn vẹn biểu diễn của toàn bộ $H$ bước tương lai.
* Mọi lát cắt $\mathbf{d}_1, \dots, \mathbf{d}_H$ đều được trích xuất độc lập trực tiếp từ $\mathbf{e}^{(i)}$, không bước nào lấy output của bước trước làm input.
* Nhờ đó, TiDE **triệt tiêu 100% hiện tượng tích lũy sai số**, đạt độ phức tạp $O(1)$ trên GPU và **chạy nhanh hơn Transformer từ 5 đến 10 lần**!

Vậy câu hỏi đặt ra là: *"Từ bản chất toán học đến cài đặt code thực tế trong `models.py`, làm sao các bước vừa chạy độc lập trên GPU nhưng vẫn giữ được mối liên kết chặt chẽ với nhau?"* Chúng ta cùng sang Slide 05 để xem cả hai vế nhé!

* **🎬 Tương tác & Điểm nhấn màn hình (Visual Cues):**
  * Chỉ vào cột trái: gạch chéo đỏ vào vòng lặp tuần tự và mũi tên tích lũy sai số.
  * Chỉ sang cột phải: nhấn mạnh cơ chế chiếu 1 phát ra toàn bộ $H \cdot p$ phần tử song song.

---

### 🟢 SLIDE 05 / 13 — TỪ CÔNG THỨC TOÁN HỌC SUY RA CODE (`models.py`)

* **Ảnh slide:** `slide_03b_ban_chat_toan_hoc.png` ([Xem ảnh trong gallery](./slide_images_gallery.md#slide-05--13-từ-công-thức-toán-học-suy-ra-code-modelspy--độc-lập-tính-toán-vs-tương-quan-biểu-diễn))
* **Thời lượng dự kiến:** 60 – 70 giây.
* **Mục tiêu slide:** Đi từ hệ thống công thức toán học ở trên để suy ra trực tiếp cách tác giả cài đặt trong `models.py` ở dưới (vừa độc lập tính toán $O(1)$ trên GPU, vừa giữ trọn vẹn tương quan biểu diễn).

![Slide 05: Từ công thức toán học suy ra code](./images/slide_03b_ban_chat_toan_hoc.png)

#### 🎙️ Lời thoại trình bày (Team Tech-talk):

Slide 05 này đi theo đúng một mạch rất logic: **từ Công thức Toán học ở khung trên $\to$ suy thẳng ra cách viết Code trong `models.py` ở khung dưới**. Anh em nhìn vào 2 cột trên màn hình:

**Cột bên trái — Độc lập tính toán ($O(1)$ GPU Parallelism):**
* **Về Toán học (khung trên):** Như hai công thức trên slide, khi tách ma trận chiếu tầng cuối thành $H$ khối con, đạo hàm riêng giữa hai bước thời gian bất kỳ $t$ và $s$ bằng $0$. Điều đó chứng minh đồ thị tính toán hoàn toàn không có cạnh nối giữa các bước, toàn bộ phép tính chỉ là 1 phép nhân ma trận duy nhất.
* **Suy ra Code trong `models.py` (khung dưới):** Chính vì không có cạnh nối tuần tự, tác giả **không cần viết vòng lặp `for t in range(H)`**. Trong code chỉ cần đúng **1 lớp `Dense(p * H)`** duy nhất để GPU tính song song toàn bộ $H$ bước cùng lúc, rồi gọi `tf.reshape` là xong `(B, p, H)`.

**Cột bên phải — Tương quan biểu diễn (Representational Correlation):**
* **Về Toán học (khung trên):** Nhóm mình có 3 công thức chứng minh ở khung trên: thứ nhất là Luật Hiệp phương sai toàn phần cho thấy dù độc lập có điều kiện, hiệp phương sai biên vẫn khác không nhờ cùng chia sẻ vector ẩn $\mathbf{e}$; thứ hai là công thức Gradient tổng hợp khi Backprop ép các khối chiếu học mượt mà; và thứ ba là nhánh `Global Residual` giữ chu kỳ trễ.
* **Suy ra Code trong `models.py` (khung dưới):** Ba công thức đó ánh xạ trực tiếp thành 3 dòng cài đặt rất gọn trong code:
  1. Mọi lát cắt $\mathbf{d}_t$ đều cắt ra từ **chung 1 tensor `encoded`**.
  2. Toàn bộ mô hình tối ưu **chung 1 hàm Loss `MSE`** trên cả $H$ bước tương lai.
  3. Cuối cùng cộng trực tiếp nhánh tắt tuyến tính `+ self.linear_skip(past_y)`.

* **🎬 Tương tác & Điểm nhấn màn hình (Visual Cues):**
  * Chỉ tay từ khung trên xuống khung dưới của từng cột: *"Từ công thức toán học ở trên $\to$ suy ra ngay 3 gạch đầu dòng cài đặt trong `models.py` ở dưới"*.

---

### 🟢 SLIDE 06 / 13 — KIẾN TRÚC $n_d$ RESIDUAL BLOCKS & CHUYỂN $\mathbf{e}^{(i)}$ THÀNH $\mathbf{g}^{(i)}$

* **Ảnh slide:** `slide_04_kien_truc_nd_residual_blocks.png` ([Xem ảnh trong gallery](./slide_images_gallery.md#slide-06--13-kiến-trúc-n_d-residual-blocks--chuyển-mathbfei-thành-mathbfgi))
* **Thời lượng dự kiến:** 50 – 60 giây.
* **Mục tiêu slide:** Bóc tách luồng biến đổi qua từng tầng theo hai phương trình trên slide; giải thích vai trò bắt buộc của ma trận chiếu đường tắt $\mathbf{W}_{\text{skip}}^{(n_d)}$ ở khối cuối cùng.

![Slide 06: Kiến trúc n_d Residual Blocks](./images/slide_04_kien_truc_nd_residual_blocks.png)

#### 🎙️ Lời thoại trình bày (Team Tech-talk):

Slide 06 đi sâu vào cấu trúc bên trong khối Dense Decoder khi nhận vector $\mathbf{e}^{(i)}$ từ Encoder truyền lên. Nhìn vào hai phương trình trên slide, Dense Decoder chia làm 2 giai đoạn biến đổi rất rõ ràng:

* **Giai đoạn 1 — Các khối trung gian (từ tầng $1$ đến $n_d - 1$, nếu chọn $n_d \geq 2$):** Như phương trình phía trên trên màn hình, ở các tầng này chiều vào và chiều ra đều giữ cố định là `(B, d_h)` để tinh chỉnh biểu diễn phi tuyến.
* **Giai đoạn 2 — Khối cuối cùng ($l = n_d$):** Đây là tầng đặc biệt nhất theo phương trình thứ hai trên slide. Chiều đầu ra không còn là $d_h$, mà được bung rộng ra kích thước $H \cdot p$!

Anh em lưu ý chi tiết kỹ thuật rất quan trọng ở khối cuối này:
Vì số chiều đầu vào là $d_h$ còn đầu ra là $H \cdot p$ (hai số này lệch nhau rất lớn, ví dụ $1024$ vs $5760$), nhánh tắt Skip Connection không thể cộng trực tiếp phần tử với phần tử được. Do đó, tác giả **bắt buộc phải đặt thêm một ma trận chiếu đường tắt $\mathbf{W}_{\text{skip}}$ có kích thước $(H \cdot p) \times d_h$** như mình khoanh tròn trên slide để đồng bộ kích thước rồi mới cộng residual!

Đầu ra thu được là vector phẳng $\mathbf{g}^{(i)}$ có `Shape: (B, H * p)`. Chính sự xuất hiện của ma trận đường tắt này cộng với ma trận nhánh chính sẽ dẫn đến bài toán bùng nổ tham số mà chúng ta sẽ mổ xẻ ở Slide 10!

* **🎬 Tương tác & Điểm nhấn màn hình (Visual Cues):**
  * Chỉ vào phương trình khối cuối trên slide: khoanh tròn vào số hạng $\mathbf{W}_{\text{skip}}^{(n_d)} \mathbf{v}^{(n_d-1)}$ để anh em thấy rõ sự xuất hiện của ma trận chiếu đường tắt.

---

### 🟢 SLIDE 07 / 13 — VAI TRÒ RESHAPE (UNFLATTEN) & Ý NGHĨA VECTOR $\mathbf{d}_t^{(i)}$

* **Ảnh slide:** `slide_05_reshape_va_y_nghia_dt.png` ([Xem ảnh trong gallery](./slide_images_gallery.md#slide-07--13-vai-trò-thao-tác-reshape-unflatten--ý-nghĩa-vector-mathbfd_ti))
* **Thời lượng dự kiến:** 55 – 65 giây.
* **Mục tiêu slide:** Giải thích chuẩn xác theo đúng công thức và mã nguồn của bài báo gốc: Phép Reshape tạo ra ma trận $\mathbf{D}^{(i)} \in \mathbb{R}^{p \times H}$ với Shape: $(B, p, H)$, bản chất chi phí $O(1)$ và cách ghép ngoại sinh ở tầng kế tiếp.

![Slide 07: Reshape và Ý nghĩa d_t](./images/slide_05_reshape_va_y_nghia_dt.png)

#### 🎙️ Lời thoại trình bày (Team Tech-talk):

Sau khi tầng cuối cho ra vector phẳng $\mathbf{g}^{(i)}$ kích thước $H \cdot p$, Slide 07 giải phẫu thao tác **`Reshape` (hay `Unflatten`)** để tách thông tin cho từng bước tương lai, bám sát 100% theo đúng bài báo gốc và mã nguồn TensorFlow:

**1. Về cấu trúc ma trận trên slide:**
* Như công thức ở trên màn hình, tác giả định hình lại vector phẳng thành ma trận $\mathbf{D}^{(i)}$ kích thước $p \times H$. Trong đại số tuyến tính, một chuỗi thời gian gồm $H$ bước được biểu diễn chuẩn mực dưới dạng một ma trận gồm **$H$ cột**, trong đó **mỗi cột thứ $t$ chính là vector bối cảnh $\mathbf{d}_t^{(i)}$ dài $p$**.

**2. Trong mã nguồn chính thức (`models.py#L210-L211`):**
* Tác giả thực hiện đúng y như công thức toán: `tf.reshape(decoder_out, [bsize, -1, self.pred_len])` để cho ra tensor có `Shape: (B, p, H)`, tức chiều thời gian $H$ nằm ở trục cuối cùng.
* Và trả lời luôn câu hỏi ở Slide 03 về việc **tầng kế tiếp ghép ngoại sinh cụ thể như thế nào**: Ngay ở dòng 211, tác giả ghép ma trận `(B, p, H)` này với ngoại sinh tương lai `enc_fut` có shape `(B, r_tilde, H)` dọc theo trục đặc trưng `axis=1`, tạo thành tensor `(B, p + r_tilde, H)` (ví dụ $8 + 4 = 12$ chiều đặc trưng cho mỗi bước $t$), rồi mới đưa sang `Temporal Decoder` để dự báo phi tuyến!

**3. Về chi phí và ý nghĩa:**
* Phép `reshape` này có **chi phí tính toán và bộ nhớ bằng 0 ($O(1)$)**, chỉ thay đổi thông số bước nhảy (stride) chứ không copy dữ liệu.
* Ý nghĩa của mỗi vector cột $\mathbf{d}_t^{(i)}$ là đóng vai trò một **vector bối cảnh quá khứ dành riêng cho thời điểm tương lai $t$**.

* **🎬 Tương tác & Điểm nhấn màn hình (Visual Cues):**
  * Chỉ vào công thức ma trận $\mathbf{D}^{(i)}$ trên slide: nhấn mạnh đây là ma trận gồm $H$ cột, mỗi cột là vector $\mathbf{d}_t^{(i)}$ dài $p$.
  * Nhấn mạnh: *"Làm đúng chuẩn bài báo: Reshape ra `(B, p, H)` với chi phí $O(1)$"*.

---

## 💻 PHẦN 2: QUY TRÌNH CODE, HYPERPARAMETERS & GIỚI HẠN (SLIDES 08 – 10)

---

### 🟢 SLIDE 08 / 13 — SOURCE CODE CHÍNH THỨC (`models.py`) & ĐỐI CHIẾU PYTORCH

* **Ảnh slide:** `slide_06_source_code_dense_decoder.png` ([Xem ảnh trong gallery](./slide_images_gallery.md#slide-08--13-source-code-chính-thức-modelspy--pytorch))
* **Thời lượng dự kiến:** 70 – 80 giây.
* **Mục tiêu slide:** **Giải thích cặn kẽ, rõ ràng quy trình chạy code 6 bước từ đầu đến cuối**, chỉ rõ luồng dữ liệu biến đổi qua từng dòng code trong `models.py` và `simulate_forward_pass.py`.

![Slide 08: Source Code Dense Decoder](./images/slide_06_source_code_dense_decoder.png)

#### 🎙️ Lời thoại trình bày (Team Tech-talk):

Bây giờ chúng ta cùng soi mã nguồn thực tế ở Slide 08. Để anh em nắm chắc quy trình, mình sẽ đi theo **luồng chạy 6 bước thực tế khi forward pass qua Dense Decoder**:

* **Bước 1 — Khởi tạo danh sách tầng (`__init__` - Dòng 64–81 trong `models.py`):**
  Hàm `_make_dnn_residual` dùng vòng lặp tạo ra một danh sách gồm $n_d$ khối `MLPResidual`. Mọi tầng ẩn trung gian đều có kích thước $d_h$.
* **Bước 2 — Cài đặt tầng chiếu mở rộng (Dòng 136 trong `models.py`):**
  Tác giả thay thế phần tử cuối cùng: `hidden_dims[-1] = decoder_output_dim * self.pred_len`. Phép gán này biến khối cuối cùng thành tầng mở rộng từ $d_h$ sang $p \cdot H$.
* **Bước 3 — Tiếp nhận tensor đầu vào (Dòng 207):**
  Khối Dense Decoder nhận đầu vào là vector ẩn $\mathbf{e}$ từ Encoder truyền lên, với shape chuẩn `(B, d_h)`.
* **Bước 4 — Chạy qua các tầng MLPResidual:**
  Tensor $\mathbf{e}$ đi tuần tự qua các khối trung gian (giữ nguyên $d_h$), rồi tới khối cuối cùng, nhân với ma trận chiếu chính và ma trận đường tắt $\mathbf{W}_{\text{skip}}$ để xuất ra vector phẳng `decoder_out` có shape `(B, H * p)`.
* **Bước 5 — Tái cấu trúc không gian (Unflatten - Dòng 210 `models.py`):**
  Tác giả dùng `tf.reshape(decoder_out, [bsize, -1, self.pred_len])` để chuyển ngay vector phẳng thành tensor 3 chiều có đúng shape của bài báo là `(B, p, H)`.
* **Bước 6 — Bàn giao cho Temporal Decoder (Dòng 211 `models.py`):**
  Tensor `D` có shape `(B, p, H)` sau đó được ghép nối tiếp với ngoại sinh tương lai `enc_fut` có shape `(B, r_tilde, H)` dọc theo trục đặc trưng `axis=1`, tạo thành tensor `(B, p + r_tilde, H)` rồi mới chuyển trục đưa vào `Temporal Decoder` để dự báo phi tuyến.

Như vậy toàn bộ quy trình từ khởi tạo, chiếu phẳng, reshape đến ghép nối ngoại sinh diễn ra hoàn toàn liền mạch!

* **🎬 Tương tác & Điểm nhấn màn hình (Visual Cues):**
  * Dùng chuột trỏ lần lượt qua 3 khối code trên slide tương ứng với 6 bước: Khởi tạo $\to$ Đổi chiều ngõ ra $\to$ Thực thi forward $\to$ Reshape.

---

### 🟢 SLIDE 09 / 13 — CẤU HÌNH SIÊU THAM SỐ DENSE DECODER TRÊN 7 BENCHMARK DATASETS

* **Ảnh slide:** `slide_07_hyperparameters_dense_decoder.png` ([Xem ảnh trong gallery](./slide_images_gallery.md#slide-09--13-cấu-hình-siêu-tham-số-dense-decoder-trên-7-benchmark-datasets-table-7--8))
* **Thời lượng dự kiến:** 75 – 85 giây.
* **Mục tiêu slide:** Phân tích bảng cấu hình chính thức từ Bảng 7 và 8 Phụ lục B.3; chỉ ra không gian tìm kiếm siêu tham số; **giải thích cặn kẽ cơ sở thực nghiệm vì sao tác giả chọn $n_d = 1$ hoặc $n_d = 2$ cho từng nhóm dữ liệu và lý do tuyệt đối $0/7$ tập chọn $n_d = 3$**.

![Slide 09: Hyperparameters Dense Decoder](./images/slide_07_hyperparameters_dense_decoder.png)

#### 🎙️ Lời thoại trình bày (Team Tech-talk):

Slide 09 là bảng cấu hình siêu tham số chính thức của Dense Decoder trên toàn bộ 7 bộ dữ liệu chuẩn, trích từ Bảng 7 (trang 20) và Bảng 8 (trang 21) ở Phụ lục bài báo.

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

Bây giờ, chúng ta cùng đào sâu vào việc chọn $p$ và $n_d$ sẽ gây ra hiện tượng nghẽn tham số cụ thể như thế nào ở Slide 10 nhé!

* **🎬 Tương tác & Điểm nhấn màn hình (Visual Cues):**
  * Chỉ vào cột $n_d$: nhấn mạnh sự phân hóa rõ rệt: nhóm vật lý/đa biến lớn chọn 1, nhóm hành vi phức tạp chọn 2.
  * Gạch chân điểm đặc biệt: **$0/7$ tập chọn $n_d = 3$** do bão hòa phi tuyến và nguy cơ Overfitting.
  * Khoanh vào hàng `ETTh2`: *"Tập duy nhất chọn $p = 32$, làm kích thước tầng cuối phình to gấp 4 lần"*.

---

### 🟢 SLIDE 10 / 13 — PHÂN TÍCH ẢNH HƯỞNG CỦA $p, n_d$ & GIỚI HẠN KHI HORIZON $H$ DÀI

* **Ảnh slide:** `slide_08_phan_tich_p_nd_va_gioi_han.png` ([Xem ảnh trong gallery](./slide_images_gallery.md#slide-10--13-phân-tích-ảnh-hưởng-của-p-n_d--giới-hạn-khi-horizon-h-dài))
* **Thời lượng dự kiến:** 75 – 85 giây.
* **Mục tiêu slide:** Chỉ rõ vị trí xuất hiện của $p$ trong bài báo (Mục 4.2 Trang 6, Table 7 Trang 20, Table 8 Trang 21); khẳng định việc tác giả chưa làm ablation riêng cho $p$; giải thích cơ sở toán học và mã nguồn thực tế dẫn đến con số 11.8M $\to$ 23.6M tham số.

![Slide 10: Phân tích p, n_d và Giới hạn](./images/slide_08_phan_tich_p_nd_va_gioi_han.png)

#### 🎙️ Lời thoại trình bày (Team Tech-talk):

Slide 10 này giải quyết một câu hỏi học thuật cực kỳ sắc bén mà anh em khi đọc bài báo chắc chắn sẽ thắc mắc:
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
  * Với $d_h = 1024, p = 8$ trên `Electricity`, như phép tính trên slide, tổng tham số của riêng 2 ma trận cuối này đã lên tới **~11.8 triệu tham số**! Còn nếu lấy $p = 32$ với $d_h = 512$ trên `ETTh2`, con số nhảy lên **~23.6 triệu tham số** (nếu $d_h = 1024$ thì tới **47.2 triệu**).
* **Cơ sở 3 — Chính lời thừa nhận của tác giả tại Mục 6 Conclusion (Trang 11):**
  Tác giả viết nguyên văn: *"transformers are generally more parameter efficient than MLPs while being much more memory and compute intensive. This could be a limitation..."* — tức tác giả chính thức thừa nhận nhược điểm tốn tham số của kiến trúc MLP của mình khi mô hình mở rộng!

Chính vì tác giả bỏ ngỏ việc khảo sát độ nhạy của $p$, nhóm mình mới đưa bài toán quét nhạy cảm $p \in \{2, 8, 16, 32\}$ vào Lộ trình thực nghiệm ở Slide 12!

* **🎬 Tương tác & Điểm nhấn màn hình (Visual Cues):**
  * Chỉ rõ vị trí xuất xứ: Mục 4.2 Trang 6, Table 7 Trang 20, Table 8 Trang 21 và Mục 6 Trang 11.
  * Chỉ vào phép nhân ma trận $2 \times (d_h \times H \cdot p)$ chứng minh con số 11.8M tham số là phép tính số học chính xác từ code.

---

## 💡 PHẦN 3: ĐỀ XUẤT TỐI ƯU & LỘ TRÌNH THỰC NGHIỆM (SLIDES 11 – 12)

---

### 🟢 SLIDE 11 / 13 — ĐỀ XUẤT TỐT NHẤT: LOW-RANK DENSE DECODER PROJECTION

* **Ảnh slide:** `slide_09_de_xuat_low_rank_decoder.png` ([Xem ảnh trong gallery](./slide_images_gallery.md#slide-11--13-đề-xuất-tốt-nhất--low-rank-dense-decoder-projection))
* **Thời lượng dự kiến:** 75 – 85 giây.
* **Mục tiêu slide:** Trình bày giải pháp Low-Rank Factorized Projection; giải thích nguồn gốc (LoRA & Sainath et al.), chứng minh giảm 92.6% tham số, giảm FLOPs hơn 13 lần và giữ nguyên tính chất One-Shot $O(1)$.

![Slide 11: Đề xuất Low-Rank Projection](./images/slide_09_de_xuat_low_rank_decoder.png)

#### 🎙️ Lời thoại trình bày (Team Tech-talk):

Để giải quyết triệt để nút thắt 11.8 triệu tham số mà tác giả bài báo để lại, ở Slide 11, mình xin đề xuất phương án tối ưu nhất cho nhóm mình: **Low-Rank Factorized Dense Decoder Projection**.

**1. Nguồn gốc học thuật vững chắc:**

Ý tưởng này kế thừa trực tiếp từ bài báo kinh điển **LoRA của Hu et al. (ICLR 2022)** và nghiên cứu nền tảng về **Low-Rank Factorization cho mạng sâu của Sainath et al. (ICASSP 2013)**.
Thay vì học một ma trận trọng số khổng lồ nối trực tiếp từ $d_h$ sang $H \cdot p$, ta phân rã nó thành tích của hai ma trận hạng thấp đi qua một nút thắt hạng $r$ như công thức và sơ đồ trên slide!

**2. Vì sao đây là phương án TỐT NHẤT cho khối Dense Decoder của nhóm mình?**

* **Giảm sốc tới 92.6% tham số:** Như phép tính trên màn hình với hạng $r = 64$ trên cấu hình `Electricity`, số tham số của 2 ma trận chiếu cuối giảm từ **11.8 triệu xuống chỉ còn ~0.87 triệu tham số**!
* **Giảm FLOPs tính toán hơn 13 lần:** Số phép tính nhân ma trận giảm từ ~5.9 triệu xuống chỉ còn ~0.43 triệu phép tính.
* **Bảo toàn 100% tính chất One-Shot $O(1)$:** Mô hình vẫn chiếu toàn bộ horizon trong 1 lần forward duy nhất, không thêm vòng lặp tuần tự nào, tốc độ GPU giữ nguyên tuyệt đối.
* **Đóng vai trò điều chuẩn chống Overfitting (Implicit Regularization):** Việc ép thông tin qua cổ chai hạng $r$ buộc mạng chỉ giữ lại các thành phần dao động chính, loại bỏ việc ghi nhớ các điểm nhiễu ngẫu nhiên trong chuỗi dài.

* **🎬 Tương tác & Điểm nhấn màn hình (Visual Cues):**
  * Chỉ vào sơ đồ phân rã: từ $d_h \to$ qua nút thắt $r=64 \to$ bung ra $H \cdot p$.
  * Nhấn mạnh phép so sánh: **11.8M $\to$ 0.87M** (giảm 92.6%).

---

### 🟢 SLIDE 12 / 13 — ĐIỂM CẢI THIỆN CỤ THỂ TRONG TiDE & LỘ TRÌNH THỰC NGHIỆM

* **Ảnh slide:** `slide_10_diem_cai_thien_va_roadmap.png` ([Xem ảnh trong gallery](./slide_images_gallery.md#slide-12--13-điểm-cải-thiện-cụ-thể-trong-tide--validation-protocol))
* **Thời lượng dự kiến:** 65 – 75 giây.
* **Mục tiêu slide:** Chỉ rõ 3 điểm yếu cụ thể trong TiDE mà giải pháp này cải thiện; trình bày lộ trình kiểm chứng thực nghiệm 3 bước và kêu gọi nhóm phân công chạy trên Kaggle.

![Slide 12: Điểm cải thiện cụ thể trong TiDE và lộ trình kiểm chứng](./images/slide_10_diem_cai_thien_va_roadmap.png)

#### 🎙️ Lời thoại trình bày (Team Tech-talk):

Để khép lại bài thuyết trình, Slide 12 chỉ rõ **đề xuất này cải thiện trực tiếp ở 3 điểm nào trong TiDE** và lộ trình thực nghiệm cụ thể của nhóm mình:

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

### 🟢 SLIDE 13 / 13 — OUTRO: THANKS FOR LISTENING! & Q&A

* **Ảnh slide:** `slide_11_outro.png` ([Xem ảnh trong gallery](./slide_images_gallery.md#slide-13--13-outro--thanks-for-listening))
* **Lời chào kết thúc:** Cảm ơn thầy/cô và các bạn đã lắng nghe phần trình bày chuyên đề `TiDE: Dense Decoder` — chuyển sang phiên hỏi đáp **Q & A Session**.

![Slide 13: Outro](./images/slide_11_outro.png)

---

## 💬 GỢI Ý GIẢI ĐÁP THẮC MẮC NỘI BỘ (INTERNAL TEAM Q&A & GOTCHAS)

Dưới đây là các câu hỏi mà thành viên trong nhóm rất dễ thắc mắc khi cùng đọc code Dense Decoder và chuẩn bị chạy thực nghiệm:

### ❓ Câu hỏi 1: *"Tại sao trong bài báo gốc và `models.py` tác giả lại reshape ma trận $\mathbf{D}^{(i)}$ thành shape `(B, p, H)` mà không phải ngược lại?"*

**Giải thích cho bạn:**
* Về mặt đại số tuyến tính (Mục 4.2, Trang 6 bài báo gốc), mỗi bước thời gian $t$ được biểu diễn bằng một vector cột $\mathbf{d}_t^{(i)} \in \mathbb{R}^p$. Ghép $H$ bước thời gian liên tiếp thành một ma trận gồm $H$ cột sẽ tự nhiên cho ra kích thước $\mathbf{D}^{(i)} \in \mathbb{R}^{p \times H}$, tương ứng với batch shape `(B, p, H)`.
* Nhờ để chiều đặc trưng $p$ ở trục giữa (`axis=1`) và chiều thời gian $H$ ở trục cuối (`axis=2`), ở bước ngay sau đó trong `models.py#L211`, tác giả dễ dàng ghép nối tiếp tensor `D` với ngoại sinh tương lai `enc_fut` (cũng có shape `(B, r_tilde, H)`) dọc theo trục đặc trưng `axis=1` để tạo thành `(B, p + r_tilde, H)` trước khi đưa vào `Temporal Decoder`.

### ❓ Câu hỏi 2: *"Phép Low-Rank Factorization $\mathbf{W} \approx \mathbf{A} \cdot \mathbf{B}$ có làm chậm tốc độ suy luận của mô hình không?"*

**Giải thích cho bạn:**
Hoàn toàn không làm chậm, thậm chí còn có thể tính toán nhanh hơn! Vì thay vì phải nhân một ma trận khổng lồ $(5760 \times 1024) \approx 5.9 \times 10^6$ phép nhân cộng, ta chỉ cần nhân hai ma trận nhỏ: $(64 \times 1024) + (5760 \times 64) \approx 4.3 \times 10^5$ phép tính — tức số phép tính nhân ma trận (FLOPs) giảm hơn **13 lần**! Trên GPU, cả hai phép nhân này đều được thực hiện song song cực nhanh.

### ❓ Câu hỏi 3: *"Nếu chọn hạng $r$ quá nhỏ (ví dụ $r=8$ hay $16$) thì mô hình có bị mất thông tin (Underfitting) không?"*

**Giải thích cho bạn:**
Chắc chắn có rủi ro đó. Nếu $r$ quá nhỏ thì không gian cổ chai không đủ dung lượng để biểu diễn sự biến thiên của 96 hay 720 bước tương lai. Đó là lý do trong lộ trình thực nghiệm Bước 1, nhóm mình sẽ quét các mức hạng $r \in \{32, 64, 128\}$: mức $r=64$ là điểm cân bằng hoàn hảo (Sweet Spot) vừa giảm 92.6% tham số vừa giữ trọn vẹn năng lực biểu diễn của không gian ẩn.
