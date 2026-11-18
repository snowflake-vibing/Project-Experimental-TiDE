# KỊCH BẢN THUYẾT TRÌNH CHI TIẾT TỪNG SLIDE (SEMINAR 10 SLIDE — KHỐI DENSE DECODER)

> **Thời lượng dự kiến:** ~8–9 phút cho toàn bộ 10 Slide (tốc độ vừa phải, rõ mạch logic từ nền tảng $\rightarrow$ code & siêu tham số $\rightarrow$ phân tích hạn chế $\rightarrow$ đề xuất cải thiện có cơ sở khoa học).

---

## PHẦN 1: THUẬT NGỮ, KIẾN TRÚC & TOÁN HỌC (SLIDE 01 – 05)

### Slide 01 / 10 — Thuật ngữ cốt lõi cho Dense Decoder (~40s)

> “Chào thầy/cô và các bạn. Hôm nay phần trình bày của em tập trung đi sâu vào một khối đóng vai trò giải mã trung tâm của mô hình TiDE, đó là **Dense Decoder** — cách TiDE chuyển biểu diễn ẩn thành các vector bối cảnh cho từng bước dự báo trong tương lai.
> Trước khi đi vào chi tiết kiến trúc, xin điểm qua nhanh 5 thuật ngữ nền tảng:
> Thứ nhất, **MLPResidual** là đơn vị cơ sở cấu thành nên Dense Decoder, gồm tầng ẩn ReLU nối Linear, Dropout, cộng với nhánh tắt tuyến tính `lin_res` và LayerNorm.
> Thứ hai, **vector ẩn toàn cục** $\mathbf{e}^{(i)}$ có shape tổng quát là $(B, d_h)$ — đây là đầu ra duy nhất từ Dense Encoder, nén toàn bộ chuỗi quá khứ, biến ngoại sinh và thuộc tính tĩnh thành một vector phẳng.
> Thứ ba, **Dense Decoder** là phần trọng tâm hôm nay: nó xếp chồng $n_d$ khối MLPResidual để mở rộng không gian từ $d_h$ thành vector phẳng $\mathbf{g}^{(i)}$ có kích thước $H \times p$ chỉ trong một lần forward.
> Thứ tư, **thao tác Reshape / Unflatten** giúp khôi phục trục thời gian $H$ từ vector phẳng $\mathbf{g}^{(i)}$ thành ma trận $\mathbf{D}^{(i)}$ có shape $(B, H, p)$.
> Và cuối cùng, mỗi lát cắt thứ $t$ của ma trận này là **vector bối cảnh** $\mathbf{d}_t^{(i)}$ có shape $(B, p)$, chứa thông tin quá khứ dành riêng cho bước tương lai $t$ để sẵn sàng đi tiếp sang tầng ghép ngoại sinh.”

### Slide 02 / 10 — Vị trí khối Dense Decoder & Unflatten trên Figure 1 (~40s)

> “Nhìn lên Figure 1 trang 5 của bài báo gốc để định vị đúng phạm vi của chúng ta:
> Phía dưới là Dense Encoder nén mọi tín hiệu đầu vào thành một vector duy nhất $\mathbf{e}^{(i)}$ — em đóng khung xanh lá ở đây, shape $(B, d_h)$.
> Ngay phía trên là **khối Dense Decoder** đóng khung cam gồm $n_d$ tầng Residual Block. Khối này giải nén $\mathbf{e}^{(i)}$ thành vector phẳng $\mathbf{g}^{(i)}$ có shape $(B, H \cdot p)$.
> Ngay sau đó, khối **Unflatten** reshape $\mathbf{g}^{(i)}$ để tách thành $H$ vector $\mathbf{d}_t^{(i)}$ xếp theo từng bước thời gian — nằm ở phần trên của khối Stack đóng khung xanh dương — chuẩn bị ghép với biến ngoại sinh $\tilde{\mathbf{x}}_{L+t}^{(i)}$ ở tầng kế tiếp.”

### Slide 03 / 10 — Cách Dense Decoder tạo biểu diễn toàn horizon (One-Shot vs Tuần tự) (~45s)

> “Điểm ăn tiền của Dense Decoder nằm ở cách nó tạo biểu diễn cho toàn bộ horizon $H$ chỉ trong một lần chạy, trái ngược hoàn toàn với cơ chế Autoregressive truyền thống.
> Ở cột trái màu đỏ — cách sinh tuần tự — mô hình phải đoán bước 1, rồi nạp lại chính dự đoán bước 1 đó để đoán bước 2, lặp đủ $H$ vòng. Cách này vướng hai nhược điểm chí mạng: sai số nhỏ ở đầu bị khuếch đại qua từng vòng lặp (tích lũy sai số), và GPU bị ép đợi tuần tự với độ trễ $O(H)$.
> Ngược lại, ở cột phải màu xanh — **Dense Decoder dùng cơ chế Non-autoregressive One-Shot**: ngay tại tầng chiếu cuối, từ một vector $\mathbf{e}^{(i)}$ duy nhất, nó xuất thẳng vector phẳng $\mathbf{g}^{(i)}$ cho cả $H$ bước cùng lúc. Vì mọi $\mathbf{d}_t^{(i)}$ đều đọc trực tiếp từ $\mathbf{e}^{(i)}$ chứ không phụ thuộc dự đoán bước trước, hiện tượng tích lũy sai số bị triệt tiêu và độ phức tạp tuần tự giảm xuống còn $O(1)$, tận dụng tối đa khả năng tính toán song song của GPU — giúp TiDE nhanh hơn Transformer từ 5 đến 10 lần.”

### Slide 04 / 10 — Kiến trúc $n_d$ Residual Blocks & Chuyển $\mathbf{e}^{(i)}$ thành $\mathbf{g}^{(i)}$ (~55s)

> “Đi sâu vào cấu trúc toán học bên trong $n_d$ khối Residual Block:
> Mọi chuỗi biến đổi đều bắt đầu từ $\mathbf{v}^{(0)} = \mathbf{e}^{(i)}$ với shape $(B, d_h)$.
> Với $n_d - 1$ khối trung gian đầu tiên (nếu có, vì nếu cấu hình $n_d = 1$ thì bỏ qua tầng này đi thẳng vào khối cuối), mỗi khối thực hiện công thức $\mathrm{LayerNorm}$ của nhánh phi tuyến ReLU–Linear–Dropout cộng với đường tắt tuyến tính $\mathbf{W}_{\mathrm{res}}^{(l)}\mathbf{v}^{(l-1)}$. Toàn bộ giai đoạn này giữ nguyên shape $(B, d_h)$ để tinh chỉnh đặc trưng ẩn.
> Điều đặc biệt nằm ở **khối cuối cùng** $l = n_d$: đây là cầu nối chuyển đổi $\mathbf{e}^{(i)}$ (hoặc $\mathbf{v}^{(n_d-1)}$) từ chiều $d_h$ thành vector phẳng $\mathbf{g}^{(i)}$ chiều $H \cdot p$. Vì $d_h \neq H \cdot p$, cả nhánh chính lẫn đường tắt đều **bắt buộc** phải dùng ma trận chiếu kích thước $(H \cdot p) \times d_h$ ($\mathbf{W}_{\mathrm{skip}}^{(n_d)}$) để cộng residual hợp lệ, tạo ra $\mathbf{g}^{(i)}$ có shape $(B, H \cdot p)$.”

### Slide 05 / 10 — Vai trò Reshape (Unflatten) & Ý nghĩa vector $\mathbf{d}_t^{(i)}$ (~45s)

> “Có được vector phẳng $\mathbf{g}^{(i)}$ chiều $(B, H \cdot p)$ rồi, bước tiếp theo là **Reshape / Unflatten** thành ma trận $\mathbf{D}^{(i)}$.
> Trong PyTorch (`g.view(-1, H, p)`), ta thu được tensor shape tổng quát $(B, H, p)$ (còn trong bài báo gốc và code TensorFlow, thứ tự trục là $(B, p, H)$). Ví dụ minh họa trên tập Electricity theo Table 8: nếu $B=512, d_h=1024, H=96, p=8$ thì $H \cdot p = 768$.
> Vai trò của thao tác này có hai điểm đáng nhớ: thứ nhất, nó khôi phục lại trục thời gian $H$ vốn đã bị gộp phẳng từ tầng Encoder với **chi phí bộ nhớ bằng** $O(1)$ (chỉ đổi stride metadata, không copy dữ liệu). Thứ hai, xét ý nghĩa: mỗi lát cắt $\mathbf{d}_t^{(i)}$ (shape $(B, p)$) chính là tóm tắt bối cảnh quá khứ **dành riêng cho thời điểm tương lai** $t$, mang thông tin chu kỳ và xu hướng tương ứng với độ trễ $t$ để chuẩn bị ghép ngoại sinh.”

---

## PHẦN 2: SOURCE CODE, HYPERPARAMETER & PHÂN TÍCH (SLIDE 06 – 08)

### Slide 06 / 10 — Source code chính thức (`models.py`) & PyTorch (~50s)

> “Để đối chiếu lý thuyết với triển khai thực tế, bên trái slide là nguyên văn code TensorFlow chính thức từ repo `google-research/tide/models.py` cùng phần PyTorch tương ứng:
> Ở cụm màu cam `[1]` (`models.py` dòng 72–74), hàm `_make_dnn_residual` lặp qua danh sách chiều để tạo chuỗi $n_d$ khối `MLPResidual` nối tiếp.
> Ở cụm xanh dương `[2]` (dòng 133–138), tác giả khởi tạo `self.decoder` bằng cách giữ nguyên `hidden_dims[:-1]` cho các lớp trung gian, nhưng **thay phần tử cuối bằng tích `decoder_output_dim * self.pred_len`** — chính là $p \cdot H$ để tạo $\mathbf{g}^{(i)}$ có shape $(B, H \cdot p)$.
> Cuối cùng, cụm xanh lá `[3]` (dòng 207–209 trong `models.py` và dòng 217–218 trong `simulate_forward_pass.py`) gọi `self.decoder(encoding)` rồi dùng `tf.reshape` / `view` tách thành $\mathbf{D}^{(i)}$ sẵn sàng cho từng bước $t$.”

### Slide 07 / 10 — Cấu hình Siêu tham số Dense Decoder trên 7 Datasets (~45s)

> “Chuyển sang bảng siêu tham số thực nghiệm chuẩn từ Phụ lục B.3 — Table 7 và Table 8 của bài báo gốc trên đủ cả 7 benchmark datasets:
> Trong không gian tìm kiếm Table 7, tác giả thử `hiddenSize` trong $\{256, 512, 1024\}$, `numDecoderLayers` ($n_d$) trong $\{1, 2, 3\}$, và `decoderOutputDim` ($p$) trong $\{4, 8, 16, 32\}$.
> Kết quả tối ưu trên Table 8 cho thấy hai quy luật cực kỳ rõ rệt: Toàn bộ 7/7 bộ dữ liệu chỉ chọn $n_d = 1$ (3 tập) hoặc $n_d = 2$ (4 tập) — không có bất kỳ tập nào cần tới $n_d = 3$. Về chiều $p$, giá trị phổ biến nhất là $p = 8$ và $p = 16$; chỉ riêng tập nhỏ ETTh2 vọt lên $p = 32$, kéo theo chi phí tham số tầng cuối tăng mạnh khi $H = 720$.”

### Slide 08 / 10 — Phân tích ảnh hưởng của $p$, số lớp $n_d$ & Giới hạn khi $H$ dài (~55s)

> “Từ số liệu Table 8 đó, ta rút ra phân tích trực tiếp về ảnh hưởng của hai siêu tham số cốt lõi:
> Về `decoderOutputDim` ($p$): nếu $p$ quá nhỏ ($1, 2$), chiều thông tin bị bó hẹp dẫn đến **nghẽn cổ chai (Information Bottleneck)**; ngược lại nếu $p$ quá lớn ($32, 64$), ma trận ngõ ra $H \cdot p$ phình to gây **bùng nổ tham số**, tốn VRAM và dễ Overfitting. Điểm cân bằng đẹp nằm ở $p \in [8, 16]$.
> Về số lớp $n_d$: $n_d = 1$ rất nhẹ nhàng; $n_d = 2$ là chuẩn mực đủ sâu; còn $n_d \geq 3$ bị bão hòa vì nhánh cộng tuyến tính toàn cục đã gánh phần xu hướng lớn.
> Tuy nhiên, thiết kế này để lộ **hai giới hạn rõ rệt**: thứ nhất là **cố định cứng Horizon** $H$ lúc khởi tạo ma trận cuối (`train H=96` không thể tự suy luận `H=192`); thứ hai là **bùng nổ tham số ở tầng cuối**: vì mỗi khối `MLPResidual` có tới hai ma trận dày đặc `lin_b` và `lin_res` cùng chiếu $d_h \rightarrow H \cdot p$, trên Electricity ở $H = 720$, riêng tầng cuối này chiếm tới $2 \times 5{,}760 \times 1{,}024 \approx 11.8\text{ triệu tham số}$. Đây chính là bằng chứng cho câu kết luận của tác giả ở Mục 6: ‘MLP kém hiệu quả tham số hơn Transformer’.”

---

## PHẦN 3: ĐỀ XUẤT TỐT NHẤT & LỘ TRÌNH KIỂM CHỨNG (SLIDE 09 – 10)

### Slide 09 / 10 — Đề xuất tốt nhất: Low-Rank Dense Decoder Projection (~55s)

> “Xuất phát trực tiếp từ đúng điểm yếu mà tác giả thừa nhận tại Mục 6 và tại dòng 136 của `models.py`, phương án cải thiện **duy nhất và thuyết phục nhất** mà em đề xuất cho khối Dense Decoder là **Low-Rank Factorized Dense Decoder Projection**, kế thừa từ bài báo **LoRA (Hu et al., ICLR 2022)** và **Low-Rank Matrix Factorization for DNN (Sainath et al., ICASSP 2013)**.
> Ý tưởng cốt lõi: thay vì chiếu dày đặc trực tiếp $\mathbf{W} \in \mathbb{R}^{(H \cdot p) \times d_h}$, ta phân rã thành hai ma trận hạng thấp $\mathbf{W} = \mathbf{A} \cdot \mathbf{B}$ đi qua một không gian cổ chai hạng $r = 64$ ($d_h \rightarrow r \rightarrow H \cdot p$).
> Lý do phương án này tốt nhất có 3 điểm rõ ràng: (1) giải quyết trúng điểm nghẽn tham số 11.8M được tác giả thừa nhận; (2) với $r = 64$ ở Electricity $H = 720$, tổng tham số hai ma trận tầng cuối giảm từ **11.8 triệu xuống chỉ còn khoảng 0.87 triệu — tiết kiệm ~92.6%**; (3) bảo toàn 100% cơ chế giải mã song song **One-Shot** $O(1)$ và tạo hiệu ứng điều chuẩn hạng thấp chống Overfitting.”

### Slide 10 / 10 — Điểm cải thiện cụ thể trong TiDE & Lộ trình kiểm chứng (~45s)

> “Chốt lại ở Slide cuối, đề xuất Low-Rank Projection mang lại cải thiện tại 3 điểm rất cụ thể trong TiDE: một là nén gọn tầng chiếu cuối ngay tại `models.py#L136`; hai là gỡ nút thắt VRAM & Overfitting ở $H = 720$, mở đường nâng dung lượng $p = 16, 32$ trên tập nhỏ như ETTh2; và ba là bổ sung trọn vẹn phần Ablation Study còn khuyết ở Mục 5.4 riêng cho cặp siêu tham số $(p, n_d)$.
> Để kiểm chứng thuyết phục, lộ trình thực nghiệm gồm 3 bước: chạy đối chứng 5 seeds Baseline vs Low-Rank với $r \in \{32, 64, 128\}$ trên Electricity, ETTh2, ETTm1 ở $H \in \{336, 720\}$; quét độ nhạy $p \in \{2..32\}$ và $n_d \in \{1..3\}$; và nghiệm thu với mục tiêu giảm tối thiểu **85% tham số tầng chiếu**, giữ nguyên tốc độ $O(1)$ cùng MSE/MAE tương đương hoặc tốt hơn bản gốc.
> Phần trình bày chuyên đề khối Dense Decoder của em đến đây là kết thúc. Em xin cảm ơn thầy/cô và các bạn đã lắng nghe!”
