# Thuật Ngữ: Feature Projection (Chiếu Giảm Chiều Đặc Trưng Theo Thời Gian)

> **Phương pháp tiếp cận Feynman (Feynman Technique)**:  
> *"Nếu bạn không thể giải thích một điều gì đó cho một đứa trẻ 10 tuổi hiểu, thì chính bạn cũng chưa thực sự hiểu nó."*  
> Bài viết này bóc tách khái niệm **Feature Projection** trong mô hình TiDE từ câu chuyện đóng gói hành lý du lịch, sau đó phân tích bản chất toán học và tác dụng sống còn của nó đối với bộ nhớ máy tính.

---

## 1. Giải Thích Cho Một Đứa Trẻ 10 Tuổi: Chiếc Balo Đi Cắm Trại Và Túi Hút Chân Không

Hãy tưởng tượng bạn chuẩn bị đi cắm trại liên tục trong vòng **30 ngày**. Mỗi ngày, bạn phải chuẩn bị sẵn **10 món đồ lặt vặt** khác nhau: đèn pin, áo ấm, pin dự phòng, la bàn, hộp quẹt, bình xịt côn trùng, kem chống nắng, v.v.

Bây giờ bạn gặp một rắc rối lớn:
- Nếu bạn cứ để nguyên 10 món đồ cồng kềnh cho từng ngày rồi nhét hết vào balo:  
  **30 ngày $\times$ 10 món = 300 món đồ đồ sộ!**  
  Chiếc balo của bạn sẽ rách toạc ngay lập tức, và bạn cũng không thể nào vác nổi trên vai.

```mermaid
flowchart TD
    subgraph RacRoi ["Rắc Rối: Hành Lý Quá Cồng Kềnh"]
        Raw["30 ngày x 10 món đồ = 300 món đồ khổng lồ! (Balo rách toạc)"]
    end

    subgraph GiaiPhap ["Giải Pháp: Túi Hút Chân Không (Feature Projection)"]
        Raw --> Pump["Máy Hút Chân Không: Nén 10 món đồ của mỗi ngày thành 1 gói nén nhỏ gọn"]
        Pump --> NeatBag["30 ngày x 1 gói nén = 30 gói siêu gọn (Vừa vặn trong balo)"]
    end
```

**Cách giải quyết thông minh**:
Bạn mua một chiếc **máy hút chân không**. Với đồ đạc của mỗi ngày, bạn ép chặt 10 món đó lại thành **1 gói nén tí hon**, chỉ giữ lại những gì thực sự quan trọng và chiếm rất ít thể tích. Khi đó, balo của bạn chỉ cần chứa 30 gói nén nhỏ nhắn, nhẹ nhàng và tiện lợi.

> 💡 **Định nghĩa kiểu Feynman**:  
> **Feature Projection** trong TiDE chính là chiếc **"máy hút chân không"**. Nó nén các thông tin ngoại sinh cồng kềnh tại mỗi bước thời gian thành một gói đặc trưng siêu nhỏ gọn trước khi xếp vào bộ nhớ của mô hình.

---

## 2. Bóc Tách Từng Từ Ngữ Chuyên Ngành (Deconstructing the Jargon)

Tại sao lại gọi là **Feature Projection** mà không gọi đơn giản là "nén dữ liệu"?

```mermaid
flowchart LR
    Term["Feature Projection"]
    Term --> P1["Feature (Đặc trưng)"]
    Term --> P2["Projection (Phép chiếu)"]

    P1 --> D1["Các biến ngoại sinh mô tả: thời tiết, ngày lễ, giá cả"]
    P2 --> D2["Ánh xạ toán học từ không gian nhiều chiều xuống không gian ít chiều"]
```

| Thành phần | Nghĩa gốc | Ý nghĩa trong Machine Learning | Phản nghĩa / Đối lập |
|:---|:---|:---|:---|
| **Feature** (*Đặc trưng*) | Dấu hiệu nhận biết, thuộc tính | Các biến số ngoại sinh mô tả bối cảnh tại thời điểm $t$ (nhiệt độ, độ ẩm, mã sự kiện...). | **Raw Data** (*Dữ liệu thô chưa qua chọn lọc*). |
| **Projection** (*Phép chiếu*) | Chiếu bóng, rọi ánh sáng | Ánh xạ một vector từ không gian có số chiều lớn ($r$) sang không gian có số chiều nhỏ hơn ($\tilde{r} \ll r$), tương tự chiếc bóng 2D của một bàn tay 3D in lên tường. | **Expansion** (*Phép mở rộng số chiều*). |

---

## 3. Trực Quan Hóa Bằng Bảng Dữ Liệu (The "Aha!" Moment)

Hãy nhìn vào sự bùng nổ số chiều khủng khiếp nếu không có Feature Projection:

Giả sử ta xét bài toán dự báo phụ tải điện với:
- Cửa sổ nhìn lại: $L = 720$ bước (30 ngày).
- Khoảng dự báo tương lai: $H = 720$ bước (30 ngày).
- Tổng số bước thời gian cần xử lý: $L + H = 1440$ bước thời gian.
- Số lượng biến ngoại sinh tại mỗi bước: $r = 20$ chiều (giờ, ngày, tháng, nhiệt độ, tốc độ gió, giá điện, ngày nghỉ...).

| Phương Án | Số Bước Thời Gian | Kích Thước Biến Ngoại Sinh | Tổng Số Chiều Khi Trải Phẳng (Flatten) | Tình Trạng Bộ Nhớ & Mô Hình |
|:---|:---:|:---:|:---:|:---|
| **Không dùng Projection** | 1,440 | $r = 20$ | **$1440 \times 20 = 28,800$ chiều!** | 💥 Quá tải bộ nhớ, bùng nổ tham số, mạng học vẹt (overfitting). |
| **Có Feature Projection (TiDE)** | 1,440 | $\tilde{r} = 4$ *(nén bằng Residual Block)* | **$1440 \times 4 = 5,760$ chiều!** | ✅ Gọn nhẹ gấp **5 lần**, tính toán mượt mà, chống quá khớp vượt bậc. |

> 🎯 **Khoảnh khắc "Aha!"**:  
> Nếu bạn trải phẳng 28,800 con số để đưa vào một tầng ẩn có 512 nơ-ron, riêng tầng đầu tiên đã tốn tới:  
> $28,800 \times 512 \approx \mathbf{14.7 \text{ triệu tham số}}$!  
> Nhưng khi dùng Feature Projection nén xuống $\tilde{r} = 4$, số tham số giảm ngay xuống chỉ còn:  
> $5,760 \times 512 \approx \mathbf{2.9 \text{ triệu tham số}}$ (giảm hơn **80%** gánh nặng tính toán)!

---

## 4. Tại Sao Các Mô Hình Khác Lại Bị Hạn Chế?

1. **Các mô hình MLP đơn giản**: Nếu không có cơ chế nén theo từng bước thời gian mà cứ thế nối toàn bộ $x_{1:L+H}$ vào đầu vào, mạng sẽ bị hiện tượng **"Lời nguyền số chiều" (Curse of Dimensionality)**. Số lượng tham số quá lớn khiến mô hình học vẹt trên tập huấn luyện và sai lệch hoàn toàn khi gặp dữ liệu thực tế.
2. **Các mô hình Transformer (Informer, Autoformer)**: Cố gắng mã hóa từng bước thời gian thành các vector nhúng (embedding) có kích thước lớn ($d_{\text{model}} = 512$), khiến ma trận Attention giữa các bước thời gian phình to thành $1440 \times 1440 \times 512$, dẫn tới tràn bộ nhớ GPU (Out-Of-Memory).

---

## 5. Mô Hình TiDE Triển Khai Feature Projection Như Thế Nào?

TiDE thiết kế một khối chiếu đặc trưng cực kỳ thanh lịch và độc lập theo từng bước thời gian:

```mermaid
flowchart LR
    X_t["Vector biến ngoại sinh x_t tại thời điểm t\n(Kích thước gốc: r chiều)"]
    RB["Khối Residual Block nhỏ\n(temporalWidth = r_tilde)"]
    X_tilde["Vector đặc trưng đã nén x_tilde_t\n(Kích thước nén: r_tilde << r)"]

    X_t --> RB
    RB --> X_tilde
```

Công thức toán học:

$$
\tilde{x}_t^{(i)} = \text{ResidualBlock}(x_t^{(i)}) \in \mathbb{R}^{\tilde{r}} \quad \forall t \in \{1, 2, \dots, L+H\}
$$

- Điểm mấu chốt: **Trọng số của Residual Block này được chia sẻ chung cho tất cả các bước thời gian $t$**.
- Nhờ đó, mô hình học được một bộ quy tắc nén đặc trưng nhất quán, bảo toàn thông tin quan trọng nhất của các biến ngoại sinh mà chi phí tham số gần như bằng 0.

### 5.1. Cơ Chế Đại Số Tuyến Tính: Làm Sao Từ $r$ Chiều Xuống $r'$ Chiều Được?

Nhiều người thắc mắc: *"Làm thế nào một mạng nơ-ron có thể ép một vector từ $r$ biến thô (ví dụ $r=7$) xuống còn $r'$ biến nén ($r'=4$) mà vẫn giữ được thông tin?"*

Bên trong khối `ResidualBlock` nhận $d_{\text{in}} = r$ và xuất ra $d_{\text{out}} = r'$, cơ chế nén diễn ra đồng thời trên cả hai nhánh:

1. **Nhánh Biến Đổi Phi Tuyến (MLP Branch):**
   * **Tầng Linear 1 ($r \to d_{\text{hidden}}$):** Nhân với ma trận trọng số $W_1 \in \mathbb{R}^{d_{\text{hidden}} \times r}$ ($1024 \times 7$) và cộng bias $b_1$, sau đó qua hàm kích hoạt $\text{ReLU}$. Bước này phóng đại các đặc trưng thô lên không gian đa chiều để chúng tương tác tự do với nhau.
   * **Tầng Linear 2 ($d_{\text{hidden}} \to r'$):** Nhân với ma trận $W_2 \in \mathbb{R}^{r' \times d_{\text{hidden}}}$ ($4 \times 1024$) và cộng bias $b_2$. Vì ma trận $W_2$ chỉ có đúng $r'$ hàng, mỗi phần tử đầu ra là một tổ hợp tuyến tính cô đọng từ $1024$ nơ-ron ẩn, ép số chiều về đúng $r'$:
     $$F(x_t) = \text{Dropout}\Big( W_2 \cdot \text{ReLU}(W_1 x_t + b_1) + b_2 \Big) \in \mathbb{R}^{r'}$$

2. **Nhánh Tắt Tuyến Tính (Skip Connection Branch):**
   * Vì số chiều đầu vào lệch với đầu ra ($r \ne r'$, ví dụ $7 \ne 4$), nhánh skip không thể dùng phép đồng nhất trực tiếp (`Identity`). Thay vào đó, nó sử dụng một **ma trận chiếu tuyến tính** $W_{\text{skip}} \in \mathbb{R}^{r' \times r}$ (kích thước $4 \times 7$):
     $$\text{Skip}(x_t) = W_{\text{skip}} \cdot x_t \in \mathbb{R}^{r'}$$

3. **Cộng Thặng Dư & Chuẩn Hóa:**
   * Cả hai nhánh lúc này đều có kích thước $r' = 4$. Chúng được cộng trực tiếp với nhau và chuẩn hóa qua $\text{LayerNorm}$:
     $$\tilde{x}_t = \text{LayerNorm}\Big( F(x_t) + W_{\text{skip}} x_t \Big) \in \mathbb{R}^{r'}$$

> 💡 **Bản chất ý nghĩa**:  
> Các biến ngoại sinh trong thực tế (như giờ trong ngày, nhiệt độ, thứ trong tuần, cờ ngày lễ) thường có **tính tương quan cao và dư thừa thông tin**. Việc chiếu từ $r \to r'$ hoạt động như một thuật toán **PCA phi tuyến (Non-linear PCA / Embedding)**: nó tự động khử nhiễu, chắt lọc các biến có chung quy luật và chỉ giữ lại $r'$ đặc trưng cô đọng, giàu ngữ cảnh nhất.

```python
# Cài đặt thực tế trong PyTorch:
# Khởi tạo:
self.feature_projection = ResidualBlock(
    in_dim=r,              # Ví dụ: 7 chiều thô
    hidden_dim=1024,       # Không gian ẩn nội bộ
    out_dim=temporal_width # r' (ví dụ: 4 chiều nén)
)

# Thực thi trên toàn bộ tensor 3D:
# (Batch, L + H, r) -> (Batch, L + H, temporal_width)
x_proj = self.feature_projection(x) # (32, 816, 7) -> (32, 816, 4)
```

---

## 6. Tổng Kết 30 Giây (The Elevator Pitch)

1. **Feature Projection là gì?**  
   Là bước nén số chiều của các biến ngoại sinh tại từng thời điểm $t$ từ không gian lớn $r$ xuống không gian nhỏ $\tilde{r} \ll r$ (thường chọn $\tilde{r} = 4$).
2. **Giải quyết vấn đề gì?**  
   Ngăn chặn hiện tượng bùng nổ số chiều khi trải phẳng $(L+H) \times r$ điểm thời gian vào mạng nơ-ron, giảm hơn 80% số tham số không cần thiết.
3. **Tại sao TiDE vượt trội?**  
   Dùng một khối Residual Block nhỏ dùng chung cho mọi bước thời gian, vừa đảm bảo tính phi tuyến tính vừa giữ cho mô hình chạy nhanh gấp 5-10 lần các mạng Transformer.
