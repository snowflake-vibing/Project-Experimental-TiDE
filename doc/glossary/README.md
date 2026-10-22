# Từ Điển Thuật Ngữ TiDE Theo Kỹ Thuật Feynman (Feynman Glossary)

> [!NOTE]
> **Điều hướng thư mục**: [🏠 Trang Chủ Repo](../../README.md) • [📖 Tổng Quan Tài Liệu](../00_tong_quan_tai_lieu.md) • [🔍 Chuyên Đề Phân Tích](../explanation/README.md)  
> **Mục tiêu**: Giải thích các khái niệm kỹ thuật cốt lõi trong mô hình TiDE và bài toán dự báo chuỗi thời gian bằng **Kỹ thuật Feynman (The Feynman Technique)** — dùng câu chuyện đời thường, ví dụ quen thuộc cho đứa trẻ 10 tuổi hiểu bản chất, không dùng biệt ngữ để định nghĩa biệt ngữ, kèm bảng dữ liệu mẫu và tổng kết 30 giây.

---

## 🎯 Triết Lý Feynman

> *"Nếu bạn không thể giải thích một khái niệm phức tạp bằng ngôn từ đơn giản cho một đứa trẻ 10 tuổi hiểu, thì chính bạn cũng chưa thực sự hiểu nó."*  
> — **Richard Feynman** (Giải Nobel Vật lý 1965)

Mỗi thuật ngữ trong từ điển đều tuân thủ nghiêm ngặt **Quy chuẩn Glossary (`RULE.md`)** với khung 6 phần:
1. **Giải thích cho đứa trẻ 10 tuổi**: Câu chuyện ngụ ngôn đời thực có nhân vật và bối cảnh sinh động.
2. **Bóc tách từng từ ngữ chuyên ngành**: Chiết tự gốc từ tiếng Anh, nghĩa tiếng Việt và khái niệm đối lập.
3. **Trực quan hóa bằng bảng dữ liệu**: Bảng số liệu mô phỏng ranh giới Quá khứ ($t \le L$) và Tương lai ($t > L$).
4. **Thách thức kỹ thuật**: Vì sao các phương pháp truyền thống (ARIMA, Transformer, PatchTST) gặp khó khăn.
5. **TiDE giải quyết vấn đề như thế nào**: Cơ chế kiến trúc cụ thể và ẩn dụ kỹ thuật.
6. **Tổng kết 30 giây**: 3 gạch đầu dòng cô đọng nhất phục vụ phỏng vấn hoặc ghi nhớ nhanh.

---

## 📚 Danh Mục Thuật Ngữ Đã Sắp Xếp Theo Bảng Chữ Cái (Alphabetical Order)

> 💡 *Quy chuẩn*: `RULE.md` đặt ở vị trí tham chiếu đầu tiên, tiếp theo là toàn bộ các thuật ngữ được sắp xếp theo thứ tự bảng chữ cái A-Z (Alphabetical order).

| STT | Thuật Ngữ (A-Z) | Ẩn Dụ Đời Thực | Khái Niệm Kỹ Thuật Trong TiDE |
|:---:|:---|:---|:---|
| **--** | [`RULE.md`](./RULE.md) | **Nguyên tắc Feynman** | Quy chuẩn bắt buộc về cấu trúc 6 phần, ngôn ngữ tiếng Việt có dấu, sơ đồ Mermaid và KaTeX. |
| **01** | [`co_adaptation.md`](./co_adaptation.md) | **Nhóm bạn làm bài tập dựa dẫm** | Hiện tượng các nơ-ron cấu kết bù trừ lỗi cho nhau dẫn đến học vẹt; hóa giải bằng Dropout $p=0.3 - 0.5$. |
| **02** | [`dynamic_covariates.md`](./dynamic_covariates.md) | **Quầy kem của bác Ba** | Biến ngoại sinh động thay đổi theo thời gian ($x_t$), biết trước cả trong quá khứ lẫn tương lai (ngày lễ, thời tiết). |
| **03** | [`feature_projection.md`](./feature_projection.md) | **Chiếc balo & túi hút chân không** | Khối Residual Block chiếu nén covariates từ $r$ chiều xuống $\tilde{r} = 4$ chiều, giải quyết bùng nổ số chiều. |
| **04** | [`hadamard_product.md`](./hadamard_product.md) | **Máy tính tiền siêu thị & nút vặn volume** | Ký hiệu phép nhân từng phần tử tương ứng ($\odot$), cơ chế scale & shift ($\gamma \odot x + \beta$) trong Layer Normalization. |
| **05** | [`overfitting.md`](./overfitting.md) | **Học sinh học vẹt đề cương thi** | Hiện tượng quá khớp mô hình và 5 tầng phòng thủ vững chắc của TiDE (MLP tối giản, Dropout, Weight Decay, RevIN, Skip). |
| **06** | [`relu.md`](./relu.md) | **Chiếc van một chiều chống ngập nước** | Hàm kích hoạt tuyến tính từng đoạn $\max(0, x)$, nguồn gốc phi tuyến tính duy nhất giúp mạng sâu uốn lượn không bị triệt tiêu gradient. |
| **07** | [`residual_block.md`](./residual_block.md) | **Bức tranh chú mèo trên tấm kính trong suốt** | Khối nơ-ron phần dư dày đặc, truyền thẳng dữ liệu gốc qua skip connection và chỉ học phần chênh lệch (residual). |
| **08** | [`revin.md`](./revin.md) | **Đổi tiền khi đi du lịch nước ngoài** | Chuẩn hóa thực thể đảo nghịch 2 chiều (Reversible Instance Normalization), trị căn bệnh trôi dạt phân phối (Non-stationarity). |
| **09** | [`temporal_decoder.md`](./temporal_decoder.md) | **Người đưa thư hỏa tốc tiệm pizza** | Bộ giải mã theo thời gian kết hợp ma trận dự báo và làn đường cao tốc (Covariate Highway) phản ứng tức thì với tương lai. |

---

> [!TIP]
> **Đóng góp thuật ngữ mới**: Trước khi khởi tạo bài viết mới trong thư mục này, vui lòng tham khảo tiêu chuẩn trong [`RULE.md`](./RULE.md).
