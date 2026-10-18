# Chương 1: Giới Thiệu (Introduction)

## 1.1. Bối Cảnh Bài Toán (Context & Motivation)

Dự báo chuỗi thời gian dài hạn (**Long-term Time-Series Forecasting - LTTF**) là nhiệm vụ cốt lõi trong phân tích dữ liệu, đóng vai trò quan trọng trong việc lập kế hoạch và ra quyết định chiến lược ở nhiều lĩnh vực:
- **Năng lượng**: Dự báo phụ tải điện lưới quốc gia, sản lượng điện mặt trời, điện gió trước nhiều ngày hoặc nhiều tuần.
- **Giao thông**: Dự báo lưu lượng xe cộ trên hệ thống đường cao tốc để điều phối luồng xe và giảm tắc nghẽn.
- **Bán lẻ & Chuỗi cung ứng**: Dự báo nhu cầu mua sắm các mặt hàng tại hệ thống siêu thị theo mùa vụ, sự kiện khuyến mãi.
- **Tài chính & Kinh tế**: Dự báo các chỉ số kinh tế vĩ mô, biến động giá hàng hóa.

Trong bài toán LTTF, mô hình được cung cấp một cửa sổ quan sát quá khứ dài (**Look-back window** $L$) và cần dự đoán chính xác chuỗi giá trị tương lai trên một khoảng thời gian dài (**Horizon** $H$), thường với $H \ge 96$ bước thời gian (thậm chí lên tới 720 bước).

---

## 1.2. Nghịch Lý Giữa Transformer Và Mô Hình Tuyến Tính

Trong giai đoạn 2020 - 2022, cộng đồng nghiên cứu chứng kiến sự bùng nổ của các mô hình dựa trên kiến trúc **Transformer**:
- Các biến thể như **Informer** (Zhou et al., 2021), **Autoformer** (Wu et al., 2021), **FEDformer** (Zhou et al., 2022) liên tục công bố đạt hiệu năng SOTA (State-of-the-Art) trên các tập benchmark công khai.
- Giả định chung khi đó là: cơ chế **Self-Attention** là chìa khóa then chốt để nắm bắt các phụ thuộc xa trong chuỗi thời gian.

Tuy nhiên, nghiên cứu mang tính bước ngoặt của **Zeng et al. (2023)** với mô hình **DLinear** đã tạo nên một làn sóng tranh luận lớn khi chứng minh rằng:
> Một mô hình tuyến tính cực kỳ đơn giản (thậm chí chỉ là 1 lớp Linear duy nhất ánh xạ trực tiếp từ $L \to H$) có thể **đánh bại hoàn toàn** các mô hình Transformer phức tạp trên hầu hết các bộ benchmark LTTF phổ biến!

Nguyên nhân được chỉ ra là:
1. Cơ chế Attention hoán vị bất biến (permutation-invariant) làm mất mát thứ tự thời gian tự nhiên nếu không có mã hóa vị trí hoàn hảo.
2. Dữ liệu chuỗi thời gian liên tục thường có xu hướng biến thiên trơn và tính chu kỳ mạnh, khiến các phép biến đổi tuyến tính đơn giản nắm bắt xu hướng tốt hơn cơ chế Attention phức tạp vốn dễ bị quá khớp (overfitting).

---

## 1.3. Hạn Chế Của Các Mô Hình Tuyến Tính Đơn Giản

Mặc dù DLinear rất nhanh và hiệu quả trên các benchmark chuẩn, nó bộc lộ những nhược điểm chí mạng trong thực tế ứng dụng:
1. **Không xử lý được phi tuyến tính**: Trong nhiều bài toán thực tế, mối quan hệ giữa tương lai và quá khứ có tính phi tuyến mạnh (ví dụ: bão hòa nhu cầu, ngưỡng nhiệt độ kích hoạt tiêu thụ điện lạnh).
2. **Không thể tận dụng Covariates**: 
   - Các mô hình như DLinear buộc phải loại bỏ các biến phụ (covariates) vì khi đưa vào mô hình tuyến tính, chúng làm giảm hiệu năng.
   - Trong thực tế, các biến ngoại sinh như ngày lễ, sự kiện thể thao, chương trình giảm giá, hoặc thuộc tính phân loại (loại cửa hàng, vùng miền) mang lại lượng thông tin sống còn cho việc dự báo.

---

## 1.4. Đóng Góp Chính Của TiDE (Key Contributions)

Nhóm tác giả đề xuất **TiDE (Time-series Dense Encoder)** để kết hợp những gì tốt nhất của cả hai thế giới: **tốc độ và sự đơn giản của mô hình tuyến tính** cộng với **khả năng học biểu diễn phi tuyến và xử lý biến ngoại sinh của Deep Learning**.

Các đóng góp chính bao gồm:
1. **Kiến trúc Encoder-Decoder hoàn toàn dựa trên MLP**:
   - Sử dụng các khối Residual Block dày đặc (Dense MLPs).
   - Chi phí tính toán và bộ nhớ tăng tuyến tính $O(L + H)$, tránh được sự bùng nổ bậc hai $O(L^2)$ hoặc $O(L \cdot H)$ của Transformer.
2. **Tích hợp toàn diện Covariates**:
   - Thiết kế khối **Feature Projection** giúp nén chiều không gian các biến động (dynamic covariates) theo từng bước thời gian.
   - Thiết kế khối **Temporal Decoder** tạo một "đường cao tốc" (highway) dẫn truyền trực tiếp ảnh hưởng của các biến tương lai (như ngày lễ tại thời điểm $L+t$) vào kết quả dự báo tại thời điểm đó.
3. **Bảo toàn mô hình tuyến tính qua Global Skip Connection**:
   - Thêm một nhánh tắt tuyến tính nối thẳng từ $y_{1:L}$ sang $y_{L+1:L+H}$.
   - Nhờ đó, mô hình tuyến tính tối ưu luôn là một trường hợp con (subclass) của TiDE, đảm bảo TiDE không bao giờ kém hơn mô hình tuyến tính thuần túy.
4. **Cơ sở lý thuyết vững chắc**:
   - Chứng minh về mặt toán học rằng đối với Hệ động lực tuyến tính (Linear Dynamical Systems - LDS), một mô hình tự hồi quy tuyến tính với cửa sổ nhìn lại hữu hạn có thể đạt tỷ lệ lỗi xấp xỉ tối ưu.
5. **Hiệu năng thực nghiệm ấn tượng**:
   - Đạt kết quả tương đương hoặc vượt trội so với PatchTST và N-HiTS trên 7 tập dữ liệu benchmark lớn.
   - Nhanh hơn **5 - 10 lần** về thời gian suy luận và tiêu thụ ít bộ nhớ GPU hơn nhiều lần so với PatchTST.
   - Đạt độ chính xác cao hơn DeepAR **20%** trên bài toán dự báo bán lẻ cạnh tranh quy mô lớn M5.
