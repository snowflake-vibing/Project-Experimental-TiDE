# Chương 6: Kết Luận & Hướng Phát Triển (Conclusion & Future Work)

Chương này tổng kết lại những thành tựu của nghiên cứu TiDE, phân tích khách quan các ưu điểm và hạn chế, đồng thời gợi mở các hướng nghiên cứu tiềm năng trong tương lai.

---

## 6.1. Những Đột Phá Chính Của Nghiên Cứu

Nghiên cứu về TiDE đã mang lại những đóng góp có ý nghĩa căn bản cho lĩnh vực học máy trên dữ liệu chuỗi thời gian:

1. **Phá vỡ định kiến về sự vượt trội bắt buộc của Attention**:
   - Nghiên cứu chứng minh rằng không cần đến cơ chế Self-Attention phức tạp và tốn kém để đạt được độ chính xác SOTA trong bài toán dự báo chuỗi thời gian dài hạn.
   - Kiến trúc MLP đơn giản nếu được thiết kế đúng cách (với Residual Blocks, Feature Projection và Temporal Decoder) hoàn toàn có thể vượt qua các Transformer hàng đầu.

2. **Cầu nối hoàn hảo giữa tốc độ tuyến tính và sức mạnh phi tuyến**:
   - Kế thừa tốc độ tính toán tuyến tính $O(L + H)$ và bộ nhớ gọn nhẹ của các mô hình tuyến tính.
   - Đồng thời sở hữu khả năng biểu diễn phi tuyến tính mạnh mẽ để mô hình hóa các quy luật phức tạp.

3. **Khả năng khai thác Covariates vô địch**:
   - Khắc phục hoàn toàn điểm yếu "mù covariates" của các mô hình tuyến tính (DLinear) và Transformer hiện đại (PatchTST).
   - Cơ chế **Temporal Decoder** cung cấp giải pháp thanh lịch để xử lý các sự kiện đột biến trong tương lai (ngày lễ, khuyến mãi, thay đổi giá).

4. **Hiệu quả thực tế và triển khai công nghiệp**:
   - Tốc độ huấn luyện và suy luận nhanh hơn 5-10 lần, không bị lỗi Out-of-Memory trên GPU thông thường.
   - Rất phù hợp để triển khai thực tế trên quy mô sản xuất (production) với hàng triệu chuỗi thời gian cần dự báo định kỳ mỗi ngày.

---

## 6.2. Hạn Chế Của Nghiên Cứu (Limitations)

Dù đạt được nhiều thành tựu vượt bậc, nhóm tác giả và các nhà nghiên cứu độc lập cũng chỉ ra một số hạn chế của TiDE:

1. **Chưa tường minh mô hình hóa tương quan giữa các kênh (Cross-Channel Correlation)**:
   - Chiến lược độc lập kênh (Channel Independence) giúp chống overfitting nhưng bỏ qua các mối tương quan động giữa các cảm biến hoặc sản phẩm thay thế lẫn nhau.
2. **Phụ thuộc vào kích thước Look-back cố định**:
   - Mô hình đòi hỏi cửa sổ đầu vào $L$ có kích thước cố định khi khởi tạo mạng MLP, chưa linh hoạt tự thích ứng với các chuỗi có độ dài lịch sử biến đổi liên tục như RNN.
3. **Bài toán dữ liệu cực kỳ thưa (Extremely Sparse Data)**:
   - Trên các chuỗi bán lẻ bán chậm (intermittent / sporadic demand với nhiều số 0), hàm mất mát MSE thông thường có thể tạo ra dự báo bị trơn hóa quá mức.

---

## 6.3. Hướng Nghiên Cứu & Phát Triển Trong Tương Lai

1. **Tích hợp Tương quan Kênh Nhẹ (Lightweight Cross-Channel Mixing)**:
   - Nghiên cứu cơ chế trộn thông tin giữa các kênh (ví dụ: Channel MLP hoặc nhóm phân cụm kênh) mà vẫn giữ được độ phức tạp tuyến tính $O(L + H)$.
2. **Dự Báo Không Cần Tinh Chỉnh (Zero-Shot & Foundation Models)**:
   - Huấn luyện TiDE trên các tập dữ liệu khổng lồ đa lĩnh vực để tạo thành các mô hình nền tảng (Foundation Models) cho Time-Series.
3. **Dự Báo Phân Phối Xác Suất (Probabilistic Forecasting)**:
   - Mở rộng tầng đầu ra của Temporal Decoder để dự đoán các tham số phân phối (như Normal, Student-t, Negative Binomial) hoặc các mức phân vị (Quantile Forecasting), phục vụ bài toán quản trị rủi ro.
