# Sprint 3: Huấn Luyện, Đánh Giá Đối Chuẩn & Thí Nghiệm Đột Phá (Training & Benchmarking)

## Mục tiêu
- Huấn luyện mô hình TiDE trên các benchmark LTTF tiêu chuẩn (Electricity, Traffic, Weather, ETTh1, ETTm1...).
- Thực nghiệm so sánh độ đo MSE/MAE với các mô hình baseline (DLinear, PatchTST, Informer, Autoformer).
- Phân tích Ablation Study và đánh giá tốc độ/bộ nhớ GPU ($O(L+H)$).

## Danh sách công việc (Tasks)
- [ ] Thiết lập pipeline huấn luyện (Loss function MSE, Adam optimizer, Cosine Annealing LR).
- [ ] Chạy thực nghiệm trên benchmark chính (Horizon $H \in \{96, 192, 336, 720\}$).
- [ ] Thu thập chỉ số MSE, MAE và so sánh với bảng kết quả trong paper.
- [ ] Đo lường tốc độ suy luận (Inference time) và mức tiêu thụ GPU VRAM.
- [ ] Viết báo cáo tổng kết thực nghiệm và đúc kết kết quả.
