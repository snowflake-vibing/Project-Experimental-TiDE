# Sprint 2: Triển Khai Kiến Trúc Mô Hình & Tiền Xử Lý Dữ Liệu (Model Implementation)

## Mục tiêu
- Xây dựng mã nguồn PyTorch mô hình TiDE hoàn chỉnh (Residual Block, Feature Projection, Dense Encoder, Dense Decoder, Temporal Decoder, Global Skip).
- Tích hợp lớp chuẩn hóa thích nghi RevIN (Reversible Instance Normalization).
- Xây dựng Data Loader hỗ trợ Channel Independence và Covariates động/tĩnh.

## Danh sách công việc (Tasks)
- [ ] Xây dựng module `ResidualBlock` kèm LayerNorm và Dropout.
- [ ] Xây dựng module `FeatureProjection` nén $r \to \tilde{r} = 4$.
- [ ] Xây dựng module `DenseEncoder` và `DenseDecoder`.
- [ ] Xây dựng module `TemporalDecoder` kết hợp bối cảnh $d_t$ và ngoại sinh tương lai.
- [ ] Tích hợp `RevIN` 2 chiều (Normalize lúc vào, Denormalize lúc ra).
- [ ] Hoàn thiện class tổng thể `TiDE` và kiểm tra luồng tensor (`shape`).
