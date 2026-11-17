# 🎞️ BỘ SLIDE THUYẾT TRÌNH 3 PHẦN: TEMPORAL DECODER & GLOBAL RESIDUAL CONNECTION

> **Phạm vi & Bố cục (Không tính slide Intro/Outro):** Tập trung vào **hai cơ chế trực tiếp tạo ra dự báo cuối cùng** trong TiDE, chia thành **3 phần rõ ràng** theo đúng bố cục bạn yêu cầu — đáp ứng trọn vẹn **5 tiêu chuẩn bắt buộc** và **7 nội dung cốt lõi**. Mỗi slide đều có sẵn hình ảnh PNG độ phân giải cao (180 DPI), sơ đồ Mermaid, nội dung súc tích chiếu trên slide, vị trí dòng code chính xác và kịch bản lời nói (Speaker Notes).

---

## 📑 Danh Sách 3 Phần Slide Chính Thức

| Phần | File Nội Dung Chi Tiết | Nội Dung & Yêu Cầu Đáp Ứng | Hình Ảnh Đính Kèm (`images/`) |
| :---: | :--- | :--- | :--- |
| **Phần 1** | 📖 [**`slide_01_phan_1_ly_thuyet_va_toan_hoc.md`**](./slide_01_phan_1_ly_thuyet_va_toan_hoc.md) | **Lý thuyết & Toán học:**<br>• Cách Temporal Decoder kết hợp $d_t^{(i)}$ với future covariates tại từng thời điểm $t$.<br>• Lý do dùng Temporal Decoder ("Covariate Highway") thay vì dùng thẳng Dense Decoder.<br>• Cách Global Residual ánh xạ tuyến tính Lookback $\to$ Horizon.<br>• Phép cộng tổng hợp Nonlinear + Linear & mối liên hệ với **DLinear**.<br>• Bảng Tensor Shape đầu vào/đầu ra. | • [`fig_p0_figure1_highlight.png`](./images/fig_p0_figure1_highlight.png)<br>• [`fig_p1_temporal_and_global_residual.png`](./images/fig_p1_temporal_and_global_residual.png) |
| **Phần 2** | 💻 [**`slide_02_phan_2_source_code_va_phan_tich.md`**](./slide_02_phan_2_source_code_va_phan_tich.md) | **Source code & Phân tích (Ablation Study):**<br>• Code chính xác từ Repo gốc Google Research (`models.py#L141-L155`, `#L203`, `#L211-L214`) và PyTorch (`simulate_forward_pass.py#L232-L266`).<br>• Ablation Study 1 (Figure 3): Thí nghiệm sự kiện Type A/B sau 1 Epoch.<br>• Ablation Study 2 (Table 5): Số liệu MSE/MAE trên Electricity (`TiDE` vs `TiDE no res.`).<br>• Giới hạn của phép cộng cứng `1:1`. | • [`fig_p2_ablation_residual_results.png`](./images/fig_p2_ablation_residual_results.png) |
| **Phần 3** | 💡 [**`slide_03_phan_3_de_xuat_cai_tien.md`**](./slide_03_phan_3_de_xuat_cai_tien.md) | **Đề xuất & Đóng góp (Contributions):**<br>• Đề xuất 1 (3.1): *Adaptive Gated Fusion* ($\alpha_t \cdot \hat{\mathbf{y}}_{\mathrm{NL}} + (1 - \alpha_t) \cdot \hat{\mathbf{y}}_{\mathrm{Lin}}$), chỉ $+21$ tham số.<br>• Đề xuất 2 (3.2): *Trend-Aware Dual-Stream* tách Trend kiểu DLinear.<br>• Đóng góp 3.3 (Contributions): Chính xác đa chế độ • Giải thích được $\alpha_t$ • Lấp khoảng trống 6/7 tập Ablation. | • [`fig_p3_adaptive_gated_fusion.png`](./images/fig_p3_adaptive_gated_fusion.png)<br>• [`fig_p3_2_trend_aware_dual_stream.png`](./images/fig_p3_2_trend_aware_dual_stream.png)<br>• [`fig_p3_3_contributions_and_roadmap.png`](./images/fig_p3_3_contributions_and_roadmap.png) |

---

## 🖼️ Thư Viện Hình Ảnh Trình Chiếu (`images/` & [`slide_images_gallery.md`](./slide_images_gallery.md))
👉 **Xem toàn bộ 10 ảnh xếp theo đúng từng mục nội dung trong 1 file duy nhất:** [**`slide_images_gallery.md`**](./slide_images_gallery.md)

Tất cả 10 file ảnh PNG sắc nét (180–200 DPI) đã nằm sẵn trong [`images/`](./images/) để bạn kéo thả trực tiếp vào PowerPoint, Canva hoặc Google Slides:
* [`fig_p0_figure1_highlight.png`](./images/fig_p0_figure1_highlight.png) — Vị trí khoanh vùng Temporal Decoder & Global Residual trên Figure 1 gốc
* [`fig_p1_temporal_decoder_highway.png`](./images/fig_p1_temporal_decoder_highway.png) — Tách 3 Card riêng (Dense Encoder • Dense Decoder • Temporal Decoder) & Cơ chế Highway
* [`fig_p1_temporal_and_global_residual.png`](./images/fig_p1_temporal_and_global_residual.png) — Sơ đồ 2 nhánh song song tạo dự báo cuối cùng
* [`fig_p1_tensor_shapes_and_dlinear.png`](./images/fig_p1_tensor_shapes_and_dlinear.png) — Chứng minh DLinear là lớp con của TiDE & Luồng Tensor Shapes
* [`fig_p2_source_code_mapping.png`](./images/fig_p2_source_code_mapping.png) — Show nguyên đoạn Code gốc (`models.py#L128-L215`) ánh xạ thẳng ra Equations
* [`fig_p2_ablation_temporal_decoder.png`](./images/fig_p2_ablation_temporal_decoder.png) — Ablation Study 1 (Figure 3 Trang 10 Paper): Temporal Decoder sau 1 Epoch
* [`fig_p2_ablation_residual_results.png`](./images/fig_p2_ablation_residual_results.png) — Ablation Study 2 (Ảnh cắt gốc Table/Figure 5 Trang 11 Paper)
* [`fig_p3_adaptive_gated_fusion.png`](./images/fig_p3_adaptive_gated_fusion.png) — Đề xuất 1 (3.1): Adaptive Gated Fusion so sánh với TiDE gốc
* [`fig_p3_2_trend_aware_dual_stream.png`](./images/fig_p3_2_trend_aware_dual_stream.png) — Đề xuất 2 (3.2): Trend-Aware Dual-Stream tách MovingAvg kiểu DLinear
* [`fig_p3_3_contributions_and_roadmap.png`](./images/fig_p3_3_contributions_and_roadmap.png) — Contributions & Lộ trình kiểm chứng thực nghiệm (3.3)
