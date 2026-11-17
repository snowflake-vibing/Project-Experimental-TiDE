# 🎯 CHUYÊN ĐỀ THUYẾT TRÌNH 3 PHẦN: TEMPORAL DECODER & GLOBAL RESIDUAL CONNECTION

> [!IMPORTANT]
> **Trọng tâm chuyên đề:** Tập trung vào **hai cơ chế trực tiếp tạo ra dự báo cuối cùng** trong TiDE (*Temporal Decoder* & *Global Residual Connection*).  
> **Cấu trúc theo yêu cầu:** Chia thành đúng **3 phần mạch lạc** — đáp ứng trọn vẹn **5 yêu cầu bắt buộc** của hội đồng và toàn bộ danh mục nội dung cần trình bày.

---

## 📂 Danh Mục 3 Phần Trình Bày Chính Thức

| Phần | Tên File Chuyên Đề | Yêu Cầu Bắt Buộc Đáp Ứng | Hình Ảnh Đính Kèm |
| :---: | :--- | :--- | :--- |
| **Phần 1** | 📖 [**`phan_1_ly_thuyet_va_toan_hoc.md`**](./phan_1_ly_thuyet_va_toan_hoc.md) | **1. Lý thuyết** & **2. Toán học** | [`fig_p1_temporal_and_global_residual.png`](./images/fig_p1_temporal_and_global_residual.png) |
| **Phần 2** | 💻 [**`phan_2_source_code_va_phan_tich.md`**](./phan_2_source_code_va_phan_tich.md) | **3. Source code** & **4. Phân tích (kèm Ablation Study)** | [`fig_p2_ablation_residual_results.png`](./images/fig_p2_ablation_residual_results.png) |
| **Phần 3** | 💡 [**`phan_3_de_xuat.md`**](./phan_3_de_xuat.md) | **5. Đề xuất giả thuyết cải tiến thực nghiệm** | [`fig_p3_adaptive_gated_fusion.png`](./images/fig_p3_adaptive_gated_fusion.png) |

---

## ✅ Bảng Kiểm Tra Bao Phủ Toàn Bộ Nội Dung Được Giao

| STT | Nội Dung Cần Trình Bày | Phần Giải Quyết | Chi Tiết Hiện Thực |
| :---: | :--- | :---: | :--- |
| **1** | Cách `Temporal Decoder` kết hợp $d_t^{(i)}$ với *future covariates* tại thời điểm tương ứng. | **Phần 1 & 2** | Ghép cặp $\big[d_t^{(i)} ; \tilde{x}_{L+t}^{(i)}\big]$ theo từng bước $t$, chia sẻ trọng số qua `MLPResidual(..., output_dim=1)` (`models.py#L150-L155`, `#L211-L213`). |
| **2** | Vì sao TiDE bổ sung `Temporal Decoder` thay vì dùng trực tiếp đầu ra `Dense Decoder`? | **Phần 1 & 2** | Tạo **"Covariate Highway"** giúp tín hiệu ngoại sinh mạnh (ngày lễ, sự kiện Type A) không bị làm mờ qua nút thắt Encoder–Decoder và giúp mô hình không trật nhịp sau sự kiện. |
| **3** | Cách `Global Residual Connection` ánh xạ tuyến tính lookback thành horizon. | **Phần 1 & 2** | Tầng tuyến tính thuần túy `Dense(H)` chiếu trực tiếp $y_{1:L}^{(i)} \to \mathbb{R}^H$ (`models.py#L141-L144`, `#L203`), không kích hoạt phi tuyến, không Dropout. |
| **4** | Cách tổng hợp đầu ra *nonlinear* và *linear* để tạo dự báo cuối cùng. | **Phần 1 & 2** | Cộng trực tiếp $\hat{y}_{\text{norm}} = \hat{y}_{\text{NL}} + \hat{y}_{\text{Lin}}$ (`models.py#L214`), sau đó đảo chuẩn hóa RevIN về thang đo gốc. |
| **5** | Mối liên hệ giữa nhánh residual tuyến tính của TiDE với **DLinear**. | **Phần 1** | Chứng minh toán học vì sao **DLinear là lớp con nghiêm ngặt (strict subclass)** của TiDE khi nhánh phi tuyến triệt tiêu ($\hat{y}_{\text{NL}} \equiv 0$). |
| **6** | Phân tích vai trò của `Temporal Decoder` và `Global Residual` qua **Ablation Study**. | **Phần 2** | Đối chiếu số liệu thật từ Mục 5.4 Paper gốc: **Figure 3** (Type A/B events sau 1 epoch) & **Table 5** (Electricity MSE/MAE `TiDE` vs `TiDE no res.` across $H=96..720$). |
| **7** | **Hướng nghiên cứu tiếp:** Cải tiến cơ chế kết hợp nhánh phi tuyến và nhánh tuyến tính. | **Phần 3** | Đề xuất 2 giả thuyết: **(H1)** *Adaptive Gated Fusion* ($\alpha_t \cdot y_{\text{NL}} + (1 - \alpha_t) \cdot y_{\text{Lin}}$) chỉ tốn 21 tham số & **(H2)** *Trend-Aware Dual-Stream Mixing* kiểu DLinear; kèm PyTorch code & Ablation Plan. |
