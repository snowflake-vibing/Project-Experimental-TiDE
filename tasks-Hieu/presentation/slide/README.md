# 🎞️ BỘ SLIDE THUYẾT TRÌNH CHI TIẾT: DENSE DECODER (TiDE)

> **Phạm vi:** Chỉ bao gồm **5 Slide nội dung cốt lõi** (không tính slide Intro/Outro theo đúng yêu cầu), trình bày trọn vẹn 6 gạch đầu dòng phân công và 5 tiêu chí đánh giá của hội đồng. Mỗi slide đều đính kèm **hình ảnh minh họa PNG độ phân giải cao**, **sơ đồ kiến trúc Mermaid**, **nội dung súc tích hiển thị trên slide**, và **kịch bản lời nói (Speaker Notes)** để thành viên nghe hiểu tường tận.

---

## 📑 Danh Sách 5 Slide Nội Dung Chính Thức

| STT | Tên File Slide | Trọng Tâm Nội Dung | Hình Ảnh Đính Kèm |
| :---: | :--- | :--- | :--- |
| **Slide 01** | 🖼️ [**`slide_01_kien_truc_va_ly_thuyet.md`**](./slide_01_kien_truc_va_ly_thuyet.md) | **Kiến trúc $n_d$ Residual Blocks & Lý do thiết kế:** Vị trí khối Dense Decoder, cấu tạo các khối trung gian vs khối cuối, lý do triệt tiêu tích lũy sai số & tối ưu GPU. | [`fig1_dense_decoder_position.png`](./images/fig1_dense_decoder_position.png) |
| **Slide 02** | 📐 [**`slide_02_toan_hoc_va_unflatten.md`**](./slide_02_toan_hoc_va_unflatten.md) | **Toán học giải nén ($e \to g$), Unflatten & Ý nghĩa $d_t$:** Công thức chiếu $H \times p$, vai trò $O(1)$ của `Reshape`, ý nghĩa $d_t$ để ghép ngoại sinh tương lai, bảng Tensor shape. | [`fig2_unflatten_tensor_flow.png`](./images/fig2_unflatten_tensor_flow.png) |
| **Slide 03** | 💻 [**`slide_03_source_code_va_non_autoregressive.md`**](./slide_03_source_code_va_non_autoregressive.md) | **Source code gốc & Cơ chế Non-autoregressive:** Đối chiếu PyTorch code trong `google-research/tide/model.py`, so sánh sinh song song One-shot vs sinh tuần tự Autoregressive. | [`fig3_non_autoregressive_vs_autoregressive.png`](./images/fig3_non_autoregressive_vs_autoregressive.png) |
| **Slide 04** | 📊 [**`slide_04_phan_tich_hieu_nang_va_gioi_han.md`**](./slide_04_phan_tich_hieu_nang_va_gioi_han.md) | **Phân tích siêu tham số ($p$, $n_d$) & Giới hạn:** Điểm ngọt $p \in [4, 16]$ vs bùng nổ $47.18\text{M}$ params ($H=720, p=64$); vai trò $n_d=2$; 3 giới hạn tiềm ẩn. | [`fig4_hyperparameter_p_impact.png`](./images/fig4_hyperparameter_p_impact.png) |
| **Slide 05** | 💡 [**`slide_05_de_xuat_cai_tien_thuc_nghiem.md`**](./slide_05_de_xuat_cai_tien_thuc_nghiem.md) | **Đề xuất giả thuyết cải tiến thực nghiệm:** Low-Rank Factorized Projection giảm $93.2\%$ tham số khi $H=720$ + Depthwise Conv1D làm mịn $d_t$ + Ablation Study. | [`fig5_low_rank_proposal.png`](./images/fig5_low_rank_proposal.png) |

---

## 🖼️ Thư Viện Hình Ảnh Trình Chiếu (`images/`)
Tất cả hình ảnh minh họa chất lượng cao (180 DPI, định dạng PNG, chữ tiếng Việt rõ nét) đều nằm sẵn trong thư mục [`images/`](./images/) để bạn kéo thả trực tiếp vào PowerPoint, Canva hoặc Google Slides:
- [`fig1_dense_decoder_position.png`](./images/fig1_dense_decoder_position.png)
- [`fig2_unflatten_tensor_flow.png`](./images/fig2_unflatten_tensor_flow.png)
- [`fig3_non_autoregressive_vs_autoregressive.png`](./images/fig3_non_autoregressive_vs_autoregressive.png)
- [`fig4_hyperparameter_p_impact.png`](./images/fig4_hyperparameter_p_impact.png)
- [`fig5_low_rank_proposal.png`](./images/fig5_low_rank_proposal.png)
