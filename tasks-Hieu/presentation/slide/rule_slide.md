# 📌 `rule_slide.md` — BỘ QUY TẮC THIẾT KẾ SLIDE SEMINAR CHUẨN MỰC

> [!IMPORTANT]
> Tài liệu này chuẩn hóa toàn bộ **yêu cầu bắt buộc về hình thức, bố cục, trích nguồn và nội dung** khi thiết kế hoặc cập nhật bộ slide thuyết trình Seminar trong [`tasks-Hieu/presentation/slide/`](./README.md), đồng thời tuân thủ song song các quy định Markdown/KaTeX/Mermaid chung tại [`RULE.md`](../../../RULE.md).

---

## 🧭 MỤC LỤC
1. [Quy tắc 1: Khung hình, Typography & Card Layout (Tính chuyên nghiệp Seminar)](#quy-tắc-1-khung-hình-typography--card-layout)
2. [Quy tắc 2: Công thức Toán học & Ký hiệu Hình dạng Tensor (`Shape`)](#quy-tắc-2-công-thức-toán-học--ký-hiệu-hình-dạng-tensor)
3. [Quy tắc 3: Hình ảnh từ Bài báo gốc & Cách chú thích](#quy-tắc-3-hình-ảnh-từ-bài-báo-gốc--cách-chú-thích)
4. [Quy tắc 4: Cấu trúc Nội dung bắt buộc cho từng loại Slide](#quy-tắc-4-cấu-trúc-nội-dung-bắt-buộc-cho-từng-loại-slide)
5. [Quy tắc 5: Phần Đề xuất Cải tiến & Cơ sở Thuyết phục](#quy-tắc-5-phần-đề-xuất-cải-tiến--cơ-sở-thuyết-phục)
6. [Checklist Nghiệm thu trước khi Commit](#checklist-nghiệm-thu-trước-khi-commit)

---

## 🎨 Quy tắc 1: Khung hình, Typography & Card Layout

* **Chuẩn khung hình thống nhất:** Toàn bộ slide sử dụng tỷ lệ màn hình rộng **16:9** (`13.333 × 7.5 in`, độ phân giải tối thiểu **180 DPI**).
* **Đồng bộ Header & Footer trên 100% slide:**
  * **Header:** Có huy hiệu phần (`PHẦN 1 / 2 / 3`) cùng bảng màu nhất quán theo từng phần; tiêu đề chính thống nhất cùng 1 cỡ chữ (`15 pt` in đậm).
  * **Footer bắt buộc ghi rõ nguồn gốc cụ thể:** Phải ghi chính xác vị trí xuất xứ của nội dung trên bài báo (Tên mục, Số `Figure`/`Table`, Số trang) và/hoặc vị trí dòng code trong repository (VD: `models.py#L141-L214`, `simulate_forward_pass.py#L228-L266`).
  * **Đánh số slide rõ ràng:** Góc dưới bên phải bắt buộc có huy hiệu đánh số liên tục dạng `Slide XX / NN` (VD: `Slide 01 / 11` → `Slide 11 / 11`).
* **Typography & Khoảng cách trong Card (Rất quan trọng):**
  * **Cỡ chữ đủ lớn để chiếu hội trường:** Nội dung text trong card tối thiểu **`10.1–10.8 pt`**; tiêu đề card **`11.2–11.8 pt`**; khoảng cách dòng thoáng (`linespacing = 1.48–1.58`). Tránh viết chữ quá nhỏ hoặc nhồi quá nhiều chữ vào 1 hộp.
  * **Thu hẹp khoảng cách giữa Title và Text:** Nội dung trong card phải được neo từ trên xuống (`va='top'`) ngay bên dưới tiêu đề (`Title`), tuyệt đối không căn giữa theo chiều dọc khiến phần đầu và thân card bị tách bởi khoảng trống lớn thừa thãi.
  * **Card vừa khít nội dung:** Điều chỉnh kích thước card (*fit*) cân đối với lượng chữ, giữ lề đồng đều giữa các cột/hàng.
  * **Sạch cú pháp thô:** Khi render ảnh PNG (Matplotlib/Pillow), tuyệt đối không để sót ký tự Markdown thô như `**in đậm**` hay `` `code` ``.

---

## 📐 Quy tắc 2: Công thức Toán học & Ký hiệu Hình dạng Tensor

* **Toán học chuẩn Equation:** Mọi biểu thức toán trên slide và trong Markdown phải hiển thị đúng dạng phương trình LaTeX/MathText ($\hat{\mathbf{y}}_{\mathrm{NL}, L+t}^{(i)}$, $\mathbf{W}_{\mathrm{res}}\mathbf{y}_{1:L}^{(i)}$, $\alpha_t \in (0, 1)$), không viết dạng văn bản thô thiếu định dạng.
* **Bắt buộc ghi rõ chữ `Shape:`:** Mỗi khi mô tả kích thước tensor trong sơ đồ hoặc bảng, phải ghi tường minh chữ **`Shape:`** đi kèm (VD: `Shape: (B, H)`, `Shape: (B, L)`, `Shape: (B, H, p + r_tilde)`).
* **Ưu tiên Shape tổng quát kèm chú giải chiều:** Không chỉ ghi số cứng gây khó hiểu (như `(32, 720)` hay `(32, 96)` đứng trơ trọi). Phải trình bày **shape tổng quát bằng ký hiệu** kèm định nghĩa các chiều (`B = Batch`, `L = Lookback`, `H = Horizon`, `p = DecoderDim`, `r_tilde = CovWidth`); nếu muốn minh họa số thực tế thì chỉ ghi thêm một dòng ví dụ phụ bên dưới (VD: `VD Electricity: B=512, L=720, H=96`).

---

## 🖼️ Quy tắc 3: Hình ảnh từ Bài báo gốc & Cách chú thích

* **Ưu tiên ảnh gốc từ Bài báo:** Khi kiến trúc, đồ thị hoặc bảng số liệu đã có sẵn trong bài báo (như `Figure 1`, `Figure 3`, `Table 5`, `Table 8`), **bắt buộc cắt trực tiếp từ bài báo** với độ phân giải cao. Chỉ tự thiết kế sơ đồ bổ trợ khi bài báo không có hình minh họa riêng.
* **Không bê nguyên văn đoạn caption tiếng Anh dài của bài báo lên slide:** Chỉ cắt phần hình vẽ hoặc bảng số liệu sạch, sau đó đặt các **thẻ nhận xét ngắn gọn, súc tích bằng tiếng Việt** bên cạnh hoặc bên dưới để khán giả nắm ngay ý chính (cách đọc bảng, con số nổi bật, kết luận thực nghiệm).

---

## 📑 Quy tắc 4: Cấu trúc Nội dung bắt buộc cho từng loại Slide

* **Slide mở đầu — Thuật ngữ cốt lõi (`1.0`):**
  * Phải giải thích rõ ràng các thuật ngữ nền tảng trước khi đi sâu vào kiến trúc (VD: `MLPResidual`, `Dense Encoder`, `Dense Decoder`, `Temporal Decoder`, `Covariate Highway`).
  * Trình bày bằng các **thẻ độc lập tách bạch, không vẽ mũi tên nối rườm rà** giữa các card định nghĩa.
* **Slide định vị kiến trúc (`1.1`):**
  * Khoanh vùng chính xác phạm vi mình trình bày trên sơ đồ tổng thể (`Figure 1`).
  * **Loại bỏ mọi nhãn thừa** không liên quan đến nội dung chiếu (như *"Nội dung nói (1), (2)"* hay *"PHẦN DƯỚI (của thành viên khác)"*).
* **Slide so sánh cơ chế (`1.2`):**
  * Không lặp lại các khối định nghĩa đã giải thích ở Slide mở đầu.
  * **Chia 2 cột đối chiếu trực diện** giữa *"KHÔNG CÓ"* và *"CÓ"* cơ chế đang phân tích để làm nổi bật lý do thiết kế (VD: `Covariate Highway`).
* **Slide Source Code (`2.1`):**
  * Show nguyên đoạn code liên quan trong repository chính thức (`models.py`) kèm số dòng cụ thể, **đánh dấu nổi bật phần đang phân tích và ánh xạ mũi tên trực tiếp** sang từng phương trình toán học tương ứng.
* **Slide Hyperparameters (`2.2`):**
  * Bắt buộc có bảng cấu hình siêu tham số chính thức của các benchmark (`Table 8`, `Table 7`), khoanh rõ đâu là siêu tham số chung (*Global*) và đâu là siêu tham số riêng của cơ chế đang thuyết trình.
* **Slide Ablation Study (`2.3–2.4`):**
  * Đưa ra ảnh/bảng số liệu gốc + nhận xét trực tiếp nguyên nhân và ý nghĩa con số.
  * Nếu bài báo có 7 benchmark nhưng phần Ablation chỉ báo cáo trên 1 tập (như `Electricity` ở Table 5), phải **giải thích rõ lý do ngay trên slide** để tránh gây hiểu nhầm.

---

## 💡 Quy tắc 5: Phần Đề xuất Cải tiến & Cơ sở Thuyết phục

* **Chỉ chọn DUY NHẤT 1 đề xuất tốt nhất:** Không dàn trải nhiều hướng mờ nhạt; tập trung phân tích sâu giải pháp có tính khả thi và tác động cao nhất.
* **Bắt buộc trả lời thuyết phục 3 câu hỏi cốt lõi ngay trên slide:**
  1. **Lấy từ bài báo nào?** (Trích dẫn chính xác tên bài báo, tác giả, năm và cơ chế kế thừa — VD: *Gated Residual Network trong TFT, Lim et al., IJF 2021*; *Highway Networks, Srivastava et al., 2015*).
  2. **Vì sao chọn phương án này là tốt nhất?** (Nêu rõ điểm đánh trúng kiến trúc, chi phí tham số/tính toán tối thiểu, và giá trị cộng thêm như khả năng giải thích $\alpha_t$).
  3. **Nó sẽ cải thiện ở điểm nào trong mô hình hiện tại?** (Chỉ đích danh dòng code tắc nghẽn như `models.py#L214`, các benchmark yếu thế tại `Table 2`/`Figure 3`/`Table 5` và khoảng trống Ablation cần hoàn thiện kèm lộ trình kiểm chứng cụ thể).

---

## ✅ Checklist Nghiệm thu trước khi Commit

- [ ] Tất cả slide dùng chung chuẩn **16:9**, đồng nhất Header/Footer, cỡ chữ nội dung `>= 10 pt` và khoảng cách Title-to-Body gọn chặt.
- [ ] Đã đánh số thứ tự chính xác `Slide 01 / NN` → `Slide NN / NN` cả trong ảnh PNG lẫn toàn bộ file `.md` (`README.md`, `slide_01..03`, `slide_images_gallery.md`, `script_thuyet_trinh_tung_slide.md`).
- [ ] Đã mở kiểm tra trực quan **từng ảnh PNG** (không chữ tràn viền, không mũi tên đè chữ, không ký tự `**` / `` ` `` thô).
- [ ] Đã kiểm tra tự động toàn bộ liên kết `./images/*.png` không bị gãy link nào và tuân thủ [`RULE.md`](../../../RULE.md).
- [ ] Đã đồng bộ sang thư mục mirror và đẩy đầy đủ lên nhánh GitHub chính thức.
