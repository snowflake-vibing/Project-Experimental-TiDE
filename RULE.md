# 📌 RULE.md - BỘ QUY TẮC BẮT BUỘC DÀNH CHO AI KHI THAO TÁC TRÊN REPOSITORY

> [!IMPORTANT]
> **THÔNG ĐIỆP BẮT BUỘC DÀNH CHO TẤT CẢ CÁC AI AGENT (Antigravity, Cursor, Copilot, ChatGPT, Claude, Gemini, v.v.):**  
> Trước khi tạo mới, chỉnh sửa hoặc tạo tài liệu/code trong repository này, AI **BẮT BUỘC PHẢI ĐỌC VÀ TUÂN THỦ TUYỆT ĐỐI 100%** các quy tắc dưới đây. Không được tự ý vi phạm hoặc bỏ qua bất kỳ quy tắc nào.

---

## 🧭 MỤC LỤC
1. [Quy Tắc 1: Ngôn Ngữ Tiếng Việt Chuẩn & Mermaid Diagrams](#quy-tắc-1-ngôn-ngữ-tiếng-việt-chuẩn--mermaid-diagrams)
2. [Quy Tắc 2: Định Dạng Công Thức Toán Học (LaTeX / KaTeX trên GitHub)](#quy-tắc-2-định-dạng-công-thức-toán-học-latex--katex-trên-github)
3. [Quy Tắc 3: Cấu Trúc Tài Liệu Kiến Trúc & Pipeline (3 Phần Bắt Buộc)](#quy-tắc-3-cấu-trúc-tài-liệu-kiến-trúc--pipeline-3-phần-bắt-buộc)
4. [Quy Tắc 4: Chuẩn Mực Code & Kiểm Thử Tensor](#quy-tắc-4-chuẩn-mực-code--kiểm-thử-tensor)

---

## 🎯 Quy Tắc 1: Ngôn Ngữ Tiếng Việt Chuẩn & Mermaid Diagrams

* **Bắt buộc dùng Tiếng Việt có dấu:** Toàn bộ tài liệu, giải thích, đề mục, chú thích code và sơ đồ phải được viết bằng Tiếng Việt chuẩn có dấu (UTF-8).
* **Nghiêm cấm viết Tiếng Việt không dấu:** Tuyệt đối không viết kiểu `Khoi du bao`, `Khong gian an`, phải viết đầy đủ: `Khối dự báo`, `Không gian ẩn`.
* **Quy chuẩn khi viết sơ đồ Mermaid:**
  * Toàn bộ nhãn node, text chú thích, văn bản trên mũi tên phải là **Tiếng Việt có dấu**.
  * Để tránh lỗi phân tích cú pháp (syntax error) của Mermaid khi có dấu cách, dấu tiếng Việt hoặc ký tự đặc biệt, **BẮT BUỘC phải bọc toàn bộ nội dung nhãn trong cặp dấu ngoặc kép `["..."]`**.
  
  ```mermaid
  %% ✅ ĐÚNG: Nhãn Tiếng Việt có dấu, bọc trong ngoặc kép ["..."]
  graph TD
      A["Dữ liệu chuỗi lịch sử (Lookback)"] --> B["Chuẩn hóa RevIN"]
      B --> C["Bộ mã hóa Dense Encoder"]
      C --> D["Kết quả dự báo tương lai"]
  ```

  ```mermaid
  %% ❌ SAI: Tiếng Việt không dấu, không dùng ngoặc kép
  graph TD
      A[Du lieu qua khu] --> B[Chuan hoa]
  ```

---

## 📐 Quy Tắc 2: Định Dạng Công Thức Toán Học (LaTeX / KaTeX trên GitHub)

GitHub sử dụng trình phân tích Markdown kết hợp MathJax/KaTeX. Để tránh lỗi công thức bị hiển thị dưới dạng văn bản thô `$$...$$` hoặc bị vỡ layout, AI phải tuân thủ nghiêm ngặt:

### 1. Khối công thức hiển thị riêng (`$$...$$` Display Math)
* **Tuyệt đối không thụt lùi dòng (No Indentation):** Ký hiệu `$$` mở đầu và `$$` kết thúc **bắt buộc phải nằm ở cột đầu tiên (Cột 0 - sát lề trái)**.
* **Luôn có dòng trống cách ly:** Bắt buộc phải có **1 dòng trống trước `$$`** và **1 dòng trống sau `$$`**.
* **Nghiêm cấm lồng `$$...$$` vào danh sách:** KHÔNG BAO GIỜ đặt `$$...$$` bên trong danh sách thụt lùi (`- `, `* `, `1. `, `2. `). Nếu cần trình bày theo từng bước, hãy dùng tiêu đề in đậm riêng biệt (ví dụ: `**Bước 1: Tính toán kỳ vọng:**`).

```markdown
<!-- ✅ ĐÚNG: Sát lề trái, có dòng trống cách ly, dùng tiêu đề in đậm -->
**Bước 1: Tính toán giá trị trung bình:**

$$
\mu^{(i)} = \frac{1}{L} \sum_{t=1}^L y_t^{(i)}
$$

**Bước 2: Chuẩn hóa dữ liệu:**

$$
y_{\text{norm}, t}^{(i)} = \frac{y_t^{(i)} - \mu^{(i)}}{\sigma^{(i)}}
$$
```

```markdown
<!-- ❌ SAI: Thụt lùi bên trong danh sách đánh số hoặc bullet point (GitHub sẽ render ra text thô $$) -->
1. Bước 1: Tính toán trung bình:
   $$\mu = \frac{1}{L} \sum y_t$$
2. Bước 2: Chuẩn hóa:
   $$y_{\text{norm}} = \frac{y - \mu}{\sigma}$$
```

### 2. Công thức toán nội dòng (`$...$` Inline Math)
* **Không bọc `$math$` bên trong in nghiêng hoặc in đậm:** Ví dụ `*($\tilde{r}=4$)*` hoặc `**$y_t$**` sẽ làm trình biên dịch Markdown hiểu nhầm dấu `*` và ngắt gãy KaTeX.  
  * 👉 **Đúng:** `với số chiều $\tilde{r} = 4$` hoặc đặt trong ngoặc thường `($\tilde{r} = 4$)`.
* **Biểu diễn kích thước & phép tính số học:**
  * Với các biểu thức kích thước tensor hoặc phép nhân số nguyên (ví dụ: `(720 + 96) * 4 = 3,264`), hãy ưu tiên dùng inline code block: `` `(720 + 96) * 4 = 3,264` `` hoặc công thức có khoảng cách rõ ràng `$(720 + 96) \times 4 = 3264$`.
  * Với các định danh tham số, hãy dùng mã code: `` `hidden_size = 1024` ``, `` `temporal_width = 4` ``, tránh viết dính liền kiểu `$\text{hiddenSize}=1024$`.

---

## 🏛️ Quy Tắc 3: Cấu Trúc Tài Liệu Kiến Trúc & Pipeline (3 Phần Bắt Buộc)

Khi tài liệu hóa kiến trúc, quy trình xử lý, hoặc luồng dữ liệu (Forward Pass), **tuyệt đối không làm sơ sài, qua loa**. Đối với mỗi giai đoạn biến đổi dữ liệu, AI bắt buộc phải phân tích đầy đủ theo cấu trúc 3 phần chuẩn sau:

1. **`#### 📥 1. Input (Đầu vào)`**:
   * Liệt kê đầy đủ mọi tensor tiếp nhận.
   * Ghi rõ ý nghĩa vật lý/nghiệp vụ của tensor.
   * Ghi rõ kích thước toán học và shape cụ thể trong code: ví dụ `(B, L) = (32, 720)`.
2. **`#### ⚙️ 2. Công thức xử lý`**:
   * Công thức toán học tường minh, định nghĩa đầy đủ các biến và hằng số (ví dụ $\epsilon = 10^{-5}$).
   * Phân tích bản chất kỹ thuật: Tại sao kiến trúc lại chọn phép biến đổi này? Tác dụng giảm chiều, chống trôi phân phối, hay kết nối tắt (Highway/Skip) như thế nào so với Paper gốc?
   * Tính toán số chiều cụ thể qua từng bước.
3. **`#### 📤 3. Output (Đầu ra)`**:
   * Kích thước tensor đầu ra chính xác `(B, ...)`.
   * Ý nghĩa biểu diễn của vector/ma trận vừa sinh ra.
   * Chỉ rõ tensor này sẽ được chuyển giao đến giai đoạn nào tiếp theo trong mạng.

---

## 💻 Quy Tắc 4: Chuẩn Mực Code & Kiểm Thử Tensor

* **Đồng bộ với Paper:** Mọi thiết lập siêu tham số (`lookback L`, `horizon H`, `hidden_size`, `r`, `p`, `rev_in`) phải bám sát bảng thực nghiệm chính thức từ bài báo khoa học (*Table 8 trong paper TiDE*).
* **Kiểm thử kích thước (Shape Verification):** Khi viết code mô phỏng hay kiểm thử, luôn in ra shape tại từng bước chuyển giao tensor để đối chiếu với tài liệu lý thuyết.
* **Tối giản & Sạch sẽ:** Không tạo tràn lan các file python trùng lặp chức năng, giữ cấu trúc thư mục gọn gàng, có tài liệu hướng dẫn và liên kết rõ ràng.
* **Đồng bộ hóa:** Nếu có bản sao tài liệu ở các thư mục dự án phụ thuộc, phải đồng bộ nội dung ngay lập tức.
