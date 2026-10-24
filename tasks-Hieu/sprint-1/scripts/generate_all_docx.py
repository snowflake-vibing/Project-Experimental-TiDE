# -*- coding: utf-8 -*-
import os
import docx
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import nsdecls, qn

def set_cell_background(cell, hex_color):
    shading_elm = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{hex_color}"/>')
    cell._tc.get_or_add_tcPr().append(shading_elm)

def set_cell_margins(cell, top=100, bottom=100, left=150, right=150):
    tcPr = cell._tc.get_or_add_tcPr()
    tcMar = OxmlElement('w:tcMar')
    for m, val in [('top', top), ('bottom', bottom), ('left', left), ('right', right)]:
        node = OxmlElement(f'w:{m}')
        node.set(qn('w:w'), str(val))
        node.set(qn('w:type'), 'dxa')
        tcMar.append(node)
    tcPr.append(tcMar)

def set_cell_border_left_only(cell, color="1B365D", sz="24"):
    tcPr = cell._tc.get_or_add_tcPr()
    tcBorders = parse_xml(f'''
        <w:tcBorders {nsdecls("w")}>
            <w:top w:val="none"/>
            <w:left w:val="single" w:sz="{sz}" w:space="0" w:color="{color}"/>
            <w:bottom w:val="none"/>
            <w:right w:val="none"/>
        </w:tcBorders>
    ''')
    tcPr.append(tcBorders)

def set_table_borders(table, color="D3D3D3"):
    tblPr = table._tbl.tblPr
    borders = parse_xml(f'''
        <w:tblBorders {nsdecls("w")}>
            <w:top w:val="single" w:sz="4" w:space="0" w:color="{color}"/>
            <w:left w:val="single" w:sz="4" w:space="0" w:color="{color}"/>
            <w:bottom w:val="single" w:sz="4" w:space="0" w:color="{color}"/>
            <w:right w:val="single" w:sz="4" w:space="0" w:color="{color}"/>
            <w:insideH w:val="single" w:sz="4" w:space="0" w:color="{color}"/>
            <w:insideV w:val="single" w:sz="4" w:space="0" w:color="{color}"/>
        </w:tblBorders>
    ''')
    tblPr.append(borders)

def setup_document(doc):
    for section in doc.sections:
        section.top_margin = Inches(0.8)
        section.bottom_margin = Inches(0.8)
        section.left_margin = Inches(0.8)
        section.right_margin = Inches(0.8)
    
    normal_style = doc.styles['Normal']
    normal_style.font.name = 'Calibri'
    normal_style.font.size = Pt(11)
    normal_style.font.color.rgb = RGBColor(0x22, 0x22, 0x22)
    normal_style.paragraph_format.line_spacing = 1.15
    normal_style.paragraph_format.space_after = Pt(4)

def add_title(doc, title_text, subtitle=None):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.add_run(title_text)
    run.font.name = 'Calibri'
    run.font.size = Pt(17)
    run.font.bold = True
    run.font.color.rgb = RGBColor(0x1B, 0x36, 0x5D)
    p.paragraph_format.space_before = Pt(6)
    p.paragraph_format.space_after = Pt(4)
    
    if subtitle:
        p2 = doc.add_paragraph()
        p2.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run2 = p2.add_run(subtitle)
        run2.font.name = 'Calibri'
        run2.font.size = Pt(11)
        run2.font.italic = True
        run2.font.color.rgb = RGBColor(0x55, 0x55, 0x55)
        p2.paragraph_format.space_after = Pt(14)

def add_heading_1(doc, text):
    p = doc.add_paragraph()
    run = p.add_run(text)
    run.font.name = 'Calibri'
    run.font.size = Pt(13.5)
    run.font.bold = True
    run.font.color.rgb = RGBColor(0x1B, 0x36, 0x5D)
    p.paragraph_format.space_before = Pt(14)
    p.paragraph_format.space_after = Pt(4)

def add_heading_2(doc, text):
    p = doc.add_paragraph()
    run = p.add_run(text)
    run.font.name = 'Calibri'
    run.font.size = Pt(12)
    run.font.bold = True
    run.font.color.rgb = RGBColor(0x2A, 0x4B, 0x7C)
    p.paragraph_format.space_before = Pt(10)
    p.paragraph_format.space_after = Pt(3)

def add_heading_3(doc, text):
    p = doc.add_paragraph()
    run = p.add_run(text)
    run.font.name = 'Calibri'
    run.font.size = Pt(11)
    run.font.bold = True
    run.font.color.rgb = RGBColor(0x3B, 0x59, 0x98)
    p.paragraph_format.space_before = Pt(6)
    p.paragraph_format.space_after = Pt(2)

def add_paper_location_box(doc, location_text, note_text=None):
    """
    Theo đúng yêu cầu của người dùng:
    - Vị trí bài báo: CHỈ GHI VỊ TRÍ, KHÔNG GHI NỘI DUNG NGUYÊN VĂN.
    """
    table = doc.add_table(rows=1, cols=1)
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = False
    
    cell = table.cell(0, 0)
    cell.width = Inches(6.9)
    set_cell_background(cell, "F0F4F8")
    set_cell_border_left_only(cell, color="1B365D", sz="24")
    set_cell_margins(cell, top=100, bottom=100, left=160, right=160)
    
    p = cell.paragraphs[0]
    p.paragraph_format.space_after = Pt(2)
    p.paragraph_format.line_spacing = 1.15
    run_icon = p.add_run("📄 Vị trí trong bài báo gốc: ")
    run_icon.bold = True
    run_icon.font.color.rgb = RGBColor(0x1B, 0x36, 0x5D)
    
    run_loc = p.add_run(location_text)
    run_loc.font.color.rgb = RGBColor(0x22, 0x22, 0x22)
    
    if note_text:
        p2 = cell.add_paragraph()
        p2.paragraph_format.space_before = Pt(2)
        p2.paragraph_format.space_after = Pt(0)
        p2.paragraph_format.line_spacing = 1.15
        run_note_title = p2.add_run("💡 Ghi chú kỹ thuật: ")
        run_note_title.bold = True
        run_note_title.font.color.rgb = RGBColor(0x2A, 0x4B, 0x7C)
        run_note = p2.add_run(note_text)
        run_note.font.color.rgb = RGBColor(0x33, 0x33, 0x33)

    p_after = doc.add_paragraph()
    p_after.paragraph_format.space_before = Pt(2)
    p_after.paragraph_format.space_after = Pt(2)

def add_bullet(doc, title, content):
    p = doc.add_paragraph(style='List Bullet')
    p.paragraph_format.space_after = Pt(2)
    p.paragraph_format.line_spacing = 1.15
    run_title = p.add_run(title)
    run_title.bold = True
    run_title.font.color.rgb = RGBColor(0x1B, 0x36, 0x5D)
    run_content = p.add_run(content)
    run_content.font.color.rgb = RGBColor(0x22, 0x22, 0x22)

def add_code_block(doc, code_str):
    table = doc.add_table(rows=1, cols=1)
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = False
    cell = table.cell(0, 0)
    cell.width = Inches(6.9)
    set_cell_background(cell, "F6F8FA")
    set_cell_margins(cell, top=90, bottom=90, left=140, right=140)
    
    tcPr = cell._tc.get_or_add_tcPr()
    borders = parse_xml(f'''
        <w:tcBorders {nsdecls("w")}>
            <w:top w:val="single" w:sz="4" w:space="0" w:color="D0D7DE"/>
            <w:left w:val="single" w:sz="4" w:space="0" w:color="D0D7DE"/>
            <w:bottom w:val="single" w:sz="4" w:space="0" w:color="D0D7DE"/>
            <w:right w:val="single" w:sz="4" w:space="0" w:color="D0D7DE"/>
        </w:tcBorders>
    ''')
    tcPr.append(borders)
    
    p = cell.paragraphs[0]
    p.paragraph_format.space_after = Pt(0)
    p.paragraph_format.line_spacing = 1.0
    run = p.add_run(code_str.strip())
    run.font.name = 'Consolas'
    run.font.size = Pt(9.0)
    run.font.color.rgb = RGBColor(0x24, 0x29, 0x2F)
    
    p_after = doc.add_paragraph()
    p_after.paragraph_format.space_before = Pt(2)
    p_after.paragraph_format.space_after = Pt(2)

def add_plain_text_diagram(doc, diagram_str, caption=None):
    table = doc.add_table(rows=1, cols=1)
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = False
    cell = table.cell(0, 0)
    cell.width = Inches(6.9)
    set_cell_background(cell, "F8F9FA")
    set_cell_margins(cell, top=100, bottom=100, left=140, right=140)
    
    tcPr = cell._tc.get_or_add_tcPr()
    borders = parse_xml(f'''
        <w:tcBorders {nsdecls("w")}>
            <w:top w:val="single" w:sz="8" w:space="0" w:color="1B365D"/>
            <w:left w:val="single" w:sz="8" w:space="0" w:color="1B365D"/>
            <w:bottom w:val="single" w:sz="8" w:space="0" w:color="1B365D"/>
            <w:right w:val="single" w:sz="8" w:space="0" w:color="1B365D"/>
        </w:tcBorders>
    ''')
    tcPr.append(borders)
    
    p = cell.paragraphs[0]
    p.paragraph_format.space_after = Pt(0)
    p.paragraph_format.line_spacing = 1.0
    run = p.add_run(diagram_str.strip())
    run.font.name = 'Consolas'
    run.font.size = Pt(8.5)
    run.font.color.rgb = RGBColor(0x1B, 0x36, 0x5D)
    
    if caption:
        p_cap = doc.add_paragraph()
        p_cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run_cap = p_cap.add_run(caption)
        run_cap.font.name = 'Calibri'
        run_cap.font.size = Pt(9.5)
        run_cap.font.italic = True
        run_cap.font.color.rgb = RGBColor(0x55, 0x55, 0x55)
        p_cap.paragraph_format.space_before = Pt(3)
        p_cap.paragraph_format.space_after = Pt(6)

def add_math_formula(doc, formula_str):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(4)
    p.paragraph_format.space_after = Pt(6)
    run = p.add_run(formula_str)
    run.font.name = 'Cambria Math'
    run.font.size = Pt(11)
    run.font.bold = True
    run.font.color.rgb = RGBColor(0x11, 0x22, 0x44)

def add_styled_table(doc, headers, data_rows, col_widths=None):
    table = doc.add_table(rows=len(data_rows) + 1, cols=len(headers))
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = False
    set_table_borders(table, color="C0C8D0")
    
    # Header row
    hdr_cells = table.rows[0].cells
    for i, title in enumerate(headers):
        hdr_cells[i].text = title
        set_cell_background(hdr_cells[i], "1B365D")
        set_cell_margins(hdr_cells[i], top=90, bottom=90, left=100, right=100)
        p = hdr_cells[i].paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        for r in p.runs:
            r.font.name = 'Calibri'
            r.font.size = Pt(9.5)
            r.font.bold = True
            r.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
            
    # Data rows
    for r_idx, row in enumerate(data_rows):
        row_cells = table.rows[r_idx + 1].cells
        bg_color = "F7F9FB" if r_idx % 2 == 1 else "FFFFFF"
        for c_idx, val in enumerate(row):
            row_cells[c_idx].text = str(val)
            set_cell_background(row_cells[c_idx], bg_color)
            set_cell_margins(row_cells[c_idx], top=70, bottom=70, left=90, right=90)
            p = row_cells[c_idx].paragraphs[0]
            p.paragraph_format.space_after = Pt(0)
            p.paragraph_format.line_spacing = 1.1
            for r in p.runs:
                r.font.name = 'Calibri'
                r.font.size = Pt(9.0)
                r.font.color.rgb = RGBColor(0x22, 0x22, 0x22)
                
    # Column widths
    if col_widths:
        for row in table.rows:
            for c_idx, w in enumerate(col_widths):
                row.cells[c_idx].width = Inches(w)
                
    p_after = doc.add_paragraph()
    p_after.paragraph_format.space_before = Pt(2)
    p_after.paragraph_format.space_after = Pt(4)


# ==========================================
# BUILD DOCUMENT 1: ARCHITECTURE DIAGRAM
# ==========================================
def generate_doc_1(output_path, image_path):
    doc = docx.Document()
    setup_document(doc)
    
    add_title(doc, 
              "PHẦN 1: SƠ ĐỒ KIẾN TRÚC TỔNG THỂ & GIẢI PHẪU KẾT CẤU TĨNH CỦA TiDE (STATIC ARCHITECTURE)",
              "Bản vẽ thiết kế kết cấu phần cứng tĩnh • def __init__(self, ...) • TiDE Google Research (2023)")
    
    # Metadata Box
    add_paper_location_box(doc, 
                           "Trang 5, Figure 1 & Mục 4 \"Architecture\"; Phụ lục B.1 & B.3",
                           "Tài liệu mô tả chi tiết 7 mô-đun phần cứng tĩnh, ma trận trọng số và bảng tham số phục vụ lập trình mô hình TiDE.")
    
    add_heading_1(doc, "🧭 Tôn Chỉ Thiết Kế: Bản Vẽ Kết Cấu Tĩnh (Static Blueprint)")
    p = doc.add_paragraph("Trong kỹ thuật phần mềm và Học sâu (Deep Learning):")
    add_bullet(doc, "Phần 1 (Tài liệu này): ", 
               "Đóng vai trò như Bản vẽ thiết kế kết cấu phần cứng tĩnh: Trả lời câu hỏi \"Mô hình gồm những linh kiện/mô-đun nào, bên trong từng linh kiện cấu tạo ra sao, ma trận trọng số có kích thước thế nào, và chúng được đấu nối với nhau ra sao trên không gian?\" (Tương ứng với hàm khởi tạo def __init__(self, ...) trong PyTorch).")
    add_bullet(doc, "Phần 2 (Quy trình vận hành động): ", 
               "Đóng vai trò như Quy trình vận hành của dây chuyền sản xuất: Trả lời câu hỏi \"Dữ liệu và Tensor di chuyển qua các linh kiện theo thứ tự nào, hình dạng tensor (shape) biến đổi ra sao qua từng dòng code?\" (Tương ứng với hàm thực thi def forward(self, y, x, a) trong PyTorch).")
    
    add_heading_1(doc, "1. Sơ Đồ Kiến Trúc Gốc Của TiDE (Figure 1 từ Google Research)")
    add_paper_location_box(doc, "Trang 5, Figure 1 (Chính giữa trang) & Mục 4 \"Architecture\".")
    
    if os.path.exists(image_path):
        p_img = doc.add_paragraph()
        p_img.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p_img.paragraph_format.space_before = Pt(6)
        p_img.paragraph_format.space_after = Pt(2)
        doc.add_picture(image_path, width=Inches(6.4))
        p_cap = doc.add_paragraph()
        p_cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r_cap = p_cap.add_run("Hình 1: Tổng quan sơ đồ kiến trúc của mô hình TiDE (Time-series Dense Encoder) từ Google Research gồm 5 khối mô-đun chính và cấu trúc chi tiết của khối cơ sở Residual Block (khung bên phải).")
        r_cap.font.italic = True
        r_cap.font.size = Pt(9.5)
        r_cap.font.color.rgb = RGBColor(0x55, 0x55, 0x55)
    
    add_heading_2(doc, "Sơ Đồ Khối Kiến Trúc Tổng Thể (Vẽ Bằng Văn Bản Thuần)")
    topology_ascii = """┌─────────────────────────────────────────────────────────────────────────────┐
│                       1. CÁC CỔNG DỮ LIỆU ĐẦU VÀO                            │
│  [Chuỗi Lịch Sử y(1:L)]     [Ngoại Sinh Động x(1:L+H)]    [Thuộc Tính Tĩnh a]│
└────────────┬─────────────────────────────┬──────────────────────────┬───────┘
             │                             │                          │
             ▼                             ▼                          │
┌──────────────────────────┐  ┌──────────────────────────┐            │
│ Cổng RevIN (Chuẩn Hóa)   │  │ Feature Projection (Res) │            │
│      y_norm = (y-μ)/σ    │  │       r  ──►  r_tilde    │            │
└────────────┬─────────────┘  └────────────┬─────────────┘            │
             │                             │                          │
             ├─────────────────────────────┼──────────────────────────┘
             │                             ▼
             │                ┌──────────────────────────┐
             │                │ Siêu Vector: enc_in      │
             │                │ [y_norm ; vec(x_proj) ; a]│
             │                └────────────┬─────────────┘
             │                             ▼
             │                ┌──────────────────────────┐
             │                │ Dense Encoder (n_e khối) │
             │                │ d_enc  ──►  d_hidden     │
             │                └────────────┬─────────────┘
             │                             ▼
             │                ┌──────────────────────────┐
             │                │ Dense Decoder (n_d khối) │
             │                │ d_hidden  ──►  H · p     │
             │                └────────────┬─────────────┘
             │                             ▼
             │                ┌──────────────────────────┐   ┌──────────────────────────┐
             │                │ Unflatten / Reshape      │   │ Ngoại sinh tương lai     │
             │                │ Ma trận D: (H x p)       │   │ x_proj(L+1:L+H): (H x r̃) │
             │                └────────────┬─────────────┘   └────────────┬─────────────┘
             │                             │                              │
             │                             └──────────────┬───────────────┘
             │                                            ▼
             │                               ┌──────────────────────────┐
             │                               │ Temporal Decoder         │
             │                               │ [d_t ; x̃_future] ──► y_NL │
             │                               └────────────┬─────────────┘
             ▼                                            │
┌──────────────────────────┐                              │
│ Global Linear Skip       │                              │
│ W_skip · y_norm + b_skip │                              │
└────────────┬─────────────┘                              │
             │                                            │
             └──────────────────────┬─────────────────────┘
                                    ▼
                         ┌──────────────────────┐
                         │ Phép Cộng Thặng Dư   │
                         │ y_hat_norm = NL + Lin│
                         └──────────┬───────────┘
                                    ▼
                         ┌──────────────────────┐
                         │ Cổng RevIN (Giải Chuẩn)│
                         │ y_hat = y_hat_norm·σ+μ│
                         └──────────┬───────────┘
                                    ▼
                         ┌──────────────────────┐
                         │ DỰ BÁO HOÀN CHỈNH    │
                         │   y_hat(L+1:L+H)     │
                         └──────────────────────┘"""
    add_plain_text_diagram(doc, topology_ascii, "Sơ đồ kiến trúc tổng thể của TiDE (Bản vẽ phẳng văn bản thuần)")

    add_heading_1(doc, "2. Giải Phẫu Cấu Tạo Chi Tiết Từng Mô-Đun Tĩnh (Static Module Anatomy)")
    doc.add_paragraph("Toàn bộ mô hình TiDE được cấu thành từ 7 mô-đun tĩnh độc lập:")

    # Mô-đun 1
    add_heading_2(doc, "🧩 Mô-Đun 1: Khối Nơ-ron Thặng Dư Cơ Sở (Residual Block - Khung Bên Phải Hình 1)")
    add_paper_location_box(doc, "Trang 5, Cột trái, Mục 4.1 \"Residual Block\", Đoạn 2; Hình 1 (Khung bên phải); Phụ lục B.3, Trang 18, Bảng 10.")
    
    resblock_ascii = """                       ┌───────────────────────────────┐
                       │    Đầu Vào u (Chiều d_in)     │
                       └──────┬─────────────────┬──────┘
                              │                 │
    [Nhánh Biến Đổi Phi Tuyến]│                 │ [Nhánh Skip Connection]
                              ▼                 │
               ┌──────────────────────────────┐ │
               │ Tầng Tuyến Tính 1 (Dense 1): │ │
               │ W1 (d_hidden x d_in), b1     │ │
               └──────────────┬───────────────┘ │
                              ▼                 │
               ┌──────────────────────────────┐ │
               │ Kích Hoạt Phi Tuyến ReLU     │ │
               │ max(0, z)                    │ │
               └──────────────┬───────────────┘ │
                              ▼                 │
               ┌──────────────────────────────┐ │
               │ Tầng Tuyến Tính 2 (Dense 2): │ │
               │ W2 (d_out x d_hidden), b2    │ │
               └──────────────┬───────────────┘ │
                              ▼                 │
               ┌──────────────────────────────┐ │ (Chiếu W_skip nếu d_in ≠ d_out,
               │ Tầng Dropout                 │ │  hoặc Identity nếu d_in == d_out)
               │ Xác suất p_drop              │ │
               └──────────────┬───────────────┘ │
                              │                 │
                              ▼                 ▼
                       ┌───────────────────────────────┐
                       │        Phép Cộng (+)          │
                       │     z = F(u) + Skip(u)        │
                       └──────────────┬────────────────┘
                                      ▼
                       ┌───────────────────────────────┐
                       │   Tầng Layer Normalization    │
                       │ v = LayerNorm(z)              │
                       └──────────────┬────────────────┘
                                      ▼
                       ┌───────────────────────────────┐
                       │    Đầu Ra v (Chiều d_out)     │
                       └───────────────────────────────┘"""
    add_plain_text_diagram(doc, resblock_ascii, "Sơ đồ giải phẫu nội bộ khối Residual Block (Văn bản thuần)")

    add_heading_3(doc, "Cấu Tạo Nội Tại Của Khối:")
    add_bullet(doc, "1. Tầng Tuyến Tính 1 (Dense 1): ", "Ma trận trọng số W1 ∈ R^(d_hidden x d_in), vector bias b1 ∈ R^(d_hidden). Ánh xạ từ số chiều đầu vào d_in lên không gian ẩn nội bộ d_hidden.")
    add_bullet(doc, "2. Hàm Kích Hoạt Phi Tuyến ReLU: ", "Áp dụng phép cắt max(0, z) tạo tính phi tuyến và tính thưa thớt (sparsity) cho biểu diễn đặc trưng.")
    add_bullet(doc, "3. Tầng Tuyến Tính 2 (Dense 2): ", "Ma trận trọng số W2 ∈ R^(d_out x d_hidden), vector bias b2 ∈ R^(d_out). Ánh xạ từ không gian ẩn sang số chiều đầu ra mong muốn d_out.")
    add_bullet(doc, "4. Tầng Dropout: ", "Ngắt ngẫu nhiên các kết nối với tỷ lệ p_drop ∈ [0.1, 0.5] trong quá trình huấn luyện nhằm triệt tiêu hiện tượng nơ-ron đồng phụ thuộc (co-adaptation).")
    add_math_formula(doc, "F(u) = Dropout( W2 · ReLU(W1 · u + b1) + b2 )")
    
    add_bullet(doc, "5. Nhánh Tắt Bảo Tồn Tín Hiệu (Residual Skip): ", "Nếu d_in ≠ d_out: Dùng ma trận chiếu tuyến tính W_skip ∈ R^(d_out x d_in) để đồng bộ số chiều. Nếu d_in == d_out: Đường dẫn tắt danh tính trực tiếp (nn.Identity()), không tốn tham số.")
    add_math_formula(doc, "z = F(u) + Skip(u)")
    add_bullet(doc, "6. Lớp Layer Normalization: ", "Chuẩn hóa theo chiều đặc trưng cuối cùng với scale γ ∈ R^(d_out) và shift β ∈ R^(d_out).")
    add_math_formula(doc, "v = LayerNorm(z) = [ (z - E[z]) / √(Var[z] + ε) ] ⊙ γ + β")

    # Mô-đun 2
    add_heading_2(doc, "🧩 Mô-Đun 2: Khối Chiếu Giảm Chiều Ngoại Sinh (Feature Projection Module)")
    add_paper_location_box(doc, "Trang 5, Cột phải, Mục 4.1 \"Feature Projection\", Đoạn 1, Công thức (3); Hình 1.")
    add_bullet(doc, "Vị trí trong kiến trúc: ", "Đứng độc lập ở cổng tiếp nhận biến ngoại sinh động x_(1:L+H).")
    add_bullet(doc, "Cấu tạo phần cứng: ", "Đúng 1 khối Residual Block cơ sở với tham số: d_in = r (số biến ngoại sinh thô), d_out = r̃ (temporalWidth), d_hidden = hiddenSize (1024).")
    add_bullet(doc, "Chia sẻ trọng số thời gian (Time-Distributed): ", "Cùng 1 khối nơ-ron được quét lặp lại cho từng bước thời gian t ∈ {1, ..., L+H}, tương đương phép tích chập 1x1 Conv1D. Tiết kiệm hơn 80% bộ nhớ tham số.")

    # Mô-đun 3
    add_heading_2(doc, "🧩 Mô-Đun 3: Khối Mã Hóa Dày Đặc (Dense Encoder Module)")
    add_paper_location_box(doc, "Trang 5, Cột phải, Mục 4.1 \"Dense Encoder\", Đoạn 2-3, Công thức (4); Phụ lục B.3, Trang 18, Bảng 10.")
    add_bullet(doc, "Vị trí trong kiến trúc: ", "Trung tâm xử lý thông tin lịch sử và ngoại sinh toàn cục.")
    add_bullet(doc, "Cấu tạo phần cứng: ", "Gồm chuỗi n_e khối Residual Blocks xếp chồng liên tiếp (numEncoderLayers = 1 hoặc 2):")
    add_bullet(doc, "  • Khối Encoder 1 (Cổng vào): ", "d_in = d_enc = L + (L+H)r̃ + p_static (3,984 chiều trên Electricity), d_out = d_hidden (1024).")
    add_bullet(doc, "  • Khối Encoder 2...n_e: ", "d_in = d_hidden (1024), d_out = d_hidden (1024), Skip là nn.Identity().")
    add_bullet(doc, "Nhiệm vụ: ", "Nén siêu vector 3,984 chiều thành vector bối cảnh ẩn duy nhất e ∈ R^(d_hidden).")

    # Mô-đun 4
    add_heading_2(doc, "🧩 Mô-Đun 4: Khối Giải Mã Dày Đặc (Dense Decoder Module)")
    add_paper_location_box(doc, "Trang 6, Cột trái, Mục 4.2 \"Decoder\", Đoạn 1-2; Phụ lục B.3, Trang 18, Bảng 10.")
    add_bullet(doc, "Vị trí trong kiến trúc: ", "Nối tiếp ngay sau Dense Encoder.")
    add_bullet(doc, "Cấu tạo phần cứng: ", "Gồm chuỗi n_d khối Residual Blocks xếp chồng liên tiếp (numDecoderLayers = 1 hoặc 2):")
    add_bullet(doc, "  • Khối Decoder 1...(n_d - 1): ", "d_in = d_hidden (1024), d_out = d_hidden (1024).")
    add_bullet(doc, "  • Khối Decoder cuối cùng (n_d): ", "d_in = d_hidden (1024), d_out = H · p (96 x 8 = 768).")
    add_bullet(doc, "Nhiệm vụ: ", "Giải nén vector ẩn e thành vector phẳng g ∈ R^(H·p), sẵn sàng bẻ thành ma trận H x p.")

    # Mô-đun 5
    add_heading_2(doc, "🧩 Mô-Đun 5: Khối Giải Mã Theo Thời Gian (Temporal Decoder Module)")
    add_paper_location_box(doc, "Trang 6, Cột trái, Mục 4.2 \"Temporal Decoder\", Đoạn 3, Công thức (5); Hình 1.")
    add_bullet(doc, "Vị trí trong kiến trúc: ", "Giai đoạn tổng hợp cuối cùng trước khi xuất kết quả phi tuyến.")
    add_bullet(doc, "Cấu tạo phần cứng: ", "Gồm 1 khối Residual Block (d_in = p + r̃, d_out = d_hidden) kết hợp với 1 tầng tuyến tính ngõ ra nn.Linear(d_hidden, 1) để thu về 1 con số vô hướng cho mỗi bước thời gian tương lai.")
    add_bullet(doc, "Đặc tính: ", "Chia sẻ trọng số cho toàn bộ H bước thời gian tương lai, đảm bảo tính bất biến tịnh tiến.")

    # Mô-đun 6
    add_heading_2(doc, "🧩 Mô-Đun 6: Nhánh Tắt Tuyến Tính Toàn Cục (Global Linear Skip Connection Module)")
    add_paper_location_box(doc, "Trang 6, Cột phải, Mục 4.2 \"Linear Skip\", Đoạn 1, Dòng 1-8.")
    add_bullet(doc, "Vị trí trong kiến trúc: ", "Đi song song, bắc cầu trực tiếp từ đầu vào lịch sử y_(1:L) sang tương lai ŷ_(L+1:L+H).")
    add_bullet(doc, "Cấu tạo phần cứng: ", "Đúng 1 tầng tuyến tính đơn lẻ nn.Linear(L, H) với ma trận W_skip ∈ R^(H x L) và bias b_skip ∈ R^H. Không có phi tuyến, không dropout.")
    add_bullet(doc, "Vai trò: ", "Học xu hướng nền tuyến tính và tự hồi quy (tương tự DLinear), giải phóng gánh nặng cho mạng MLP sâu.")

    # Mô-đun 7
    add_heading_2(doc, "🧩 Mô-Đun 7: Khối Chuẩn Hóa Thực Thể Khả Nghịch (RevIN Module)")
    add_paper_location_box(doc, "Trang 15 - 16, Phụ lục B.1 \"Reversible Instance Normalization\" & Bảng 8 (Table 8).")
    add_bullet(doc, "Vị trí trong kiến trúc: ", "Đặt tại 2 cửa ngõ: Cổng vào (chuẩn hóa y) và Cổng ra (giải chuẩn hóa ŷ).")
    add_bullet(doc, "Cấu tạo: ", "Bộ nhớ đệm không học (non-trainable buffer) lưu tạm kỳ vọng μ và độ lệch chuẩn σ của từng mẫu trong batch, cùng 2 tham số affine học được tùy chọn (γ, β).")

    # Mục 3: Bảng tổng hợp tham số
    add_heading_1(doc, "3. Bảng Tổng Hợp Tham Số & Khai Báo Trong __init__() (PyTorch Specification)")
    add_paper_location_box(doc, "Trang 18, Phụ lục B.3, Bảng 10 (Cấu hình tham chiếu Electricity: L=720, H=96, r=7, r̃=4, p=8, d_hidden=1024, n_e=2, n_d=2, p_drop=0.3).")
    
    headers_3 = ["Tên Mô-Đun / Lớp", "Kiểu Lớp PyTorch", "Trọng Số (W)", "Bias (b)", "Số Tham Số", "Vai Trò Kiến Trúc"]
    rows_3 = [
        ["feature_projection", "ResidualBlock", "D1: (1024, 7)\nD2: (4, 1024)\nSkip: (4, 7)", "b1: (1024)\nb2: (4)\nbskip: (4)", "~12,324", "Nén ngoại sinh r=7 -> r̃=4 độc lập từng bước thời gian."],
        ["encoder.block_0", "ResidualBlock", "D1: (1024, 3984)\nD2: (1024, 1024)\nSkip: (1024, 3984)", "b1: (1024)\nb2: (1024)\nbskip: (1024)", "~9,216,000", "Nhận siêu vector 3,984 chiều, chiếu vào không gian ẩn 1024."],
        ["encoder.block_1", "ResidualBlock", "D1: (1024, 1024)\nD2: (1024, 1024)\nSkip: Identity", "b1: (1024)\nb2: (1024)\nKhông có", "~2,099,200", "Trích xuất tương tác phi tuyến sâu giữa các thành phần bối cảnh."],
        ["decoder.block_0", "ResidualBlock", "D1: (1024, 1024)\nD2: (1024, 1024)\nSkip: Identity", "b1: (1024)\nb2: (1024)\nKhông có", "~2,099,200", "Tái cấu trúc thông tin tiềm ẩn của chuỗi."],
        ["decoder.block_1", "ResidualBlock", "D1: (1024, 1024)\nD2: (768, 1024)\nSkip: (768, 1024)", "b1: (1024)\nb2: (768)\nbskip: (768)", "~2,624,000", "Ánh xạ về không gian bối cảnh tương lai H·p = 96x8 = 768."],
        ["temporal_decoder", "ResidualBlock + Linear", "ResBlock: (1024, 12)...\nLinear: (1, 1024)", "bres...\nbout: (1)", "~100,000", "Ghép bối cảnh p với ngoại sinh x̃, dự báo phi tuyến ŷ_NL."],
        ["linear_skip", "nn.Linear", "(96, 720)", "(96)", "69,216", "Nhánh tắt tuyến tính trực tiếp từ L bước quá khứ sang H bước tương lai."],
        ["revin", "RevIN", "γ: (1)", "β: (1)", "2", "Chuẩn hóa cục bộ và giải chuẩn hóa chống trôi dạt phân phối."],
        ["TỔNG CỘNG", "Toàn Bộ TiDE", "—", "—", "~16.2 Triệu", "Toàn bộ mô hình chạy thuần MLP siêu tốc O(L + H)."]
    ]
    add_styled_table(doc, headers_3, rows_3, [1.2, 1.2, 1.4, 0.9, 0.9, 1.3])

    # Mục 4: Bảng tra cứu ký hiệu
    add_heading_1(doc, "4. Bảng Tra Cứu Toàn Bộ Ký Hiệu Xuất Hiện Trong Hình 1 (Figure 1)")
    add_paper_location_box(doc, "Trang 4 - 5, Mục 3 \"Problem Setting\" & Mục 4 \"Architecture\".")
    
    headers_4 = ["Ký Hiệu", "Vị Trí Trong Paper", "Tên Linh Kiện / Khối", "Định Nghĩa Kiến Trúc", "Kích Thước Toán Học"]
    rows_4 = [
        ["y_(1:L)", "Section 3, Trang 4", "Lookback Series", "Cổng tiếp nhận chuỗi lịch sử mục tiêu L bước quá khứ.", "R^L"],
        ["x_(1:L+H)", "Section 3, Trang 4", "Dynamic Covariates", "Cổng tiếp nhận ma trận ngoại sinh biến thiên theo thời gian.", "R^((L+H) x r)"],
        ["a", "Section 3, Trang 4", "Static Attributes", "Cổng tiếp nhận vector thuộc tính tĩnh không đổi theo thời gian.", "R^(p_static)"],
        ["x̃_(1:L+H)", "Section 4.1, Trang 5", "Projected Covariates", "Ngõ ra của khối Feature Projection sau khi nén đặc trưng.", "R^((L+H) x r̃)"],
        ["e", "Section 4.1, Trang 5", "Context Embedding", "Ngõ ra của khối Dense Encoder, biểu diễn trừu tượng toàn cục.", "R^(d_hidden)"],
        ["g", "Section 4.2, Trang 6", "Decoded Flat Vector", "Ngõ ra của khối Dense Decoder trước khi bẻ thành ma trận.", "R^(H · p)"],
        ["d_(L+1:L+H)", "Section 4.2, Trang 6", "Decoded Matrix D", "Ma trận gồm H hàng, mỗi hàng là một vector bối cảnh d_t.", "R^(H x p)"],
        ["x̃_(L+1:L+H)", "Section 4.2, Trang 6", "Future Covariates", "Nhánh ngoại sinh tương lai đưa thẳng vào Temporal Decoder.", "R^(H x r̃)"],
        ["ŷ_NL", "Section 4.2, Trang 6", "Nonlinear Prediction", "Đầu ra dự báo của nhánh phi tuyến (Temporal Decoder).", "R^H"],
        ["ŷ_Lin", "Section 4.2, Trang 6", "Linear Prediction", "Đầu ra dự báo của nhánh tuyến tính (Global Linear Skip).", "R^H"],
        ["ŷ_(L+1:L+H)", "Section 4.2, Trang 6", "Final Prediction", "Kết quả dự báo tương lai cuối cùng xuất ra khỏi mô hình.", "R^H"],
        ["n_e, n_d", "Section 4.1 & 4.2", "Layer Counts", "Số lượng khối Residual Block xếp chồng trong Encoder và Decoder.", "1 hoặc 2"]
    ]
    add_styled_table(doc, headers_4, rows_4, [1.0, 1.2, 1.3, 2.3, 1.1])

    doc.save(output_path)
    print(f"Document 1 saved successfully to: {output_path}")


# ==========================================
# BUILD DOCUMENT 2: FORWARD PASS WALKTHROUGH
# ==========================================
def generate_doc_2(output_path):
    doc = docx.Document()
    setup_document(doc)
    
    add_title(doc, 
              "PHẦN 2: GIẢI PHẪU DÒNG CHẢY DỮ LIỆU ĐỘNG FORWARD PASS (DYNAMIC RUNTIME EXECUTION)",
              "Quy trình vận hành động của dây chuyền sản xuất • def forward(self, y, x, a) • TiDE Google Research (2023)")
    
    # Metadata Box
    add_paper_location_box(doc, 
                           "Trang 5 - 6, Section 4 \"Architecture\" & Trang 15, Phụ lục B.1 \"RevIN\"",
                           "Tài liệu theo dõi chính xác từng thao tác nguyên tử trong RAM, sự biến đổi kích thước Tensor qua từng dòng code PyTorch.")
    
    add_heading_1(doc, "🧭 Tôn Chỉ Thiết Kế: Dòng Chảy Dữ Liệu Động (Dynamic Runtime Execution)")
    p = doc.add_paragraph("Trong khi Phần 1 tập trung vào Bản vẽ kết cấu phần cứng tĩnh, thì Phần 2 này tập trung 100% vào Hành trình sống của dữ liệu khi di chuyển qua mô hình:")
    add_bullet(doc, "Câu hỏi trọng tâm: ", 
               "\"Khi một batch dữ liệu thực tế được nạp vào, các tensor di chuyển qua những bước nào, gặp các phép toán động gì (mean, flatten, cat, view, squeeze), và kích thước hình dạng (Tensor Shapes) biến đổi chính xác ra sao ở từng dòng code PyTorch?\"")
    add_bullet(doc, "Phương thức tương ứng: ", "def forward(self, y, x, a) trong PyTorch.")

    add_heading_1(doc, "1. Thiết Lập Dữ Liệu Đầu Vào Thực Nghiệm (Runtime Input Tensors)")
    add_paper_location_box(doc, "Trang 4, Mục 3 \"Problem Setting\" (Đoạn 1-2); Phụ lục B.3, Trang 18, Bảng 10 (Cấu hình chuẩn trên benchmark Electricity).")
    
    add_bullet(doc, "Batch size (B): ", "32 chuỗi thời gian.")
    add_bullet(doc, "Cửa sổ lịch sử (L): ", "720 bước quan sát quá khứ (Lookback horizon).")
    add_bullet(doc, "Cửa sổ dự báo (H): ", "96 bước dự báo tương lai (Forecast horizon).")
    add_bullet(doc, "Biến ngoại sinh động thô (r): ", "7 đặc trưng thời gian (giờ trong ngày, ngày trong tuần, tháng...).")
    add_bullet(doc, "Ngoại sinh sau nén (r̃): ", "4 đặc trưng (temporalWidth = 4).")
    add_bullet(doc, "Bối cảnh giải mã (p): ", "8 đặc trưng (decoderOutputDim = 8).")
    add_bullet(doc, "Kích thước ẩn (d_hidden): ", "1024 nơ-ron (hiddenSize = 1024).")

    input_flow_ascii = """┌─────────────────────────────────┐
│ y: Chuỗi Lịch Sử                │
│ Kích thước: (B, L) = (32, 720)   │───────┐
└─────────────────────────────────┘       │
                                          │
┌─────────────────────────────────┐       │
│ x: Ngoại Sinh Động Toàn Cầu     │       │       ┌─────────────────────────────────┐
│ Kích thước: (B, L+H, r)         │───────┼──────►│ HÀM FORWARD(y, x, a)            │
│           = (32, 816, 7)        │       │       │ (11 Thao Tác Tuần Tự Trong RAM) │
└─────────────────────────────────┘       │       └────────────────┬────────────────┘
                                          │                        │
┌─────────────────────────────────┐       │                        ▼
│ a: Thuộc Tính Tĩnh              │       │       ┌─────────────────────────────────┐
│ Kích thước: (B, s) = (32, 0)    │───────┘       │ y_hat: Dự Báo Hoàn Chỉnh        │
└─────────────────────────────────┘               │ Kích thước: (B, H) = (32, 96)   │
                                                  └─────────────────────────────────┘"""
    add_plain_text_diagram(doc, input_flow_ascii, "Sơ đồ dòng chảy dữ liệu đầu vào của hàm forward()")

    add_heading_1(doc, "2. Bảng Đối Chiếu Song Song 11 Bước Nguyên Tử (Paper Math vs PyTorch Code)")
    add_paper_location_box(doc, "Trang 5 - 6, Toàn bộ Mục 4 \"Architecture\" & Trang 15, Phụ lục B.1 \"RevIN\".")

    headers_steps = ["Bước", "Tên Thao Tác", "Vị Trí Trong Paper", "Công Thức Toán Học Chính Thức", "Dòng Code PyTorch Thực Thi", "Tensor Shape Biến Đổi"]
    rows_steps = [
        ["0", "RevIN Norm", "Trang 15, Phụ lục B.1", "y_norm = (y - μ) / σ", "y_norm = (y - mean) / std", "(B, L) -> (B, L)"],
        ["1", "Feature Proj", "Trang 5, Công thức (3)", "x̃_t = Proj(x_t) ∈ R^r̃", "x_proj = self.feature_projection(x)", "(B, L+H, r) -> (B, L+H, r̃)"],
        ["2", "Flatten", "Trang 5, Dòng 8-12", "vec(x̃) ∈ R^((L+H)r̃)", "x_flat = x_proj.flatten(start_dim=1)", "(B, L+H, r̃) -> (B, (L+H)r̃)"],
        ["3", "Concat", "Trang 5, Công thức (4)", "u_enc = [y_norm ; vec(x̃) ; a]", "enc_in = torch.cat([y_norm, x_flat, a], -1)", "Ghép thành (B, d_enc)"],
        ["4", "Dense Enc", "Trang 5, Công thức (4)", "e = Encoder(u_enc)", "e = self.encoder(enc_in)", "(B, d_enc) -> (B, d_hidden)"],
        ["5", "Dense Dec", "Trang 6, Dòng 1-6", "g = Decoder(e) ∈ R^(H·p)", "g = self.decoder(e)", "(B, d_hidden) -> (B, H·p)"],
        ["6", "Unflatten", "Trang 6, Dòng 7-14", "D = Reshape(g) ∈ R^(H x p)", "d = g.view(B, H, p)", "(B, H·p) -> (B, H, p)"],
        ["7", "Future Stack", "Trang 6, Đoạn 3", "[d_t ; x̃_(L+t)] tại mỗi t", "temp_in = torch.cat([d, x_future], -1)", "Ghép thành (B, H, p + r̃)"],
        ["8", "Temp Dec", "Trang 6, Công thức (5)", "ŷ_NL,t = TempDec([d_t ; x̃_(L+t)])", "y_nl = self.temporal_decoder(temp_in).squeeze(-1)", "(B, H, p + r̃) -> (B, H)"],
        ["9", "Linear Skip", "Trang 6, Dòng 1-8", "ŷ_Lin = W_skip · y_norm + b", "y_lin = self.linear_skip(y_norm)", "(B, L) -> (B, H)"],
        ["10", "Residual Add", "Trang 6, Dòng 9-11", "ŷ_norm = ŷ_NL + ŷ_Lin", "y_hat_norm = y_nl + y_lin", "(B, H) + (B, H) -> (B, H)"],
        ["11", "RevIN Denorm", "Trang 15, Phụ lục B.1", "ŷ = ŷ_norm · σ + μ", "y_hat = y_hat_norm * std + mean", "(B, H) -> (B, H)"]
    ]
    add_styled_table(doc, headers_steps, rows_steps, [0.5, 0.9, 1.1, 1.4, 1.7, 1.3])

    add_heading_1(doc, "3. Giải Phẫu Chi Tiết Dòng Chảy Dữ Liệu Qua 11 Bước Nguyên Tử")

    # Bước 0
    add_heading_2(doc, "Bước 0: Chuẩn Hóa Cục Bộ RevIN (Reversible Instance Normalization)")
    add_paper_location_box(doc, "Trang 15 - 16, Phụ lục B.1 \"Reversible Instance Normalization\" & Bảng 8 (Table 8).")
    add_bullet(doc, "1. Bản chất toán học: ", "Tính kỳ vọng μ và độ lệch chuẩn σ trên từng chuỗi con độc lập: μ = (1/L) ∑ y_t, σ = √((1/L) ∑ (y_t - μ)^2 + ε), y_norm = (y - μ) / σ.")
    add_bullet(doc, "2. Mã nguồn PyTorch: ", "")
    add_code_block(doc, "mean = y.mean(dim=1, keepdim=True)        # Kích thước: (32, 1)\nstd = y.std(dim=1, keepdim=True) + 1e-5   # Kích thước: (32, 1)\ny_norm = (y - mean) / std                # Kích thước: (32, 720)")
    add_bullet(doc, "3. Bản chất trong RAM: ", "Biến đổi phân phối của batch chuỗi lịch sử về kỳ vọng 0 và phương sai 1. Cặp giá trị (μ, σ) được lưu vào RAM để hoàn nguyên ở Bước 11.")

    # Bước 1
    add_heading_2(doc, "Bước 1: Chiếu Giảm Chiều Ngoại Sinh Động (Feature Projection)")
    add_paper_location_box(doc, "Trang 5, Cột phải, Mục 4.1 \"Feature Projection\", Đoạn 1, Công thức (3).")
    add_bullet(doc, "1. Bản chất toán học: ", "x̃_t = FeatureProjection(x_t) ∈ R^r̃ ∀ t ∈ {1, ..., L+H}.")
    add_bullet(doc, "2. Mã nguồn PyTorch: ", "")
    add_code_block(doc, "x_proj = self.feature_projection(x)      # (32, 816, 7) -> (32, 816, 4)")
    add_bullet(doc, "3. Bản chất trong RAM: ", "Cùng 1 khối nơ-ron quét qua toàn bộ 816 lát cắt thời gian, nén vector 7 chiều về 4 chiều giàu thông tin nhất.")

    # Bước 2 & 3
    add_heading_2(doc, "Bước 2 & 3: Trải Phẳng (Flatten) & Ghép Nối Siêu Vector Đầu Vào Encoder")
    add_paper_location_box(doc, "Trang 5, Cột phải, Mục 4.1 \"Dense Encoder\", Dòng 8-15 & Công thức (4).")
    add_bullet(doc, "1. Bản chất toán học: ", "u_enc = [ y_norm,1:L ; vec(x̃_1:L+H) ; a ] ∈ R^(L + (L+H)r̃ + p_static).")
    add_bullet(doc, "2. Mã nguồn PyTorch: ", "")
    add_code_block(doc, "x_flat = x_proj.flatten(start_dim=1)            # (32, 816, 4) -> (32, 3264)\nenc_in = torch.cat([y_norm, x_flat, a], dim=-1) # (32, 720 + 3264 + 0) = (32, 3984)")
    add_bullet(doc, "3. Bản chất trong RAM: ", "Biến đổi cấu trúc không gian - thời gian thành 1 vector 1D phẳng duy nhất, sẵn sàng đưa vào MLP.")

    # Bước 4
    add_heading_2(doc, "Bước 4: Mã Hóa Dày Đặc (Dense Encoder)")
    add_paper_location_box(doc, "Trang 5, Cột phải, Mục 4.1 \"Dense Encoder\", Công thức (4) & Đoạn 3; Trang 18, Bảng 10.")
    add_bullet(doc, "1. Bản chất toán học: ", "e = DenseEncoder(u_enc) ∈ R^(d_hidden).")
    add_bullet(doc, "2. Mã nguồn PyTorch: ", "")
    add_code_block(doc, "e = self.encoder(enc_in)                         # (32, 3984) -> (32, 1024)")
    add_bullet(doc, "3. Bản chất trong RAM: ", "Nén 3,984 chiều thông tin hỗn hợp thành 1,024 chiều biểu diễn đặc trưng trừu tượng (Context Embedding).")

    # Bước 5
    add_heading_2(doc, "Bước 5: Giải Mã Dày Đặc (Dense Decoder)")
    add_paper_location_box(doc, "Trang 6, Cột trái, Mục 4.2 \"Decoder\", Đoạn 1, Dòng 1-6; Trang 18, Bảng 10.")
    add_bullet(doc, "1. Bản chất toán học: ", "g = DenseDecoder(e) ∈ R^(H · p).")
    add_bullet(doc, "2. Mã nguồn PyTorch: ", "")
    add_code_block(doc, "g = self.decoder(e)                              # (32, 1024) -> (32, 768)")
    add_bullet(doc, "3. Bản chất trong RAM: ", "Vector g có độ dài H·p = 96 x 8 = 768, mang toàn bộ bối cảnh dự báo cho 96 bước tương lai.")

    # Bước 6
    add_heading_2(doc, "Bước 6: Tái Cấu Trúc (Unflatten / Reshape) Thành Ma Trận Bối Cảnh")
    add_paper_location_box(doc, "Trang 6, Cột trái, Mục 4.2 \"Decoder\", Đoạn 2, Dòng 7-14.")
    add_bullet(doc, "1. Bản chất toán học: ", "D = Reshape(g, (H, p)) = [d_1 ; d_2 ; ... ; d_H] ∈ R^(H x p).")
    add_bullet(doc, "2. Mã nguồn PyTorch: ", "")
    add_code_block(doc, "d = g.view(B, self.H, self.p)                   # (32, 768) -> (32, 96, 8)")
    add_bullet(doc, "3. Bản chất trong RAM: ", "Khôi phục lại trục thời gian tương lai: mỗi bước t ∈ [1, 96] nhận riêng một vector bối cảnh d_t dài 8 chiều.")

    # Bước 7 & 8
    add_heading_2(doc, "Bước 7 & 8: Làn Đường Cao Tốc (Covariate Highway) & Giải Mã Thời Gian (Temporal Decoder)")
    add_paper_location_box(doc, "Trang 6, Cột trái, Mục 4.2 \"Temporal Decoder\", Đoạn 3 & Công thức (5).")
    add_bullet(doc, "1. Bản chất toán học: ", "ŷ_NL,t = TemporalDecoder([ d_t ; x̃_(L+t) ]) ∈ R ∀ t ∈ {1, ..., H}.")
    add_bullet(doc, "2. Mã nguồn PyTorch: ", "")
    add_code_block(doc, "x_future = x_proj[:, self.L:, :]                 # Cắt 96 bước tương lai: (32, 96, 4)\ntemp_in = torch.cat([d, x_future], dim=-1)       # Ghép bối cảnh: (32, 96, 12)\ny_nl = self.temporal_decoder(temp_in).squeeze(-1)# Dự báo phi tuyến: (32, 96)")
    add_bullet(doc, "3. Bản chất trong RAM: ", "Đưa trực tiếp đặc trưng tương lai vào ngõ ra, giúp mô hình phản ứng tức thời với ngoại sinh đã biết trước.")

    # Bước 9
    add_heading_2(doc, "Bước 9: Nhánh Tắt Tuyến Tính Toàn Cục (Global Linear Skip)")
    add_paper_location_box(doc, "Trang 6, Cột phải, Mục 4.2 \"Linear Skip\", Đoạn 1, Dòng 1-8.")
    add_bullet(doc, "1. Bản chất toán học: ", "ŷ_Lin = W_skip · y_norm + b_skip ∈ R^H.")
    add_bullet(doc, "2. Mã nguồn PyTorch: ", "")
    add_code_block(doc, "y_lin = self.linear_skip(y_norm)                 # (32, 720) -> (32, 96)")
    add_bullet(doc, "3. Bản chất trong RAM: ", "Học xu hướng tăng giảm tự hồi quy tuyến tính cơ bản, tương tự như mô hình DLinear.")

    # Bước 10 & 11
    add_heading_2(doc, "Bước 10 & 11: Hợp Nhất Thặng Dư & Hoàn Nguyên Thang Đo Thực RevIN")
    add_paper_location_box(doc, "Trang 6, Cột phải, Dòng 9-11 & Trang 15, Phụ lục B.1 \"RevIN\".")
    add_bullet(doc, "1. Bản chất toán học: ", "ŷ_norm = ŷ_NL + ŷ_Lin, ŷ_(L+1:L+H) = ŷ_norm ⊙ σ + μ.")
    add_bullet(doc, "2. Mã nguồn PyTorch: ", "")
    add_code_block(doc, "y_hat_norm = y_nl + y_lin                        # (32, 96) + (32, 96) = (32, 96)\ny_hat = y_hat_norm * std + mean                  # Nhân lại std, cộng lại mean: (32, 96)")
    add_bullet(doc, "3. Bản chất trong RAM: ", "Hoàn nguyên kết quả dự báo từ miền chuẩn về đúng đơn vị vật lý ban đầu (Megawatt, lượng xe/giờ...).")

    # Mục 4: Bảng nhật ký biến đổi tensor
    add_heading_1(doc, "4. Bảng Nhật Ký Biến Đổi Tensor Shapes Động (Runtime Tensor Log)")
    add_paper_location_box(doc, "Trang 18, Phụ lục B.3, Bảng 10 (Siêu tham số thực nghiệm benchmark Electricity: B=32, L=720, H=96, r=7, r̃=4, p=8).")

    headers_log = ["Bước Thời Gian", "Tên Biến Tensor", "Shape Thực Nghiệm", "Kiểu Dữ Liệu", "Trạng Thái Dữ Liệu / Ý Nghĩa Bộ Nhớ"]
    rows_log = [
        ["Khởi điểm", "y", "torch.Size([32, 720])", "float32", "Chuỗi lịch sử thực tế chưa chuẩn hóa."],
        ["Khởi điểm", "x", "torch.Size([32, 816, 7])", "float32", "Ngoại sinh động thô 7 chiều (720 quá khứ + 96 tương lai)."],
        ["Khởi điểm", "a", "torch.Size([32, 0])", "float32", "Thuộc tính tĩnh (rỗng với Electricity)."],
        ["Sau Bước 0", "mean, std", "torch.Size([32, 1])", "float32", "Cặp thống kê kỳ vọng và độ lệch chuẩn lưu vào RAM."],
        ["Sau Bước 0", "y_norm", "torch.Size([32, 720])", "float32", "Chuỗi lịch sử đã đưa về Mean=0, Std=1."],
        ["Sau Bước 1", "x_proj", "torch.Size([32, 816, 4])", "float32", "Ngoại sinh đã nén từ 7 -> 4 chiều tại mỗi bước thời gian."],
        ["Sau Bước 2", "x_flat", "torch.Size([32, 3264])", "float32", "Ma trận ngoại sinh được duỗi thẳng thành vector 1D phẳng."],
        ["Sau Bước 3", "enc_in", "torch.Size([32, 3984])", "float32", "Siêu vector ghép: 720 + 3264 + 0 = 3984 phần tử."],
        ["Sau Bước 4", "e", "torch.Size([32, 1024])", "float32", "Vector tiềm ẩn toàn cục xuất ra từ Dense Encoder."],
        ["Sau Bước 5", "g", "torch.Size([32, 768])", "float32", "Vector giải mã phẳng xuất ra từ Dense Decoder (96 x 8)."],
        ["Sau Bước 6", "d", "torch.Size([32, 96, 8])", "float32", "Ma trận bối cảnh tương lai sau phép view(B, H, p)."],
        ["Sau Bước 7", "x_future", "torch.Size([32, 96, 4])", "float32", "Lát cắt 96 bước tương lai của biến ngoại sinh đã nén."],
        ["Sau Bước 7", "temp_in", "torch.Size([32, 96, 12])", "float32", "Ghép bối cảnh và ngoại sinh tương lai: 8 + 4 = 12 chiều."],
        ["Sau Bước 8", "y_nl", "torch.Size([32, 96])", "float32", "Thành phần dự báo phi tuyến từ Temporal Decoder."],
        ["Sau Bước 9", "y_lin", "torch.Size([32, 96])", "float32", "Thành phần dự báo tuyến tính từ Global Linear Skip."],
        ["Sau Bước 10", "y_hat_norm", "torch.Size([32, 96])", "float32", "Kết quả dự báo chuẩn hóa hoàn chỉnh: y_nl + y_lin."],
        ["Sau Bước 11", "y_hat", "torch.Size([32, 96])", "float32", "KẾT QUẢ DỰ BÁO CUỐI CÙNG sau khi giải chuẩn hóa RevIN."]
    ]
    add_styled_table(doc, headers_log, rows_log, [1.0, 1.2, 1.6, 0.9, 2.2])

    # Mục 5: Mã nguồn PyTorch
    add_heading_1(doc, "5. Mã Nguồn Hàm forward() Chuẩn Mực Trong PyTorch")
    code_forward = """def forward(self, y: torch.Tensor, x: torch.Tensor, a: torch.Tensor = None) -> torch.Tensor:
    \"\"\"
    Quy trình tính toán thuận (Forward Pass) chuẩn mực của TiDE:
    Args:
        y: Chuỗi giá trị mục tiêu trong quá khứ, shape (B, L)
        x: Biến ngoại sinh động toàn thời gian, shape (B, L + H, r)
        a: Thuộc tính tĩnh không đổi theo thời gian, shape (B, s)
    Returns:
        y_hat: Giá trị dự báo tương lai, shape (B, H)
    \"\"\"
    B = y.shape[0]

    # --- Bước 0: Chuẩn Hóa Cục Bộ RevIN ---
    if self.use_revin:
        mean = y.mean(dim=1, keepdim=True)
        std = y.std(dim=1, keepdim=True) + 1e-5
        y_norm = (y - mean) / std
    else:
        y_norm = y

    # --- Bước 1: Chiếu Giảm Chiều Biến Ngoại Sinh Động ---
    # Shape: (B, L + H, r) -> (B, L + H, temporal_width)
    x_proj = self.feature_projection(x)

    # --- Bước 2 & 3: Trải Phẳng & Ghép Nối Siêu Vector Đầu Vào ---
    x_flat = x_proj.flatten(start_dim=1)  # (B, (L + H) * temporal_width)
    if a is not None and a.numel() > 0:
        enc_in = torch.cat([y_norm, x_flat, a], dim=-1)
    else:
        enc_in = torch.cat([y_norm, x_flat], dim=-1)

    # --- Bước 4: Mã Hóa Dày Đặc (Dense Encoder) ---
    # Shape: (B, d_enc) -> (B, hidden_size)
    e = self.encoder(enc_in)

    # --- Bước 5: Giải Mã Dày Đặc (Dense Decoder) ---
    # Shape: (B, hidden_size) -> (B, H * decoder_output_dim)
    g = self.decoder(e)

    # --- Bước 6: Tái Cấu Trúc (Reshape) Thành Ma Trận Bối Cảnh ---
    # Shape: (B, H * p) -> (B, H, decoder_output_dim)
    d = g.view(B, self.H, self.p)

    # --- Bước 7: Trích Xuất Ngoại Sinh Tương Lai & Ghép Cặp ---
    x_future = x_proj[:, self.L:, :]  # (B, H, temporal_width)
    temp_in = torch.cat([d, x_future], dim=-1)  # (B, H, p + temporal_width)

    # --- Bước 8: Giải Mã Thời Gian (Temporal Decoder) ---
    # Shape: (B, H, p + temporal_width) -> (B, H, 1) -> (B, H)
    y_nl = self.temporal_decoder(temp_in).squeeze(-1)

    # --- Bước 9: Nhánh Tắt Tuyến Tính Toàn Cục (Global Linear Skip) ---
    # Shape: (B, L) -> (B, H)
    y_lin = self.linear_skip(y_norm)

    # --- Bước 10: Hợp Nhất Thặng Dư ---
    # Shape: (B, H)
    y_hat_norm = y_nl + y_lin

    # --- Bước 11: Hoàn Nguyên Thang Đo Thực RevIN ---
    if self.use_revin:
        y_hat = y_hat_norm * std + mean
    else:
        y_hat = y_hat_norm

    return y_hat"""
    add_code_block(doc, code_forward)

    doc.save(output_path)
    print(f"Document 2 saved successfully to: {output_path}")

if __name__ == "__main__":
    base_dir = r"d:\PTDLKD\Project-Experimental-TiDE\tasks-Hieu\sprint-1"
    img_file = os.path.join(base_dir, "images", "tide_architecture.png")
    doc1_out = os.path.join(base_dir, "01_architecture_diagram.docx")
    doc2_out = os.path.join(base_dir, "02_forward_pass_walkthrough.docx")
    
    generate_doc_1(doc1_out, img_file)
    generate_doc_2(doc2_out)
    print("ALL DOCUMENTS GENERATED SUCCESSFULLY!")
