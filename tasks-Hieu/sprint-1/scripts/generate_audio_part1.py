# -*- coding: utf-8 -*-
"""
Script tạo file MP4 âm thanh thuyết minh (chỉ âm thanh, giọng đọc nam trầm ấm, phát âm thuật ngữ tiếng Anh theo kiểu Tiếng Việt, ngắt nghỉ tự nhiên)
cho tài liệu: tasks-Hieu/sprint-1/01_architecture_diagram.md
"""
import os
import sys
import subprocess
import shutil
from gtts import gTTS

# Reconfigure stdout for utf-8
sys.stdout.reconfigure(encoding='utf-8')

BASE_DIR = r"d:\PTDLKD\Project-Experimental-TiDE\tasks-Hieu\sprint-1"
TEMP_DIR = os.path.join(BASE_DIR, "audio_temp")
OUTPUT_MP4 = os.path.join(BASE_DIR, "01_architecture_diagram.mp4")
SYNC_MP4 = r"d:\PTDLKD\project\tasks-Hieu\sprint-1\01_architecture_diagram.mp4"

SECTIONS = [
    {
        "id": "sec_00_intro",
        "title": "Phần mở đầu và Tôn chỉ thiết kế",
        "text": (
            "Chào các bạn. Hôm nay, chúng ta cùng nhau tìm hiểu Phần một: "
            "Sơ đồ kiến trúc tổng thể và giải phẫu kết cấu tĩnh của mô hình Tai-Đờ, "
            "viết tắt của Tham-xi-ri Đen-xơ En-cô-đơ, "
            "công trình nghiên cứu nổi tiếng của Gúc-gồ Rì-sớt vào năm 2023. "
            "Trong kỹ thuật phần mềm và học sâu, tài liệu này đóng vai trò như một bản vẽ thiết kế kết cấu phần cứng tĩnh. "
            "Bản vẽ này sẽ trả lời rõ ràng các câu hỏi cốt lõi: "
            "Mô hình gồm những linh kiện, mô-đun nào? "
            "Cấu tạo bên trong từng khối ra sao? "
            "Kích thước ma trận trọng số như thế nào? "
            "Và các khối được đấu nối với nhau ra sao trên không gian? "
            "Trong mã nguồn Pai-toóc, toàn bộ nội dung của phần này tương ứng trực tiếp với hàm khởi tạo đép in-nít. "
            "Khi nắm vững kết cấu tĩnh này, bạn sẽ dễ dàng bước tiếp sang Phần hai: "
            "Quy trình vận hành dữ liệu động, tương ứng với hàm pho-guất của mô hình."
        )
    },
    {
        "id": "sec_01_topology",
        "title": "Mục 1: Sơ đồ kiến trúc gốc và Topology tổng thể",
        "text": (
            "Bây giờ, chúng ta bước vào Mục một: Sơ đồ kiến trúc gốc của mô hình Tai-Đờ. "
            "Trong bài báo khoa học, toàn bộ kiến trúc này được trình bày tại Trang 5, Hình 1 và Mục 4. "
            "Nhìn vào sơ đồ khối tổng thể từ trên xuống dưới, hệ thống được phân thành sáu tầng chức năng mạch lạc: "
            "Tầng thứ nhất là Ba cổng tiếp nhận dữ liệu đầu vào. "
            "Bao gồm: Chuỗi mục tiêu lịch sử ký hiệu là y từ một đến eo. "
            "Ma trận biến ngoại sinh động biến thiên theo thời gian ký hiệu là x từ một đến eo cộng hát. "
            "Và véc-tơ thuộc tính tĩnh ký hiệu là a. "
            "Tầng thứ hai là Tiền xử lý và nén chiều ngoại sinh. "
            "Chuỗi quá khứ y được đưa qua cổng chuẩn hóa Rê-vịn. "
            "Trong khi đó, biến ngoại sinh x được đưa qua khối Phi-chơ Pờ-rô-déc-sơn "
            "để nén số chiều thô e-rờ xuống số chiều tinh gọn e-rờ uốn. "
            "Tầng thứ ba là Bộ xử lý bối cảnh toàn cục. "
            "Gồm Khối mã hóa dày đặc Đen-xơ En-cô-đơ và Khối giải mã dày đặc Đen-xơ Đi-cô-đơ. "
            "Tầng thứ tư là Làn đường cao tốc và khối giải mã theo thời gian Tem-pô-rồ Đi-cô-đơ, "
            "kết hợp bối cảnh dự báo với thông tin ngoại sinh tương lai. "
            "Tầng thứ năm là Nhánh tắt tuyến tính toàn cục Li-ni-ơ Xíp, "
            "chạy song song độc lập từ đầu vào lịch sử sang thẳng đầu ra tương lai. "
            "Và cuối cùng, Tầng thứ sáu là Khối hợp nhất cộng thặng dư và cổng giải chuẩn hóa Rê-vịn, "
            "đưa tín hiệu về thang đo thực để xuất ra kết quả dự báo hoàn chỉnh y mũ từ eo cộng một đến eo cộng hát."
        )
    },
    {
        "id": "sec_02_resblock",
        "title": "Mục 2: Giải phẫu Mô-đun 1 - Khối Residual Block",
        "text": (
            "Tiếp theo, chúng ta cùng giải phẫu chi tiết bảy mô-đun tĩnh bên trong mô hình. "
            "Mô-đun số một: Khối nơ-ron thặng dư cơ sở, Rê-zi-đu-ồ Blốc. "
            "Đây chính là viên gạch nền tảng cấu thành nên hầu hết các thành phần của Tai-Đờ. "
            "Cấu trúc của khối này gồm ba thành phần chính: "
            "Thành phần thứ nhất là Nhánh biến đổi phi tuyến Em-Eo-Pi. "
            "Đầu vào u có số chiều đê in, trước hết đi qua tầng tuyến tính một Đen-xơ 1 "
            "với ma trận trọng số vê kép 1 kích thước đê hít-đừn nhân đê in và véc-tơ bi-át b 1. "
            "Tầng này mở rộng dữ liệu lên không gian ẩn đê hít-đừn, thường là một nghìn không trăm hai mươi tư chiều. "
            "Tiếp theo, hàm kích hoạt phi tuyến Rê-lu cắt bỏ phần âm để tạo tính thưa và tính phi tuyến. "
            "Sau đó, tầng tuyến tính hai Đen-xơ 2 với ma trận trọng số vê kép 2 kích thước đê ao nhân đê hít-đừn "
            "sẽ thu gọn về số chiều đầu ra đê ao. "
            "Cuối cùng, tầng Đờ-róp-ao ngắt ngẫu nhiên các nơ-ron với tỷ lệ từ không phẩy một đến không phẩy năm, "
            "nhằm triệt tiêu hiện tượng đồng phụ thuộc cô-a-đáp-tây-sơn, chống học vẹt. "
            "Thành phần thứ hai là Nhánh tắt bảo tồn tín hiệu Xíp Con-néc-sơn. "
            "Nếu số chiều đầu vào lệch với số chiều đầu ra, một ma trận chiếu tuyến tính vê kép xíp "
            "kích thước đê ao nhân đê in sẽ khớp chiều trực tiếp. "
            "Nếu số chiều đã bằng nhau, dữ liệu được truyền thẳng nguyên vẹn qua đường dẫn ai-đen-ti-ti mà không tốn bất kỳ tham số nào. "
            "Thành phần thứ ba là Phép cộng thặng dư và Chuẩn hóa tầng. "
            "Tín hiệu từ nhánh phi tuyến và nhánh tắt được cộng dồn lại với nhau theo công thức z bằng F u cộng Xíp u, "
            "rồi đi qua tầng chuẩn hóa Lây-ơ Noóc-mờ với hai tham số học được là gam-ma và bê-ta, "
            "giúp ổn định dòng phân phối dữ liệu đầu ra."
        )
    },
    {
        "id": "sec_03_feature_proj",
        "title": "Mục 2: Giải phẫu Mô-đun 2 - Feature Projection",
        "text": (
            "Mô-đun số hai: Khối chiếu giảm chiều ngoại sinh Phi-chơ Pờ-rô-déc-sơn. "
            "Vị trí nằm ngay ở cổng tiếp nhận biến ngoại sinh động x từ một đến eo cộng hát. "
            "Khối này có cấu tạo phần cứng là đúng một khối Rê-zi-đu-ồ Blốc cơ sở. "
            "Với số chiều đầu vào đê in bằng e-rờ, tức số biến ngoại sinh thô ban đầu, ví dụ bảy chiều trên tập dữ liệu điện lực. "
            "Và số chiều đầu ra đê ao bằng e-rờ uốn, nén xuống chỉ còn bốn chiều. "
            "Đặc tính kiến trúc cốt lõi ở đây là Chia sẻ trọng số toàn thời gian, Thao-đít-xtri-biu-tịt Guết Se-rinh. "
            "Cùng một khối nơ-ron này được áp dụng lặp đi lặp lại cho từng bước thời gian độc lập, "
            "tương đương với một phép tích chập một chiều Côn-vơ một đê kích thước cửa sổ bằng một. "
            "Cơ chế này giúp mô hình giảm kích thước tham số tới hơn tám mươi phần trăm "
            "mà không hề phụ thuộc vào độ dài cửa sổ thời gian."
        )
    },
    {
        "id": "sec_04_enc_dec",
        "title": "Mục 2: Giải phẫu Mô-đun 3 và 4 - Dense Encoder và Decoder",
        "text": (
            "Bây giờ, chúng ta đến với trái tim của mô hình: "
            "Mô-đun số ba: Khối mã hóa dày đặc Đen-xơ En-cô-đơ. "
            "Cấu tạo gồm chuỗi en-nơ e khối Rê-zi-đu-ồ Blốc xếp chồng liên tiếp. "
            "Khối đầu tiên tiếp nhận một siêu véc-tơ bối cảnh phẳng, ghép từ chuỗi lịch sử đã chuẩn hóa, "
            "toàn bộ ma trận ngoại sinh trải phẳng và thuộc tính tĩnh. "
            "Siêu véc-tơ này có số chiều lên tới ba nghìn chín trăm tám mươi tư chiều trên bài toán điện lực. "
            "En-cô-đơ nén toàn bộ thông tin khổng lồ này về một véc-tơ bối cảnh ẩn duy nhất e "
            "có kích thước một nghìn không trăm hai mươi tư chiều. "
            "Nối tiếp ngay sau đó là Mô-đun số bốn: Khối giải mã dày đặc Đen-xơ Đi-cô-đơ. "
            "Gồm chuỗi en-nơ đê khối Rê-zi-đu-ồ Blốc xếp chồng. "
            "Khối Đi-cô-đơ cuối cùng có nhiệm vụ giải nén véc-tơ ẩn một nghìn không trăm hai mươi tư chiều "
            "thành một véc-tơ phẳng kích thước hát nhân pê, "
            "ví dụ chín mươi sáu bước tương lai nhân tám đặc trưng bằng bảy trăm sáu mươi tám chiều. "
            "Véc-tơ này sau đó được bẻ thành ma trận bối cảnh D gồm chín mươi sáu hàng, "
            "sẵn sàng phân phát cho từng bước thời gian tương lai."
        )
    },
    {
        "id": "sec_05_temporal_skip_revin",
        "title": "Mục 2: Giải phẫu Mô-đun 5, 6 và 7",
        "text": (
            "Tiếp theo là ba mô-đun quan trọng ở giai đoạn đầu ra: "
            "Mô-đun số năm: Khối giải mã theo thời gian Tem-pô-rồ Đi-cô-đơ. "
            "Cấu tạo gồm một khối Rê-zi-đu-ồ Blốc kết hợp với một tầng tuyến tính ngõ ra li-ni-ơ. "
            "Tại mỗi bước thời gian tương lai, lát cắt bối cảnh tám chiều được ghép cặp trực tiếp "
            "với bốn chiều ngoại sinh tương lai từ làn đường cao tốc, tạo thành véc-tơ mười hai chiều. "
            "Tầng li-ni-ơ ngõ ra sẽ chiếu véc-tơ này về đúng một con số vô hướng duy nhất, "
            "đại diện cho thành phần dự báo phi tuyến y mũ nơ-ron. "
            "Mô-đun số sáu: Nhánh tắt tuyến tính toàn cục Li-ni-ơ Xíp. "
            "Đây là điểm độc đáo của Tai-Đờ, lấy cảm hứng từ mô hình Đê-Li-ni-ơ. "
            "Nó là đúng một tầng tuyến tính đơn lẻ với ma trận vê kép xíp kích thước hát nhân eo, "
            "tức chín mươi sáu nhân bảy trăm hai mươi. Hoàn toàn không có phi tuyến, không có đờ-róp-ao. "
            "Nhánh này gánh vác việc học xu hướng nền tuyến tính và mùa vụ của chuỗi, giải phóng gánh nặng cho mạng nơ-ron sâu. "
            "Mô-đun số bảy: Khối chuẩn hóa thực thể khả nghịch Rê-vịn. "
            "Đặt tại cổng vào và cổng ra. "
            "Rê-vịn tính toán giá trị trung bình muy và độ lệch chuẩn xích-ma cục bộ trên từng chuỗi con để triệt tiêu hiện tượng trôi dạt phân phối. "
            "Cặp thống kê này được lưu vào thanh ghi đệm bắp-phơ trong bộ nhớ để hoàn nguyên thang đo vật lý gốc ở bước cuối cùng."
        )
    },
    {
        "id": "sec_06_params_summary",
        "title": "Mục 3 và 4: Tổng hợp tham số, bảng ký hiệu và Lời kết",
        "text": (
            "Chúng ta cùng nhìn lại bức tranh toàn cảnh ở Mục ba: Bảng tổng hợp tham số phần cứng. "
            "Trên tập dữ liệu chuẩn mực Ê-lếch-tri-xi-ti: "
            "Khối Phi-chơ Pờ-rô-déc-sơn chiếm khoảng mười hai nghìn tham số. "
            "Khối En-cô-đơ hai tầng chiếm khoảng mười một triệu tham số. "
            "Khối Đi-cô-đơ hai tầng chiếm khoảng bốn phẩy bảy triệu tham số. "
            "Khối Tem-pô-rồ Đi-cô-đơ chiếm khoảng một trăm nghìn tham số. "
            "Và Nhánh Li-ni-ơ Xíp chiếm đúng sáu mươi chín nghìn hai trăm mười sáu tham số. "
            "Tổng cộng toàn bộ mô hình Tai-Đờ chỉ có khoảng mười sáu phẩy hai triệu tham số, "
            "và đặc biệt là chạy thuần mạng Per-xép-tơ-rôn đa tầng Em-Eo-Pi siêu tốc, "
            "đạt độ phức tạp tính toán tuyến tính O của eo cộng hát, "
            "nhanh gấp từ năm đến mười lần so với các kiến trúc Trán-xơ-pho-mơ phức tạp. "
            "Tại Mục bốn của tài liệu, bạn có thể dễ dàng tra cứu lại toàn bộ các ký hiệu toán học "
            "từ y một đến eo, x, a, cho tới y mũ dự báo cuối cùng. "
            "Như vậy, chúng ta đã nắm vững trọn vẹn Bản vẽ thiết kế kết cấu tĩnh của mô hình Tai-Đờ. "
            "Để khám phá cách các ten-sơ thực tế di chuyển, biến đổi hình dạng qua từng dòng mã nguồn trong bộ nhớ ram, "
            "xin mời bạn đón nghe tiếp Phần hai: Giải phẫu dòng chảy dữ liệu động Pho-guất Pát. Cảm ơn các bạn đã lắng nghe!"
        )
    }
]

def generate_all():
    os.makedirs(TEMP_DIR, exist_ok=True)
    raw_files = []
    male_files = []
    
    print("=== BẮT ĐẦU TỔNG HỢP ÂM THANH THUYẾT MINH PHẦN 1 ===")
    
    # 1. Tạo audio thô tiếng Việt cho từng phần bằng gTTS
    for idx, sec in enumerate(SECTIONS):
        raw_file = os.path.join(TEMP_DIR, f"raw_{idx:02d}_{sec['id']}.mp3")
        male_file = os.path.join(TEMP_DIR, f"male_{idx:02d}_{sec['id']}.mp3")
        
        print(f"[{idx+1}/{len(SECTIONS)}] Tổng hợp: {sec['title']}...")
        tts = gTTS(text=sec['text'].strip(), lang='vi', slow=False)
        tts.save(raw_file)
        
        # 2. Chuyển đổi giọng sang giọng nam trầm ấm (pitch=0.82) bằng librubberband chuyên nghiệp
        subprocess.run([
            'ffmpeg', '-y', '-i', raw_file,
            '-af', 'rubberband=pitch=0.82',
            '-q:a', '2', male_file
        ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        raw_files.append(raw_file)
        male_files.append(male_file)
        print(f"  -> Hoàn thành giọng nam: {male_file}")
        
    print("\nToàn bộ 7 đoạn âm thanh giọng nam đã được tạo thành công!")
    
    # 3. Tạo đoạn ngắt nghỉ tĩnh lặng 0.8s giữa các phần
    silence_file = os.path.join(TEMP_DIR, "silence.mp3")
    subprocess.run([
        'ffmpeg', '-y', '-f', 'lavfi', '-i', 'anullsrc=r=24000:cl=mono',
        '-t', '0.8', '-q:a', '9', silence_file
    ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    # 4. Tạo file danh sách ghép nối
    concat_list = os.path.join(TEMP_DIR, "concat_list.txt")
    with open(concat_list, "w", encoding="utf-8") as f:
        for i, mf in enumerate(male_files):
            f.write(f"file '{mf.replace(os.sep, '/')}'\n")
            if i < len(male_files) - 1:
                f.write(f"file '{silence_file.replace(os.sep, '/')}'\n")
                
    # 5. Ghép nối và xuất ra file MP4 chỉ âm thanh (AAC audio stream, 192kbps)
    print(f"\nĐang ghép nối và đóng gói vào file MP4: {OUTPUT_MP4}...")
    temp_merged = os.path.join(TEMP_DIR, "final_merged.mp3")
    subprocess.run([
        'ffmpeg', '-y', '-f', 'concat', '-safe', '0',
        '-i', concat_list, '-c', 'copy', temp_merged
    ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    subprocess.run([
        'ffmpeg', '-y', '-i', temp_merged,
        '-c:a', 'aac', '-b:a', '192k', '-vn',
        OUTPUT_MP4
    ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    print(f"\nĐÃ TẠO THÀNH CÔNG FILE MP4: {OUTPUT_MP4}")
    
    # 6. Đồng bộ sang thư mục dự án
    shutil.copy2(OUTPUT_MP4, SYNC_MP4)
    print(f"Đã đồng bộ sang: {SYNC_MP4}")
    
    # 7. Kiểm tra thông tin thời lượng bằng ffprobe
    result = subprocess.run(['ffprobe', '-hide_banner', OUTPUT_MP4], capture_output=True, text=True)
    print("\nThông số kỹ thuật file MP4 tạo thành:")
    print(result.stderr)

if __name__ == "__main__":
    generate_all()
