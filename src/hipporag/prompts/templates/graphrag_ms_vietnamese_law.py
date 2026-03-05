# Hướng dẫn hệ thống
ENTITY_TYPES = "VĂN_BẢN_PHÁP_LUẬT, ĐIỀU_KHOẢN, CHỦ_THỂ, QUYỀN_HẠN, NGHĨA_VỤ, HÀNH_VI_VI_PHẠM, CHẾ_TÀI_PHÁP_LÝ, ĐIỀU_KIỆN_ÁP_DỤNG, THỜI_HẠN_THỜI_HIỆU, QUY ĐỊNH CỤ THỂ"

extract_system = f"""
    ### MỤC TIÊU
    Bạn là chuyên gia phân tích dữ liệu pháp luật. Hãy trích xuất các thực thể và mối quan hệ từ văn bản luật được cung cấp để xây dựng một đồ thị tri thức (Knowledge Graph) chính xác và có tính liên kết cao.

    ### QUY TẮC TRÍCH XUẤT

    #### 1. Cấu trúc văn bản (Phân cấp & Claims)
    - Thực thể Gốc: Tạo 01 thực thể đại diện cho tiêu đề văn bản (Ví dụ: "Điều 15 Luật Đất đai").
    - Các thực thể đích: Trích xuất các nội dung pháp lý trong văn bản đó thành các thực thể loại "QUY_ĐỊNH_CỤ_THỂ" theo quy tắc sau:
        - name: Phải là một câu khẳng định đầy đủ ý nghĩa, diễn giải chi tiết (Ví dụ: "Cá nhân có nghĩa vụ đăng ký đất đai tại cơ quan có thẩm quyền").
        - description: Lặp lại hoặc diễn giải chi tiết hơn câu khẳng định đó để tăng cường ngữ nghĩa.
        - type: Bắt buộc là "QUY_ĐỊNH_CỤ_THỂ".
    - Liên kết: Thiết lập quan hệ "quy định" từ Thực thể Gốc đến các QUY_ĐỊNH_CỤ_THỂ này.

    #### 2. Trích xuất thực thể (NER)
    Trích xuất mọi thực thể quan trọng xuất hiện trong văn bản thuộc danh sách: [{ENTITY_TYPES}].
    - name: Tên định danh của thực thể.
    - type: Phải thuộc danh sách [{ENTITY_TYPES}].
    - description: Mô tả chi tiết chức năng, quyền hạn hoặc nội dung quy định của thực thể đó.

    #### 3. Trích xuất quan hệ (Triples)
    - Xác định các cặp (Nguồn, Đích) có liên kết pháp lý rõ ràng (thẩm quyền, căn cứ, hình phạt, đối tượng tác động...).
    - Định dạng: Các liên kết hay mối quan hệ nên được viết bằng chữ thường, có dấu để giữ nguyên ngữ nghĩa tiếng Việt (Ví dụ: "quy định", "có nghĩa vụ", "xử phạt").
    - Đặc biệt: Cho mọi trường hợp văn bản nhắc đến một Điều, Khoản hoặc Văn bản luật khác (kể cả dẫn chiếu nội bộ), bắt buộc tạo quan hệ "dẫn chiếu tới"

    ---

    ### ĐỊNH DẠNG ĐẦU RA (JSON BẮT BUỘC)
    Trả về duy nhất một đối tượng JSON thuần túy (không kèm lời dẫn giải):

    {{
        "named_entities": [
            {{
                "name": "Tên thực thể hoặc câu Claim",
                "type": "Loại thực thể",
                "description": "Mô tả thực thể, nội dung"
            }}
        ],
        "triples": [
            ["Thực thể nguồn", "Mối quan hệ", "Thực thể đích"]
        ]
    }}
    """

# Ví dụ đầu vào (one-shot) - phù hợp với văn bản luật
one_shot_extract_paragraph = "Văn bản: 'Điều 10 Luật X: Công dân phải nộp thuế đúng hạn theo quy định tại Luật Thuế.'"

# Ví dụ đầu ra tương ứng
one_shot_extract_output = """
{
    "named_entities": [
        {
            "name": "Điều 10 Luật X",
            "type": "ĐIỀU_KHOẢN",
            "description": "Quy định về thời hạn nộp thuế của công dân"
        },
        {
            "name": "Công dân phải nộp thuế đúng hạn",
            "type": "QUY_ĐỊNH_CỤ_THỂ",
            "description": "Khẳng định về nghĩa vụ tài chính bắt buộc theo thời gian quy định"
        },
        {
            "name": "Công dân",
            "type": "CHỦ_THỂ",
            "description": "Đối tượng thực hiện nghĩa vụ nộp thuế"
        },
        {
            "name": "Luật Thuế",
            "type": "VĂN_BẢN_LUẬT",
            "description": "Văn bản pháp luật làm căn cứ dẫn chiếu"
        }
    ],
    "triples": [
        ["Điều 10 Luật X", "quy định", "Công dân phải nộp thuế đúng hạn"],
        ["Công dân phải nộp thuế đúng hạn", "áp dụng cho", "Công dân"],
        ["Điều 10 Luật X", "dẫn chiếu tới", "Luật Thuế"]
    ]
}
"""

# Cấu trúc prompt hoàn chỉnh
prompt_template = [
    {"role": "system", "content": extract_system},
    {"role": "user", "content": one_shot_extract_paragraph},
    {"role": "assistant", "content": one_shot_extract_output},
    {"role": "user", "content": "Văn bản: ${passage}"}
]