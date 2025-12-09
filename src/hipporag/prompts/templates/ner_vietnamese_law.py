# Hướng dẫn hệ thống
ner_system = """Nhiệm vụ của bạn là trích xuất các thực thể có tên (Named Entities) từ đoạn văn bản được cung cấp.
Trả lời bằng một danh sách JSON chứa các thực thể.
"""

# Ví dụ đầu vào (one-shot) - phù hợp với văn bản luật
one_shot_ner_paragraph = """Bộ luật Dân sự số 91/2015/QH13 được Quốc hội nước Cộng hòa xã hội chủ nghĩa Việt Nam khóa XIII thông qua ngày 24 tháng 11 năm 2015."""


# Ví dụ đầu ra tương ứng
one_shot_ner_output = """{"named_entities":
    ["Bộ luật Dân sự", "91/2015/QH13", "Quốc hội", "Cộng hòa xã hội chủ nghĩa Việt Nam", "Khoá XIII", "24 tháng 11 năm 2015"]
}
"""


# Cấu trúc prompt hoàn chỉnh
prompt_template = [
    {"role": "system", "content": ner_system},
    {"role": "user", "content": one_shot_ner_paragraph},
    {"role": "assistant", "content": one_shot_ner_output},
    {"role": "user", "content": "${passage}"}
]