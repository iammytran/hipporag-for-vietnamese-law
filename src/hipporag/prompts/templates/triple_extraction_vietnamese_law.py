# Sử dụng lại các ví dụ từ prompt NER tiếng Việt để đảm bảo tính nhất quán
from .ner_vietnamese_law import one_shot_ner_paragraph, one_shot_ner_output
from ...utils.llm_utils import convert_format_to_template

# Hướng dẫn hệ thống
ner_conditioned_re_system = """Nhiệm vụ của bạn là xây dựng một đồ thị RDF (Resource Description Framework) từ đoạn văn bản và danh sách thực thể có tên được cung cấp.
Trả lời bằng một danh sách JSON chứa các bộ ba (triples), mỗi bộ ba đại diện cho một mối quan hệ trong đồ thị RDF.

Hãy chú ý đến các yêu cầu sau:
- Mỗi bộ ba nên chứa ít nhất một, và tốt nhất là hai, thực thể có tên trong danh sách được cung cấp.
- Hãy giải quyết các đại từ (ví dụ: nó, ông ấy, bà ấy) thành tên cụ thể của chúng để duy trì sự rõ ràng.
"""

# Khung prompt cho người dùng
ner_conditioned_re_frame = """Dựa vào Đoạn văn và Danh sách thực thể có tên dưới đây, hãy trích xuất các mối quan hệ dưới dạng một danh sách các bộ ba (triples).
Đoạn văn:
```
{passage}
```

{named_entity_json}
"""

# Ví dụ đầu vào (one-shot)
ner_conditioned_re_input = ner_conditioned_re_frame.format(passage=one_shot_ner_paragraph, named_entity_json=one_shot_ner_output)

# Ví dụ đầu ra tương ứng với ví dụ đầu vào
ner_conditioned_re_output = """{"triples": [
            ["Bộ luật Dân sự", "có số hiệu", "91/2015/QH13"],
            ["Bộ luật Dân sự", "được thông qua bởi", "Quốc hội"],
            ["Quốc hội", "thuộc", "Cộng hòa xã hội chủ nghĩa Việt Nam"],
            ["Quốc hội", "là", "khóa XIII"],
            ["Bộ luật Dân sự", "được thông qua vào ngày", "24 tháng 11 năm 2015"]
    ]
}
"""

# Cấu trúc prompt hoàn chỉnh
prompt_template = [
    {"role": "system", "content": ner_conditioned_re_system},
    {"role": "user", "content": ner_conditioned_re_input},
    {"role": "assistant", "content": ner_conditioned_re_output},
    {"role": "user", "content": convert_format_to_template(original_string=ner_conditioned_re_frame, placeholder_mapping=None, static_values=None)}
]