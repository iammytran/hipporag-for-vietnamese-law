import json

best_dspy_prompt_vn_law = {'prog': {'lm': None, 'traces': [], 'train': [], 'demos': [
    {
        'question': 'Bản án 65/2025/HNGĐ-ST thuộc lĩnh vực nào?',
        'fact_before_filter': json.dumps({
            "fact": [
                ["Bản án 65/2025/HNGĐ-ST", "thuộc lĩnh vực", "Hôn nhân và gia đình"],
                ["Tòa án nhân dân tối cao", "có trụ sở tại", "Hà Nội"],
                ["Bộ luật Dân sự", "được thông qua bởi", "Quốc hội"]
            ]
        }),
        'fact_after_filter': json.dumps({
            "fact": [
                ["Bản án 65/2025/HNGĐ-ST", "thuộc lĩnh vực", "Hôn nhân và gia đình"]
            ]
        })
    },
    {
        'question': 'Ai là người ban hành Bộ luật Dân sự 2015?',
        'fact_before_filter': json.dumps({
            "fact": [
                ["Bộ luật Dân sự", "có số hiệu", "91/2015/QH13"],
                ["Bộ luật Dân sự", "được thông qua bởi", "Quốc hội"],
                ["Luật Hôn nhân và gia đình", "có hiệu lực từ", "01/01/2015"],
                ["Quốc hội", "thuộc", "Cộng hòa xã hội chủ nghĩa Việt Nam"]
            ]
        }),
        'fact_after_filter': json.dumps({
            "fact": [
                ["Bộ luật Dân sự", "được thông qua bởi", "Quốc hội"],
                ["Quốc hội", "thuộc", "Cộng hòa xã hội chủ nghĩa Việt Nam"]
            ]
        })
    },
    {
        'question': 'Lãi suất ngân hàng nhà nước hiện tại là bao nhiêu?',
        'fact_before_filter': json.dumps({
            "fact": [
                ["Bộ luật Dân sự", "quy định về", "quan hệ tài sản"],
                ["Luật các tổ chức tín dụng", "áp dụng cho", "ngân hàng thương mại"]
            ]
        }),
        'fact_after_filter': json.dumps({
            "fact": []
        })
    },
    {
        'question': 'Khi nào Luật Hôn nhân và gia đình 2014 có hiệu lực?',
        'fact_before_filter': json.dumps({
            "fact": [
                ["Luật Hôn nhân và gia đình 2014", "có hiệu lực từ", "ngày 01 tháng 01 năm 2015"],
                ["Luật Hôn nhân và gia đình 2014", "được thông qua ngày", "19 tháng 6 năm 2014"],
                ["Bộ luật Lao động", "có hiệu lực từ", "ngày 01 tháng 01 năm 2021"]
            ]
        }),
        'fact_after_filter': json.dumps({
            "fact": [
                ["Luật Hôn nhân và gia đình 2014", "có hiệu lực từ", "ngày 01 tháng 01 năm 2015"]
            ]
        })
    }
], 'signature': 
    {'instructions': 'Bạn là một thành phần quan trọng của một hệ thống hỏi-đáp phức tạp. Nhiệm vụ của bạn là lọc các fact (bộ ba thông tin) dựa trên sự liên quan của chúng đến một câu hỏi cho trước. Câu hỏi có thể đòi hỏi phân tích và suy luận đa bước để kết nối các mẩu thông tin khác nhau. Bạn phải chọn tối đa 4 fact liên quan nhất từ danh sách ứng viên được cung cấp để hỗ trợ việc suy luận và đưa ra câu trả lời chính xác. Kết quả phải ở định dạng JSON, ví dụ: {"fact": [["chủ thể 1", "quan hệ 1", "đối tượng 1"], ["chủ thể 2", "quan hệ 2", "đối tượng 2"]]}, và nếu không có fact nào liên quan, hãy trả về một danh sách rỗng, {"fact": []}. Độ chính xác của bạn là tối quan trọng. Bạn chỉ được sử dụng các fact từ danh sách ứng viên và không được tự tạo ra fact mới.', 
     'fields': [
         {'prefix': 'Câu hỏi:', 'description': 'Câu hỏi cần truy vấn'}, 
         {'prefix': 'Fact trước khi lọc:', 'description': 'Các fact ứng viên cần được lọc'}, 
         {'prefix': 'Fact sau khi lọc:', 'description': 'Các fact đã được lọc ở định dạng JSON'}]
    }, 
    'system': """Các trường đầu vào của bạn là:
1. `question` (str): Câu hỏi cần truy vấn
2. `fact_before_filter` (str): Các fact ứng viên cần được lọc

Trường đầu ra của bạn là:
1. `fact_after_filter` (Fact): Các fact đã được lọc ở định dạng JSON

Lưu ý quan trọng: Giá trị bạn tạo ra cho `fact_after_filter` phải luôn là một JSON hợp lệ. JSON này phải chứa một key duy nhất là "fact". Giá trị của "fact" là một danh sách các fact, trong đó mỗi fact là một danh sách con chứa đúng 3 chuỗi theo thứ tự: [chủ thể, quan hệ, đối tượng]. Nếu không có fact nào liên quan, trả về một danh sách rỗng: `{"fact": []}`. **KHÔNG** được thêm bất kỳ giải thích hay văn bản nào khác ngoài khối mã JSON.
Ví dụ của 1 giá trị của fact_after_filter: 
        "fact_after_filter": [
            ["Luật Hôn nhân và gia đình 2014", "có hiệu lực từ", "ngày 01 tháng 01 năm 2015"]
        ]
    
Tuân thủ nghiêm ngặt cấu trúc và các ví dụ đã cho. Mục tiêu của bạn là:
Bạn là một thành phần quan trọng của một hệ thống hỏi-đáp phức tạp. Nhiệm vụ của bạn là lọc các fact (bộ ba thông tin) dựa trên sự liên quan của chúng đến một câu hỏi cho trước. Câu hỏi có thể đòi hỏi phân tích và suy luận đa bước để kết nối các mẩu thông tin khác nhau. Bạn phải chọn tối đa 4 fact liên quan nhất từ danh sách ứng viên được cung cấp để hỗ trợ việc suy luận và đưa ra câu trả lời chính xác. Kết quả phải ở định dạng JSON, ví dụ: {"fact": [["chủ thể 1", "quan hệ 1", "đối tượng 1"], ["chủ thể 2", "quan hệ 2", "đối tượng 2"]]}, và nếu không có fact nào liên quan, hãy trả về một danh sách rỗng, {"fact": []}. Độ chính xác của bạn là tối quan trọng. Bạn chỉ được sử dụng các fact từ danh sách ứng viên và không được tự tạo ra fact mới.
---
Bây giờ, hãy thực hiện cho trường hợp sau. Hãy nhớ, chỉ trả về khối mã JSON.

Câu hỏi:
{question}

Fact trước khi lọc:
{fact_before_filter}

Fact sau khi lọc:
# """
    }}