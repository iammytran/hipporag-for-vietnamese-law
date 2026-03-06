import json

best_dspy_prompt_vn_law = {
  "prog": {
    "lm": None,
    "traces": [],
    "train": [],
    "demos": [
      {
        "question": "Bản án 65/2025/HNGĐ-ST thuộc lĩnh vực nào?",
        "fact_before_filter": "{\"fact\": [[\"Bản án 65/2025/HNGĐ-ST\", \"thuộc lĩnh vực\", \"Hôn nhân và gia đình\"], [\"Tòa án nhân dân tối cao\", \"có trụ sở tại\", \"Hà Nội\"], [\"Bộ luật Dân sự\", \"được thông qua bởi\", \"Quốc hội\"]]}",
        "fact_after_filter": "{\"fact\": [[\"Bản án 65/2025/HNGĐ-ST\", \"thuộc lĩnh vực\", \"Hôn nhân và gia đình\"]]}"
      },
      {
        "question": "Ai là người ban hành Bộ luật Dân sự 2015?",
        "fact_before_filter": "{\"fact\": [[\"Bộ luật Dân sự\", \"có số hiệu\", \"91/2015/QH13\"], [\"Bộ luật Dân sự\", \"được thông qua bởi\", \"Quốc hội\"], [\"Luật Hôn nhân và gia đình\", \"có hiệu lực từ\", \"01/01/2015\"], [\"Quốc hội\", \"thuộc\", \"Cộng hòa xã hội chủ nghĩa Việt Nam\"]]}",
        "fact_after_filter": "{\"fact\": [[\"Bộ luật Dân sự\", \"được thông qua bởi\", \"Quốc hội\"], [\"Quốc hội\", \"thuộc\", \"Cộng hòa xã hội chủ nghĩa Việt Nam\"]]}"
      },
      {
        "question": "Lãi suất ngân hàng nhà nước hiện tại là bao nhiêu?",
        "fact_before_filter": "{\"fact\": [[\"Bộ luật Dân sự\", \"quy định về\", \"quan hệ tài sản\"], [\"Luật các tổ chức tín dụng\", \"áp dụng cho\", \"ngân hàng thương mại\"]]}",
        "fact_after_filter": "{\"fact\": []}"
      },
      {
        "question": "Khi nào Luật Hôn nhân và gia đình 2014 có hiệu lực?",
        "fact_before_filter": "{\"fact\": [[\"Luật Hôn nhân và gia đình 2014\", \"có hiệu lực từ\", \"ngày 01 tháng 01 năm 2015\"], [\"Luật Hôn nhân và gia đình 2014\", \"được thông qua ngày\", \"19 tháng 6 năm 2014\"], [\"Bộ luật Lao động\", \"có hiệu lực từ\", \"ngày 01 tháng 01 năm 2021\"]]}",
        "fact_after_filter": "{\"fact\": [[\"Luật Hôn nhân và gia đình 2014\", \"có hiệu lực từ\", \"ngày 01 tháng 01 năm 2015\"]]}"
      }
    ],
    "signature": {
      "instructions": "Bạn là một thành phần quan trọng của một hệ thống hỏi-đáp phức tạp. Nhiệm vụ của bạn là lọc các fact (bộ ba thông tin) dựa trên sự liên quan của chúng đến một câu hỏi cho trước. Câu hỏi có thể đòi hỏi phân tích và suy luận đa bước để kết nối các mẩu thông tin khác nhau. Bạn phải chọn tối đa 4 fact liên quan nhất từ danh sách ứng viên được cung cấp để hỗ trợ việc suy luận và đưa ra câu trả lời chính xác. Kết quả phải ở định dạng JSON, ví dụ: {\"fact\": [[\"chủ thể 1\", \"quan hệ 1\", \"đối tượng 1\"], [\"chủ thể 2\", \"quan hệ 2\", \"đối tượng 2\"]]}, và nếu không có fact nào liên quan, hãy trả về một danh sách rỗng, {\"fact\": []}. Độ chính xác của bạn là tối quan trọng. Bạn chỉ được sử dụng các fact từ danh sách ứng viên và không được tự tạo ra fact mới.",
      "fields": [
        {
          "prefix": "Câu hỏi:",
          "description": "Câu hỏi cần truy vấn"
        },
        {
          "prefix": "Fact trước khi lọc:",
          "description": "Các fact ứng viên cần được lọc"
        },
        {
          "prefix": "Fact sau khi lọc:",
          "description": "Các fact đã được lọc ở định dạng JSON"
        }
      ]
    },
    "system": "Các trường đầu vào của bạn là:\n1. `question` (str): Truy vấn để tìm kiếm\n2. `fact_before_filter` (str): Các sự kiện ứng viên cần được lọc\n\nCác trường đầu ra của bạn là:\n1. `fact_after_filter` (Fact): Các sự kiện đã lọc ở định dạng JSON\n\nTất cả các tương tác sẽ được cấu trúc theo cách sau, với các giá trị thích hợp được điền vào.\n\n[[ ## question ## ]]\n{question}\n\n[[ ## fact_before_filter ## ]]\n{fact_before_filter}\n\n[[ ## fact_after_filter ## ]]\n{fact_after_filter}        # lưu ý: giá trị bạn tạo ra phải có thể phân tách (parse) được theo cấu trúc JSON sau: {\"type\": \"object\", \"properties\": {\"fact\": {\"type\": \"array\", \"description\": \"Một danh sách các sự kiện, mỗi sự kiện là một danh sách gồm 3 chuỗi: [chủ thể, vị ngữ, đối tượng]\", \"items\": {\"type\": \"array\", \"items\": {\"type\": \"string\"}}, \"title\": \"Sự kiện\"}}, \"required\": [\"fact\"], \"title\": \"Sự kiện\"}\n\n[[ ## completed ## ]]\n\nĐể tuân thủ cấu trúc này, mục tiêu của bạn là: \n        Bạn là một thành phần quan trọng của hệ thống trả lời câu hỏi có tính rủi ro cao, được sử dụng bởi các nhà nghiên cứu và những người ra quyết định hàng đầu trên toàn thế giới. Nhiệm vụ của bạn là lọc các sự kiện dựa trên mức độ liên quan của chúng với một truy vấn nhất định, đảm bảo rằng thông tin quan trọng nhất được trình bày cho các bên liên quan này. Truy vấn yêu cầu phân tích cẩn thận và có thể cần suy luận đa bước (multi-hop reasoning) để kết nối các phần thông tin khác nhau. Bạn phải chọn tối đa 4 sự kiện liên quan từ danh sách ứng viên được cung cấp có kết nối chặt chẽ với truy vấn, nhằm hỗ trợ việc suy luận và đưa ra câu trả lời chính xác. Đầu ra phải ở định dạng JSON, ví dụ: {\"fact\": [[\"s1\", \"p1\", \"o1\"], [\"s2\", \"p2\", \"o2\"]]}, và nếu không có sự kiện nào liên quan, hãy trả về một danh sách trống, {\"fact\": []}. Độ chính xác trong câu trả lời của bạn là tối quan trọng, vì nó sẽ ảnh hưởng trực tiếp đến các quyết định của các bên liên quan cấp cao này. Bạn chỉ được sử dụng các sự kiện từ danh sách ứng viên và không được tự tạo ra các sự kiện mới. Tương lai của việc ra quyết định quan trọng phụ thuộc vào khả năng lọc và trình bày thông tin liên quan một cách chính xác của bạn."  }
}