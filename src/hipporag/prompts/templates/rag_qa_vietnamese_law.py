# ==========================================================================================
# PHẦN 1: ĐỊNH NGHĨA VÍ DỤ MẪU (ONE-SHOT DEMO)
# ==========================================================================================

# Bối cảnh mẫu: Bao gồm các đoạn trích từ các văn bản luật khác nhau.
# Thay thế các bài viết Wikipedia bằng các điều luật thực tế.
mot_vi_du_mau_boi_canh = (
    """Văn bản: Bộ luật Dân sự 2015
Điều 117. Điều kiện có hiệu lực của giao dịch dân sự
1. Giao dịch dân sự có hiệu lực khi có đủ các điều kiện sau đây:
a) Chủ thể có năng lực pháp luật dân sự, năng lực hành vi dân sự phù hợp với giao dịch dân sự được xác lập;
b) Chủ thể tham gia giao dịch dân sự hoàn toàn tự nguyện;
c) Mục đích và nội dung của giao dịch dân sự không vi phạm điều cấm của luật, không trái đạo đức xã hội.
---
Văn bản: Luật Hôn nhân và gia đình 2014
Điều 8. Điều kiện kết hôn
1. Nam, nữ kết hôn với nhau phải tuân theo các điều kiện sau đây:
a) Nam từ đủ 20 tuổi trở lên, nữ từ đủ 18 tuổi trở lên;
b) Việc kết hôn do nam và nữ tự nguyện quyết định;
c) Không bị mất năng lực hành vi dân sự;
d) Việc kết hôn không thuộc một trong các trường hợp cấm kết hôn theo quy định tại Điều 5 của Luật này.
---
Văn bản: Luật Hôn nhân và gia đình 2014
Điều 51. Quyền yêu cầu giải quyết ly hôn
1. Vợ, chồng hoặc cả hai người có quyền yêu cầu Tòa án giải quyết ly hôn.
2. Cha, mẹ, người thân thích khác có quyền yêu cầu Tòa án giải quyết ly hôn khi một bên vợ, chồng do bị bệnh tâm thần hoặc mắc bệnh khác mà không thể nhận thức, làm chủ được hành vi của mình, đồng thời là nạn nhân của bạo lực gia đình do chồng, vợ của họ gây ra làm ảnh hưởng nghiêm trọng đến tính mạng, sức khỏe, tinh thần của họ.
"""
)

# ==========================================================================================
# PHẦN 2: ĐỊNH NGHĨA HƯỚNG DẪN VÀ CẤU TRÚC PROMPT
# ==========================================================================================

# Hướng dẫn hệ thống: Yêu cầu mô hình đóng vai trò là chuyên gia luật và tuân theo định dạng.
# "Thought" được dịch thành "Suy nghĩ", "Answer" thành "Trả lời".
huong_dan_he_thong = (
    'Bạn là một trợ lý AI chuyên về luật pháp Việt Nam. Nhiệm vụ của bạn là phân tích kỹ lưỡng các đoạn văn bản luật và câu hỏi tương ứng. '
    'Câu trả lời của bạn phải bắt đầu sau mục "Suy nghĩ:", nơi bạn sẽ trình bày quá trình suy luận để đi đến câu trả lời. '
    'Kết thúc bằng mục "Trả lời:" để đưa ra câu trả lời cuối cùng, ngắn gọn và chính xác.'
)

# Đầu vào mẫu: Kết hợp bối cảnh và câu hỏi mẫu.
mot_vi_du_mau_dau_vao = (
    f"{mot_vi_du_mau_boi_canh}"
    "\n\nCâu hỏi: "
    "Độ tuổi kết hôn hợp pháp đối với nữ giới là bao nhiêu?"
    '\nSuy nghĩ: '
)

# Đầu ra mẫu: Cung cấp câu trả lời mẫu với quá trình suy luận.
mot_vi_du_mau_dau_ra = (
    "Câu hỏi yêu cầu về độ tuổi kết hôn của nữ giới. Tôi cần tìm trong các văn bản được cung cấp. "
    'Trong "Luật Hôn nhân và gia đình 2014", Điều 8, khoản 1, điểm a có ghi: "nữ từ đủ 18 tuổi trở lên". '
    "Vậy câu trả lời là từ đủ 18 tuổi trở lên."
    "\nTrả lời: Nữ từ đủ 18 tuổi trở lên."
)

# ==========================================================================================
# PHẦN 3: TỔNG HỢP THÀNH PROMPT TEMPLATE HOÀN CHỈNH
# ==========================================================================================

# Cấu trúc prompt cuối cùng:
# 1. Hướng dẫn hệ thống.
# 2. Ví dụ mẫu của người dùng (user).
# 3. Ví dụ mẫu của trợ lý (assistant).
# 4. Placeholder cho câu hỏi thực tế của người dùng.
prompt_template = [
    {"role": "system", "content": huong_dan_he_thong},
    {"role": "user", "content": mot_vi_du_mau_dau_vao},
    {"role": "assistant", "content": mot_vi_du_mau_dau_ra},
    {"role": "user", "content": "${prompt_user}"}
]