def get_query_instruction(linking_method):
    instructions = {
        'ner_to_node': 'Given a phrase, retrieve synonymous or relevant phrases that best match this phrase.',
        'query_to_node': 'Given a question, retrieve relevant phrases that are mentioned in this question.',
        'query_to_fact': 'Given a question, retrieve relevant triplet facts that matches this question.',
        'query_to_sentence': 'Given a question, retrieve relevant sentences that best answer the question.',
        'query_to_passage': 'Given a question, retrieve relevant documents that best answer the question.',
    }
    default_instruction = 'Given a question, retrieve relevant documents that best answer the question.'
    return instructions.get(linking_method, default_instruction)

def get_query_instruction_vn(linking_method):
    instructions = {
        'ner_to_node': 'Tìm các thuật ngữ pháp lý đồng nghĩa hoặc có liên quan chặt chẽ với cụm từ sau.',
        'query_to_node': 'Trích xuất các thuật ngữ pháp lý và khái niệm trọng tâm xuất hiện trong câu hỏi sau.',
        'query_to_fact': 'Xác định các thực thể, hành vi và mối quan hệ pháp lý trọng tâm được đề cập và có liên quan để trả lời câu hỏi sau.',
        'query_to_sentence': 'Truy xuất các quy định hoặc trích dẫn luật pháp giải quyết trực tiếp vấn đề sau.',
        'query_to_passage': 'Truy xuất các tài liệu pháp luật có liên quan để trả lời câu hỏi sau.',
    }
    default_instruction = 'Truy xuất các tài liệu pháp luật có liên quan để trả lời câu hỏi sau.'
    return instructions.get(linking_method, default_instruction)
    