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
        'ner_to_node': 'Cho một cụm từ, truy xuất các cụm từ đồng nghĩa hoặc có liên quan phù hợp nhất với cụm từ này.',
        'query_to_node': 'Cho một câu hỏi, truy xuất các cụm từ có liên quan được đề cập trong câu hỏi này.',
        'query_to_fact': 'Cho một câu hỏi, truy xuất các bộ ba dữ kiện có liên quan phù hợp với câu hỏi này.',
        'query_to_sentence': 'Cho một câu hỏi, truy xuất các câu có liên quan trả lời câu hỏi một cách tốt nhất.',
        'query_to_passage': 'Cho một câu hỏi, truy xuất các tài liệu có liên quan trả lời câu hỏi một cách tốt nhất.',
    }
    default_instruction = 'Cho một câu hỏi, truy xuất các tài liệu có liên quan trả lời câu hỏi một cách tốt nhất.'
    return instructions.get(linking_method, default_instruction)
    