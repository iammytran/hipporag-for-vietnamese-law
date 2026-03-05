import json
from typing import Dict, Tuple
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

from ..llm import BaseLLM
from ..utils.misc_utils  import Ner, NerRawOutput, TripleRawOutput
from ..utils.logging_utils import get_logger
from .openie_openai import OpenIE

logger = get_logger(__name__)

class GraphRagMSExtractor(OpenIE):
    def __init__(self, llm):
        super().__init__(llm)
        logger.info("Using GraphRagMSExtractor")

    def _create_prompt(self, text: str) -> str:
        ENTITY_TYPES = "VĂN_BẢN_PHÁP_LUẬT, ĐIỀU_KHOẢN, CHỦ_THỂ, QUYỀN_HẠN, NGHĨA_VỤ, HÀNH_VI_VI_PHẠM, CHẾ_TÀI_PHÁP_LÝ, ĐIỀU_KIỆN_ÁP_DỤNG, THỜI_HẠN_THỜI_HIỆU, QUY ĐỊNH CỤ THỂ"

        return f"""
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
                        "description": "Mô tả thực thể, nội dung, ”
                    }}
                ],
                "triples": [
                    ["Thực thể nguồn", "Mối quan hệ", "Thực thể đích"]
                ]
            }}

            ---

            ### VÍ DỤ MINH HỌA
            Văn bản: "Điều 10 Luật X: Công dân phải nộp thuế đúng hạn theo quy định tại Luật Thuế."

            Output:
            {{
                "named_entities": [
                    {{
                        "name": "Điều 10 Luật X",
                        "type": "ĐIỀU_KHOẢN",
                        "description": "Quy định về thời hạn nộp thuế của công dân"
                    }},
                    {{
                        "name": "Công dân phải nộp thuế đúng hạn",
                        "type": "QUY_ĐỊNH_CỤ_THỂ",
                        "description": "Khẳng định về nghĩa vụ tài chính bắt buộc theo thời gian quy định"
                    }},
                    {{
                        "name": "Công dân",
                        "type": "CHỦ_THỂ",
                        "description": "Đối tượng thực hiện nghĩa vụ nộp thuế"
                    }},
                    {{
                        "name": "Luật Thuế",
                        "type": "VĂN_BẢN_LUẬT",
                        "description": "Văn bản pháp luật làm căn cứ dẫn chiếu"
                    }}
                ],
                "triples": [
                    ["Điều 10 Luật X", "quy định", "Công dân phải nộp thuế đúng hạn"],
                    ["Công dân phải nộp thuế đúng hạn", "áp dụng cho", "Công dân"],
                    ["Điều 10 Luật X", "dẫn chiếu tới", "Luật Thuế"]
                ]
            }}

            ---

            ### DỮ LIỆU THỰC TẾ
            Văn bản: {text}
            Output:
            """
        
    def batch_openie(self, documents: Dict[str, Dict]) -> Tuple[Dict[str, NerRawOutput], Dict[str, TripleRawOutput]]:
        ner_results = {}
        triple_results = {}

        def process_single_chunk(hash_id, doc_info):
            content = doc_info["content"]
            # Sử dụng GRAPH_PROMPT mới (One-pass: trả về cả entities và triples)
            prompt_messages = self._create_prompt(content)
            
            try:
                # LLM INFERENCE (Gọi 1 lần duy nhất)
                raw_response, metadata, cache_hit = self.llm_model.infer(messages=prompt_messages)
                metadata['cache_hit'] = cache_hit

                # Làm sạch chuỗi JSON từ LLM
                cleaned_str = raw_response.strip()
                if "```json" in cleaned_str:
                    cleaned_str = cleaned_str.split("```json")[-1].split("```")[0].strip()
                elif "```" in cleaned_str:
                    cleaned_str = cleaned_str.split("```")[1].strip()
                
                response_json = json.loads(cleaned_str)
                
                # --- XỬ LÝ NER ---
                structured_entities = []
                for ent in response_json.get("named_entities", []):
                    # Khởi tạo object Ner theo đúng dataclass của My
                    ner_obj = Ner(
                        name=ent.get("name", ""),
                        type=ent.get("type", ""),
                        description=ent.get("description", "")
                    )
                    structured_entities.append(ner_obj)

                # Loại bỏ trùng lặp dựa trên 'name'
                unique_entities_dict = {e.name: e for e in structured_entities}
                final_entities = list(unique_entities_dict.values())

                ner_output = NerRawOutput(
                    chunk_id=hash_id,
                    response=raw_response,
                    unique_entities=final_entities,
                    metadata=metadata
                )

                # --- XỬ LÝ TRIPLES ---
                raw_triples = response_json.get("triples", [])
                # Filter để đảm bảo triple đúng định dạng [s, p, o]
                valid_triples = [t for t in raw_triples if isinstance(t, list) and len(t) >= 3]
                
                triple_output = TripleRawOutput(
                    chunk_id=hash_id,
                    response=raw_response,
                    triples=valid_triples,
                    metadata=metadata
                )

                return hash_id, ner_output, triple_output

            except Exception as e:
                logger.error(f"Lỗi tại chunk {hash_id}: {e}")
                # Trả về kết quả rỗng nếu lỗi để không làm sập cả batch
                return hash_id, NerRawOutput(hash_id, raw_response, [], {"error": str(e)}), \
                            TripleRawOutput(hash_id, raw_response, [], {"error": str(e)})

        # CHẠY ĐA LUỒNG (Multithreading)
        # Giới hạn max_workers để tránh lỗi Rate Limit của OpenAI
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(process_single_chunk, hid, dinfo) for hid, dinfo in documents.items()]
            
            pbar = tqdm(as_completed(futures), total=len(futures), desc="Extracting Graph Data")
            for future in pbar:
                hid, ner_out, tri_out = future.result()
                ner_results[hid] = ner_out
                triple_results[hid] = tri_out

        return ner_results, triple_results