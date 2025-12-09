import json
from typing import Dict, Tuple
import os 
from tqdm import tqdm

from ..information_extraction import OpenIE
from .openie_openai import ChunkInfo
from ..utils.misc_utils import NerRawOutput, TripleRawOutput
from ..utils.logging_utils import get_logger
from ..prompts import PromptTemplateManager
from ..llm.transformers_offline import TransformersOffline
from .. import TransformersOfflineOpenIE


logger = get_logger(__name__)


class TransformersOfflineOpenIEVnLaw(TransformersOfflineOpenIE):
    def __init__(self, global_config):
        super().__init__(global_config)

    def batch_openie(self, chunks: Dict[str, ChunkInfo]) -> Tuple[Dict[str, NerRawOutput], Dict[str, TripleRawOutput]]:
        """
        Conduct batch OpenIE synchronously using vLLM offline batch mode, including NER and triple extraction

        Args:
            chunks (Dict[str, ChunkInfo]): chunks to be incorporated into graph. Each key is a hashed chunk
            and the corresponding value is the chunk info to insert.

        Returns:
            Tuple[Dict[str, NerRawOutput], Dict[str, TripleRawOutput]]:
                - A dict with keys as the chunk ids and values as the NER result instances.
                - A dict with keys as the chunk ids and values as the triple extraction result instances.
        """

        # Extract passages from the provided chunks
        chunk_passages = {chunk_key: chunk["content"] for chunk_key, chunk in chunks.items()}

        logger.info(f"Bắt đầu trích xuất NER cho {len(chunk_passages)} chunk...")
        ner_input_messages = [self.prompt_template_manager.render(name='ner_vietnamese_law', passage=p) for p in chunk_passages.values()]
        ner_output, ner_output_metadata = self.llm_model.batch_infer(ner_input_messages, json_template='ner', max_tokens=2048)
        logger.info("Hoàn thành NER. Bắt đầu trích xuất Triple...")

        triple_extract_input_messages = [self.prompt_template_manager.render(
            name='triple_extraction_vietnamese_law',
            passage=passage,
            named_entity_json=named_entities
        ) for passage, named_entities in zip(chunk_passages.values(), ner_output)]
        triple_output, triple_output_metadata = self.llm_model.batch_infer(triple_extract_input_messages, json_template='triples', max_tokens=2048)
        logger.info("Hoàn thành trích xuất Triple. Bắt đầu xử lý kết quả...")

        ner_raw_outputs = []
        for idx, ner_output_instance in enumerate(tqdm(ner_output, desc="Đang xử lý kết quả NER")):
            chunk_id = list(chunks.keys())[idx]
            response = ner_output_instance
            try:
                unique_entities = json.loads(response)["named_entities"]
            except Exception as e:
                unique_entities = []
                logger.warning(f"Could not parse response from OpenIE: {e}")
            if len(unique_entities) == 0:
                logger.warning("No entities extracted for chunk_id: {}".format(chunk_id))
            ner_raw_output = NerRawOutput(chunk_id, response, unique_entities, {})
            ner_raw_outputs.append(ner_raw_output)
        ner_results_dict = {chunk_key: ner_raw_output for chunk_key, ner_raw_output in zip(chunks.keys(), ner_raw_outputs)}

        triple_raw_outputs = []
        for idx, triple_output_instance in enumerate(tqdm(triple_output, desc="Đang xử lý kết quả Triple")):
            chunk_id = list(chunks.keys())[idx]
            response = triple_output_instance
            try:
                triples = json.loads(response)["triples"]
            except Exception as e:
                triples = []
                logger.warning(f"Could not parse response from OpenIE: {e}")
            if len(triples) == 0:
                logger.warning("No triples extracted for chunk_id: {}".format(chunk_id))
            triple_raw_output = TripleRawOutput(chunk_id, response, triples, {})
            triple_raw_outputs.append(triple_raw_output)
        triple_results_dict = {chunk_key: triple_raw_output for chunk_key, triple_raw_output in zip(chunks.keys(), triple_raw_outputs)}

        # --- THÊM CODE ĐỂ DEBUG ---
        # Lưu kết quả NER và Triple vào thư mục 'outputs' để kiểm tra
        try:
            output_dir = "outputs"
            # Dòng os.makedirs sẽ không làm gì nếu thư mục đã tồn tại
            os.makedirs(output_dir, exist_ok=True)

            # Tạo đường dẫn đầy đủ đến file debug NER
            ner_debug_path = os.path.join(output_dir, "debug_ner_results.json")
            ner_to_save = {chunk_id: ner_output.__dict__ for chunk_id, ner_output in ner_results_dict.items()}
            with open(ner_debug_path, "w", encoding="utf-8") as f:
                json.dump(ner_to_save, f, ensure_ascii=False, indent=4)
            logger.info(f"Đã lưu kết quả NER vào file: {ner_debug_path}")

            # Tạo đường dẫn đầy đủ đến file debug Triple
            triple_debug_path = os.path.join(output_dir, "debug_triple_results.json")
            triple_to_save = {chunk_id: triple_output.__dict__ for chunk_id, triple_output in triple_results_dict.items()}
            with open(triple_debug_path, "w", encoding="utf-8") as f:
                json.dump(triple_to_save, f, ensure_ascii=False, indent=4)
            logger.info(f"Đã lưu kết quả Triple vào file: {triple_debug_path}")
        except Exception as e:
            logger.error(f"Lỗi khi lưu file JSON để debug: {e}")
        # --- KẾT THÚC CODE DEBUG ---

        return ner_results_dict, triple_results_dict
