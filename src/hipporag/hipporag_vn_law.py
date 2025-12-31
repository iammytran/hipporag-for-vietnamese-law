import json
import os
from dataclasses import dataclass, field, asdict
from datetime import datetime
import pprint
from typing import Union, Optional, List, Set, Dict, Any, Tuple, Literal
import numpy as np
import importlib
from collections import defaultdict
from transformers import HfArgumentParser
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from igraph import Graph
import igraph as ig
import numpy as np
from collections import defaultdict
import re
import time

from .llm import _get_llm_class, BaseLLM
from .embedding_model import _get_embedding_model_class, BaseEmbeddingModel
from .embedding_store import EmbeddingStore
from .information_extraction import OpenIE
from .information_extraction.openie_vllm_offline import VLLMOfflineOpenIE
from .information_extraction.openie_transformers_offline_vn_law import TransformersOfflineOpenIEVnLaw
from .evaluation.retrieval_eval import RetrievalRecall
from .evaluation.qa_eval import QAExactMatch, QAF1Score
from .prompts.linking import get_query_instruction_vn
from .prompts.prompt_template_manager import PromptTemplateManager
from .rerank_vn_law import DSPyFilterVnLaw
from .utils.misc_utils import *
from .utils.misc_utils import NerRawOutput, TripleRawOutput
from .utils.embed_utils import retrieve_knn
from .utils.logging_utils import get_logger
from .utils.typing import Triple
from .utils.config_utils import BaseConfig
from .HippoRAG import HippoRAG

logger = get_logger(__name__)

class HippoRAGVnLaw(HippoRAG):
    def __init__(self,
                 global_config=None,
                 save_dir=None,
                 llm_model_name=None,
                 llm_base_url=None,
                 embedding_model_name=None,
                 embedding_base_url=None,
                 azure_endpoint=None,
                 azure_embedding_endpoint=None):
        super().__init__(global_config,
                        save_dir,
                        llm_model_name,
                        llm_base_url,
                        embedding_model_name,
                        embedding_base_url,
                        azure_endpoint,
                        azure_embedding_endpoint)
        if self.global_config.openie_mode == 'online':
            self.openie = OpenIE(llm_model=self.llm_model)
        elif self.global_config.openie_mode == 'offline':
            self.openie = VLLMOfflineOpenIE(self.global_config)
        elif self.global_config.openie_mode ==  'Transformers-offline':
            self.openie = TransformersOfflineOpenIEVnLaw(self.global_config)

        self.rerank_filter = DSPyFilterVnLaw(self)

    def index(self, docs: List[str]):
        """
        Indexes the given documents based on the HippoRAG 2 framework which generates an OpenIE knowledge graph
        based on the given documents and encodes passages, entities and facts separately for later retrieval.

        Parameters:
            docs : List[str]
                A list of documents to be indexed.
        """

        logger.info(f"Indexing Documents")

        logger.info(f"Performing OpenIE")

        if self.global_config.openie_mode == 'offline':
            self.pre_openie(docs)

        self.chunk_embedding_store.insert_strings(docs)
        chunk_to_rows = self.chunk_embedding_store.get_all_id_to_rows()
        logger.debug(f"chunk_to_rows: {json.dumps(chunk_to_rows, indent=4, ensure_ascii=False)}")

        all_openie_info, chunk_keys_to_process = self.load_existing_openie(chunk_to_rows.keys())
        new_openie_rows = {k : chunk_to_rows[k] for k in chunk_keys_to_process}
        logger.debug(f"new_openie_rows: {json.dumps(new_openie_rows, indent=4, ensure_ascii=False)}")

        if len(chunk_keys_to_process) > 0:
            new_ner_results_dict, new_triple_results_dict = self.openie.batch_openie(new_openie_rows)
            self.merge_openie_results(all_openie_info, new_openie_rows, new_ner_results_dict, new_triple_results_dict)

        if self.global_config.save_openie:
            self.save_openie_results(all_openie_info)

        ner_results_dict, triple_results_dict = reformat_openie_results(all_openie_info)
        
        # logging
        serializable_ner_results = {k: v.__dict__ for k, v in ner_results_dict.items()}
        serializable_triple_results = {k: v.__dict__ for k, v in triple_results_dict.items()}

        logger.debug(f"ner_results_dict: {json.dumps(serializable_ner_results, indent=4, ensure_ascii=False)}")
        logger.debug(f"triple_results_dict: {json.dumps(serializable_triple_results, indent=4, ensure_ascii=False)}")

        assert len(chunk_to_rows) == len(ner_results_dict) == len(triple_results_dict), f"len(chunk_to_rows): {len(chunk_to_rows)}, len(ner_results_dict): {len(ner_results_dict)}, len(triple_results_dict): {len(triple_results_dict)}"

        # prepare data_store
        chunk_ids = list(chunk_to_rows.keys())

        chunk_triples = [[text_processing(t) for t in triple_results_dict[chunk_id].triples] for chunk_id in chunk_ids]
        entity_nodes, chunk_triple_entities = extract_entity_nodes(chunk_triples)
        facts = flatten_facts(chunk_triples)

        logger.debug(f"chunk_triples: {json.dumps(chunk_triples, indent=4, ensure_ascii=False)}")
        logger.debug(f"entity_nodes: {json.dumps(entity_nodes, indent=4, ensure_ascii=False)}")
        logger.debug(f"chunk_triple_entities: {json.dumps(chunk_triple_entities, indent=4, ensure_ascii=False)}")
        logger.debug(f"facts: {json.dumps(facts, indent=4, ensure_ascii=False)}")

        logger.info(f"Encoding Entities")
        self.entity_embedding_store.insert_strings(entity_nodes)

        logger.info(f"Encoding Facts")
        self.fact_embedding_store.insert_strings([str(fact) for fact in facts])

        logger.info(f"Constructing Graph")

        self.node_to_node_stats = {}
        self.ent_node_to_chunk_ids = {}

        self.add_fact_edges(chunk_ids, chunk_triples)
        num_new_chunks = self.add_passage_edges(chunk_ids, chunk_triple_entities)

        if num_new_chunks > 0:
            logger.info(f"Found {num_new_chunks} new chunks to save into graph.")
            self.add_synonymy_edges()

            self.augment_graph()
            self.save_igraph()

    def rag_qa(self,
               queries: List[str|QuerySolution],
               gold_docs: List[List[str]] = None,
               gold_answers: List[List[str]] = None) -> Tuple[List[QuerySolution], List[str], List[Dict]] | Tuple[List[QuerySolution], List[str], List[Dict], Dict, Dict]:
        """
        Performs retrieval-augmented generation enhanced QA using the HippoRAG 2 framework.

        This method can handle both string-based queries and pre-processed QuerySolution objects. Depending
        on its inputs, it returns answers only or additionally evaluate retrieval and answer quality using
        recall @ k, exact match and F1 score metrics.

        Parameters:
            queries (List[Union[str, QuerySolution]]): A list of queries, which can be either strings or
                QuerySolution instances. If they are strings, retrieval will be performed.
            gold_docs (Optional[List[List[str]]]): A list of lists containing gold-standard documents for
                each query. This is used if document-level evaluation is to be performed. Default is None.
            gold_answers (Optional[List[List[str]]]): A list of lists containing gold-standard answers for
                each query. Required if evaluation of question answering (QA) answers is enabled. Default
                is None.

        Returns:
            Union[
                Tuple[List[QuerySolution], List[str], List[Dict]],
                Tuple[List[QuerySolution], List[str], List[Dict], Dict, Dict]
            ]: A tuple that always includes:
                - List of QuerySolution objects containing answers and metadata for each query.
                - List of response messages for the provided queries.
                - List of metadata dictionaries for each query.
                If evaluation is enabled, the tuple also includes:
                - A dictionary with overall results from the retrieval phase (if applicable).
                - A dictionary with overall QA evaluation metrics (exact match and F1 scores).

        """
        if gold_answers is not None:
            qa_em_evaluator = QAExactMatch(global_config=self.global_config)
            qa_f1_evaluator = QAF1Score(global_config=self.global_config)

        # Retrieving (if necessary)
        overall_retrieval_result = None

        if not isinstance(queries[0], QuerySolution):
            if gold_docs is not None:
                print(f"Có Gold_Docs")
                queries, overall_retrieval_result = self.retrieve(queries=queries, gold_docs=gold_docs)
            else:
                print(f"Khôngg có Gold_Docs")
                queries = self.retrieve(queries=queries)

        # Performing QA
        queries_solutions, all_response_message, all_metadata = self.qa(queries)
        logger.debug(f"queries_solutions ater qa {queries_solutions}")
        logger.debug(f"all_response_message after qa: {all_response_message}")
        logger.debug(f"all_metadata after qa: {all_metadata}")

        # Save results to a JSON file
        results_dir = 'outputs/results'
        os.makedirs(results_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_path = os.path.join(results_dir, f"qa_results_{timestamp}.json")

        results_to_save = []
        for solution in queries_solutions:
            top_3_docs = [
                    {"doc": doc, "score": float(score)} 
                    for doc, score in zip(solution.docs[:3], solution.doc_scores[:3])
                ]
            
            results_to_save.append({
                "question": solution.question,
                "answer": solution.answer,
                "retrieved_docs": top_3_docs
            })

        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(results_to_save, f, ensure_ascii=False, indent=4)
        
        logger.info(f"QA results saved to {file_path}")

        # Evaluating QA
        if gold_answers is not None:
            overall_qa_em_result, example_qa_em_results = qa_em_evaluator.calculate_metric_scores(
                gold_answers=gold_answers, predicted_answers=[qa_result.answer for qa_result in queries_solutions],
                aggregation_fn=np.max)
            overall_qa_f1_result, example_qa_f1_results = qa_f1_evaluator.calculate_metric_scores(
                gold_answers=gold_answers, predicted_answers=[qa_result.answer for qa_result in queries_solutions],
                aggregation_fn=np.max)

            # round off to 4 decimal places for QA results
            overall_qa_em_result.update(overall_qa_f1_result)
            overall_qa_results = overall_qa_em_result
            overall_qa_results = {k: round(float(v), 4) for k, v in overall_qa_results.items()}
            logger.info(f"Evaluation results for QA: {overall_qa_results}")

            # Save retrieval and QA results
            for idx, q in enumerate(queries_solutions):
                q.gold_answers = list(gold_answers[idx])
                if gold_docs is not None:
                    q.gold_docs = gold_docs[idx]

            return queries_solutions, all_response_message, all_metadata, overall_retrieval_result, overall_qa_results
        else:
            return queries_solutions, all_response_message, all_metadata

    def qa(self, queries: List[QuerySolution]) -> Tuple[List[QuerySolution], List[str], List[Dict]]:
        """
        Executes question-answering (QA) inference using a provided set of query solutions and a language model.

        Parameters:
            queries: List[QuerySolution]
                A list of QuerySolution objects that contain the user queries, retrieved documents, and other related information.

        Returns:
            Tuple[List[QuerySolution], List[str], List[Dict]]
                A tuple containing:
                - A list of updated QuerySolution objects with the predicted answers embedded in them.
                - A list of raw response messages from the language model.
                - A list of metadata dictionaries associated with the results.
        """
        #Running inference for QA
        logger.debug(f"Queries before qa: {queries}")
        all_qa_messages = []

        for query_solution in tqdm(queries, desc="Collecting QA prompts"):

            # obtain the retrieved docs
            retrieved_passages = query_solution.docs[:self.global_config.qa_top_k]

            prompt_user = ''
            for passage in retrieved_passages:
                prompt_user += f'Văn bản: {passage}\n\n'
            prompt_user += 'Câu hỏi: ' + query_solution.question + '\nSuy nghĩ: '

            if self.prompt_template_manager.is_template_name_valid(name=f'rag_qa_{self.global_config.dataset}'):
                # find the corresponding prompt for this dataset
                prompt_dataset_name = self.global_config.dataset
            else:
                # the dataset does not have a customized prompt template yet
                logger.debug(
                    f"rag_qa_{self.global_config.dataset} does not have a customized prompt template.")
                prompt_dataset_name = 'vietnamese_law'
            all_qa_messages.append(
                self.prompt_template_manager.render(name=f'rag_qa_{prompt_dataset_name}', prompt_user=prompt_user))

        all_qa_results = [self.llm_model.infer(qa_messages) for qa_messages in tqdm(all_qa_messages, desc="QA Reading")]

        all_response_message, all_metadata, all_cache_hit = zip(*all_qa_results)
        all_response_message, all_metadata = list(all_response_message), list(all_metadata)

        #Process responses and extract predicted answers.
        queries_solutions = []
        for query_solution_idx, query_solution in tqdm(enumerate(queries), desc="Extraction Answers from LLM Response"):
            response_content = all_response_message[query_solution_idx]
            try:
                # pred_ans = response_content.strip()
                pred_ans = response_content.rpartition("\n\nTrả lời:")[2].strip()
            except Exception as e:
                logger.warning(f"Error in parsing the answer from the raw LLM QA inference response: {str(e)}!")
                pred_ans = response_content

            query_solution.answer = pred_ans
            queries_solutions.append(query_solution)

        return queries_solutions, all_response_message, all_metadata
    
    def retrieve(self,
                 queries: List[str],
                 num_to_retrieve: int = None,
                 gold_docs: List[List[str]] = None) -> List[QuerySolution] | Tuple[List[QuerySolution], Dict]:
        retrieve_start_time = time.time()  # Record start time

        if num_to_retrieve is None:
            num_to_retrieve = self.global_config.retrieval_top_k

        if gold_docs is not None:
            retrieval_recall_evaluator = RetrievalRecall(global_config=self.global_config)

        if not self.ready_to_retrieve:
            print("prepare_retrieval_objects dc activated!")
            self.prepare_retrieval_objects()

        #self.get_query_embeddings(queries)
        # add for debug
        self.get_query_embeddings_through_retrieved_facts(queries)

        # Convert numpy arrays to lists for JSON serialization
        serializable_embeddings = {
            'triple': {
                query: embedding.tolist() if isinstance(embedding, np.ndarray) else embedding
                for query, embedding in self.query_to_embedding.get('triple', {}).items()
            },
            'passage': {
                query: embedding.tolist() if isinstance(embedding, np.ndarray) else embedding
                for query, embedding in self.query_to_embedding.get('passage', {}).items()
            }
        }

        # Define directory and filename
        output_dir = 'outputs/embeddings'
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_path = os.path.join(output_dir, f"query_embeddings_{timestamp}.json")
        facts_output_dir = 'outputs/retrieved_facts'
        os.makedirs(facts_output_dir, exist_ok=True)
        facts_file_path = os.path.join(facts_output_dir, f"retrieved_facts_{timestamp}.txt")

        # Save to JSON
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(serializable_embeddings, f, ensure_ascii=False, indent=4)
        
        logger.info(f"Query embeddings saved to {file_path}")

        retrieval_results = []

        for q_idx, query in tqdm(enumerate(queries), desc="Retrieving", total=len(queries)):
            rerank_start = time.time()
            query_fact_scores = self.get_fact_scores(query)
            top_k_fact_indices, top_k_facts, rerank_log = self.rerank_facts(query, query_fact_scores)

            logger.info(f"Top retrieved facts for query '{query}': {top_k_facts}")
            with open(facts_file_path, 'a', encoding='utf-8') as f:
                f.write(f"--- Query: {query} ---\n")
                if top_k_facts:
                    for fact in top_k_facts:
                        f.write(f"{fact}\n")
                else:
                    f.write("No facts found.\n")
                f.write("\n")
            
            rerank_end = time.time()

            self.rerank_time += rerank_end - rerank_start

            if len(top_k_facts) == 0:
                logger.info(f'For {query}, no facts found after reranking, return DPR results')
                sorted_doc_ids, sorted_doc_scores = self.dense_passage_retrieval(query)
            else:
                logger.info(f'For {query}, do graph search')
                sorted_doc_ids, sorted_doc_scores = self.graph_search_with_fact_entities(query=query,
                                                                                         link_top_k=self.global_config.linking_top_k,
                                                                                         query_fact_scores=query_fact_scores,
                                                                                         top_k_facts=top_k_facts,
                                                                                         top_k_fact_indices=top_k_fact_indices,
                                                                                         passage_node_weight=self.global_config.passage_node_weight)

            top_k_docs = [self.chunk_embedding_store.get_row(self.passage_node_keys[idx])["content"] for idx in sorted_doc_ids[:num_to_retrieve]]

            retrieval_results.append(QuerySolution(question=query, docs=top_k_docs, doc_scores=sorted_doc_scores[:num_to_retrieve]))
            logger.debug(f"retrieval_results: {retrieval_results}")

        retrieve_end_time = time.time()  # Record end time

        self.all_retrieval_time += retrieve_end_time - retrieve_start_time

        logger.info(f"Total Retrieval Time {self.all_retrieval_time:.2f}s")
        logger.info(f"Total Recognition Memory Time {self.rerank_time:.2f}s")
        logger.info(f"Total PPR Time {self.ppr_time:.2f}s")
        logger.info(f"Total Misc Time {self.all_retrieval_time - (self.rerank_time + self.ppr_time):.2f}s")

        # Evaluate retrieval
        if gold_docs is not None:
            k_list = [1, 2, 5, 10, 20, 30, 50, 100, 150, 200]
            overall_retrieval_result, example_retrieval_results = retrieval_recall_evaluator.calculate_metric_scores(gold_docs=gold_docs, retrieved_docs=[retrieval_result.docs for retrieval_result in retrieval_results], k_list=k_list)
            logger.info(f"Evaluation results for retrieval: {overall_retrieval_result}")

            return retrieval_results, overall_retrieval_result
        else:
            return retrieval_results
    
    def get_query_embeddings(self, queries: List[str] | List[QuerySolution]):
        """
        Retrieves embeddings for given queries and updates the internal query-to-embedding mapping. The method determines whether each query
        is already present in the `self.query_to_embedding` dictionary under the keys 'triple' and 'passage'. If a query is not present in
        either, it is encoded into embeddings using the embedding model and stored.

        Args:
            queries List[str] | List[QuerySolution]: A list of query strings or QuerySolution objects. Each query is checked for
            its presence in the query-to-embedding mappings.
        """

        all_query_strings = []
        for query in queries:
            if isinstance(query, QuerySolution) and (
                    query.question not in self.query_to_embedding['triple'] or query.question not in
                    self.query_to_embedding['passage']):
                all_query_strings.append(query.question)
            elif query not in self.query_to_embedding['triple'] or query not in self.query_to_embedding['passage']:
                all_query_strings.append(query)

        if len(all_query_strings) > 0:
            # get all query embeddings
            logger.info(f"Encoding {len(all_query_strings)} queries for query_to_fact.")
            query_embeddings_for_triple = self.embedding_model.batch_encode(all_query_strings,
                                                                            instruction=get_query_instruction_vn('query_to_fact'),
                                                                            norm=True)
            for query, embedding in zip(all_query_strings, query_embeddings_for_triple):
                self.query_to_embedding['triple'][query] = embedding

            logger.info(f"Encoding {len(all_query_strings)} queries for query_to_passage.")
            query_embeddings_for_passage = self.embedding_model.batch_encode(all_query_strings,
                                                                             instruction=get_query_instruction_vn('query_to_passage'),
                                                                             norm=True)
            for query, embedding in zip(all_query_strings, query_embeddings_for_passage):
                self.query_to_embedding['passage'][query] = embedding

    def get_query_embeddings_through_retrieved_facts(self, queries: List[str] | List[QuerySolution]):
        """
        Another way to get query_embeddings. Instead of using embedding model to extract triples and embed at the same time, we separate that into 2 steps. 
        """
        all_query_strings = []
        for query in queries:
            if isinstance(query, QuerySolution) and (
                    query.question not in self.query_to_embedding['triple'] or query.question not in
                    self.query_to_embedding['passage']):
                all_query_strings.append(query.question)
            elif query not in self.query_to_embedding['triple'] or query not in self.query_to_embedding['passage']:
                all_query_strings.append(query)

        if len(all_query_strings) > 0:
            # construct input for batch_openie
            logger.info(f"Encoding query using new get_query_embedding function!")
            queries_for_openie = {}
            for query in all_query_strings:
                query_hash_id = compute_mdhash_id(query, prefix=("query-"))
                queries_for_openie[query_hash_id] = {"hash_id": query, "content": query}

            queries_ner_results_dict, queries_triple_results_dict = self.openie.batch_openie(queries_for_openie)
            logger.debug(f"queries_ner_results_dict: {json_dumps_readable(queries_ner_results_dict)}")
            logger.debug(f"queries_triple_results_dict: {json_dumps_readable(queries_triple_results_dict)}")
            
            query_ids = list(queries_for_openie.keys())
            chunk_triples = [[text_processing(t) for t in queries_triple_results_dict[query_id].triples] for query_id in query_ids]

            # entity_nodes, chunk_triple_entities = extract_entity_nodes(chunk_triples)
            # facts = flatten_facts(chunk_triples)
            query_to_triples = dict()
            for query, extracted_triples in zip(all_query_strings, chunk_triples):
                query_to_triples[query] = extracted_triples
            
            logger.debug(f"result extract triples from queries: {json_dumps_readable(query_to_triples)}")

            # Encoding query_to_fact
            logger.info(f"Encoding {len(all_query_strings)} queries for query_to_fact.")
            query_embeddings_for_triple = ""
            for query, triples_in_query in query_to_triples.items():
                logger.info(f"query: {query}")
                logger.info(f"triples_in_query: {triples_in_query}")
                if len(triples_in_query) == 0:
                    query_embeddings_for_triple = self.embedding_model.batch_encode(query,
                                                                                instruction=get_query_instruction_vn('query_to_fact'),
                                                                                norm=True)
                else:
                    query_embeddings_for_triple = self.embedding_model.batch_encode(stringify_fact(triples_in_query),
                                                                                instruction=get_query_instruction_vn('fact_query_to_fact'),
                                                                                norm=True)
            for query, embedding in zip(all_query_strings, query_embeddings_for_triple):
                self.query_to_embedding['triple'][query] = embedding

            # Encoding query_to_passage
            logger.info(f"Encoding {len(all_query_strings)} queries for query_to_passage.")
            query_embeddings_for_passage = self.embedding_model.batch_encode(all_query_strings,
                                                                                instruction=get_query_instruction_vn('query_to_passage'),
                                                                                norm=True)
            for query, embedding in zip(all_query_strings, query_embeddings_for_passage):
                self.query_to_embedding['passage'][query] = embedding

        # logger.info(f"Encoding query using new get_query_embedding function!")
        # queries_for_openie = {}
        # for query in queries:
        #     query_hash_id = compute_mdhash_id(query, prefix=("query-"))
        #     queries_for_openie[query_hash_id] = {"hash_id": query, "content": query}

        # queries_ner_results_dict, queries_triple_results_dict = self.openie.batch_openie(queries_for_openie)
        # logger.debug(f"queries_ner_results_dict: {json_dumps_readable(queries_ner_results_dict)}")
        # logger.debug(f"queries_triple_results_dict: {json_dumps_readable(queries_triple_results_dict)}")
        
        # query_ids = list(queries_for_openie.keys())
        # chunk_triples = [[text_processing(t) for t in queries_triple_results_dict[query_id].triples] for query_id in query_ids]

        # # entity_nodes, chunk_triple_entities = extract_entity_nodes(chunk_triples)
        # # facts = flatten_facts(chunk_triples)
        # query_to_triples = dict()
        # for query, extracted_triples in zip(queries, chunk_triples):
        #     query_to_triples[query] = extracted_triples
        
        # logger.debug(f"result extract triples from queries: {json_dumps_readable(query_to_triples)}")

        # # Encoding query_to_fact
        # logger.info(f"Encoding {len(queries)} queries for query_to_fact.")
        # for query, triples_in_query in query_to_triples.items():
        #     logger.info(f"query: {query}")
        #     logger.info(f"triples_in_query: {triples_in_query}")
        #     if len(triples_in_query) == 0:
        #         query_embeddings_for_triple = self.embedding_model.batch_encode(query,
        #                                                                     instruction=get_query_instruction_vn('query_to_fact'),
        #                                                                     norm=True)
        #     else:
        #         query_embeddings_for_triple = self.embedding_model.batch_encode(stringify_fact(triples_in_query),
        #                                                                     instruction=get_query_instruction_vn('fact_query_to_fact'),
        #                                                                     norm=True)
        # for query, embedding in zip(queries, query_embeddings_for_triple):
        #     self.query_to_embedding['triple'][query] = embedding

        # # Encoding query_to_passage
        # logger.info(f"Encoding {len(all_query_strings)} queries for query_to_passage.")
        # query_embeddings_for_passage = self.embedding_model.batch_encode(all_query_strings,
        #                                                                     instruction=get_query_instruction_vn('query_to_passage'),
        #                                                                     norm=True)
        # for query, embedding in zip(all_query_strings, query_embeddings_for_passage):
        #     self.query_to_embedding['passage'][query] = embedding

    def get_fact_scores(self, query: str) -> np.ndarray:
        """
        Retrieves and computes normalized similarity scores between the given query and pre-stored fact embeddings.

        Parameters:
        query : str
            The input query text for which similarity scores with fact embeddings
            need to be computed.

        Returns:
        numpy.ndarray
            A normalized array of similarity scores between the query and fact
            embeddings. The shape of the array is determined by the number of
            facts.

        Raises:
        KeyError
            If no embedding is found for the provided query in the stored query
            embeddings dictionary.
        """
        query_embedding = self.query_to_embedding['triple'].get(query, None)
        if query_embedding is None:
            query_embedding = self.embedding_model.batch_encode(query,
                                                                instruction=get_query_instruction_vn('query_to_fact'),
                                                                norm=True)

        # Check if there are any facts
        if len(self.fact_embeddings) == 0:
            logger.warning("No facts available for scoring. Returning empty array.")
            return np.array([])
            
        try:
            query_fact_scores = np.dot(self.fact_embeddings, query_embedding.T) # shape: (#facts, )
            query_fact_scores = np.squeeze(query_fact_scores) if query_fact_scores.ndim == 2 else query_fact_scores
            query_fact_scores = min_max_normalize(query_fact_scores)
            return query_fact_scores
        except Exception as e:
            logger.error(f"Error computing fact scores: {str(e)}")
            return np.array([])

    def dense_passage_retrieval(self, query: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Conduct dense passage retrieval to find relevant documents for a query.

        This function processes a given query using a pre-trained embedding model
        to generate query embeddings. The similarity scores between the query
        embedding and passage embeddings are computed using dot product, followed
        by score normalization. Finally, the function ranks the documents based
        on their similarity scores and returns the ranked document identifiers
        and their scores.

        Parameters
        ----------
        query : str
            The input query for which relevant passages should be retrieved.

        Returns
        -------
        tuple : Tuple[np.ndarray, np.ndarray]
            A tuple containing two elements:
            - A list of sorted document identifiers based on their relevance scores.
            - A numpy array of the normalized similarity scores for the corresponding
              documents.
        """
        query_embedding = self.query_to_embedding['passage'].get(query, None)
        if query_embedding is None:
            query_embedding = self.embedding_model.batch_encode(query,
                                                                instruction=get_query_instruction_vn('query_to_passage'),
                                                                norm=True)
        query_doc_scores = np.dot(self.passage_embeddings, query_embedding.T)
        query_doc_scores = np.squeeze(query_doc_scores) if query_doc_scores.ndim == 2 else query_doc_scores
        query_doc_scores = min_max_normalize(query_doc_scores)

        sorted_doc_ids = np.argsort(query_doc_scores)[::-1]
        sorted_doc_scores = query_doc_scores[sorted_doc_ids.tolist()]
        return sorted_doc_ids, sorted_doc_scores
