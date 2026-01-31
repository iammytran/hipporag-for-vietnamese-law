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
            logger.debug(f"num_to_retrieve: {num_to_retrieve}")

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
            logger.debug(f"query_fact_scores: {query_fact_scores}")
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
            logger.debug(f"top_k_docs: {top_k_docs}")

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
            logger.debug(f"len(fact_embeddings): {len(self.fact_embeddings)}")
            logger.debug(f"fact_embeddings: {self.fact_embeddings}")
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

    def graph_search_with_fact_entities(self, query: str,
                                        link_top_k: int,
                                        query_fact_scores: np.ndarray,
                                        top_k_facts: List[Tuple],
                                        top_k_fact_indices: List[str],
                                        passage_node_weight: float = 0.05) -> Tuple[np.ndarray, np.ndarray]:
        """
        Computes document scores based on fact-based similarity and relevance using personalized
        PageRank (PPR) and dense retrieval models. This function combines the signal from the relevant
        facts identified with passage similarity and graph-based search for enhanced result ranking.

        Parameters:
            query (str): The input query string for which similarity and relevance computations
                need to be performed.
            link_top_k (int): The number of top phrases to include from the linking score map for
                downstream processing.
            query_fact_scores (np.ndarray): An array of scores representing fact-query similarity
                for each of the provided facts.
            top_k_facts (List[Tuple]): A list of top-ranked facts, where each fact is represented
                as a tuple of its subject, predicate, and object.
            top_k_fact_indices (List[str]): Corresponding indices or identifiers for the top-ranked
                facts in the query_fact_scores array.
            passage_node_weight (float): Default weight to scale passage scores in the graph.

        Returns:
            Tuple[np.ndarray, np.ndarray]: A tuple containing two arrays:
                - The first array corresponds to document IDs sorted based on their scores.
                - The second array consists of the PPR scores associated with the sorted document IDs.
        """

        #Assigning phrase weights based on selected facts from previous steps.
        linking_score_map = {}  # from phrase to the average scores of the facts that contain the phrase
        phrase_scores = {}  # store all fact scores for each phrase regardless of whether they exist in the knowledge graph or not
        phrase_weights = np.zeros(len(self.graph.vs['name']))
        passage_weights = np.zeros(len(self.graph.vs['name']))
        number_of_occurs = np.zeros(len(self.graph.vs['name']))

        phrases_and_ids = set()

        for rank, f in enumerate(top_k_facts):
            subject_phrase = f[0].lower()
            predicate_phrase = f[1].lower()
            object_phrase = f[2].lower()
            fact_score = query_fact_scores[
                top_k_fact_indices[rank]] if query_fact_scores.ndim > 0 else query_fact_scores

            for phrase in [subject_phrase, object_phrase]: # phrase: ("f[0], f[1]" trong fact)
                phrase_key = compute_mdhash_id(
                    content=phrase,
                    prefix="entity-"
                )
                phrase_id = self.node_name_to_vertex_idx.get(phrase_key, None) #phase_key: 1 hash_id

                if phrase_id is not None:
                    weighted_fact_score = fact_score

                    if len(self.ent_node_to_chunk_ids.get(phrase_key, set())) > 0:
                        weighted_fact_score /= len(self.ent_node_to_chunk_ids[phrase_key])

                    phrase_weights[phrase_id] += weighted_fact_score # dict: (phrase_id: weights)
                    number_of_occurs[phrase_id] += 1

                phrases_and_ids.add((phrase, phrase_id))

        # lấy điểm trung bình phrase weights
        phrase_weights /= number_of_occurs

        for phrase, phrase_id in phrases_and_ids:
            if phrase not in phrase_scores:
                phrase_scores[phrase] = []
            logger.debug(f"phrase: {phrase}")
            phrase_scores[phrase].append(phrase_weights[phrase_id])
        logger.debug(f"phrase_scores: {phrase_scores}")

        # calculate average fact score for each phrase
        for phrase, scores in phrase_scores.items():
            linking_score_map[phrase] = float(np.mean(scores))

        if link_top_k:
            phrase_weights, linking_score_map = self.get_top_k_weights(link_top_k,
                                                                           phrase_weights,
                                                                           linking_score_map)  # at this stage, the length of linking_scope_map is determined by link_top_k

        logger.debug(f"linking_score_map phrase_node: {linking_score_map}")

        #Get passage scores according to chosen dense retrieval model
        dpr_sorted_doc_ids, dpr_sorted_doc_scores = self.dense_passage_retrieval(query)
        normalized_dpr_sorted_scores = min_max_normalize(dpr_sorted_doc_scores)

        for i, dpr_sorted_doc_id in enumerate(dpr_sorted_doc_ids.tolist()):
            passage_node_key = self.passage_node_keys[dpr_sorted_doc_id]
            passage_dpr_score = normalized_dpr_sorted_scores[i]
            passage_node_id = self.node_name_to_vertex_idx[passage_node_key]
            passage_weights[passage_node_id] = passage_dpr_score * passage_node_weight # dict(chunk_id: weights)
            passage_node_text = self.chunk_embedding_store.get_row(passage_node_key)["content"]
            linking_score_map[passage_node_text] = passage_dpr_score * passage_node_weight # dict(chunk_text: weights)
        
        

        #Combining phrase and passage scores into one array for PPR
        node_weights = phrase_weights + passage_weights # node_weights: dict (node_id: weight); node có thể là phrase hay passage
        logger.debug(f"node_weights: {node_weights}")

        #Recording top 30 facts in linking_score_map
        if len(linking_score_map) > 30:
            linking_score_map = dict(sorted(linking_score_map.items(), key=lambda x: x[1], reverse=True)[:30])
        
        logger.debug(f"linking_score_map passage_node: {linking_score_map}")

        assert sum(node_weights) > 0, f'No phrases found in the graph for the given facts: {top_k_facts}'

        #Running PPR algorithm based on the passage and phrase weights previously assigned
        ppr_start = time.time()
        ppr_sorted_doc_ids, ppr_sorted_doc_scores = self.run_ppr(node_weights, damping=self.global_config.damping)
        ppr_end = time.time()

        self.ppr_time += (ppr_end - ppr_start)

        assert len(ppr_sorted_doc_ids) == len(
            self.passage_node_idxs), f"Doc prob length {len(ppr_sorted_doc_ids)} != corpus length {len(self.passage_node_idxs)}"

        return ppr_sorted_doc_ids, ppr_sorted_doc_scores
    
    def run_ppr(self,
                reset_prob: np.ndarray,
                damping: float =0.5) -> Tuple[np.ndarray, np.ndarray]:
        """
        Runs Personalized PageRank (PPR) on a graph and computes relevance scores for
        nodes corresponding to document passages. The method utilizes a damping
        factor for teleportation during rank computation and can take a reset
        probability array to influence the starting state of the computation.

        Parameters:
            reset_prob (np.ndarray): A 1-dimensional array specifying the reset
                probability distribution for each node. The array must have a size
                equal to the number of nodes in the graph. NaNs or negative values
                within the array are replaced with zeros.
            damping (float): A scalar specifying the damping factor for the
                computation. Defaults to 0.5 if not provided or set to `None`.

        Returns:
            Tuple[np.ndarray, np.ndarray]: A tuple containing two numpy arrays. The
                first array represents the sorted node IDs of document passages based
                on their relevance scores in descending order. The second array
                contains the corresponding relevance scores of each document passage
                in the same order.
        """

        if damping is None: damping = 0.5 # for potential compatibility
        reset_prob = np.where(np.isnan(reset_prob) | (reset_prob < 0), 0, reset_prob)

        logger.debug(f"node_name_to_vertex_idx: {self.node_name_to_vertex_idx}")
        logger.debug(f"reset_prob: {reset_prob}")

        pagerank_scores = self.graph.personalized_pagerank(
            vertices=range(len(self.node_name_to_vertex_idx)),
            damping=damping,
            directed=False,
            weights='weight',
            reset=reset_prob,
            implementation='prpack'
        )

        logger.debug(f"pagerank_scores: {pagerank_scores}")

        doc_scores = np.array([pagerank_scores[idx] for idx in self.passage_node_idxs])
        logger.debug(f"doc_scores: {doc_scores}")
        sorted_doc_ids = np.argsort(doc_scores)[::-1]
        sorted_doc_scores = doc_scores[sorted_doc_ids.tolist()]

        return sorted_doc_ids, sorted_doc_scores
    
    def rerank_facts(self, query: str, query_fact_scores: np.ndarray) -> Tuple[List[int], List[Tuple], dict]:
        """

        Args:

        Returns:
            top_k_fact_indicies:
            top_k_facts:
            rerank_log (dict): {'facts_before_rerank': candidate_facts, 'facts_after_rerank': top_k_facts}
                - candidate_facts (list): list of link_top_k facts (each fact is a relation triple in tuple data type).
                - top_k_facts:


        """
        # load args
        link_top_k: int = self.global_config.linking_top_k
        
        # Check if there are any facts to rerank
        if len(query_fact_scores) == 0 or len(self.fact_node_keys) == 0:
            logger.warning("No facts available for reranking. Returning empty lists.")
            return [], [], {'facts_before_rerank': [], 'facts_after_rerank': []}
            
        try:
            # Get the top k facts by score
            if len(query_fact_scores) <= link_top_k:
                # If we have fewer facts than requested, use all of them
                candidate_fact_indices = np.argsort(query_fact_scores)[::-1].tolist()
            else:
                # Otherwise get the top k
                candidate_fact_indices = np.argsort(query_fact_scores)[-link_top_k:][::-1].tolist()
                
            # Get the actual fact IDs
            real_candidate_fact_ids = [self.fact_node_keys[idx] for idx in candidate_fact_indices]
            fact_row_dict = self.fact_embedding_store.get_rows(real_candidate_fact_ids)
            candidate_facts = [eval(fact_row_dict[id]['content']) for id in real_candidate_fact_ids]
            
            # Rerank the facts
            top_k_fact_indices, top_k_facts, reranker_dict = self.rerank_filter(query,
                                                                                candidate_facts,
                                                                                candidate_fact_indices,
                                                                                len_after_rerank=link_top_k)
            
            rerank_log = {'facts_before_rerank': candidate_facts, 'facts_after_rerank': top_k_facts}
            logger.debug(f"rerank_log: {rerank_log}")
            
            return top_k_fact_indices, top_k_facts, rerank_log
            
        except Exception as e:
            logger.error(f"Error in rerank_facts: {str(e)}")
            return [], [], {'facts_before_rerank': [], 'facts_after_rerank': [], 'error': str(e)}