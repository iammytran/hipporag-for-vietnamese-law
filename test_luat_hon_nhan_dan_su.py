import os
from typing import List
import json
import argparse
import logging
import requests
from bs4 import BeautifulSoup
import re
import io
from pypdf import PdfReader
import time
import tiktoken 

from src.hipporag import HippoRAGVnLaw
from src.hipporag.utils.config_utils import BaseConfig

# Đặt ở đầu file thực thi chính của bạn
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def chunk_text_by_tokens(text: str, encoding: tiktoken.Encoding, chunk_size: int = 10) -> List[str]:
    """
    Chia một văn bản lớn thành các chunk nhỏ hơn dựa trên số lượng token.
    """
    if not text:
        return []
        
    tokens = encoding.encode(text)
    chunks = []
    for i in range(0, len(tokens), chunk_size):
        chunk_tokens = tokens[i:i + chunk_size]
        chunk_text = encoding.decode(chunk_tokens)
        chunks.append(chunk_text)
    print(f"Đã chia văn bản gốc thành {len(chunks)} chunk.")
    return chunks

def main():
    # Lấy 3 văn bản lớn
    content_marriage_law, content_civil_law, full_text = load_law_docs()
    
    # --- BẮT ĐẦU CHIA CHUNK ---
    print("\n--- Bắt đầu quá trình chia văn bản thành các chunk ---")
    # Lấy encoding phù hợp với các model của OpenAI
    encoding = tiktoken.get_encoding("cl100k_base")
    CHUNK_SIZE = 256

    all_chunks = []
    
    print("\n[1] Đang xử lý Luật Hôn nhân và gia đình...")
    marriage_chunks = chunk_text_by_tokens(content_marriage_law, encoding, CHUNK_SIZE)
    all_chunks.extend(marriage_chunks)

    print("\n[2] Đang xử lý Bộ luật Dân sự...")
    civil_chunks = chunk_text_by_tokens(content_civil_law, encoding, CHUNK_SIZE)
    all_chunks.extend(civil_chunks)

    print("\n[3] Đang xử lý nội dung Bản án...")
    case_chunks = chunk_text_by_tokens(full_text, encoding, CHUNK_SIZE)
    all_chunks.extend(case_chunks)

    print(f"\n>>> Tổng cộng có {len(all_chunks)} chunk sẽ được đưa vào index.")
    print("--- Hoàn thành quá trình chia chunk ---\n")

    # Thay thế list `docs` cũ bằng list `all_chunks` đã được xử lý
    docs = all_chunks

    save_dir = 'outputs/transformer_test'  # Define save directory for HippoRAG objects (each LLM/Embedding model combination will create a new subdirectory)
    llm_model_name = 'Transformers/Qwen/Qwen2.5-3B-Instruct' # Any OpenAI model name
    embedding_model_name = 'Transformers/BAAI/bge-m3' # Embedding model name (NV-Embed, GritLM or Contriever for now)

    global_config = BaseConfig(
        openie_mode='Transformers-offline',
        information_extraction_model_name='Transformers/Qwen/Qwen2.5-3B-Instruct'
    )

    # Startup a HippoRAG instance
    hipporag = HippoRAGVnLaw(global_config,
                        save_dir=save_dir,
                        llm_model_name=llm_model_name,
                        embedding_model_name=embedding_model_name,
                        )

    # Run indexing
    hipporag.index(docs=docs)

    # Separate Retrieval & QA
    queries = [
        "Bản án 65/2025/HNGĐ-ST thuộc lĩnh vực nào?",
    ]

    # For Evaluation
    answers = [
        ["Hôn nhân và gia đình"],
    ]

    print(hipporag.rag_qa(queries=queries,
                                  gold_answers=answers)[-2:])


def load_law_docs():
    headers = {
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7',
        'Accept-Encoding': 'identity',
        'Accept-Language': 'en-US,en;q=0.9,vi;q=0.8',
        'Cache-Control': 'max-age=0',
        'Referer': 'https://www.google.com/',
        'Sec-Ch-Ua': '"Not/A)Brand";v="8", "Chromium";v="126", "Google Chrome";v="126"',
        'Sec-Ch-Ua-Mobile': '?0',
        'Sec-Ch-Ua-Platform': '"macOS"',
        'Sec-Fetch-Dest': 'document',
        'Sec-Fetch-Mode': 'navigate',
        'Sec-Fetch-Site': 'cross-site',
        'Sec-Fetch-User': '?1',
        'Upgrade-Insecure-Requests': '1',
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36',
    }
    
    marriage_url = 'https://thuvienphapluat.vn/van-ban/Quyen-dan-su/Luat-Hon-nhan-va-gia-dinh-2014-238640.aspx'
    civil_url = 'https://thuvienphapluat.vn/van-ban/Quyen-dan-su/Bo-luat-dan-su-2015-296215.aspx'

    content_marriage_law = ""
    content_civil_law = ""

    try:
        # --- Lấy nội dung Luật Hôn nhân và Gia đình ---
        print(f"Đang cào dữ liệu từ: {marriage_url}")
        response_marriage = requests.get(marriage_url, headers=headers)
        # response_marriage.raise_for_status() # Báo lỗi nếu status không phải 2xx
        response_marriage.encoding = 'utf-8'
        soup_marriage = BeautifulSoup(response_marriage.content, 'html.parser')

        parent_div_marriage = soup_marriage.find('div', id ='divContentDoc')

        if parent_div_marriage:
            # SỬA ĐỔI LOGIC: Lấy toàn bộ văn bản từ div cha một cách chính xác
            content_marriage_law = parent_div_marriage.get_text(separator='\n', strip=True)
            print("--- Lấy thành công nội dung Luật Hôn nhân và gia đình ---")
        else:
            print(f"Lỗi: Không tìm thấy div nội dung (class='content1') tại {marriage_url}")

        # --- Lấy nội dung Bộ luật Dân sự ---
        print(f"Đang cào dữ liệu từ: {civil_url}")
        response_civil = requests.get(civil_url, headers=headers)
        response_civil.raise_for_status()
        response_civil.encoding = 'utf-8'
        soup_civil = BeautifulSoup(response_civil.text, 'html.parser')

        # SỬA ĐỔI QUAN TRỌNG: Tìm theo class='content1'
        parent_div_civil = soup_civil.find('div', id='divContentDoc')

        if parent_div_civil:
            content_civil_law = parent_div_civil.get_text(separator='\n', strip=True)
            print("--- Lấy thành công nội dung Bộ luật Dân sự ---")
        else:
            print(f"Lỗi: Không tìm thấy div nội dung (class='content1') tại {civil_url}")

    except requests.exceptions.HTTPError as e:
        print(f"Lỗi HTTP: {e.response.status_code}. Trang web có thể đã chặn yêu cầu.")
    except requests.exceptions.RequestException as e:
        print(f"Lỗi kết nối mạng: {e}")
    except Exception as e:
        print(f"Đã xảy ra lỗi không xác định khi cào dữ liệu: {e}")

    # --- Đọc nội dung bản án từ file PDF ---
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Kết hợp đường dẫn thư mục với tên file PDF để có đường dẫn tuyệt đối
    file_path = os.path.join(script_dir, 'data', 'ban_an_hon_nhan_1.pdf')
    full_text = ""
    try:
        print(f"Đang đọc file PDF: {file_path}")
        with open(file_path, 'rb') as f:
            reader = PdfReader(f)
            for page in reader.pages:
                text = page.extract_text()
                if text:
                    full_text += text + "\n"
        print("--- Đọc thành công file PDF ---")
    except FileNotFoundError:
        print(f"Lỗi: Không tìm thấy file '{file_path}'. Hãy chắc chắn rằng file này nằm cùng thư mục với script đang chạy.")
    except Exception as e:
        print(f"Đã xảy ra lỗi khi đọc file PDF: {e}")
        
    return content_marriage_law, content_civil_law, full_text

if __name__ == "__main__":
    main()
    # a, b, c = load_law_docs()
    # print(c)