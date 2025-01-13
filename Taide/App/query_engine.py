import re
import logging
from llama_index.core.query_engine import RetrieverQueryEngine
from settings import (initialize_index, 
                      initialize_reranker, 
                      initialize_synthesizer,
                      classify_query)

# 初始化所需的模型和工具
index_Pediatric, index_Surgical, index_Oncology = initialize_index()
reranker = initialize_reranker()
response_synthesizer = initialize_synthesizer()
logger = logging.getLogger(__name__)

def process_query(query: str) -> str:
    """
    根據用戶輸入的查詢執行文檔檢索和回答生成。
    :param query: 用戶輸入的問題
    :return: 檢索結果的回答
    """

    logger.info(f"收到查詢: {query}")

    category = re.sub(r"[，。！？…《》“”‘’（）\[\]{}：；、,.!?()]", "", classify_query(query))

    logger.info(f"查詢分類: {category}")

    if category == "Pediatric":
        retriever = index_Pediatric.as_retriever(similarity_top_k = 10)
    elif category == "Surgical":
        retriever = index_Surgical.as_retriever(similarity_top_k = 10)
    elif category == "Oncology":
        retriever = index_Oncology.as_retriever(similarity_top_k = 10)
    else:
        return "抱歉，我只能回答醫療相關問題。"

    query_engine = RetrieverQueryEngine(
        retriever = retriever,
        response_synthesizer = response_synthesizer,
        node_postprocessors = [reranker],
    )
    response = query_engine.query(query)

    if not response or not response.response:
        logger.warning("查詢未返回任何結果")
        return "抱歉，我無法回答相關的答案。"

    logger.info(f"生成的回應: {response.response}")
    return response.response