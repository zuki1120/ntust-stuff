from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.response_synthesizers import ResponseMode
from llama_index.core import get_response_synthesizer, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
# from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.core import PromptTemplate
from llama_index.core import VectorStoreIndex
from llama_index.core.postprocessor import SentenceTransformerRerank
from ollama import chat
import chromadb

# 嵌入模型初始化
def initialize_embedding_model():
    return HuggingFaceEmbedding(
        model_name = "intfloat/multilingual-e5-large",
        device = "cuda",
        embed_batch_size = 200,
        cache_folder = "./model_cache"
    )

# LLM 初始化
def initialize_llm():
    return Ollama(
        model = "cwchang/llama3-taide-lx-8b-chat-alpha1",
        request_timeout = 300
    )

# 索引初始化
def initialize_index():
    db = chromadb.PersistentClient(path = "./local_data")

    chroma_collection_Pediatric= db.get_or_create_collection("Pediatric")
    chroma_collection_Surgical= db.get_or_create_collection("Surgical")
    chroma_collection_Oncology= db.get_or_create_collection("Oncology")
    vector_store_Pediatric = ChromaVectorStore(chroma_collection = chroma_collection_Pediatric)
    vector_store_Surgical = ChromaVectorStore(chroma_collection = chroma_collection_Surgical)
    vector_store_Oncology = ChromaVectorStore(chroma_collection = chroma_collection_Oncology)
    return VectorStoreIndex.from_vector_store(vector_store_Pediatric), VectorStoreIndex.from_vector_store(vector_store_Surgical), VectorStoreIndex.from_vector_store(vector_store_Oncology)

# 重排序器初始化
def initialize_reranker():
    return SentenceTransformerRerank(
        model = "BAAI/bge-reranker-v2-m3", 
        top_n = 3, 
        device = "cuda"
    )

# 回應綜合器初始化
def initialize_synthesizer():
    prompt = PromptTemplate(
        "以下是信息：\n"
        "---------------------\n"
        "{context_str}\n"
        "---------------------\n"
        "請根據上述信息整理出重點後回答下列問題，僅限於所提供的內容，可以用條列式。\n"
        "請直接說答案，不需要過多的文字來解釋。\n"
        "問題：{query_str}\n"
        "回答："
    )
    
    return get_response_synthesizer(
        response_mode = ResponseMode.SIMPLE_SUMMARIZE,
        text_qa_template = prompt
    )

def classify_query(query: str) -> str:
    classification_prompt = (
        f"以下是用戶的問題：\n{query}\n"
        "請根據問題的內容，判斷應該使用以下哪個分類，如果非醫療相關分類使用NaN：\n"
        "- Pediatric\n"
        "- Surgical\n"
        "- Oncology\n"
        "- NaN\n"
        "僅從提供的四種分類中返回分類名稱，只需要回答分類，不要有其他文字，回答不要加'。'。"
    )
    response = chat(
        model = "cwchang/llama3-taide-lx-8b-chat-alpha1",
        messages=[
            {"role": "user", "content": classification_prompt}
        ]
    )

    return response.message.content

# 設置全局配置
Settings.embed_model = initialize_embedding_model()
Settings.llm = initialize_llm()