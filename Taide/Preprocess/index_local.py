import chromadb
import json
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import VectorStoreIndex
from llama_index.core import StorageContext
from llama_index.core import Document
from llama_index.core import Settings


def indexing(index_name):
    """
    將指定類別的 JSON 文檔載入並索引到 Chroma 向量資料庫中。

    此函數執行以下步驟：
    1. 從 ../documents/{index_name}.json 文件中載入文檔數據。
    2. 將每個文檔轉換為 LlamaIndex 的 Document 物件，並收集到文檔列表中。
    3. 連接到持久化的 Chroma 資料庫，並獲取或創建名為 index_name 的集合。
    4. 使用 ChromaVectorStore 作為向量儲存庫，並建立 StorageContext。
    5. 使用 VectorStoreIndex 從文檔中創建索引，並顯示進度條。
    6. 打印索引創建完成的訊息。

    參數:
        index_name (str): 要索引的文檔類別名稱（例如 "Pediatric"、"Surgical"、"Oncology"）。

    異常:
        FileNotFoundError: 如果指定的 JSON 文件不存在。
        json.JSONDecodeError: 如果 JSON 文件的格式不正確。
        chromadb.errors.ChromaDBException: 如果連接或操作 Chroma 資料庫時發生錯誤。
    """
    print(f"Processing {index_name}...")
    json_path = f"./documents/{index_name}.json"

    try:
        with open(json_path, "rb") as f:
            docs = json.load(f)
        print(f"Number of documents in {index_name}: {len(docs)}")
    except FileNotFoundError:
        print(f"Error: The file {json_path} does not exist.")
        return
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON from {json_path}: {e}")
        return

    documents = []
    for doc in docs:
        temp = Document()
        temp.text = doc.get("title", "")
        temp.metadata = doc.get("metadata", {})
        documents.append(temp)

    try:
        db = chromadb.PersistentClient(path="./local_data")
        chroma_collection = db.get_or_create_collection(index_name)
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        index = VectorStoreIndex.from_documents(
            documents, storage_context=storage_context, show_progress=True
        )
        print(f"{index_name} index created successfully.")
    except chromadb.errors.ChromaDBException as e:
        print(f"Error with ChromaDB operations: {e}")


if __name__ == "__main__":
    """
    主執行入口點。

    此部分負責：
    1. 設定索引的塊大小和嵌入模型。
    2. 調用 indexing 函數來索引 "Pediatric"、"Surgical"...。
    """
    # 設定索引的塊大小
    Settings.chunk_size = 35000

    # 設定嵌入模型為 HuggingFaceEmbedding
    Settings.embed_model = HuggingFaceEmbedding(
        model_name="intfloat/multilingual-e5-large",
        device="cuda",
        embed_batch_size = 200,
        cache_folder="./model_cache",
    )

    # 如果需要使用 OllamaEmbedding，可以取消以下註釋並進行配置
    # Settings.embed_model = OllamaEmbedding(
    #     model_name="your-model-name",
    #     embed_batch_size=200,
    # )

    # 索引不同類別的文檔
    indexing('Pediatric')
    indexing('Surgical')
    indexing('Oncology')
    indexing('IM')
    indexing('OAGD')