import os
os.environ ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import get_dataset

import time
import numpy as np
import pandas as pd
import argparse
import faiss
import glob

# Parse command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--model_type", type=str, default="semantic_model", choices=["semantic_model", "ckipbert"],
                    help="Type of model to use: 'semantic_model' or 'ckipbert'")
parser.add_argument("--top_k", type=int, default=5, help="Number of top results to return")
args = parser.parse_args()

# Prepare the model and inference function based on the model type
if args.model_type == "semantic_model":
    from semantic_model import get_semantic_model, inference
    model, tokenizer = get_semantic_model()
elif args.model_type == "ckipbert":
    from ckipbert import get_ckipbert, inference
    model, tokenizer = get_ckipbert()

# Set the embeddings directory based on model type
embeddings_dir = f'./embeddings/{args.model_type}/'

# Load pre-computed product names and embeddings
product_names = []
product_embeddings = []

student_id = "M11352032"
results_path = "./results"

# Find top100 query
# 使用 glob 將目錄中的所有 csv 檔案路徑列出來
items_folder = '../tf-idf-for-EE5327701/results'
csv_files = glob.glob(os.path.join(items_folder, '*.csv'))

# 取得檔案名與紀錄數的列表 [(檔案路徑, 紀錄數)]
file_record_counts = []
for file in csv_files:
    try:
        # 使用 pandas 讀取 CSV 檔案，並取得其行數（紀錄數）
        df = pd.read_csv(file)
        record_count = len(df)
        file_record_counts.append((file, record_count))
    except Exception as e:
        print(f"Error reading {file}: {e}")

# 按照紀錄數從多到少排序
sorted_files = sorted(file_record_counts, key=lambda x: x[1], reverse=True)

top100_query = []
top100_query_count = []
# 列印出排序後的前 100 個檔案名稱，並提取「_」之後，".csv" 之前的部分，並顯示紀錄數
for file, record_count in sorted_files[:100]:
    # 取得檔案名（不包含路徑）
    filename = os.path.basename(file)
    
    # 找到關鍵字
    if 'M11352032_tfidf_' in filename:
        extracted_name = filename.split('M11352032_tfidf_')[1].replace('.csv', '')
        top100_query.append(extracted_name)
        top100_query_count.append(record_count)
        print(f"{extracted_name}: {record_count} records")

# Ensure the embeddings directory exists
if not os.path.exists(embeddings_dir):
    raise FileNotFoundError(f"Embeddings directory '{embeddings_dir}' not found.")

# Loop through all .npy files in the embeddings directory
for file in os.listdir(embeddings_dir):
    if file.endswith('.npy'):
        embedding_file = os.path.join(embeddings_dir, file)
        csv_file = os.path.join('./random_samples_1M', file.replace('.npy', '.csv'))

        # Check if the corresponding CSV file exists
        if not os.path.exists(csv_file):
            continue

        # Load product names from the CSV file
        items_df = pd.read_csv(csv_file)
        product_names.extend(items_df['product_name'].values)

        # Load product embeddings from the .npy file
        embeddings = np.load(embedding_file)
        product_embeddings.append(embeddings)

# Concatenate all embeddings into a single numpy array
product_embeddings = np.concatenate(product_embeddings, axis=0)

print(f'Number of products: {len(product_names)}')
print(f'Number of pre-computed embeddings: {product_embeddings.shape[0]}')

# Convert embeddings to float32
product_embeddings = product_embeddings.astype('float32')

# Normalize embeddings for cosine similarity
faiss.normalize_L2(product_embeddings)

# Build FAISS index
embedding_dim = product_embeddings.shape[1]
index = faiss.IndexFlatIP(embedding_dim)  # Using Inner Product as similarity measure
index.add(product_embeddings)

print(f'FAISS index built with {index.ntotal} vectors.')

# Convert product names to pandas Series for easy indexing
product_names_series = pd.Series(product_names)

# Function to search for the top k items
def search(query, product_names_series, index, top_k=args.top_k):
    # Get the embedding for the query
    query_embedding, _ = inference(tokenizer, model, [query] , 16)
    query_embedding = np.array([query_embedding]).astype('float32')[0]

    # Normalize query embedding
    faiss.normalize_L2(query_embedding)

    # Search using the index
    scores, indices = index.search(query_embedding, top_k)

    # Retrieve search results
    top_k_names = product_names_series.iloc[indices[0]].values
    top_k_scores = scores[0]

    # Create a DataFrame with product names and their corresponding scores
    results_df = pd.DataFrame({
        'product_name': top_k_names,
        'product_score': top_k_scores
    })

    # Define the file path for saving the CSV
    file_path = os.path.join(results_path, f"{student_id}_semantic_{query}.csv")

    # Save the DataFrame to a CSV file
    results_df.to_csv(file_path, index=False, encoding='utf-8-sig')
    print(f"Results for {query} have been saved to {file_path}")

    return top_k_names, top_k_scores

# Run in interactive mode
# while True:
#     query = input('Enter query (type "exit" to quit): ')
#     if query.lower() == 'exit':
#         break

#     start_time = time.time()
#     top_k_names, scores = search(query, product_names_series, index)
#     elapsed_time = time.time() - start_time
#     print(f'Took {elapsed_time:.4f} seconds to search')

#     for i, (name, score) in enumerate(zip(top_k_names, scores)):
#         print(f'[Rank {i+1} | Score: {score:.4f}] {name}')

for i in range(len(top100_query)):
    top_k = top100_query_count[i]
    start_time = time.time()
    search(top100_query[i], product_names_series, index, top_k)
    elapsed_time = time.time() - start_time
    print(f'Took {elapsed_time:.4f} seconds to search {top100_query[i]}')
