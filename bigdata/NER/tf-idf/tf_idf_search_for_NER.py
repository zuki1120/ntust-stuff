# python .\tf_idf_revised.py ../02_程式集/Coupang_Scraping-main/results -k 5 -i
# %%
# Import required libraries
import time
from tqdm import tqdm
import numpy as np
import pandas as pd
import jieba
import os
import html
import glob
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle
from argparse import ArgumentParser
from datasets import load_dataset

# Load the Coupang Product Set 1M dataset
dataset = load_dataset('clw8998/Coupang-Product-Set-1M')
dataset = dataset['train']

# Command line arguments
argparser = ArgumentParser()
# argparser.add_argument('items_folder', type=str, help='Folder containing the items (csv files) to search.')
argparser.add_argument('-k', '--top_k', type=int, default=5, help='Number of top k items to return.')
argparser.add_argument('-f', '--file_idx', type=int, default=-1, help='File index of activate_item folder. Use -1 to load all files at once.')
argparser.add_argument('-i', '--interactive', action='store_true', help='Run in interactive mode.')
argparser.add_argument('-s', '--sample_size', type=int, default=100000, help='Number of items to sample from the dataset for TF-IDF model creation. Use -1 to load all items.')
argparser.add_argument('-a', '--all', action='store_true', help='Load all items without dropping duplicates.')
argparser.add_argument('-c', '--create', action='store_true', help='Create the TF-IDF models without using the saved models.')
argparser.add_argument('--api_server', action='store_true', help='Run in API server mode.')
args = argparser.parse_args()
# items_folder = args.items_folder

top_k = args.top_k
file_idx = args.file_idx
interactive = args.interactive
sample_size = args.sample_size
drop_duplicates = not args.all
create = args.create

student_id = "id"
results_path = "./results"


# %%
def preprocess_product_names(dataset):
    def preprocess_function(example):
        example['product_name'] = html.unescape(example['product_name']).strip()  # Unescape HTML
        return example
    
    dataset = dataset.map(preprocess_function, batched=False)
    return dataset

dataset = preprocess_product_names(dataset)

csv_file = 'id_assigned_queries.csv'
df = pd.read_csv(csv_file)
query250 = df['key_word'].to_list()
print(query250)

# %%
# Initialize tokenizer
timer_start = time.time()

# 定義簡單的分詞器
def jieba_tokenizer(text):
    # 使用 jieba 的精確模式進行分詞
    tokens = jieba.lcut(text, cut_all=False)
    # 定義不需要的符號列表
    stop_words = ['【','】','/','~','＊','、','（','）','+','‧',' ','']
    # 過濾掉不需要的符號
    tokens = [t for t in tokens if t not in stop_words]
    return tokens

# 使用 jieba_tokenizer 進行分詞
tokenizer = jieba_tokenizer


# %%

# Path to save/load the models
model_path = 'tf_idf_checkpoint.pkl'

# Function to save the models
def save_models_and_matrices(tfidf, items_tfidf_matrix, path):
    with open(path, 'wb') as file:
        pickle.dump({
            'tfidf': tfidf,
            'items_tfidf_matrix': items_tfidf_matrix,
        }, file)

# Function to load the models
def load_models_and_matrices(path):
    with open(path, 'rb') as file:
        data = pickle.load(file)
    return data['tfidf'], data['items_tfidf_matrix']


# %%

# Check if the models are already saved
if os.path.exists(model_path) and not create:
    # If saved, load the models
    tfidf, items_tfidf_matrix = load_models_and_matrices(model_path)
else:
    # If not saved, create the models
    print('TF-IDF models not found. Creating them...')

    tfidf = TfidfVectorizer(token_pattern=None, tokenizer=tokenizer, ngram_range=(1,2))
    items_tfidf_matrix = tfidf.fit_transform(tqdm(dataset['product_name']))
    
    save_models_and_matrices(tfidf, items_tfidf_matrix, model_path)

print(f'TF-IDF models loaded in {time.time() - timer_start:.2f} seconds.')


# %%

# Function to search for the top k items
def search(query, dataset):

    query_tfidf = tfidf.transform([query]) # sparse array
    scores = cosine_similarity(query_tfidf, items_tfidf_matrix)
    top_k_indices = np.argsort(-scores[0])[:top_k]
    

    top_k_names = dataset.select(top_k_indices)['product_name']
    top_k_scores = scores[0][top_k_indices]
    rank = list(range(1, 11))

    return top_k_names, rank, top_k_scores


# Run in interactive mode
if interactive and not args.api_server:
    columns = ['搜尋詞', 'Rank', 'tf-idf', 'ner_relevancy_1', 'score']
    results_df = pd.DataFrame(columns = columns)

    for i in range(len(query250)):
        top_k = 10
        start_time = time.time()
        product_name, rank, score = search(query250[i], dataset)
        row = pd.DataFrame({
            '搜尋詞': query250[i],
            'Rank': rank,
            'tf-idf': product_name,
            'ner_relevancy_1': 0,
            'score': score
        })

        results_df = pd.concat([results_df, row], ignore_index=True)

        elapsed_time = time.time() - start_time
        print(f'Took {elapsed_time:.4f} seconds to search {query250[i]}')

    output_file = 'tfidf_search_results.csv'
    results_df.to_csv(output_file, index=False, encoding='utf-8-sig')

    print(f"搜尋結果已保存到 {output_file}")
# %%



