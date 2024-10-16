# big-data
step1
> Coupang Scraper \
> repo: https://github.com/clw8998/Coupang_Scraper

修改跳過機制
```python
all_contain_string = any(search_string in file_name for file_name in csv_files)
if all_contain_string:
    print(f"Results for {query} have already been scraped. Skipping...\n")
    continue
```

step2
>Tf-idf search \
>repo: https://github.com/agbld/tf-idf-for-EE5327701

1.增加找最熱門100個關鍵字
```python
# find top100 query
# 使用 glob 將目錄中的所有 csv 檔案路徑列出來
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
    
    # 找到 "_" 之後和 ".csv" 之前的部分
    if '_' in filename:
        extracted_name = filename.split('_')[1].replace('.csv', '')
        top100_query.append(extracted_name)
        top100_query_count.append(record_count)
        print(f"{extracted_name}: {record_count} records")
```

2.修改search function，存成csv
```python
def search(query, items_df):
    data = []

    query_tfidf = tfidf.transform([query]) # sparse array
    scores = cosine_similarity(query_tfidf, items_tfidf_matrix)
    top_k_indices = np.argsort(-scores[0])[:top_k]
    
    top_k_names = items_df['product_name'].values[top_k_indices]
    top_k_scores = scores[0][top_k_indices]

    for name, score in zip(top_k_names, top_k_scores):
        data.append({'product_name': name, 'product_score': score})
        
        data_df = pd.DataFrame(data)

        # Save the results to a CSV file
        file_path = os.path.join(results_path, f"{student_id}_tfidf_{query}.csv")
        data_df.to_csv(file_path, index=False, encoding='utf-8-sig')
        print(f"Results for {query} have been saved to {file_path}")
```

3.修改執行，讀csv
```python
if interactive and not args.api_server:
    for i in range(len(top100_query)):
        top_k = top100_query_count[i]
        search(top100_query[i], items_df)
```

step3
>Semantic Search \
>repo: https://github.com/agbld/semantic-search-for-EE5327701

1.增加找最熱門100個關鍵字，同Tf-idf search \
2.修改search function，存成csv
```python
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
```
3.修改執行方式
```python
for i in range(len(top100_query)):
    top_k = top100_query_count[i]
    start_time = time.time()
    search(top100_query[i], product_names_series, index, top_k)
    elapsed_time = time.time() - start_time
    print(f'Took {elapsed_time:.4f} seconds to search {top100_query[i]}')
```

step4 比較Coupang、Tf-idf、Semantic三種搜尋方式