import os
import glob
import numpy as np
import pandas as pd
from xlsxwriter import Workbook

coupang_folder = './Coupang_Scraping-main/results'
tfidf_folder = './tf-idf-for-EE5327701/results'
ABRSS_folder = './semantic_search-for-EE5327701/results'
student_id = 'M11352032'

csv_files = glob.glob(os.path.join(tfidf_folder, '*.csv'))

query = []
# 利用tfidf results找關鍵字
for file in csv_files:
    # 取得檔案名（不包含路徑）
    filename = os.path.basename(file)
    
    # 找關鍵字
    if 'M11352032_tfidf_' in filename:
        extracted_name = filename.split('M11352032_tfidf_')[1].replace('.csv', '')
        query.append(extracted_name)

print(f'{query}')

def extract_top10_product_names(folder_path, name):

    name_type = ''
    if folder_path == coupang_folder:
        name_type = f'{student_id}_'
    elif folder_path == tfidf_folder:
        name_type = f'{student_id}_tfidf_'
    else:
        name_type = f'{student_id}_semantic_'

    # 遍歷資料夾中的所有檔案
    for file_name in os.listdir(folder_path):
        # 檢查檔案是否是 .csv 並且檔案名稱精確匹配 name_list 的元素
        expected_file_name = f"{name_type + name}.csv"
        if file_name == expected_file_name:
            df = pd.read_csv(f'{folder_path + '/' + expected_file_name}')
    
            top10_names = df['product_name'][:10].tolist()  # 提取前10個產品名稱 (忽略 header)
            rank = list(range(1, 11))

    # return top10_product_names
    return top10_names, rank

output_file_path = f'{student_id}_搜尋比較.xlsx'
with pd.ExcelWriter(output_file_path, engine='xlsxwriter') as writer:
    combined_data = pd.DataFrame()

    # 對於每一個query中的名稱，創建一個分頁
    for name in query:
        top10_coupang, rank = extract_top10_product_names(coupang_folder, name)
        top10_tf_idf, _ = extract_top10_product_names(tfidf_folder, name)
        top10_ABRSS, _ = extract_top10_product_names(ABRSS_folder, name)
        # 創建 DataFrame 來存儲結果
        df = pd.DataFrame({
            '搜尋詞': name,
            'Rank': rank,
            'Coupang': top10_coupang,
            'relevancy': [None] * 10,
            'tf-idf': top10_tf_idf,
            'relevancy ': [None] * 10,
            'semantic_model': top10_ABRSS,
            'relevancy  ': [None] * 10
        })

        combined_data = pd.concat([combined_data, df], ignore_index=True)

    # 將所有數據寫入到同一個分頁
    combined_data.to_excel(writer, sheet_name='Combined', index=False)

    # 獲取當前的 worksheet
    worksheet = writer.sheets['Combined']

    # 調整每一列的寬度，根據列中最長的字串
    for i, col in enumerate(df.columns):
        max_len = max(df[col].astype(str).map(len).max(), len(col))  # 計算列中最長的字串長度
        worksheet.set_column(i, i, max_len + 10)  # 設置列寬（稍微加點padding）

print(f"Results saved to {output_file_path}")