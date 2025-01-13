import json
import os
import re
import pandas as pd
from tqdm import tqdm

def build_json(department):
    """
    json 的格式如下：
    [
        {
            "title": "問題",
            "metadata": {
                "ask": "詳細問題",
                "answer": "答案",
                "department": "科別"
            }
        },
        ...
    ]
    """
    csv_file = f"./reference/{department}_traditional.csv"
    df = pd.read_csv(csv_file)

    # 將每一行轉換為 JSON 格式
    json_data = []
    for _, row in df.iterrows():
        entry = {
            "title": row["title"],
            "metadata": {
                "ask": row["ask"],
                "answer": row["answer"],
                "department": row["department"]
            }
        }
        json_data.append(entry)

    # 將結果保存為 JSON 文件
    output_file = f"./documents/{department}.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(json_data, f, ensure_ascii=False, indent=4)

    print(f"JSON 文件已保存到 {output_file}")

# 主程式
if __name__ == "__main__":

    build_json('IM')
    build_json('OAGD')
    build_json('Oncology')
    build_json('Pediatric')
    build_json('Surgical')