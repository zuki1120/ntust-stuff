# Dataset
[Chinese medical dialogue data 中文医疗问答数据集](https://github.com/Toyhom/Chinese-medical-dialogue-data)
### **1 資料處理**
1. Preprocess/build_json.py -> 轉換成json格式
2. Preprocess/index_local.py -> 存入 ChromaDB

### **2 TG Bot**
1. App/bot.py -> Bot設定
2. App/query_engine.py -> 處理回應邏輯
3. App/settings.py -> 設定Model