import os
import re
import nbformat as nbf

def clean_py_code(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        code = f.read()
    # Remove imports at the top since we will consolidate them
    code = re.sub(r'^(import|from\s+[\w\.]+\s+import).*$', '', code, flags=re.MULTILINE)
    # Remove empty lines left by imports
    code = re.sub(r'\n{3,}', '\n\n', code)
    # Remove __main__ block
    code = re.sub(r'if __name__ == "__main__":.*', '', code, flags=re.DOTALL)
    return code.strip()

def create_notebook():
    nb = nbf.v4.new_notebook()
    cells = []

    # 1. MARKDOWN: Intro
    cells.append(nbf.v4.new_markdown_cell("""# Full Pipeline: VNG Sentiment Analysis
Tất cả bộ source code Python nguyên gốc (`data_preprocessing.py`, `ml_models.py`, `dl_models.py`, `visualizations.py`) đã được bê nguyên vẹn vào Notebook này theo từng module chức năng để bảo quản đầy đủ kỹ thuật cốt lõi (Cross-Validation, Architecture, SMOTE)."""))

    # 2. CODE: Imports
    cells.append(nbf.v4.new_markdown_cell("## BƯỚC 1: IMPORT THƯ VIỆN CHUNG"))
    cells.append(nbf.v4.new_code_cell("""# 1. Import Thư Viện Tổng Hợp
import os
import re
import json
import time
import pickle
import numpy as np
import pandas as pd
from collections import Counter

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import MaxNLocator

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix, roc_auc_score
from sklearn.preprocessing import label_binarize
from sklearn.utils.class_weight import compute_class_weight
try:
    from imblearn.over_sampling import SMOTE
    HAS_SMOTE = True
except ImportError:
    HAS_SMOTE = False

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Conv1D, GlobalMaxPooling1D, Dense, Dropout, SpatialDropout1D, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

try:
    from underthesea import word_tokenize
    HAS_UNDERTHESEA = True
except ImportError:
    HAS_UNDERTHESEA = False

%matplotlib inline
plt.style.use('seaborn-v0_8-whitegrid')
"""))

    base_dir = os.path.dirname(os.path.abspath(__file__))

    # 3. CODE: Data Preprocessing
    cells.append(nbf.v4.new_markdown_cell("## BƯỚC 2: HÀM TIỀN XỬ LÝ DỮ LIỆU VÀ LÀM SẠCH VĂN BẢN\n(Giữ nguyên 100% logic regex, stopwords, tokenization)"))
    code_prep = clean_py_code(os.path.join(base_dir, 'data_preprocessing.py'))
    cells.append(nbf.v4.new_code_cell(code_prep))

    # 4. CODE: ML Models
    cells.append(nbf.v4.new_markdown_cell("## BƯỚC 3: HÀM MÔ HÌNH HÓA MACHINE LEARNING TIÊU CHUẨN\n(Khai báo TF-IDF, SVM, LogisticRegression, Naive Bayes, Random Forest, Tính K-Fold CV)"))
    code_ml = clean_py_code(os.path.join(base_dir, 'ml_models.py'))
    cells.append(nbf.v4.new_code_cell(code_ml))

    # 5. CODE: DL Models
    cells.append(nbf.v4.new_markdown_cell("## BƯỚC 4: HÀM MÔ HÌNH HÓA ĐỘ CẤP SÂU (DEEP LEARNING)\n(Khởi tạo Tokenizer Padding 80 ký tự, Build BiLSTM, CNN-1D, Callbacks)"))
    code_dl = clean_py_code(os.path.join(base_dir, 'dl_models.py'))
    cells.append(nbf.v4.new_code_cell(code_dl))

    # 6. CODE: Visualizations
    cells.append(nbf.v4.new_markdown_cell("## BƯỚC 5: HÀM VẼ TẤT CẢ BIỂU ĐỒ (VISUALIZATIONS)\n(Confusion Matrices, ROC, Training Histories, CrossVal Boxplots)"))
    code_viz = clean_py_code(os.path.join(base_dir, 'visualizations.py'))
    # Disable Agg for notebook
    code_viz = code_viz.replace("matplotlib.use('Agg')", "")
    cells.append(nbf.v4.new_code_cell(code_viz))

    # 7. CODE: THỰC THI (Execution block with SMOTE logic overlaid)
    cells.append(nbf.v4.new_markdown_cell("## BƯỚC 6: CHẠY THỰC THI TOÀN BỘ PIPELINE (CÓSMOTE)\nNơi giao thoa tất cả các Object đã được biên dịch bên trên."))
    cells.append(nbf.v4.new_code_cell("""# 1. LOAD DATA & TIỀN XỬ LÝ (Từ File CSV Gốc)
DATA_DIR = '../vng_reviews_data'
OUTPUT_DIR = './output'

print("--- [BƯỚC 1]: LOAD DATA ---")
# Hàm load_and_preprocess lấy trọn vẹn từ data_preprocessing.py
df, X_train, X_test, y_train, y_test, metadata = load_and_preprocess(DATA_DIR, OUTPUT_DIR)

# Vẽ Khám phá Dữ Liệu
plot_data_distribution(df, OUTPUT_DIR)
"""))

    cells.append(nbf.v4.new_code_cell("""# 2. XỬ LÝ MẤT CÂN BẰNG BẰNG SMOTE ĐỐI VỚI ML
print("--- [BƯỚC 2]: TẠO TF-IDF & CÂN BẰNG NHÃN BẰNG SMOTE ---")
X_train_tfidf, X_test_tfidf, vectorizer = create_tfidf_features(X_train, X_test)

if HAS_SMOTE:
    print(f"Phân bố gốc (Imbalance): {Counter(y_train)}")
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train_tfidf, y_train)
    print(f"Phân bố sau SMOTE (Balanced): {Counter(y_train_resampled)}")
else:
    print("Không có SMOTE, sử dụng class weights")
    X_train_resampled, y_train_resampled = X_train_tfidf, y_train
"""))

    cells.append(nbf.v4.new_code_cell("""# 3. TRAINING MACHINE LEARNING models với SMOTE
# Do train_and_evaluate_all trong ml_models nhận X_train dạng text, ta tách ra gọi hàm fit thủ công để đưa Smote vào
print("--- [BƯỚC 3]: TRAINING MACHINE LEARNING MODELS ---")
models = get_models()
ml_results = {}

for name, model in models.items():
    print(f"\\nĐào tạo: {name}")
    start_time = time.time()
    model.fit(X_train_resampled, y_train_resampled)
    
    # Eval
    result = evaluate_model(model, X_test_tfidf, y_test, name)
    result['train_time'] = time.time() - start_time
    ml_results[name] = result
    
    # In điểm
    m = result['metrics']
    print(f"Acc: {m['accuracy']:.4f} | F1: {m['f1_macro']:.4f} | Recall: {m['recall_macro']:.4f}")

# Vẽ Confusion Matrix ML
plot_confusion_matrices(ml_results, OUTPUT_DIR, prefix='ml')
"""))

    cells.append(nbf.v4.new_code_cell("""# 4. TRAINING DEEP LEARNING (BiLSTM & CNN)
print("--- [BƯỚC 4]: TRAINING DEEP LEARNING ---")
# Hàm train_and_evaluate_dl_all sẽ padding Sequences, compile Model keras và run EarlyStopping
# (Lấy trọn vẹn từ dl_models.py)
dl_results, dl_cv_results, dl_histories = train_and_evaluate_dl_all(X_train, X_test, y_train, y_test, OUTPUT_DIR)

# Biểu đồ Lịch sử Training DL
plot_dl_training_history(dl_histories, OUTPUT_DIR)
"""))

    nb['cells'] = cells
    out_file = os.path.join(base_dir, 'vng_sentiment_analysis_full_pipeline.ipynb')
    with open(out_file, 'w', encoding='utf-8') as f:
        nbf.write(nb, f)
    print(f"Bơm code thành công vào file IPYNB!")

if __name__ == "__main__":
    create_notebook()
