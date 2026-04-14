import os
import nbformat as nbf

def create_notebook():
    nb = nbf.v4.new_notebook()
    
    # Text cells
    title_md = """# VNG Apps - Sentiment Analysis Pipeline
Dự án phân tích cảm xúc (Sentiment Analysis) dữ liệu từ Google Play Store cho 10 ứng dụng của VNG.
## Các bước thực hiện:
1. Tiền xử lý dữ liệu (Data Preprocessing)
2. Machine Learning Pipeline (TF-IDF + Classifiers)
3. Deep Learning Pipeline (Word Embeddings + BiLSTM/CNN)
4. Trực quan hoá dữ liệu (Visualizations)"""

    setup_md = """### 0. Import Thư Viện
Khai báo các thư viện cần thiết. Yêu cầu đã cài đặt `scikit-learn`, `tensorflow`, `underthesea`, `pandas`, `matplotlib`, `seaborn`."""

    # Read python files
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    def get_py_code(filename):
        with open(os.path.join(base_dir, filename), 'r', encoding='utf-8') as f:
            code = f.read()
            # Remove __main__ blocks to keep it clean in Notebook
            import re
            code = re.sub(r'if __name__ == "__main__":.*', '', code, flags=re.DOTALL)
            return code
            
    prep_code = get_py_code('data_preprocessing.py')
    ml_code = get_py_code('ml_models.py')
    dl_code = get_py_code('dl_models.py')
    viz_code = get_py_code('visualizations.py')
    
    # Cells
    nb['cells'] = [
        nbf.v4.new_markdown_cell(title_md),
        
        nbf.v4.new_markdown_cell(setup_md),
        nbf.v4.new_code_cell("""%matplotlib inline
import warnings
warnings.filterwarnings('ignore')"""),

        nbf.v4.new_markdown_cell("### 1. Quy Trình Tiền Xử Lý Dữ Liệu\n(Clean text, Word tokenize, Vectorize, Labelling)"),
        nbf.v4.new_code_cell(prep_code),
        
        nbf.v4.new_markdown_cell("### 2. Mô hình Machine Learning\n(Naive Bayes, Logistic Regression, Linear SVC, Random Forest)"),
        nbf.v4.new_code_cell(ml_code),
        
        nbf.v4.new_markdown_cell("### 3. Mô hình Deep Learning\n(BiLSTM, Conv1D)"),
        nbf.v4.new_code_cell(dl_code),

        nbf.v4.new_markdown_cell("### 4. Hàm Trực Quan Hoá Đồ Thị\n(Vẽ Confusion Matrix, Training Curves, Data Distributions)"),
        nbf.v4.new_code_cell(viz_code),
        
        nbf.v4.new_markdown_cell("### 5. Thực Thi Toàn Bộ Pipeline"),
        nbf.v4.new_code_cell("""# Khởi chạy Pipeline
# Cấu hình đường dẫn (Cập nhật đường dẫn tương đối tới data của bạn)
DATA_DIR = '../vng_reviews_data'
OUTPUT_DIR = './output'

# Bước 1: Tiền Xử Lý
print("BƯỚC 1: TIỀN XỬ LÝ")
df, X_train, X_test, y_train, y_test, metadata = load_and_preprocess(DATA_DIR, OUTPUT_DIR)

# Bước 2: Huấn luyện K-Fold Pipeline Machine Learning
print("BƯỚC 2: MACHINE LEARNING")
ml_results, ml_cv, ml_models_dict, vectorizer, ml_features = train_and_evaluate_all(X_train, X_test, y_train, y_test, OUTPUT_DIR)

# Bước 3: Huấn luyện K-Fold Pipeline Deep Learning (BiLSTM)
print("BƯỚC 3: DEEP LEARNING")
dl_results, dl_cv_results, dl_histories = train_and_evaluate_dl_all(X_train, X_test, y_train, y_test, OUTPUT_DIR)

# Bước 4: Vẽ Đồ Thị (Hiển thị trực tiếp trên Notebook)
print("BƯỚC 4: RENDER BÁO CÁO")
# (Lệnh generate_all_visualizations đang lưu as PNG, chúng ta có thể gọi hàm plot truyền trực tiếp biến vào đây)
plot_data_distribution(df, OUTPUT_DIR)
plot_confusion_matrices(ml_results, OUTPUT_DIR, prefix='ml')
plot_confusion_matrices(dl_results, OUTPUT_DIR, prefix='dl')
""")
    ]
    
    with open(os.path.join(base_dir, 'vng_sentiment_analysis_pipeline.ipynb'), 'w', encoding='utf-8') as f:
        nbf.write(nb, f)
    print("Created Notebook successfully!")

if __name__ == '__main__':
    create_notebook()
