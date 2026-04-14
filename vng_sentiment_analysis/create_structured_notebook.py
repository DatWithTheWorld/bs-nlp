import os
import nbformat as nbf

def create_notebook():
    nb = nbf.v4.new_notebook()
    cells = []

    # 1. Tieu de và Library Setup
    cells.append(nbf.v4.new_markdown_cell("""# Báo Cáo Phân Tích Cảm Xúc Ứng Dụng VNG
(Sentiment Analysis for VNG Google Play Reviews)

Dự án mô phỏng lại một vòng xoay công việc NLP chuẩn mực từ EDA, Tiền xử lý, Khai phá đặc trưng tới Build Machine Learning & Deep Learning.

## BƯỚC 1: IMPORT THƯ VIỆN ĐỊNH HƯỚNG MÔI TRƯỜNG"""))
    
    cells.append(nbf.v4.new_code_cell("""# Setup libraries
import os
import re
import numpy as np
import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns

# Sklearn for Machine Learning pipeline
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression

# Keras for Deep Learning pipeline
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SpatialDropout1D, Bidirectional, LSTM, GlobalMaxPooling1D, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# Hiển thị biểu đồ rõ nét ngay trên Notebook
%matplotlib inline
plt.style.use('seaborn-v0_8-whitegrid')
"""))

    # 2. Load Data + Khám phá
    cells.append(nbf.v4.new_markdown_cell("""## BƯỚC 2: KHÁM PHÁ DỮ LIỆU ĐẦU VÀO (EDA)
Đọc cấu trúc file csv tự cào (scrape) từ PlayStore."""))
    cells.append(nbf.v4.new_code_cell("""# 1. Đọc dữ liệu thô
DATA_DIR = '../vng_reviews_data'
CSV_FILES = [f for f in os.listdir(DATA_DIR) if f.startswith('all_vng_reviews') and f.endswith('.csv')]

try:
    df = pd.read_csv(os.path.join(DATA_DIR, CSV_FILES[0]), encoding='utf-8')
    print(f"Tổng số dòng gốc thu thập: {len(df)}")
except Exception as e:
    print("Vui lòng trỏ lại biến DATA_DIR vào thư mục data tĩnh nếu không chạy.")

# Xoá bình luận lặp do thuật toán cào nhiều lần
df = df.drop_duplicates(subset=['review_id'])
df = df[df['content'].notna() & (df['content'].str.strip() != '')]
print(f"Tổng số bình luận sạch ID: {len(df)}")

# Xem qua tỷ lệ sao Rating (1->5)
fig, ax = plt.subplots(figsize=(6,4))
sns.countplot(x='score', data=df, palette='viridis', ax=ax)
ax.set_title('Bất đối xứng dữ liệu Ratings')
plt.show()
"""))

    # 3. Preprocessing
    cells.append(nbf.v4.new_markdown_cell("""## BƯỚC 3: TIỀN XỬ LÝ VÀ PHÂN NHÃN DỮ LIỆU
Ta cần khử nhiễu (Dấu phẩy, emoji, ngắt dòng) và tạo ra nhãn mục tiêu (Target labels: 0=Negative, 1=Neutral, 2=Positive)."""))
    cells.append(nbf.v4.new_code_cell("""try:
    from underthesea import word_tokenize
except ImportError:
    print("Vui lòng `pip install underthesea` nếu muốn tối ưu Tiếng Việt")
    def word_tokenize(text, format): return text

# Cấu trúc dừng Stopwords (Tối giản hoá)
STOPWORDS = set(['và', 'là', 'của', 'như', 'nhưng', 'được', 'thì', 'ở', 'đó', 'mà'])

def clean_text(text):
    text = text.lower().strip()
    text = re.sub(r'http[s]?://\S+', '', text) # Loại link
    text = re.sub(r'[^\w\s\u00C0-\u024F\u1E00-\u1EFF]', ' ', text) # Chỉ giữ kí tự UTF8 Tiếng Việt
    text = re.sub(r'\s+', ' ', text).strip() # Gom trắng vỡ
    return text

def preprocess_pipeline(text):
    text = clean_text(text)
    # Gom chữ Tiếng Việt
    text = word_tokenize(text, format="text") 
    words = [w for w in text.split() if w not in STOPWORDS]
    return ' '.join(words)

# Khởi tạo nhãn Target Y
def create_label(score):
    if score <= 2: return 0  # Negative
    elif score == 3: return 1  # Neutral
    else: return 2  # Positive

print("Bắt đầu xử lý (Sẽ tốn khoảng 30s-1p do thư viện quét cấu trúc từ tiếng Việt)...")
df['sentiment'] = df['score'].apply(create_label)
# Tạo ra cột text sạch bong để train
df['processed_text'] = df['content'].apply(preprocess_pipeline)
"""))

    # 4. Train Test Split
    cells.append(nbf.v4.new_markdown_cell("""## BƯỚC 4: PHÂN LƯỚI TẬP VÀ XỬ LÝ IMBALANCE DATA
Phân bố nhãn của chúng ta rất lệch (Lên tới 65% là Positive).
Nên ta phải bật tính năng `Stratify=y` (Tức là chia tỷ lệ 80/20 train/test mà độ thưa thớt Neutral vẫn được giữ nguyên tính đồng dạng)."""))
    cells.append(nbf.v4.new_code_cell("""# Drop rỗng
df = df[df['processed_text'].str.strip() != '']

X = df['processed_text'].values
y = df['sentiment'].values

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Kích cỡ tập TRAIN: {X_train.shape}")
print(f"Kích cỡ tập TEST (Dùng để đo lường): {X_test.shape}")

# Tự tính toán rớt trọng số bù đắp cho Label 1 (Neutral). Class này chỉ có 5% số người chọn
classes = np.unique(y_train)
weights = compute_class_weight('balanced', classes=classes, y=y_train)
class_weights_dict = dict(zip(classes, weights))
print(f"Trọng số nhân bản cân bằng Class Weights: {class_weights_dict}")
"""))

    # 5. Cân bằng nhãn với SMOTE & TF-IDF
    cells.append(nbf.v4.new_markdown_cell("""## BƯỚC 5: KHAI PHÁ ĐẶC TRƯNG (TF-IDF) & CÂN BẰNG NHÃN (SMOTE)
Do dữ liệu bị mất cân bằng nghiêm trọng (Positive chiếm >65%, Neutral chỉ 5%), nếu cho máy học ngay, máy sẽ có xu hướng đoán an toàn là Positive hết để lấy điểm cao.
**Giải pháp:** Mọi mô hình ML cổ điển đều bị bias. Ta sử dụng kỹ thuật SMOTE (Synthetic Minority Over-sampling Technique) để sinh ra thêm các mẫu (vector) nhân tạo cho nhóm 1 sao và 3 sao, ép số lượng 3 nhóm NGANG BẰNG NHAU."""))
    cells.append(nbf.v4.new_code_cell("""# 1. Trích xuất đặc trưng văn bản thành tần suất TF-IDF
vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1,2), min_df=3)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)
print(f"Bản gốc tập Train (Vector TF-IDF): {X_train_tfidf.shape}")

# 2. Sử dụng SMOTE Cân Bằng Label
try:
    from imblearn.over_sampling import SMOTE
    print(f"Phân bố nhãn TRƯỚC KHI CÂN BẰNG: {Counter(y_train)}")
    
    smote = SMOTE(random_state=42)
    # Fit SMOTE để nhân bản data tập Train
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train_tfidf, y_train)
    
    print(f"Phân bố nhãn SAU KHI CÂN BẰNG (SMOTE): {Counter(y_train_resampled)}")
    print(f"Bản mới tập Train (Đã bù đắp): {X_train_resampled.shape}")
except ImportError:
    print("Vui lòng chạy `pip install imbalanced-learn` để cài thư viện SMOTE (Cân bằng nhãn).")
    X_train_resampled, y_train_resampled = X_train_tfidf, y_train
"""))

    # 6. ML Models
    cells.append(nbf.v4.new_markdown_cell("""## BƯỚC 6: XÂY DỰNG MÔ HÌNH HÓA (MACHINE LEARNING)
Sử dụng dữ liệu Đã Cân Bằng (SMOTE) `X_train_resampled` để huấn luyện Thuật Toán Support Vector Machine và Logistic Regression."""))
    cells.append(nbf.v4.new_code_cell("""ml_models = {
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'Linear SVC': LinearSVC(max_iter=2000)
}

TARGET_NAMES = ['Negative', 'Neutral', 'Positive']

for name, model in ml_models.items():
    print(f"--- Đào tạo Model trên dữ liệu cân bằng: {name} ---")
    model.fit(X_train_resampled, y_train_resampled)
    y_pred = model.predict(X_test_tfidf)
    acc = accuracy_score(y_test, y_pred)
    
    print(f"Accuracy: {acc*100:.2f}%")
    print(classification_report(y_test, y_pred, target_names=TARGET_NAMES))
    
    # Biểu diễn Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(4,3))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=TARGET_NAMES, yticklabels=TARGET_NAMES)
    plt.title(f"Confusion Matrix: {name}")
    plt.ylabel('Thực tế')
    plt.xlabel('AI Dự đoán')
    plt.show()
"""))

    # 7. DL Models
    cells.append(nbf.v4.new_markdown_cell("""## BƯỚC 7: XÂY DỰNG MÔ HÌNH HỌC SÂU DEEP LEARNING (BiLSTM)
Vì Deep Learning không nhai được TF-IDF, ta phải dùng lệnh Tokenizer băm nhỏ văn bản thay thế mảng số nguyên, đưa qua không gian Embedding 128 Chiều."""))
    cells.append(nbf.v4.new_code_cell("""MAX_WORDS = 15000
MAX_LEN = 80

# Keras Tokenizer
tokenizer = Tokenizer(num_words=MAX_WORDS, oov_token="<OOV>")
tokenizer.fit_on_texts(X_train)

# Ép chặt mảng và cố định độ dài = MAX_LEN (Cắt rập hoặc Bù Padding Số 0)
X_train_pad = pad_sequences(tokenizer.texts_to_sequences(X_train), maxlen=MAX_LEN)
X_test_pad = pad_sequences(tokenizer.texts_to_sequences(X_test), maxlen=MAX_LEN)

# Tạo Kiến Trúc Neural Network Lớp Cắt Lớp
dl_model = Sequential([
    Embedding(MAX_WORDS, 128, input_length=MAX_LEN),
    SpatialDropout1D(0.3),
    Bidirectional(LSTM(128, dropout=0.3, recurrent_dropout=0.2)),
    Dense(64, activation='relu'),
    Dropout(0.4),
    Dense(3, activation='softmax')
])

dl_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
dl_model.summary()

# Training mượt trên môi trường Jupyter
history = dl_model.fit(
    X_train_pad, y_train,
    epochs=5, # Ở cấp độ máy local, ta test 5 Epochs 
    validation_split=0.1, # Dành 10% Tập học làm giám thị check val_loss
    batch_size=128,
    class_weight=class_weights_dict, # Đưa trọng số bù trừ 3 sao vào Kênh DL
    verbose=1
)
"""))

    # 8. DL Eval
    cells.append(nbf.v4.new_markdown_cell("""## BƯỚC 8: ĐÁNH GIÁ HISTORY CỦA NƠ-RON"""))
    cells.append(nbf.v4.new_code_cell("""# Report cuối cùng DL
dl_pred_probs = dl_model.predict(X_test_pad)
dl_pred = np.argmax(dl_pred_probs, axis=1)

print("BiLSTM - Classification Report")
print(classification_report(y_test, dl_pred, target_names=TARGET_NAMES))

# Loss Chart Tracking
plt.figure(figsize=(8,4))
plt.plot(history.history['loss'], label='Train Loss', color='blue')
plt.plot(history.history['val_loss'], label='Validation Loss', color='red')
plt.title('Đường cong rượt đuổi Overfitting Check')
plt.ylabel('Loss Value')
plt.xlabel('Vòng Học (Epoch)')
plt.legend()
plt.show()
"""))

    nb['cells'] = cells

    out_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'vng_sentiment_analysis_structured.ipynb')
    with open(out_file, 'w', encoding='utf-8') as f:
        nbf.write(nb, f)
    print(f"Created '{out_file}' successfully!")

if __name__ == "__main__":
    create_notebook()
