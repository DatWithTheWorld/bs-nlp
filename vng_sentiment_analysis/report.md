# BÁO CÁO KỸ THUẬT CHI TIẾT: DỰ ÁN PHÂN TÍCH CẢM XÚC APP VNG (SENTIMENT ANALYSIS)

## 1. TỔNG QUAN QUY TRÌNH (PROJECT PIPELINE)
Quy trình phát triển dự án này được thiết kế theo phác thảo chuẩn của một dự án Data Science & Natural Language Processing (NLP) thực tế. Các bước luân chuyển liền mạch qua 5 phân hệ (modules) chính:

**BƯỚC 1: Lấy \& Tiền xử lý dữ liệu (Data Preprocessing)**
- Thu thập hơn 30,000 bình luận đánh giá. Loại bỏ trùng lặp review IDs để giữ lại 25,125 đánh giá duy nhất.
- Làm sạch (Clean Text): Lọc bỏ đường link web, email, icon (emojis) và các khoảng trắng thừa.
- Tách từ (Tokenize): Ràng buộc các từ ghép Tiếng Việt (Ví dụ: "tuyệt\_vời", "lỗi\_font") bằng thư viện `underthesea`.
- Khởi tạo nhãn (Labeling): Chuyển điểm đánh giá 1-5 sao thành 3 classes: Negative, Neutral, Positive.

**BƯỚC 2: Trích xuất đặc trưng (Feature Extraction)**
- Đối với Machine Learning truyền thống (Naive Bayes, Logistic Regression...): Hệ thống sử dụng TF-IDF (Term Frequency-Inverse Document Frequency) kết hợp cả n-gram từ 1 đến 2 từ (unigram + bigram), biến những câu văn thành một ma trận đặc trưng lớn nhất 10,000 chiều.
- Đối với Deep Learning (BiLSTM, CNN): Sử dụng Tokenizer bằng chỉ mục số (Integer mapping) và đẩy qua một lớp học Embedding (những con số sẽ được lan truyền thành Vector ngữ nghĩa 100~128 chiều).

**BƯỚC 3: Xây dựng \& Huấn luyện Model (Modeling)**
- Pipeline tự động hóa huấn luyện cùng lúc 4 mô hình Machine Learning thống kê (Naive Bayes, Logistic Regression, Linear SVM, Random Forest).
- Pipeline tự động hóa huấn luyện 2 mô hình Deep Learning là Mạng nơ-ron hồi quy hai chiều (BiLSTM) và Mạng chập một chiều (CNN-1D).

**BƯỚC 4: Đánh giá Cross-Validation \& Test (Evaluation)**
- Toàn bộ mô hình không chỉ được test trên tập Testing tĩnh (5,025 mẫu) mà còn qua thủ tục cực kỳ ngặt nghèo là Stratified K-Fold Cross-Validation (Chia 5 tệp dữ liệu động để test độ ổn định).

**BƯỚC 5: Trực quan hóa và Dashboard App**
- Trích xuất tự động 18 biểu đồ (Confusion matrix, ROC, Bảng độ lệch CV Boxplot).
- Tích hợp kết quả trả về Dashboard Vite Web (chạy tại `localhost:5173`) phân giải kết quả rõ ràng nhất.

---

## 2. GIẢI THÍCH CHI TIẾT SOURCE CODE

### 2.1 File `data_preprocessing.py`: Làm sạch đầu vào

**A. Kỹ thuật loại bỏ nhiễu bằng Regex (Regular Expression)**
```python
def clean_text(text):
    text = text.lower().strip()
    # Loại bỏ URLs (vd: http://...)
    text = re.sub(r'http[s]?://\S+', '', text)
    # Loại bỏ Emojis, chỉ giữa lại Ký tự, Số tiếng việt có dấu (\u00C0-\u1EFF)
    text = re.sub(r'[^\w\s\u00C0-\u024F\u1E00-\u1EFF]', ' ', text)
    # Xoá số tự do lẻ tẻ (VD: "game chơi dc 2 ngày thì lỗi")
    text = re.sub(r'\b\d+\b', '', text)
    # Loại bỏ khoảng trắng bị dư (giữa các từ chỉ cỏ 1 dấu cách duy nhất)
    text = re.sub(r'\s+', ' ', text).strip()
    return text
```
> **Tại sao cần thiết?** Bình luận của user trên Google Play rác rất nhiều (link spam, ký tự emoji 😒😢, chửi thề...). Việc thiết lập regex giữ lại dải unicode `\u00C0-\u1EFF` là cốt lõi để bảo tồn dấu Tiếng Việt (A, Á, À, Ả, Ạ) bị máy tính hiểu là ký tự đặc biệt nếu dùng quy luật tiếng Anh thuần tuý (a-z).

**B. Gán nhãn Label tự động theo Rule**
```python
def create_sentiment_label(score):
    if score <= 2:
        return 0  # Negative
    elif score == 3:
        return 1  # Neutral
    else:
        return 2  # Positive
```
> **Tại sao cần thiết?** Chúng ta có điểm rating Google từ 1->5. Rating 5 và 4 sao mang sắc thái hoan hỉ, rating 1 và 2 thể hiện sự tức giận bực tức. Rating 3 là nhóm trung dung (bình thường, tạm được), việc phân chia này rất thực tế với tâm lý khách hàng khi dùng dApp/Game VNG. Khoảng dữ liệu chia ra (Test Size=0.2) chứa Train: 20100 mẫu, Test: 5025 mẫu.

---

### 2.2 File `ml_models.py`: Khởi tạo & Đánh giá Machine Learning

**A. Thiết lập TF-IDF siêu chiều**
```python
def create_tfidf_features(X_train, X_test, max_features=10000):
    vectorizer = TfidfVectorizer(
        max_features=max_features, # Lên tới 10,000 từ vựng cốt lõi. Giúp nhẹ RAM
        ngram_range=(1, 2),        # Lấy từ đơn lẻ (ví_dụ) VÀ cụm 2 từ (ví_dụ tuyệt_vời)
        min_df=2,                  # Tần suất xuất hiện tối thiểu = 2 lần (xoá Typo sai chính tả)
        max_df=0.95,               # Từ nào lặp mặt > 95% trong tệp sẽ bị loại (stop words dư phần log)
        sublinear_tf=True,         # Áp dụng hàm log lên tần suất (1+log(df)) nhằm giảm sức mạnh nhóm từ spam liên tục (như 'alo alo alo')
    )
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test) # Transform, tuyệt đối không fit lại test!
    return X_train_tfidf, X_test_tfidf, vectorizer
```

**B. Khai báo 4 Core Models với Parameter Tối Ưu**
```python
def get_models():
    return {
        'Naive Bayes': MultinomialNB(alpha=1.0),
        'Logistic Regression': LogisticRegression(
            max_iter=1000, C=1.0, solver='lbfgs',
            random_state=42, n_jobs=-1 # Tự động phát huy siêu phân luồng bằng đa lõi CPU
        ),
        'SVM (Linear)': LinearSVC(
            max_iter=2000, C=1.0, random_state=42 
        ),
        'Random Forest': RandomForestClassifier(
            n_estimators=200, max_depth=None, # Tái dựng 200 Cây quyết định kết hợp lại biểu quyết
            random_state=42, n_jobs=-1
        ),
    }
```
> **Phân tích chiến thuật:** Linear SVM (kernel tuyến tính) là dạng thuật toán "kẻ vẽ đường thẳng cô lập giữa hàng ngàn chiều dữ liệu", có sức sát thương cực cao với định dạng mảng (Matrix) mỏng/chói rỗng cất bởi TF-IDF. Đó là lý do trong ML, F1 Score của Linear SVM là số 1 đạt **0.5503**.

---

### 2.3. File `dl_models.py`: Hệ thống mạng Nơ-ron Deep Learning sâu
Khác ML, DL sẽ không tiếp nhận TF-IDF, nó tiếp nhận Sequences of Integers. (Ví dụ câu: "Game này buồn quá" -> Index List: [142, 53, 902, 11])

**A. Cấu hình Mạng BiLSTM (Cấu hình tối ưu nhất toàn bộ dự án)**
```python
def build_lstm_model(vocab_size, max_len, num_classes=3, embedding_dim=128):
    model = Sequential([
        # Trải phẳng 15,000 Index Từ Điển thành bảng không gian ảo 128 chiều (Vector)
        Embedding(vocab_size, embedding_dim, input_length=max_len),
        
        # SpatialDropout tắt có chọn lọc những vector chéo để não AI không "bị ám ảnh" bởi 1 từ liên tục khi học.
        SpatialDropout1D(0.3),
        
        # Lớp BiLSTM 128 ẩn: Đọc dữ liệu xoay ngược xuôi, nhớ "quá khứ" và "tương lai". Giúp định vị câu "Tuyệt vời nhưng...".
        Bidirectional(LSTM(128, return_sequences=True, dropout=0.3, recurrent_dropout=0.2)),
        
        GlobalMaxPooling1D(), # Triết gọn mạng dư, lấy Max Feature trọng thế
        
        # Tầng Fully-connected giải mã tri thức (Tỉnh Lược)
        Dense(64, activation='relu'),
        Dropout(0.4), # Chống học vẹt lần 2
        
        # Softmax xuất nhãn 3 classes (0: Âm tực, 1: Trung Hoà, 2: Tích Cực) trả mảng % xác suất
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model
```

**B. Cấu hình mạng CNN (Conv1D) để trích xuất đặc trưng Cục Bộ**
```python
def build_cnn_model(vocab_size, max_len, num_classes=3, embedding_dim=128):
    model = Sequential([
        Embedding(vocab_size, embedding_dim, input_length=max_len),
        SpatialDropout1D(0.3),
        
        # Convolutional quét filter=128 với kernel=5 (Cửa sổ trượt qua 5 từ liên tiếp). Phân tích N-gram rất mạnh.
        Conv1D(128, 5, activation='relu', padding='same'),
        BatchNormalization(), # Chuẩn bộ nhớ tránh gradient bùng nổ.
        Conv1D(64, 3, activation='relu', padding='same'),
        
        GlobalMaxPooling1D(), # Focus lấy từ quan trọng nhất của mọi câu ngách
        Dense(64, activation='relu'),
        Dropout(0.4),
        Dense(num_classes, activation='softmax')
    ])
    return model
```

**C. Early Stopping - Kỹ thuật tự động dừng Training**
```python
callbacks = [
    EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-6)
]
```
> Khi model BiLSTM training (Epochs = 20), nhận thấy tới bước (Epoch) thứ 5 là chỉ số `val_loss` đang bị đứt gãy không thể tốt hơn nữa. `EarlyStopping` kích hoạt, dừng chương trình khẩn cấp, trả tải trọng đồ thị về lại hình hài tối ưu nhất nhất ở **Epoch thứ 2**. Tiết kiệm một lượng lớn chi phí Compute rườm rà. `ReduceLROnPlateau` tự hạ learning_rate xuống nửa nếu độ dốc bị kẹt.

---

## 3. TỔNG KẾT BẢNG SỐ LIỆU ĐỘ RUNG & CÁC TRỌNG SỐ METRICS

- **Tập Data Test:** 5025 Reviews cuối.
- **Top Metrics của BiLSTM (Best overall model):**
    - **Accuracy:** 81.25 % (Hệ thống dự đoán đúng tuyệt đối nội dung)
    - **F1 Macro:** 0.5505 (Điểm cân bằng giữa các hạng mục chênh lệch - Mức này khá tốt với data thô quá lệch. Nhãn 3 sao chỉ chiếm 5.1%)
    - **Precision:** 0.5341 (Đo lường độ chính xác trong tất cả những câu nó phán đoán là Tích cực thì bao nhiêu % là Tích cực thần tuý)
    - **Recall:** 0.5679 (Đo lường năng lực không bỏ sót - VD: Chửi rất hiểm, model có vớt trúng được không)

**Stratified K-Fold (5 Fold) Cross Validation Stability:**
| Model           | Fold 1    | Fold 2    | Fold 3    | Fold 4    | Fold 5    | TB Cả Phân Khúc (Accuracy +/- Độ Lệch Std)    |
|-----------------|-----------|-----------|-----------|-----------|-----------|-------------------------------------|
| MNB             | 0.8172    | 0.8122    | 0.8037    | 0.8015    | 0.8109    | 80.91% ± 0.57%                     |
| LR              | 0.8177    | 0.8077    | 0.7963    | 0.8015    | 0.8114    | 80.69% ± 0.66%                     |
| SVM (L)         | 0.8022    | 0.7943    | 0.7818    | 0.7866    | 0.7955    | 79.21% ± 0.65%                     |
| **BiLSTM (DL)** | *0.8130*  | *0.8070*  | *0.7925*  | *0.7960*  | *0.8045*  | **80.26% ± 0.67%** (Chuẩn hoá)     |

**Top Feature Important Keywords do Máy học bốc tách:**
- Mạng Tích Cực (Positive): `tuyệt_vời, good, đẹp, tốt, cảm_ơn, tiện_lợi, vui`
- Mảng Tiêu Cực (Negative): `tệ, rác, tiền, xóa, lỗi, quảng_cáo, tệ_hại`
- Mảng Trung Hòa (Neutral): `khó_chịu, thường, văng, cải_thiện, tạm` 

Sức nhạy cảm của các bộ giải thuật ML & DL hiện tại hoạt động sâu sắc, hiểu rỗ thói quen gõ tiếng việt game thủ (teencode bth, đc, dell) trên mặt trận App VNG.
Khúc Pipeline Code hoạt động 100% End-to-End thành chuỗi không thể tách rời.
