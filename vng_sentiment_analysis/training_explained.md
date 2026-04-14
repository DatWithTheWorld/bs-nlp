# HƯỚNG DẪN CHI TIẾT TỪ A-Z: QUÁ TRÌNH TRAIN AI, CODE VÀ CÁCH ĐỌC BIỂU ĐỒ

Đây là chuyên đề giải thích chuyên sâu từng ngóc ngách của quá trình Huấn luyện Mô hình (Training) và Trực quan hoá dữ liệu (Visualizations) cho dự án Phân tích cảm xúc nhận xét ứng dụng (VNG App Reviews).

---

## PHẦN 1: TẠI SAO PHẢI TRAIN VÀ QUÁ TRÌNH "TRAIN" LÀ GÌ?
Hãy hiểu đơn giản, "Train" là quá trình "Dạy học". Máy tính là một vùng não trống, không hiểu chữ "Tuyệt vời" là lời khen, hay chữ "Rác rưởi" là lời chê.
Chúng ta có 20,000 bài bình luận **đã biết trước điểm sao** (nhãn). Ta đẩy 20,000 câu văn này vào hệ thống và bắt nó lặp đi lặp lại quá trình tính toán để nó **tự rút ra các quy luật ràng buộc**.
Ví dụ: Thấy chữ "lỗi", máy tính sẽ dùng ma trận số học để ghi nhớ rằng từ này có tương quan 90% với điểm 1-2 sao. Quá trình tính toán, nhớ và tối ưu đó gọi là **Train (Fit)**.

---

## PHẦN 2: GIẢI THÍCH MÃ NGUỒN MACHINE LEARNING (`ml_models.py`)

### 1. Hàm biến chữ thành số (TF-IDF Vectorizer)
Máy tính không học được chữ nên ta phải dùng `TfidfVectorizer`.
```python
vectorizer = TfidfVectorizer(
    max_features=10000, 
    ngram_range=(1, 2), 
    min_df=2,
    max_df=0.95
)
X_train_tfidf = vectorizer.fit_transform(X_train)
```
**Giải thích Code:**
- `max_features=10000`: Trong 20,000 bình luận có cả triệu chữ đan xen nhau. Lệnh này bảo máy chỉ giữ lại đúng 10,000 cụm từ mang ý nghĩa mạnh nhất, vứt bỏ các từ thừa. Giúp máy ảo không bị tràn RAM.
- `ngram_range=(1, 2)`: Cắt chữ theo kiểu cụm từ. (VD cắt chữ "Tuyệt": 1 từ, cắt chữ "Tuyệt vời": 2 từ).
- `fit_transform()`: Đoạn code này trải thẳng 20k câu văn ra thành một Bảng tính Excel có 20,000 dòng và 10,000 cột chứa mã con số xác suất của chữ.

### 2. Các thuật toán áp dụng Train bằng `.fit()`
```python
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, C=1.0, random_state=42),
    'SVM (Linear)': LinearSVC(max_iter=2000, random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=200, n_jobs=-1),
}
for name, model in models.items():
    model.fit(X_train_tfidf, y_train) # <- ĐÂY LÀ DÒNG CODE BẮT ĐẦU DẠY HỌC
```
**Giải thích thuật toán:**
- Máy sẽ lôi 1 bản nháp ra (như Logistic Regression).
- Câu lệnh `model.fit(X_train_tfidf, y_train)`: Ném tờ giấy 10.000 cột (Chữ) và đáp án (Label Khen/Chê) vào model.
- **LinearSVC:** Có nhiệm vụ tính toán không gian đa chiều, vẽ 1 vạch ranh giới (hàng rào bê tông) để chia phe: Bên trái là nhóm chê, Bến phải là nhóm khen. Những điểm mờ nhạt sẽ được vứt đi nhờ lệnh (C=1.0 - Hệ số phạt).
- **Random Forest:** Tạo ra 200 cái cây (`n_estimators=200`). Mỗi cây sẽ tự bốc ngẫu nhiên vài dòng bình luận lên thi nghiệm và học. Khi đem ra dự đoán, 200 cái cây này sẽ VOTE (bỏ phiếu) số đông.

---

## PHẦN 3: GIẢI THÍCH MÃ NGUỒN DEEP LEARNING (`dl_models.py`)
Deep Learning tinh vi hơn ở cấp độ "Neural Networks" (Mô phỏng nơ-ron não bộ). Khác biệt lớn nhất là nó không đếm chữ, nó đọc nguyên câu.

### 1. Kiến trúc Não Hai Chiều (BiLSTM)
```python
model = Sequential([
    Embedding(vocab_size, embedding_dim=128, input_length=80),
    SpatialDropout1D(0.3),
    Bidirectional(LSTM(128)),
    Dense(64, activation='relu'),
    Dropout(0.4),
    Dense(3, activation='softmax')
])
```
**Giải thích Cơ chế Não bộ:**
- `Embedding`: Biến bảng danh sách từ vựng thành Không gian ảo 128 chiều. 
- `SpatialDropout1D`: Cố hình làm "mù" ngẫu nhiên 30% mạng lưới để ngăn AI học vẹt. Tránh việc cứ thấy từ "đẹp" là phán từ khen.
- `Bidirectional(LSTM(128))`: **Cốt lõi siêu việt**. Đọc câu văn như con người, đọc từ trái qua phải, rồi ngược từ phải qua trái. Rất mạnh trong việc bắt các câu trào phúng, văn phong phủ định của tiếng Việt (VD: "Không (trái qua) - Rất Đẹp (Phải qua)").
- `Dense(3, softmax)`: Lớp nơ-ron cuối cùng gom toàn bộ nếp nhăn não phân hoá ra 3 cổng xác suất (Tỉ lệ % Negative, Neutral, Positive).

### 2. Quá trình bắt Não đi Học với `EarlyStopping`
```python
callbacks = [
    EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
]
history = model.fit(
    X_train, y_train, epochs=20, batch_size=64, 
    validation_data=(X_val, y_val), callbacks=callbacks
)
```
**Giải thích Code:**
- Mạng DL phải học đi học lại. `epochs=20` là quy luật bắt hệ thống xem tới xem lui bộ bình luận 20 lần. Mỗi đợt học nhồi nhét ngẫu nhiên 64 bình luận (`batch_size=64`) để tính Loss nghịch đảo.
- Thấy sai lệch (Loss), bộ não DL tự giật ngược dòng điện về (Back Propagation) để sửa lại. 
- `EarlyStopping`: Vừa học, ta sẽ nhét đề test vào kiểm tra chéo luôn (`validation_data`). Kẻo để máy học vẹt thuộc đề. Nếu kiểm tra 3 lần (`patience=3`) thấy não bị mụ đi (điểm thi dậm chân tại chỗ đi xuống), AI có thuật toán cưỡng chế tắt chương trình khẩn cấp lại, lùi thời gian vòng não về trạng thái làm bài hoàn mỹ nhất nhất ở quá khứ (`restore_best_weights`). Đỡ tốn tiền điện server.

---

## PHẦN 4: HƯỚNG DẪN CÁCH ĐỌC BIỂU ĐỒ AI (Visualizations)

Sau khi gọi `fit()` và dạy học thành công cỗ máy. Ta ném tập Test 5,025 bình luận (Cỗ máy chưa bao giờ thấy mặt) vào cho nó tự dự đoán.
Để xem độ thông minh, ta sinh ra các loại biểu đồ từ file `visualizations.py`:

### 1. Ma Trận Nhầm Lẫn (Confusion Matrix)
Chức năng: **Bắt quả tang máy tính ngu ở cái tật nào.**
*   *Cách xem:* Cột chữ Dọc (Y) đếm Label THỰC TẾ (Đáp án), Cột hàng ngang (X) báo Label ĐOÁN (AI).
*   *Đọc kết quả:* Một mô hình thiên tài là con số nằm trên **đường chéo từ góc trên bên trái xuống góc dưới phải** phải TO bự nhất (Màu xanh nước biển đậm đặc nhất). Tức là Thực tế là Khen - AI ngoan ngoãn đoán Khen.
*   *Lỗi hay gặp:* Ở ô chéo (Neutral - Negative) mà lên màu nhạt nhạt chứng tỏ AI của mình đang lấn cấn, thấy người dùng bình luận chung chung cũng quy về họ đang chê cty. Phải sửa data đầu vào ngay.

### 2. Biểu đồ Đường cong Loss / Accuracy (Của hệ DL)
Chức năng: **Kiểm tra đồ thị tim AI có bị bệnh "Học Vẹt" không.**
*   *Cách xem:* Có hai đường rượt đuổi nhau. Nhìn vào biểu đồ `Loss` xanh đỏ đứt nét.
*   *Đọc kết quả:* Nếu 2 đường rủ nhau cắm đầu rơi xuống đáy -> AI khoẻ đi đúng hướng trơn tru. Đột ngột đường màu Đỏ (Lúc bị ném đi test) "vót ngược hướng bay lên trần nhà" -> Báo nguy hiểm ngay lập tức! Có nghĩa là lúc ngồi nhà luyện đề AI làm được 10 điểm (Loss luyện giảm mạnh), đem ra phòng thi thực tế gặp câu rẽ hướng thì AI bị điếc ngơ ngác (Loss kiểm tra dâng cao).

### 3. Biểu đồ hộp Độ lệch Cross-Validation (Boxplot)
Chức năng: **Đo độ hên xui của mô hình.**
Giống như thi tốt nghiệp, thi trúng tủ thì ta được F1 cao nên điểm cao chưa chắc là AI khôn. K-Fold tức là bài test khốc liệt bắt nó làm bài thi 5 Lần Khác Nhau hoàn toàn.
*   *Cách xem:* Xem biểu đồ hình những cái hộp có hai sợi râu tua lên xuống.
*   *Đọc kết quả khôn khéo:* 
    - Cái hộp nào càng dẹt (ép dẹp lại), 2 sợi râu đâm đứt lìa càng ngắn -> Độ ổn định tuyệt đỉnh. Có thể ra mắt sếp ngay lập tức.
    - Hộp nào bị kéo dãn bành ra (như cái cột đình) chứng tỏ vòng thi 1 điểm 9 điểm 10, vòng thi số 2 xuống hẳn 5 điểm hên xui mù mờ cực rủi ro. Nhìn trên biểu đồ của Random Forest là ví dụ của độ rung giật nhẹ.

---
*(Bức màn AI/Machine Learning cho Dự Án đã được vén lên toàn bộ theo một cách thân thiện với thế giới ngôn ngữ người nhất dành cho Sếp và bạn)*
