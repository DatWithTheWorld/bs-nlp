# 📚 Hướng Dẫn Chuyên Sâu: Hệ Thống Business Sentiment Intelligence (BSI)

Chào mừng bạn đến với hệ thống **Phân Tích Cảm Xúc Kinh Doanh Phút Chót (Real-time Business Sentiment)**. Hệ thống này không chỉ dừng lại ở mức "đọc chữ phân loại", mà là một giải pháp **Trí Tuệ Doanh Nghiệp (Business Intelligence)** hoàn chỉnh.

---

## 1. Mục Đích & Tầm Nhìn Chiến Lược (Business Mission)
Trong kỷ nguyên số, **Dữ liệu Phản hồi khách hàng (Customer Feedback)** chính là "Dầu mỏ mới". Mục tiêu của hệ thống này là biến hàng triệu dòng bình luận hỗn loạn trên Google Play thành các **Quyết định Kinh doanh (Data-Driven Decisions)** chính xác:
- **Lắng nghe Xã hội (Social Listening):** Theo dõi sức khỏe thương hiệu 24/7 theo thời gian thực.
- **Dự báo Rời bỏ (Churn Prediction):** Phát hiện sớm các lỗi hệ thống nghiêm trọng trước khi người dùng đồng loạt xóa app.
- **Tối ưu hóa Sản phẩm (Product Growth):** Chỉ ra chính xác "Nỗi đau" (Pain Points) của khách hàng để team kỹ thuật sửa chữa ngay lập tức.

---

## 2. Hệ Thống Chỉ Số Phân Tích (Key Business Metrics) 📊
Hệ thống AI cung cấp 4 bộ chỉ số cốt lõi mà bất kỳ Giám đốc Sản phẩm (PO) hay Marketing (CMO) nào cũng cần:

1.  **Chỉ số Danh tiếng (Average Polarity Score):**
    - Điểm từ -1.0 đến 1.0.
    - Giúp so sánh sức khỏe giữa App của mình với các đối thủ (Benchmark).
2.  **Ma trận Độ ưu tiên Lỗi (Bug Priority Matrix):** 🚨
    - **CRITICAL (🚨):** Rủi ro mất tiền, hack, nuốt tiền (Cần xử lý trong 1 giờ).
    - **HIGH (🔴):** Lỗi văng app, không đăng nhập được (Cần xử lý trong 4 giờ).
    - **MEDIUM (🟡):** App chậm, lag, khó dùng (Cập nhật trong phiên bản tới).
    - **LOW (⚪):** Chê bai cảm quan cá nhân.
3.  **Tần suất Cảm xúc (Top N-Grams Distribution):**
    - Không chỉ đếm từ đơn, hệ thống bóc tách các cụm từ ngữ cảnh như: *"rất chậm"*, *"không vào được"*, *"quá tuyệt"*.
4.  **Xu hướng Biến động (Rating Time-Series):**
    - Theo dõi biểu đồ đường để xem điểm trung bình App có đang tụt dốc theo thời gian hay không.

---

## 3. Luồng Hoạt Động Hệ Thống (End-to-End Pipeline Flow) 🔄
Quy trình xử lý dữ liệu được thiết kế theo chuẩn **Học Máy Thích Ứng (Active Learning Architecture)**:

### Bước 1: Thu Thập (Live Scraper)
Khi người dùng nhập App ID (ví dụ: `com.zing.zalo`), cỗ máy `google-play-scraper` sẽ kết nối thẳng tới Google Server để kéo 150 bình luận mới nhất (Fresh Data).

### Bước 2: Tiền Xử Lý (NLP Preprocessing)
- **Noise Cleanup:** Loại bỏ URL, Email, Numbers, Emojis thừa.
- **PUNCT Boundary:** AI nhận diện dấu chấm/phẩy để chặn "nghĩa lây lan" giữa các câu.
- **Compounding:** Ghép các cụm từ tiếng Việt chuyên biệt (`an_toàn`, `không_được`) để tránh cắt sai nghĩa.

### Bước 3: Phân Tích Đa Ngữ Cảnh (Contextual Analysis)
Hệ thống sử dụng bộ quy tắc **N-Gram Sliding Window**:
- **Negation Flipping:** Gặp chữ "không" sẽ đảo ngược cảm xúc đằng sau ("không tốt" -> Cực kỳ tệ).
- **Intensifier Boost:** Gặp chữ "rất", "quá" sẽ nhân đôi trọng số điểm cảm xúc.

### Bước 4: Tự Động Huấn Luyện (ML Continuous Learning)
Đây là tính năng đắt giá nhất:
- Mọi app được tìm kiếm sẽ được lưu vào `raw/master_training_data.csv`.
- AI sẽ chạy ngầm để so sánh các từ lạ với số Sao (Rating) thực tế của khách hàng.
- Nếu một từ lạ xuất hiện nhiều lần ở mức 1-2 Sao, AI tự động "Học" đó là từ xấu và nạp vào từ điển JSON cho các lần phân tích sau.

### Bước 5: Tổng Hợp & Visual Dashboards
Toàn bộ dữ liệu được đẩy vào Global Dashboard (`processed_reviews.csv`) để so sánh tương quan giữa tất cả các App trong thị trường.

---

## 4. Hướng Dẫn Vận Hành (How to Run) 🚀

### 1. Khởi chạy Backend (Cỗ máy AI & API)
Mở Terminal, đứng tại thư mục gốc và chạy:
```bash
python -m uvicorn api.main:app --reload
```
*Hệ thống sẽ chạy tại: `http://localhost:8000`*

### 2. Khởi chạy Frontend (Giao diện Dashboard)
Mở Terminal mới, chuyển vào thư mục frontend và chạy:
```bash
cd frontend
npm run dev
```
*Truy cập Website: `http://localhost:5173`*

### 3. Nghiệm thu kết quả
- Nhập App ID bất kỳ vào ô **"Live Prediction Engine"**.
- Bấm **"Run NLP Analysis"** để xem AI đọc hiểu và chấm nhãn ưu tiên (Priority 🚨).
- Bấm **"View Global Dashboard"** để so sánh app vừa tìm với thị trường chung.

---
*Dự án được xây dựng phục vụ cho mục tiêu Phân tích Dữ liệu và Trí tuệ Nhân tạo trong Kinh doanh.*
