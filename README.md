# VNG App Review & Sentiment Analysis Pipeline 🚀

Dự án Phân tích cảm xúc (Sentiment Analysis) sử dụng Machine Learning (SVM, Logistic Regression...) và Deep Learning (BiLSTM, CNN) để phân loại 25.000+ từ khoá đánh giá của người dùng đối với các ứng dụng thuộc hệ sinh thái VNG (Zalo, Zing MP3, VNGGames...).

Được tích hợp cùng Dashboard phân tích thời gian thực chạy trên nền Vite + Chart.js.

---

## 📁 Cấu trúc Dự án (Project Structure)

Dự án bao gồm 3 phân hệ độc lập ghép lại thành 1 vòng tuần hoàn:
1. **`vng_reviews_data/`**: Chứa hơn 30,000+ dòng Data thô crawl trực tiếp từ cửa hàng Google Play Store.
2. **`vng_sentiment_analysis/`**: "Bộ não" AI của dự án (Python). Đọc, tiền xử lý ngôn ngữ Tiếng Việt, đào tạo mô hình tự động nhận diện cảm xúc.
3. **`vng-sentiment-dashboard/`**: Front-end diện mạo Data (Vite + Node). Bọc thông tin chạy từ AI trả ra giao diện Web Dark-Mode tương tác trực quan 18 loại biểu đồ.

---

## ⚙️ Yêu cầu Hệ thống (Prerequisites)
*   **Python:** Cài đặt phiên bản `3.9 - 3.11`.
*   **Node.js:** Bắt buộc phải có để chạy Web Dashboard (phiên bản `v16+`).

---

## 🛠️ HƯỚNG DẪN CÀI ĐẶT VÀ CHẠY FULL PROJECT

### BƯỚC 1: MÔI TRƯỜNG AI (PYTHON BACKEND)
Điều kiện bắt buộc là phải đẩy cho mô hình AI chạy ra được file kết quả `ml_results.json` thì Web Dashboard mới có Data để vẽ.

1. **Mở Terminal mới** nằm ngay tại thư mục Gốc của dự án (`ANI`).
2. **Cài đặt thư viện Python:**
   ```powershell
   pip install pandas numpy scikit-learn tensorflow matplotlib seaborn jupyter imbalanced-learn underthesea
   ```
3. **Di chuyển vào trung tâm não bộ AI:**
   ```powershell
   cd vng_sentiment_analysis
   ```
4. **THỰC THI CHẠY AI:**
   Bạn có **2 lựa chọn** tuỳ mục đích sử dụng:
   
   👉 **Cách A (Chạy Tự động ngầm):**
   ```powershell
   python main.py
   ```
   *(Hệ thống sẽ chạy liên hoàn Pipeline, training K-Fold, sử dụng thuật toán bù đắp mảng vỡ SMOTE, và sinh ra Toàn bộ Model + Hình ảnh lưu vào thư mục `vng_sentiment_analysis/output/`)*
   
   👉 **Cách B (Chạy thông qua báo cáo thị giác Jupyter Notebook):**
   Rất phù hợp để nộp bài tập / Thuyết trình giảng viên. Mở VS Code (đã cài đặt Extensions Jupyter), tìm đến file `vng_sentiment_analysis_full_pipeline.ipynb` và bấm lệnh **"Run All"** ở trên đỉnh.

---

### BƯỚC 2: MÔI TRƯỜNG WEB DASHBOARD (VITE FRONTEND)
Sau khi Bước 1 chạy xong (quá trình Train DL sẽ tốn của bạn cỡ 3-5 phút), ta bắt đầu mở Web để xem tổng quát trực quan.

1. **Mở một Terminal MỚI HOÀN TOÀN** (Tách biệt khỏi Terminal Python hồi nãy).
2. **Di chuyển vào thư mục Web:**
   ```powershell
   cd vng-sentiment-dashboard
   ```
3. **Cài đặt gói Modules Nodejs (Chỉ chạy lần đầu tiên):**
   ```powershell
   npm install
   ```
4. **Khởi động Giao diện:**
   ```powershell
   npm run dev
   ```
5. 🚀 Terminal sẽ báo địa chỉ Local (Thường là `http://localhost:5173/`). Bạn bấm `Ctrl + Click` vào đường link để mở bằng Trình duyệt.

---

## 📚 TÀI LIỆU CHUYÊN SÂU
Mọi diễn giải cho dự án này đều đã được Document lại rất chi tiết với chuẩn báo cáo nghiên cứu đồ án, bạn hãy đọc 2 file sau ở trong thư mục `vng_sentiment_analysis/`:
1. `report.pdf`: Outline tổng quan kiến trúc mã nguồn.
2. `training_explained.pdf`: Đi cực sâu vào phân tích Code Training ML, cách kiến trúc DL 2 chiều hoạt động, ý nghĩa kĩ thuật SMOTE và cách đọc các loại siêu Biểu Đồ đánh giá Confusion.
