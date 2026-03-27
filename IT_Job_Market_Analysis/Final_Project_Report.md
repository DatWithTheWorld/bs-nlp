# BÁO CÁO ĐỀ TÀI: PHÂN TÍCH VÀ ĐÁNH GIÁ TRẢI NGHIỆM NGƯỜI DÙNG QUA GOOGLE PLAY STORE
## (Hệ Thống Business Sentiment Intelligence - Real-time Analysis)

**Người thực hiện:** (Họ và tên sinh viên)  
**Đề tài:** Business Intelligence & Sentiment Analysis  
**Ngày thực hiện:** 27 tháng 03, 2026

---

## I. TỔNG QUAN ĐỀ TÀI (EXECUTIVE SUMMARY)
Trong bối cảnh chuyển đổi số, phản hồi khách hàng (Customer Feedback) trên các kho ứng dụng (App Store) là nguồn dữ liệu quan trọng nhất để đánh giá sức khỏe thương hiệu. Đề tài này xây dựng một hệ thống **Business Sentiment Intelligence (BSI)** nhằm tự động hóa việc thu thập, phân tích và định lượng cảm xúc người dùng từ Google Play Store theo thời gian thực. Hệ thống sử dụng các kỹ thuật Xử lý Ngôn ngữ Tự nhiên (NLP) chuyên sâu và Học Máy (Machine Learning) để hỗ trợ doanh nghiệp ra quyết định dựa trên dữ liệu.

---

## II. MỤC TIÊU & GIÁ TRỊ DOANH NGHIỆP (BUSINESS OBJECTIVES)

### 1. Quản Trị Danh Tiếng (Brand Reputation Management)
Xác định mức độ hài lòng của khách hàng thông qua chỉ số **Polarity Score (-1.0 đến 1.0)**. Giúp doanh nghiệp nhận diện ngay lập tức nếu danh tiếng thương hiệu bị giảm sút.

### 2. Quản Trị Rủi Ro & Khủng Hoảng (Crisis Management)
Tự động phân loại các phản hồi tiêu cực theo **Ma trận Độ ưu tiên (Priority Matrix)**:
- **Nguy cấp (CRITICAL 🚨):** Rủi ro tài chính, lỗi bảo mật.
- **Nghiêm trọng (HIGH 🔴):** Lỗi văng ứng dụng, không đăng nhập được.
- **Trung bình (MEDIUM 🟡):** App chậm, lag, khó sử dụng.

### 3. Tối Ưu Hóa Sản Phẩm (Product Optimization)
Phát hiện các "Điểm mù" (Pain Points) của sản phẩm để đội ngũ kỹ thuật tập trung cải thiện đúng tính năng người dùng đang phàn nàn nhiều nhất.

### 4. Phân Tích Đối Thủ (Competitor Benchmarking)
So sánh trực diện hiệu quả kinh doanh và mức độ hài lòng của khách hàng giữa các ứng dụng cùng ngành (ví dụ: Momo vs ZaloPay, Shopee vs Tiki).

---

## III. GIẢI PHÁP CÔNG NGHỆ (METHODOLOGY & ARCHITECTURE)

Hệ thống được thiết kế theo mô hình **ELT (Extract - Load - Transform)** hiện đại:

### 1. Thu Thập Dữ Liệu (Live Data Scraping)
Sử dụng thư viện `google-play-scraper` để kết nối trực tiếp với máy chủ Google, lấy dữ liệu phản hồi mới nhất mà không cần cơ sở dữ liệu tĩnh trung gian.

### 2. Xử Lý Ngôn Ngữ Tự Nhiên (Advanced NLP Pipeline)
- **Chuẩn hóa dữ liệu:** Loại bỏ nhiễu (URL, Email, Đặc ký tự) bằng Regex.
- **N-Gram Sliding Window:** Nhận diện cụm từ ngữ cảnh (vd: *"rất tệ"*, *"không hay"*).
- **Negation Flipping:** Thuật toán đảo ngược cảm xúc khi gặp các từ phủ định (*"không"*, *"chưa"*).
- **Intensifier Boost:** Nhân đôi trọng số cảm xúc khi gặp các trạng từ chỉ mức độ (*"rất"*, *"quá"*).

### 3. Học Máy Tích Cực (Active Learning / Auto-Training)
Hệ thống có khả năng **Tự học từ vựng mới**. Mỗi khi người dùng thực hiện lượt quét App mới, AI sẽ tự động so sánh văn bản với số Sao (Rating) thực tế để cập nhật từ điển cảm xúc JSON, giúp hệ thống ngày càng chính xác theo thời gian.

---

## IV. KẾT QUẢ PHÂN TÍCH (RESULTS & VISUALIZATION)

Dữ liệu được trình bày trực quan trên **Business Dashboard**:
- **Polarity Comparison:** Biểu đồ BarChart so sánh điểm danh tiếng các App.
- **Sentiment Breakdown:** Biểu đồ PieChart tỷ lệ Tích cực / Tiêu cực của thị trường.
- **Pain Points Ranking:** Bảng xếp hạng các lỗi cần ưu tiên xử lý nhất.
- **Rating Trend:** Biểu đồ LineChart theo dõi biến động điểm số theo thời gian.

---

## V. KẾT LUẬN
Hệ thống **Business Sentiment Intelligence** đã chứng minh được tính hiệu quả trong việc chuyển hóa dữ liệu văn bản thô thành các chỉ số kinh doanh có giá trị. Việc tích hợp Active Learning giúp hệ thống linh hoạt với các ngôn ngữ mới và từ lóng mạng xã hội, tạo ra một giải pháp tự động hóa hoàn toàn cho khâu giám sát trải nghiệm khách hàng của doanh nghiệp.

---

*(Hướng dẫn: Bấm Ctrl+P trên trình soạn thảo VS Code và chọn "Save as PDF" để xuất báo cáo này thành định dạng PDF chuyên nghiệp).*
