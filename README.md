# 🚀 NeoRAG Cup 2025

## 1. Giới thiệu
NeoRAG Cup 2025 là cuộc thi học thuật – kỹ thuật do **Team AI – CLB Lập trình ProPTIT** tổ chức, dành cho các bạn đam mê **Trí tuệ nhân tạo (AI)**, **Xử lý ngôn ngữ tự nhiên (NLP)** và **Kỹ thuật hệ thống**.  

Người tham gia sẽ:
- Tự thiết kế, hiện thực hóa và trình bày một **pipeline RAG** (Retrieval-Augmented Generation) với domain là thông tin của CLB ProPTIT.
- Trải nghiệm toàn bộ quy trình phát triển sản phẩm AI từ **ý tưởng → triển khai → demo**.

---

## 2. Thể lệ & Yêu cầu
**Domain:** Thông tin liên quan đến CLB ProPTIT (lịch sử, thành viên, hoạt động, dự án, tài liệu nội bộ, v.v.)

**Nhiệm vụ:**
1. Thiết kế pipeline RAG hoàn chỉnh (kiến trúc, công nghệ, chiến lược index, retrieval, reranking, generation…).
2. Triển khai code hiện thực pipeline.
3. Chuẩn bị slide thuyết trình mô tả kiến trúc, giải pháp và kết quả.
4. Chạy demo hệ thống trong buổi pitching.

**Tài nguyên BTC cung cấp:**
- Bộ dataset chuẩn về CLB PROPTIT.
- Metrics benchmark: Context Recall, Context Precision, MRR, Hit@k, …

**Hình thức dự thi:** Cá nhân.

---

## 3. Mốc thời gian
- **Tuần 0:** Phát động cuộc thi, gửi dataset & benchmark metrics.
- **Tuần 1–3:** Hoàn thiện pipeline, code và slide.
- **Ngày Pitching:**
  - Tối đa **30 phút** thuyết trình + **10 phút** Q&A.
  - Chạy demo code trực tiếp (có thể dùng Streamlit).

---

## 4. Tiêu chí chấm điểm
| Tiêu chí                  | Trọng số |
|---------------------------|----------|
| Kiến trúc pipeline        | 30%      |
| Hiệu năng benchmark       | 40%      |
| Chất lượng demo           | 20%      |
| Kỹ năng thuyết trình      | 10%      |

---

## 5. Giải thưởng
🥇 **Giải Nhất:** 200.000 VNĐ + Giấy chứng nhận  
🥈 **Giải Nhì:** 150.000 VNĐ + Giấy chứng nhận  
🥉 **Giải Ba:** 100.000 VNĐ + Giấy chứng nhận  
🌟 **Giải Tiềm Năng:** 50.000 VNĐ + Giấy chứng nhận  

---

## 6. Đối tượng tham gia
- Thành viên thuộc **Team AI – CLB Lập trình ProPTIT**

📌 Hãy sẵn sàng **sáng tạo & bứt phá** cùng NeoRAG Cup 2025!  
💬 Mọi thắc mắc vui lòng comment hoặc inbox BTC để được giải đáp.

---

## 📊 Benchmark

- Trong suốt cuộc thi, các bạn sẽ chỉ được cung cấp bộ dữ liệu train. Bộ dữ liệu test sẽ được BTC công bố vào ngày thi cuối cùng. Dưới đây là benchmark của baseline model — mục tiêu của bạn là xây dựng mô hình có hiệu năng vượt qua được baseline model.
- Nhiệm vụ: Chỉnh sửa các file main.py, metrics_rag.py, vector_db.py và embeddings.py. Trong mỗi file đã được đánh dấu rõ vị trí cần chỉnh sửa — hãy đọc kỹ và thực hiện cẩn thận.

### **Retrieval – Train (100 query)** 
| K  | hit@k | recall@k | precision@k | f1@k | map@k | mrr@k | ndcg@k | context_precision@k | context_recall@k | context_entities_recall@k |
|----|-------|----------|-------------|------|-------|-------|--------|----------------------|------------------|---------------------------|
| 3  | 0.31  | 0.19     | 0.12        | 0.15 | 0.23  | 0.23  | 0.25   | 0.63                 | 0.50             | 0.32                      |
| 5  | 0.46  | 0.28     | 0.10        | 0.15 | 0.23  | 0.27  | 0.31   | 0.56                 | 0.44             | 0.37                      |
| 7  | 0.57  | 0.35     | 0.09        | 0.15 | 0.23  | 0.28  | 0.35   | 0.54                 | 0.40             | 0.38                      |

### **LLM Answer – Train (100 query)**
| K  | string_presence@k | rouge_l@k | bleu_4@k | groundedness@k | response_relevancy@k | noise_sensitivity@k |
|----|-------------------|-----------|----------|----------------|----------------------|---------------------|
| 3  | 0.35              | 0.21      | 0.03     | 0.57           | 0.80                 | 0.51                |
| 5  | 0.40              | 0.23      | 0.03     | 0.61           | 0.80                 | 0.53                |
| 7  | 0.41              | 0.22      | 0.04     | 0.64           | 0.80                 | 0.51                |

---

### **Retrieval – Test (30 query)**
| K  | hit@k | recall@k | precision@k | f1@k | map@k | mrr@k | ndcg@k | context_precision@k | context_recall@k | context_entities_recall@k |
|----|-------|----------|-------------|------|-------|-------|--------|----------------------|------------------|---------------------------|
| 3  | 0.23  | 0.06     | 0.08        | 0.07 | 0.12  | 0.12  | 0.15   | 0.34                 | 0.32             | 0.11                      |
| 5  | 0.40  | 0.10     | 0.08        | 0.09 | 0.16  | 0.16  | 0.22   | 0.35                 | 0.29             | 0.15                      |
| 7  | 0.47  | 0.13     | 0.08        | 0.10 | 0.17  | 0.17  | 0.24   | 0.31                 | 0.27             | 0.16                      |

### **LLM Answer – Test (30 query)**
| K  | string_presence@k | rouge_l@k | bleu_4@k | groundedness@k | response_relevancy@k | noise_sensitivity@k |
|----|-------------------|-----------|----------|----------------|----------------------|---------------------|
| 3  | 0.18              | 0.14      | 0.01     | 0.33           | 0.79                 | 0.68                |
| 5  | 0.16              | 0.15      | 0.01     | 0.30           | 0.79                 | 0.71                |
| 7  | 0.21              | 0.15      | 0.02     | 0.39           | 0.80                 | 0.71                |

---

### Giải thích các metrics

**hit@k**  
- **Ý nghĩa:** Tỷ lệ truy vấn mà kết quả đúng xuất hiện trong top-k kết quả trả về.  
- **Phản ánh:** Giá trị cao nghĩa là mô hình thường tìm được câu trả lời đúng trong top-k; giá trị thấp nghĩa là mô hình bỏ sót nhiều. Ưu điểm: dễ hiểu; nhược điểm: không xét vị trí chính xác trong top-k.

**recall@k**  
- **Ý nghĩa:** Tỷ lệ các câu trả lời đúng được tìm thấy trong top-k trên tổng số câu trả lời đúng.  
- **Phản ánh:** Cao → tìm được nhiều câu trả lời đúng; thấp → bỏ sót nhiều. Ưu điểm: đánh giá độ bao phủ; nhược điểm: không phản ánh độ chính xác.

**precision@k**  
- **Ý nghĩa:** Tỷ lệ câu trả lời đúng trong top-k kết quả.  
- **Phản ánh:** Cao → ít kết quả sai; thấp → nhiều kết quả nhiễu. Ưu điểm: đo độ chính xác; nhược điểm: không phản ánh số lượng câu trả lời tìm được.

**f1@k**  
- **Ý nghĩa:** Trung bình điều hòa của precision@k và recall@k.  
- **Phản ánh:** Cao → cân bằng tốt giữa độ chính xác và độ bao phủ; thấp → mất cân bằng. Ưu điểm: cân bằng hai yếu tố; nhược điểm: khó diễn giải nếu một chỉ số quá thấp.

**map@k (Mean Average Precision)**  
- **Ý nghĩa:** Trung bình của độ chính xác tại mỗi vị trí có kết quả đúng trong top-k.  
- **Phản ánh:** Cao → mô hình trả kết quả đúng ở vị trí cao; thấp → kết quả đúng nằm sâu. Ưu điểm: xét thứ tự kết quả; nhược điểm: tính toán phức tạp.

**mrr@k (Mean Reciprocal Rank)**  
- **Ý nghĩa:** Trung bình nghịch đảo của vị trí câu trả lời đúng đầu tiên trong top-k.  
- **Phản ánh:** Cao → câu trả lời đúng thường xuất hiện sớm; thấp → xuất hiện muộn. Ưu điểm: tập trung vào câu trả lời đúng đầu tiên; nhược điểm: bỏ qua các câu trả lời đúng khác.

**ndcg@k (Normalized Discounted Cumulative Gain)**  
- **Ý nghĩa:** Đo lường độ liên quan của kết quả, có xét vị trí trong top-k.  
- **Phản ánh:** Cao → kết quả liên quan ở vị trí cao; thấp → kết quả liên quan nằm sâu. Ưu điểm: phản ánh tốt thứ hạng; nhược điểm: cần thông tin độ liên quan.

**context_precision**  
- **Ý nghĩa:** Tỷ lệ thông tin ngữ cảnh được truy xuất là chính xác.  
- **Phản ánh:** Cao → ít thông tin dư thừa; thấp → nhiều nhiễu. Ưu điểm: đo độ sạch dữ liệu ngữ cảnh; nhược điểm: không xét độ đầy đủ.

**context_recall**  
- **Ý nghĩa:** Tỷ lệ thông tin ngữ cảnh đúng được lấy ra so với tổng số thông tin đúng.  
- **Phản ánh:** Cao → lấy được nhiều thông tin quan trọng; thấp → bỏ sót nhiều. Ưu điểm: đo độ bao phủ ngữ cảnh; nhược điểm: không phản ánh độ chính xác.

**context_entities_recall@k**  
- **Ý nghĩa:** Tỷ lệ thực thể (entities) đúng xuất hiện trong ngữ cảnh top-k.  
- **Phản ánh:** Cao → hầu hết thực thể cần thiết xuất hiện; thấp → nhiều thực thể bị thiếu. Ưu điểm: phù hợp cho bài toán yêu cầu thông tin thực thể; nhược điểm: phụ thuộc vào chất lượng nhận diện thực thể.

**string_presence@k**  
- **Ý nghĩa:** Tỷ lệ câu trả lời chứa đúng chuỗi ký tự kỳ vọng trong top-k.  
- **Phản ánh:** Cao → câu trả lời khớp trực tiếp với đáp án mong muốn; thấp → ít khớp. Ưu điểm: đơn giản; nhược điểm: không xét ý nghĩa tương đồng.

**rouge_l@k**  
- **Ý nghĩa:** Độ trùng khớp theo chuỗi con chung dài nhất (Longest Common Subsequence) giữa câu trả lời và đáp án.  
- **Phản ánh:** Cao → câu trả lời gần giống đáp án; thấp → ít trùng khớp. Ưu điểm: đánh giá tốt độ bao phủ; nhược điểm: không xét thứ tự chính xác toàn phần.

**bleu_4@k**  
- **Ý nghĩa:** Độ trùng khớp n-gram (4-gram) giữa câu trả lời và đáp án.  
- **Phản ánh:** Cao → câu trả lời sát ngữ cảnh đáp án; thấp → khác biệt lớn. Ưu điểm: phổ biến trong NLP; nhược điểm: nhạy với thay đổi nhỏ về từ ngữ.

**groundedness@k**  
- **Ý nghĩa:** Mức độ câu trả lời dựa trên thông tin đã truy xuất.  
- **Phản ánh:** Cao → ít thông tin bịa; thấp → nhiều thông tin ngoài ngữ cảnh. Ưu điểm: đánh giá tính tin cậy; nhược điểm: khó đo tự động chính xác.

**response_relevancy**  
- **Ý nghĩa:** Mức độ liên quan của câu trả lời với câu hỏi.  
- **Phản ánh:** Cao → câu trả lời phù hợp; thấp → lạc đề. Ưu điểm: phản ánh trải nghiệm người dùng; nhược điểm: cần đánh giá thủ công hoặc mô hình phụ.

**noise_sensitivity@k**  
- **Ý nghĩa:** Mức độ mô hình bị ảnh hưởng bởi dữ liệu nhiễu trong top-k.  
- **Phản ánh:** Cao → dễ bị nhiễu tác động; thấp → mô hình ổn định hơn. Ưu điểm: giúp kiểm tra khả năng chống nhiễu; nhược điểm: khó tính toán nếu không có dữ liệu nhiễu rõ ràng.

