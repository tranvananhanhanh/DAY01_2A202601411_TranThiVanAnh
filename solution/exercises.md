# K3 — Ngày 1: Bài Tập & Phản Ánh
## Khám Phá LLM API | Phiếu Thực Hành

**Thời lượng:** 9h00–13h00
**Cách làm:** Trả lời từng câu ngay sau khi hoàn thành block tương ứng —
đừng để dồn hết về cuối buổi. Thay dòng `*Câu trả lời của bạn*` bằng câu
trả lời thật (chấm tự động sẽ đếm số câu đã trả lời).

---

## Block 1 — API Cơ Bản (trả lời sau Checkpoint 1)

### Câu 1.1 — Độ nhạy của temperature
Gọi `call_openai` với temperature 0.0, 0.5, 1.0 và 1.5 dùng prompt
**"Hãy kể cho tôi một sự thật thú vị về Việt Nam."**

**Bạn nhận thấy quy luật gì qua bốn phản hồi?** (2–3 câu)
> *Ở mức nhiệt độ thấp (0.0 và 0.5), phản hồi của mô hình mang tính cô đọng, đi thẳng vào các sự thật mang tính cấu trúc cao (ví dụ: thông tin về Hang Sơn Đoòng). Khi nhiệt độ tăng lên mức trung bình và cao (1.0 và 1.5), mô hình bắt đầu tự định dạng câu trả lời một cách sáng tạo hơn (ví dụ: tự thêm tiêu đề phân tích "**Selecting the Best Fact:**") hoặc chuyển hẳn sang hành văn dạng hội thoại tự nhiên bằng tiếng Việt, cho thấy tính ngẫu nhiên và độ biến thiên của từ ngữ tăng lên rõ rệt.
*

### Câu 1.2 — Chọn temperature cho sản phẩm
**Bạn sẽ đặt temperature bao nhiêu cho chatbot hỗ trợ khách hàng, và tại sao?**
> *Tôi sẽ đặt temperature ở mức thấp, từ 0.0 đến 0.2. Chatbot hỗ trợ khách hàng yêu cầu tính nhất quán, chính xác về mặt thông tin nghiệp vụ và tránh hiện tượng "ảo tưởng" (hallucination). Việc đặt nhiệt độ thấp giúp đảm bảo mô hình luôn trả về cùng một câu trả lời chuẩn xác cho các câu hỏi trùng lặp, giữ vững tính chuyên nghiệp của dịch vụ.*

### Câu 1.3 — Đánh đổi chi phí
Kịch bản: 10.000 người dùng hoạt động mỗi ngày, mỗi người gọi API 3 lần,
mỗi lần trung bình ~350 token đầu ra.

**Ước tính GPT-4o đắt hơn GPT-4o-mini bao nhiêu lần cho workload này? Nêu một
trường hợp GPT-4o xứng đáng với chi phí và một trường hợp nên dùng mini:**

> * Trong code em sử dụng model Gemini Flash và Gemini Flash Lite thay vì model GPT :  Dựa trên bảng giá của Gemini Flash và Gemini Flash Lite, Gemini Flash có chi phí cao hơn Gemini Flash Lite, đổi lại cung cấp khả năng suy luận và chất lượng phản hồi tốt hơn đối với các tác vụ phức tạp.

Trường hợp dùng Gemini Flash: Phân tích hợp đồng pháp lý, xử lý tài liệu dài hoặc giải quyết các bài toán suy luận nhiều bước, nơi độ chính xác của kết quả là yếu tố quan trọng.

Trường hợp dùng Gemini Flash Lite: Hệ thống chatbot trả lời các câu hỏi thường gặp (FAQs), phân loại hoặc gán nhãn văn bản, hỗ trợ khách hàng và các ứng dụng có lưu lượng truy cập lớn cần tối ưu tốc độ và chi phí.*

---

## Block 2 — System Prompt & Token (trả lời sau Checkpoint 2)

### Câu 2.1 — Sức mạnh của persona
Gọi `chat_with_system_prompt` hai lần với cùng câu hỏi
**"Giải thích blockchain là gì?"** nhưng hai system prompt khác nhau:
- "Bạn là giáo viên tiểu học, giải thích thật đơn giản cho trẻ 8 tuổi."
- "Bạn là chuyên gia tài chính, trả lời chuyên sâu bằng thuật ngữ kỹ thuật."

**Hai phản hồi khác nhau như thế nào (độ dài, từ vựng, ví dụ)? System prompt
ảnh hưởng đến hành vi model ra sao?** (3–4 câu)
> *Phản hồi của Persona 1 (Giáo viên tiểu học) mở đầu rất thân thiện bằng từ ngữ xưng hô gần gũi ("Chào con! Thầy/Cô rất vui...") kết hợp với cấu trúc giải thích đơn giản hóa. Ngược lại, Persona 2 (Chuyên gia tài chính) sử dụng định dạng chuyên nghiệp hơn ("**Blockchain** (Chuỗi khối)...") và đi thẳng vào các thuật ngữ chuyên ngành. System prompt đóng vai trò như một bộ lọc định hình toàn bộ phong cách ngôn ngữ, từ vựng, cấu trúc và cách tiếp cận vấn đề của mô hình ngay từ đầu đầu vào.*

### Câu 2.2 — tiktoken vs đếm từ
Chọn một đoạn văn tiếng Việt ~100 từ. So sánh số token theo `count_tokens`
(tiktoken) với ước lượng `số từ / 0.75` mà Part 1 đã dùng.

**Hai con số chênh nhau bao nhiêu phần trăm? Vì sao tiếng Việt thường tốn
nhiều token hơn tiếng Anh cùng độ dài?**
> *Hai con số chênh nhau 16.91% (Số từ thực tế là 121 từ trong khi số token thực tế đo được là 138). Tiếng Việt tốn nhiều token hơn tiếng Anh vì các bộ mã hóa (tokenizer) hiện tại của các mô hình ngôn ngữ lớn được tối ưu hóa chủ yếu dựa trên ngữ liệu tiếng Anh. Do đó, các từ tiếng Việt đa âm tiết hoặc các ký tự có dấu thanh thường bị phân tách thành nhiều mảnh từ (sub-word tokens) hoặc các ký tự byte-level nhỏ hơn thay vì được tính là một token nguyên vẹn như tiếng Anh.*

---

## Block 3 — Streaming & Độ Bền (trả lời sau Checkpoint 3)

### Câu 3.1 — Trải nghiệm người dùng với streaming
**Streaming quan trọng nhất trong trường hợp nào, và khi nào thì
non-streaming lại phù hợp hơn?** (1 đoạn văn)
> *Streaming quan trọng nhất trong các ứng dụng chatbot tương tác trực tiếp với người dùng (như ChatGPT, trợ lý CLI), nơi việc giảm "độ trễ nhận thức" (perceived latency) là yếu tố quyết định trải nghiệm người dùng — người dùng có thể bắt đầu đọc ngay khi token đầu tiên xuất hiện thay vì đợi cả đoạn văn dài xử lý xong. Ngược lại, non-streaming phù hợp hơn cho các tác vụ xử lý ngầm (background jobs), các lời gọi API tích hợp hệ thống, hoặc khi cần trích xuất dữ liệu có cấu trúc (như JSON) để hệ thống phía sau phân tích cú pháp (parse) trước khi thực hiện bước tiếp theo.*

### Câu 3.2 — Vì sao backoff theo cấp số nhân?
**So với delay cố định (ví dụ luôn chờ 1 giây), exponential backoff có lợi
thế gì khi API bị quá tải? Điều gì xảy ra nếu hàng nghìn client cùng retry
với delay cố định giống nhau?**
> *Exponential backoff giúp giãn cách thời gian giữa các lần thử lại sau mỗi thất bại, cho phép máy chủ có đủ khoảng trống thời gian để phục hồi sau sự cố quá tải. Nếu hàng nghìn client cùng retry với một khoảng trễ cố định (ví dụ 1 giây), hành vi này sẽ tạo ra hiệu ứng cộng dồn "thundering herd", liên tục gửi các đợt yêu cầu đồng thời cực lớn vào server, khiến hệ thống bị nghẽn mạch nghiêm trọng hơn và không thể phục hồi.*

---

## Block 4 — Mini-Project (trả lời sau Checkpoint 4)

### Câu 4.1 — Thiết kế persona
**Bạn chọn persona gì cho trợ lý của mình? Viết lại system prompt đó và giải
thích 1–2 lựa chọn từ ngữ quan trọng trong prompt (ví dụ: vì sao yêu cầu
"trả lời ngắn gọn", vì sao chỉ định ngôn ngữ...):**
> *System Prompt: "Bạn là một trợ giảng AI thân thiện, chuyên hỗ trợ sinh viên học lập trình Python. Hãy trả lời thật ngắn gọn, dễ hiểu, sử dụng tiếng Việt và luôn đi kèm một ví dụ code minh họa tối giản khi giải thích khái niệm kỹ thuật."
Giải thích lựa chọn từ ngữ:
Yêu cầu "ngắn gọn" để giảm thiểu chi phí token đầu ra và giúp người học dễ tiếp thu nhanh trên giao diện dòng lệnh (CLI).
Yêu cầu "luôn đi kèm một ví dụ code minh họa tối giản" giúp lý thuyết lập trình trở nên thực tế và trực quan ngay lập tức.*

### Câu 4.2 — Hạn chế & cải thiện
**Trợ lý của bạn hiện có hạn chế lớn nhất là gì (ví dụ: history chỉ 3 lượt,
không có bộ nhớ dài hạn, không kiểm duyệt nội dung...)? Đề xuất một cải
thiện cụ thể và mô tả ngắn cách triển khai:**
> *Hạn chế lớn nhất: Trợ lý hiện tại không lưu giữ được lịch sử hội thoại lâu dài giữa các phiên chạy khác nhau; mỗi lần khởi động lại chương trình, toàn bộ ngữ cảnh đã thảo luận trước đó đều bị mất sạch.
Giải pháp cải thiện: Triển khai tính năng lưu trữ lịch sử hội thoại dưới dạng file JSON cục bộ (ví dụ: chat_history.json).
Cách triển khai: Khi chương trình kết thúc, ghi toàn bộ mảng history vào file JSON. Khi khởi động lại chương trình, kiểm tra sự tồn tại của file này, đọc dữ liệu lên và nạp lại vào biến history để tiếp tục cuộc đối thoại cũ.*

---

## Danh Sách Kiểm Tra Nộp Bài

- [ ] `python grade.py` — xem điểm tự động, mục tiêu ≥ 75/100
- [ ] Cả 4 checkpoint pytest đều pass
- [ ] Tất cả 9 câu trong file này đã được trả lời
- [ ] Đã copy bài làm vào folder `solution/` và zip theo hướng dẫn README
