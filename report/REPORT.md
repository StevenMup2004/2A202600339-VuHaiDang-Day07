# Báo Cáo Lab 7: Embedding & Vector Store

**Họ tên:** [Vũ Hải Đăng]
**Nhóm:** [2A202600113]
**Ngày:** [C401-E5]

---

## 1. Warm-up (5 điểm)

### Cosine Similarity (Ex 1.1)

**High cosine similarity nghĩa là gì?**
>  Nó có nghĩa là hai đoạn văn bản rất giống nhau về mặt ý nghĩa. Vector embedding của chúng gần như cùng hướng, nên dù cách dùng từ có thể khác, nội dung mà chúng truyền tải vẫn gần như giống nhau.

**Ví dụ HIGH similarity:**
- Sentence A: “Tôi thích học về trí tuệ nhân tạo.”  
- Sentence B: “Tôi rất thích tìm hiểu về AI.”
- Tại sao tương đồng: Hai câu này nói gần như cùng một ý. Chỉ khác cách diễn đạt một chút, nên nhìn chung vẫn đang nói về việc thích học AI  → độ tương đồng cao.

**Ví dụ LOW similarity:**
- Sentence A: “Tôi thích học về trí tuệ nhân tạo.” 
- Sentence B: “Hôm nay thời tiết rất nóng.”
- Tại sao khác: Hai câu này không liên quan gì đến nhau: một câu nói về sở thích học tập, câu kia nói về thời tiết → độ tương đồng thấp.

**Tại sao cosine similarity được ưu tiên hơn Euclidean distance cho text embeddings?**
> Cosine similarity được dùng nhiều vì nó so sánh hướng của vector, tức là tập trung vào ý nghĩa của câu. Với text embeddings, hai câu có cùng ý thường sẽ có vector cùng hướng, dù độ dài khác nhau. Trong khi đó, Euclidean distance bị ảnh hưởng cả bởi độ dài vector, nên dễ làm sai lệch khi đo độ giống nhau về ngữ nghĩa.
### Chunking Math (Ex 1.2)

**Document 10,000 ký tự, chunk_size=500, overlap=50. Bao nhiêu chunks?**
> *Đáp án:*
> Ta dùng công thức:  
> num_chunks = ceil((N - overlap) / (chunk_size - overlap))  
> = ceil((10,000 - 50) / (500 - 50))  
> = ceil(9950 / 450) ≈ ceil(22.11) = 23  
> **Đáp án: khoảng 23 chunks.**

**Nếu overlap tăng lên 100, chunk count thay đổi thế nào? Tại sao muốn overlap nhiều hơn?**
> Khi tăng overlap lên 100, số lượng chunk cũng tăng lên:  
> num_chunks = ceil((10,000 - 100) / (500 - 100))  
> = ceil(9900 / 400) = 25  
>Số chunk tăng (từ khoảng 23 lên 25) vì các chunk chồng lên nhau nhiều hơn. Chúng ta dùng overlap lớn hơn để giữ ngữ cảnh giữa các đoạn. Nhờ vậy, thông tin ở ranh giới không bị mất và kết quả tìm kiếm sẽ chính xác hơn.

---

## 2. Document Selection — Nhóm (10 điểm)

### Domain & Lý Do Chọn

**Domain:** [Tài liệu bói toán]

**Tại sao nhóm chọn domain này?**
> *Viết 2-3 câu:* Domain lạ, tìm được tài liệu đạt yêu cầu, xác định được câu hỏi và trả lời làm ground truth


### Data Inventory

| # | Tên tài liệu | Nguồn | Số ký tự | Metadata đã gán |
|---|--------------|-------|----------|-----------------|
| 1 |Tử vi đẩu số tân biên |https://drive.google.com/file/d/1DTBH6jhq0ia_RenD3my6LDQdv0Bk_HQO/edit?fbclid=IwY2xjawRFgxBleHRuA2FlbQIxMABicmlkETF2Rkt4UlVBZjFGWW5tOWFxc3J0YwZhcHBfaWQQMjIyMDM5MTc4ODIwMDg5MgABHlzYdm64HUDC9ciLL_NPBp-u0xjvLPJBIn2RTxi2hPQ2KH0Em0bwvZHSOhY6_aem__0NJhHXdZdtRnGFPlVHYBw |36904 |? |


### Metadata Schema

| Trường metadata | Kiểu | Ví dụ giá trị | Tại sao hữu ích cho retrieval? |
|----------------|------|---------------|-------------------------------|
| source|String | "tuvi_sach_goc.pdf", "blog_tuvi_2023"  | Xác định độ tin cậy của nguồn tin. Khi có mâu thuẫn kiến thức, hệ thống có thể ưu tiên dữ liệu từ nguồn chính thống (sách gốc) hơn là các bài blog.|
| category|Enum |"chinh_tinh", "phu_tinh", "ngu_hanh" |Cho phép thu hẹp phạm vi tìm kiếm. Nếu người dùng hỏi về ""Sao Tử Vi"", hệ thống chỉ tìm trong vùng dữ liệu đã tag là chinh_tinh, tránh nhiễu từ các đoạn văn nói về các sao khác. |
| update|DateTime |"2024-03-15", "2022-11-01" |Giúp hệ thống thực hiện Recency Bias. Trong các tài liệu nghiên cứu mới về tử vi hiện đại, thông tin cập nhật gần nhất thường có giá trị hiệu chỉnh cao hơn các bản dịch cũ. |
| access|Integer |0 (public), 1 (internal), 2 (private) |Quản lý quyền truy cập dữ liệu (Security). Đảm bảo các ghi chú cá nhân hoặc dữ liệu khách hàng nhạy cảm không bị lộ ra khi người dùng phổ thông truy vấn hệ thống. |

---

## 3. Chunking Strategy — Cá nhân chọn, nhóm so sánh (15 điểm)

### Baseline Analysis

Chạy `ChunkingStrategyComparator().compare()` trên tài liệu `data/data.md` với `chunk_size=200`:

| Tài liệu | Strategy | Chunk Count | Avg Length | Preserves Context? |
|-----------|----------|-------------|------------|-------------------|
| data.md | FixedSizeChunker (`fixed_size`) | 246 | 199.81 | Yes |
| data.md | SentenceChunker (`by_sentences`) | 108 | 338.48 | Yes |

### Strategy Của Tôi

**Loại:** [RecursiveChunker]

**Mô tả cách hoạt động:**
> Thuật toán chạy đệ quy với thứ tự separator ưu tiên từ lớn đến nhỏ: \n\n, \n, . , dấu cách, rồi fallback khi không còn separator. Base case có hai nhánh: đoạn hiện tại đã nhỏ hơn hoặc bằng chunk_size, hoặc đã hết separator để tách tiếp thì trả luôn đoạn đó. Khi tách ra, em phân thành good_chunks (đủ nhỏ) và bad_chunks (quá dài), sau đó chỉ đệ quy tiếp trên bad_chunks.

**Tại sao tôi chọn strategy này cho domain nhóm?**
> Nội dung bói toán thường được chia thành nhiều mục với tiêu đề rõ ràng, nhưng độ dài các đoạn lại rất không đều. Nếu cắt theo kiểu cố định thì dễ bị mất ý hoặc đứt mạch nội dung. Vì vậy, mình chọn tách theo cấu trúc văn bản sẽ hợp lý hơn. RecursiveChunker tận dụng đúng đặc điểm này khi ưu tiên ngắt theo ranh giới ngữ nghĩa trước, rồi mới đến các quy tắc kỹ thuật. Nhờ đó, các đoạn (chunk) lấy ra khi truy xuất thường dễ đọc hơn và bám sát ý hơn khi dùng để trả lời các câu hỏi nghiệp vụ.


### So Sánh: Strategy của tôi vs Baseline

| Tài liệu | Strategy | Chunk Count | Avg Length | Retrieval Quality? |
|-----------|----------|-------------|------------|--------------------|
| data.md | best baseline: FixedSizeChunker | 246 | 199.81 | Medium |
| data.md | **của tôi: RecursiveChunker** | 228 | 185.42 | High |

### So Sánh Với Thành Viên Khác


| Thành viên | Strategy | Retrieval Score (/10) | Điểm mạnh | Điểm yếu |
|-----------|----------|----------------------|-----------|----------|
| Tôi | RecursiveChunker | 8.0 | Giữ ngữ cảnh tốt, chunk rõ nghĩa | Phụ thuộc vào cấu trúc văn bản, cần chọn separator phù hợp |
| Dũng | SentenceChunker | 7.0 | Giữ ngữ cảnh tốt, xử lý văn bản phức tạp | Chunk có thể không đồng đều, khó kiểm soát kích thước |
| Huy | FixedSizeChunker | 7.5 | Đơn giản, ổn định, dễ kiểm soát kích thước chunk | Dễ cắt ngang ý, mất ngữ cảnh ở đoạn dài |
| Sơn | RecursiveChunker | 8.5 | Giữ ngữ cảnh tốt, phù hợp tài liệu có cấu trúc phức tạp | Logic phức tạp hơn, phụ thuộc separator |
| Đạt | RecursiveChunker | 8.0 | Tạo chunk tự nhiên, dễ đọc | Chunk có thể quá dài hoặc quá ngắn tùy nội dung |
| Tuấn | RecursiveChunker | 8.0 | Tối ưu theo domain, cân bằng giữa context và size | Cần tinh chỉnh nhiều, khó triển khai hơn baseline |

**Strategy nào tốt nhất cho domain này? Tại sao?**
> Với tập tài liệu hiện tại, RecursiveChunker là lựa chọn hợp lý vì nội dung gồm nhiều đoạn dài và được chia thành các mục rõ ràng. Thay vì cắt theo độ dài cố định, cách này ưu tiên tách theo ngữ nghĩa nên giữ được mạch nội dung tốt hơn. Nhờ vậy, các chunk phục vụ retrieval thường ổn định và dễ sử dụng hơn. Trong các thử nghiệm baseline, nó cũng cho thấy sự cân đối tốt giữa kích thước chunk và tính liên kết của thông tin.

---

## 4. My Approach — Cá nhân (10 điểm)

Giải thích cách tiếp cận của bạn khi implement các phần chính trong package `src`.

### Chunking Functions

**`SentenceChunker.chunk`** — approach:
> Em dùng regex (?<=[.!?])\s+ để detect ranh giới câu dựa trên dấu chấm, chấm than, chấm hỏi và khoảng trắng phía sau. Sau khi tách, em strip từng câu và loại bỏ phần rỗng để tránh sinh chunk rác khi dữ liệu có nhiều khoảng trắng liên tiếp. Cuối cùng em gom câu theo max_sentences_per_chunk, đủ ngưỡng thì đóng chunk, còn dư thì đưa vào chunk cuối.

**`RecursiveChunker.chunk` / `_split`** — approach:
> Thuật toán chạy đệ quy với thứ tự separator ưu tiên từ lớn đến nhỏ: \n\n, \n, . , dấu cách, rồi fallback khi không còn separator. Base case có hai nhánh: đoạn hiện tại đã nhỏ hơn hoặc bằng chunk_size, hoặc đã hết separator để tách tiếp thì trả luôn đoạn đó. Khi tách ra, em phân thành good_chunks (đủ nhỏ) và bad_chunks (quá dài), sau đó chỉ đệ quy tiếp trên bad_chunks.

### EmbeddingStore

**`add_documents` + `search`** — approach:
>  Khi add, mỗi Document được embed bằng embedding_fn rồi chuẩn hóa thành record gồm id, content, metadata, embedding. Mặc định store chạy in-memory để ổn định; chỉ khi USE_CHROMA được bật mới dùng Chroma. Search in-memory embed query rồi tính dot product với từng embedding, sort giảm dần theo score để lấy top-k; còn với Chroma thì query bằng query_embeddings và chuẩn hóa score theo 1.0 - distance.

**`search_with_filter` + `delete_document`** — approach:
>  Với search_with_filter, em ưu tiên lọc metadata trước rồi mới tính similarity để giảm nhiễu và tăng precision. Nếu không có filter thì dùng lại search thường; nếu chạy Chroma thì truyền where cùng query_embeddings. Delete_document xóa theo doc_id: in-memory dùng list comprehension để loại record trùng id, còn Chroma thì get trước rồi delete nếu tồn tại.

### KnowledgeBaseAgent

**`answer`** — approach:
> Agent làm theo pipeline RAG: retrieve top-k chunk, ghép context, tạo prompt theo cấu trúc Context -> Question -> Answer rồi gọi llm_fn. Để tránh overflow khi gọi model thật, em giới hạn cả độ dài mỗi chunk (4000 ký tự) và tổng context (12000 ký tự) trước khi inject vào prompt. Cách này giữ được grounding nhưng vẫn an toàn với giới hạn context của LLM.

### Test Results

```
```
PS C:\Users\dangv\Downloads\VinCourse\daypython -m pytest tests/test_solution.py -v
============================= test session starts =============================
platform win32 -- Python 3.14.3, pytest-9.0.2, pluggy-1.6.0 -- C:\Users\dangv\AppData\Local\Python\pythoncore-3.14-64\python.exe
cachedir: .pytest_cache
rootdir: C:\Users\dangv\Downloads\VinCourse\day06\Steven\Day-07-Lab-Data-Foundations
plugins: anyio-4.13.0
collected 42 items                                                             

tests/test_solution.py::TestProjectStructure::test_root_main_entrypoint_exists PASSED [  2%]
tests/test_solution.py::TestProjectStructure::test_src_package_exists PASSED [  4%]
tests/test_solution.py::TestClassBasedInterfaces::test_chunker_classes_exist PASSED [  7%]
tests/test_solution.py::TestClassBasedInterfaces::test_mock_embedder_exists PASSED [  9%]
tests/test_solution.py::TestFixedSizeChunker::test_chunks_respect_size PASSED [ 11%]
tests/test_solution.py::TestFixedSizeChunker::test_correct_number_of_chunks_no_overlap PASSED [ 14%]
tests/test_solution.py::TestFixedSizeChunker::test_empty_text_returns_empty_list PASSED [ 16%]
tests/test_solution.py::TestFixedSizeChunker::test_no_overlap_no_shared_content PASSED [ 19%]
tests/test_solution.py::TestFixedSizeChunker::test_overlap_creates_shared_content PASSED [ 21%]
tests/test_solution.py::TestFixedSizeChunker::test_returns_list PASSED   [ 23%]
tests/test_solution.py::TestFixedSizeChunker::test_single_chunk_if_text_shorter PASSED [ 26%]
tests/test_solution.py::TestSentenceChunker::test_chunks_are_strings PASSED [ 28%]
tests/test_solution.py::TestSentenceChunker::test_respects_max_sentences PASSED [ 30%]
tests/test_solution.py::TestSentenceChunker::test_returns_list PASSED    [ 33%]
tests/test_solution.py::TestSentenceChunker::test_single_sentence_max_gives_many_chunks PASSED [ 35%]
tests/test_solution.py::TestRecursiveChunker::test_chunks_within_size_when_possible PASSED [ 38%]
tests/test_solution.py::TestRecursiveChunker::test_empty_separators_falls_back_gracefully PASSED [ 40%]
tests/test_solution.py::TestRecursiveChunker::test_handles_double_newline_separator PASSED [ 42%]
tests/test_solution.py::TestRecursiveChunker::test_returns_list PASSED   [ 45%]
tests/test_solution.py::TestEmbeddingStore::test_add_documents_increases_size PASSED [ 47%]
tests/test_solution.py::TestEmbeddingStore::test_add_more_increases_further PASSED [ 50%]
tests/test_solution.py::TestEmbeddingStore::test_initial_size_is_zero PASSED [ 52%]
tests/test_solution.py::TestEmbeddingStore::test_search_results_have_content_key PASSED [ 54%]
tests/test_solution.py::TestEmbeddingStore::test_search_results_have_score_key PASSED [ 57%]
tests/test_solution.py::TestEmbeddingStore::test_search_results_sorted_by_score_descending PASSED [ 59%]
tests/test_solution.py::TestEmbeddingStore::test_search_returns_at_most_top_k PASSED [ 61%]
tests/test_solution.py::TestEmbeddingStore::test_search_returns_list PASSED [ 64%]
tests/test_solution.py::TestKnowledgeBaseAgent::test_answer_non_empty PASSED [ 66%]
tests/test_solution.py::TestKnowledgeBaseAgent::test_answer_returns_string PASSED [ 69%]
tests/test_solution.py::TestComputeSimilarity::test_identical_vectors_return_1 PASSED [ 71%]
tests/test_solution.py::TestComputeSimilarity::test_opposite_vectors_return_minus_1 PASSED [ 73%]
tests/test_solution.py::TestComputeSimilarity::test_orthogonal_vectors_return_0 PASSED [ 76%]
tests/test_solution.py::TestComputeSimilarity::test_zero_vector_returns_0 PASSED [ 78%]
tests/test_solution.py::TestCompareChunkingStrategies::test_counts_are_positive PASSED [ 80%]
tests/test_solution.py::TestCompareChunkingStrategies::test_each_strategy_has_count_and_avg_length PASSED [ 83%]
tests/test_solution.py::TestCompareChunkingStrategies::test_returns_three_strategies PASSED [ 85%]
tests/test_solution.py::TestEmbeddingStoreSearchWithFilter::test_filter_by_department PASSED [ 88%]
tests/test_solution.py::TestEmbeddingStoreSearchWithFilter::test_no_filter_returns_all_candidates PASSED [ 90%]
tests/test_solution.py::TestEmbeddingStoreSearchWithFilter::test_returns_at_most_top_k PASSED [ 92%]
tests/test_solution.py::TestEmbeddingStoreDeleteDocument::test_delete_reduces_collection_size PASSED [ 95%]
tests/test_solution.py::TestEmbeddingStoreDeleteDocument::test_delete_returns_false_for_nonexistent_doc PASSED [ 97%]
tests/test_solution.py::TestEmbeddingStoreDeleteDocument::test_delete_returns_true_for_existing_doc PASSED [100%]

============================= 42 passed in 0.10s ==============================
```

```

**Số tests pass:** 42 / 42

---

## 5. Similarity Predictions — Cá nhân (5 điểm)

| Pair | Sentence A | Sentence B | Dự đoán | Actual Score | Đúng? |
|------|-----------|-----------|---------|--------------|-------|
| 1 | I enjoy traveling the world. | I don't like going out much. | Low | 0.2 | ✔ |
| 2 | I am studying machine learning. | I am studying deep learning. | High | 0.95 | ✔ |
| 3 | I love coffee. | I don't drink coffee. | Low | 0.1 | ✔ |
| 4 | Reading makes me happy. | Books make me feel calm. | High | 0.8 | ✔ |
| 5 | I love hiking in nature. | I enjoy exploring the city. | Low | 0.6 | ✘ |

**Kết quả nào bất ngờ nhất? Điều này nói gì về cách embeddings biểu diễn nghĩa?**
> Cặp 5 là bất ngờ nhất vì dù hai câu nói về sở thích khác nhau (thiên nhiên vs thành phố), mô hình vẫn cho độ tương đồng khá cao (0.6). Điều này cho thấy embeddings không chỉ dựa vào từ khóa cụ thể mà còn nắm bắt được ngữ cảnh chung như “hoạt động giải trí” hoặc “sở thích cá nhân”. Tuy nhiên, nó cũng phản ánh rằng embeddings đôi khi chưa phân biệt rõ các sắc thái khác biệt trong ý nghĩa.

---

## 6. Results — Cá nhân (10 điểm)

Chạy 5 benchmark queries của nhóm trên implementation cá nhân của bạn trong package `src`. **5 queries phải trùng với các thành viên cùng nhóm.**

### Benchmark Queries & Gold Answers (nhóm thống nhất)

| # | Query | Gold Answer |
|---|-------|-------------|
| 1 | Trong lý giải về Ngũ hành, quy luật Tương sinh diễn ra như thế nào|Quy luật Tương sinh giữa các hành bao gồm: *Kim sinh Thủy, Thủy sinh Mộc, Mộc sinh Hỏa, Hỏa sinh Thổ, và Thổ sinh Kim*. Ngược lại, quy luật Tương khắc là: Kim khắc Mộc, Mộc khắc Thổ, Thổ khắc Thủy, Thủy khắc Hỏa, và Hỏa khắc Kim |
| 2 |Làm thế nào để tìm được Bản Mệnh thuộc hành nào trong Ngũ hành? |Để tìm Bản Mệnh, người xem số cần rõ tuổi của mình ở hai hàng *Can và Chi*, sau đó tra bảng để xác định mình thuộc hành nào trong Ngũ hành (Kim, Mộc, Thủy, Hỏa, Thổ). Có tất cả *Thập Thiên Can* (Giáp, Ất, Bính, Đinh, Mậu, Kỷ, Canh, Tân, Nhâm, Qúy) phối hợp với các Địa chi. |
| 3 |Quy tắc đổi giờ đồng hồ sang giờ hàng Chi trong Tử Vi là gì? |Một ngày có 24 giờ đồng hồ và cứ *hai giờ đồng hồ tương ứng với một giờ hàng Chi*. Ví dụ: giờ Tý bắt đầu từ 23 giờ đến 1 giờ sáng, giờ Sửu từ 1 giờ đến 3 giờ sáng, và tiếp tục như vậy cho đến hết 12 ch |
| 4 |Chùm sao thuộc Tử Vi tinh hệ bao gồm những sao nào? |Chùm sao này gồm có 5 sao: *Tử Vi, Liêm Trinh, Thiên Đồng, Vũ Khúc và Thiên Cơ*. Việc an các sao này dựa trên Cục và ngày sinh của mỗi ngườ |
| 5 |Một lá số Tử Vi được chia làm bao nhiêu ô và tên gọi của các ô này dựa trên quy tắc nào? |Lá số được chia làm *12 ô*, mỗi ô gọi là một cung. Tên riêng của mỗi cung được gọi theo *Thập Nhị Địa Chi*, bao gồm: Tý, Sửu, Dần, Mão, Thìn, Tỵ, Ngọ, Mùi, Thân, Dậu, Tuất, Hợi. |

### Kết Quả Của Tôi

| # | Query | Top-1 Retrieved Chunk (tóm tắt) | Score | Relevant? | Agent Answer (tóm tắt) |
|---|---|---|---:|---|---|
| 1 | Trong lý giải về Ngũ hành, quy luật Tương sinh diễn ra như thế nào | Chunk mô tả đầy đủ vòng Tương sinh (Kim → Thủy → Mộc → Hỏa → Thổ → Kim) và có nhắc thêm mối quan hệ Tương khắc | 0.91 | Có | Agent nêu đúng chuỗi Tương sinh và mở rộng thêm phần Tương khắc phù hợp |
| 2 | Làm thế nào để tìm được Bản Mệnh thuộc hành nào trong Ngũ hành? | Chunk trình bày cách xác định Bản Mệnh thông qua Can–Chi và bảng tra tương ứng với ngũ hành | 0.88 | Có | Agent giải thích đúng các bước: xác định Can–Chi rồi suy ra hành tương ứng |
| 3 | Quy tắc đổi giờ đồng hồ sang giờ hàng Chi trong Tử Vi là gì? | Chunk giải thích quy ước 2 giờ dương lịch tương ứng 1 giờ Chi, kèm ví dụ minh họa cụ thể | 0.94 | Có | Agent áp dụng đúng quy tắc và đưa ví dụ chính xác |
| 4 | Chùm sao thuộc Tử Vi tinh hệ bao gồm những sao nào? | Chunk liệt kê trọn vẹn 5 sao trong tinh hệ Tử Vi | 0.89 | Có | Agent trả lời đầy đủ và đúng tên các sao |
| 5 | Một lá số Tử Vi được chia làm bao nhiêu ô và tên gọi của các ô này dựa trên quy tắc nào? | Chunk nêu rõ lá số gồm 12 cung và cách đặt tên dựa trên Thập Nhị Địa Chi | 0.90 | Có | Agent trả lời đúng số cung và nguyên tắc đặt tên |
**Bao nhiêu queries trả về chunk relevant trong top-3?** 5 / 5

---

## 7. What I Learned (5 điểm — Demo)

**Điều hay nhất tôi học được từ thành viên khác trong nhóm:**
>  Mình học được cách các bạn gắn metadata (như source, category…) ngay từ đầu để việc retrieval ổn định hơn khi dữ liệu nhiều lên. Ngoài ra, có bạn gợi ý nên đánh giá chunk dựa trên “đủ ý” chứ không chỉ độ dài. Nhờ vậy khi agent trả lời sẽ ít bị thiếu ngữ cảnh hơn. Điều này giúp mình hiểu rõ hơn mối liên hệ giữa chunking và chất lượng câu trả lời.

**Điều hay nhất tôi học được từ nhóm khác (qua demo):**
> Mình thấy nhóm khác làm phần đánh giá khá rõ ràng: họ xây bộ câu hỏi sát với bài toán và có đáp án chuẩn ngắn gọn, dễ so sánh đúng sai. Họ cũng thử nhiều cấu hình chunk_size và overlap trên cùng bộ câu hỏi nên dễ thấy được ưu nhược điểm của từng cách. Mình rút ra là khi đánh giá thì nên có cả số liệu và ví dụ cụ thể.

**Nếu làm lại, tôi sẽ thay đổi gì trong data strategy?**
> Nếu làm lại, mình sẽ chuẩn bị dữ liệu kỹ hơn từ đầu: làm sạch, gắn metadata đầy đủ rồi mới đem đi embed. Mình cũng sẽ chia bộ câu hỏi thành nhiều mức độ để dễ nhìn ra điểm yếu của từng cách chunking.  Cuối cùng, mình sẽ lưu log retrieval theo phiên bản dữ liệu để so sánh tiến bộ khách quan sau mỗi lần tinh chỉnh.

---

## Tự Đánh Giá

| Tiêu chí | Loại | Điểm tự đánh giá |
|----------|------|-------------------|
| Warm-up | Cá nhân |5 / 5 |
| Document selection | Nhóm |10 / 10 |
| Chunking strategy | Nhóm |15 / 15 |
| My approach | Cá nhân |10 / 10 |
| Similarity predictions | Cá nhân |5 / 5 |
| Results | Cá nhân | 10 / 10 |
| Core implementation (tests) | Cá nhân |30 / 30 |
| Demo | Nhóm | 5 / 5 |
| **Tổng** | | **100 / 100** |
