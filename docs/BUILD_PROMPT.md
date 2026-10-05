# Xây baseline v1 — Truy hồi tài liệu y khoa đa ngôn ngữ (ViBioMIR, AI GURU Stage 3)

## 1. Vai trò và cách làm việc

Bạn xây baseline cho đội thi 3 người. Tôi là người duyệt. Hãy làm theo đúng thứ tự ở mục 7 "Danh sách task".

Quy tắc bắt buộc:

- Mỗi lượt chỉ làm MỘT task. Xong task thì chạy kiểm thử của task đó trên dữ liệu mẫu, viết báo cáo ngắn rồi DỪNG, chờ tôi xác nhận mới làm task kế tiếp. Không gộp task, không làm trước.
- Báo cáo sau mỗi task gồm: đã tạo/sửa file nào, lệnh để tôi tự chạy lại, số đo thật thu được, và những điểm bạn chưa chắc. Con số nào chưa đo thì ghi rõ "chưa đo", không ước lượng thành sự thật.
- Không đổi kiến trúc đã chốt ở mục 5. Nếu thấy phương án tốt hơn, ghi vào `NOTES.md` dưới mục "Đề xuất sau baseline" và tiếp tục làm theo baseline.
- Luật thi: chỉ dùng model mở, tối đa 14B tham số, phát hành trước 1/6/2026. Không gọi API của model đóng (GPT, Gemini...). Mọi nguồn dữ liệu và model dùng thêm phải ghi vào `NOTES.md` (tên, phiên bản, đường dẫn) để viết bài mô tả phương pháp.
- Khi gặp quyết định mà hai hướng đều hợp lý và khó đảo ngược (ví dụ đổi định dạng file trung gian), hỏi tôi thay vì tự chọn.

## 2. Bài toán

Đầu vào là câu hỏi y khoa tiếng Việt. Hệ thống trả về, cho mỗi câu hỏi, danh sách tài liệu liên quan và danh sách đoạn nội dung liên quan.

Chấm điểm bằng F2 macro ở hai cấp, tài liệu và đoạn: với mỗi câu hỏi tính Precision và Recall trên TẬP kết quả trả về, F2 = 5·P·R / (4·P + R), rồi lấy trung bình cộng trên mọi câu hỏi. Kết quả nộp là tập hợp, không phải danh sách xếp hạng, nên số lượng trả về cho mỗi câu hỏi ảnh hưởng trực tiếp tới điểm.

Định dạng nộp: một file `.json` chứa một mảng, mỗi phần tử có `id` (số nguyên), `relevant_docs` (mảng chuỗi doc_id) và `relevant_chunks` (mảng các đối tượng `{"doc_id": ..., "chunk_text": ...}`). Phải có đủ mọi câu hỏi; mức nào không có kết quả thì để mảng rỗng. `chunk_text` phải là nội dung trích nguyên văn từ tài liệu gốc, không viết lại. File json được nén thành một file ZIP chứa đúng một file, không nằm trong thư mục con.

## 3. Dữ liệu có sẵn (thư mục gốc: `E:\Medical Retrieval\`)

- `Data/query.parquet`: 1.200 câu hỏi, cột `id` (int), `query` (str). Độ dài trung vị 18 từ, dài nhất 283 từ.
- `Data/links_corpus.parquet`: 4.394.718 dòng, cột `id` (int), `url` (str). `id` này chính là `doc_id` khi nộp bài (đổi sang chuỗi).
- `out/part-*.jsonl.gz`: kết quả crawl của `crawl.py`. Mỗi dòng là `{"id", "url", "host", "lang", "title", "text"}`, với `lang` là `vi` hoặc `zh`. Crawl có thể CHƯA xong, nên mọi bước phải chạy được với số file hiện có và chạy tiếp được khi có thêm file.
- `out/progress.tsv`: trạng thái từng URL. `out/html_samples/`: vài HTML gốc mỗi tên miền.

Những điều đã biết về kho (đo trên mẫu 300 bài, cần đo lại trên dữ liệu thật ở task T0):

- Khoảng 83% URL là tiếng Trung, 17% tiếng Việt; không có tiếng Anh. 97 tên miền, 10 tên miền đầu chiếm 81%.
- Trung bình khoảng 1.600 ký tự mỗi bài.
- `120ask.com` (khoảng 918k URL) có rất nhiều chữ giao diện lặp lại ở mọi trang; các trang khác tương đối sạch.
- Hai tên miền không crawl được và hiện vắng mặt: `www.familydoctor.com.cn`, `zysjonline.com` (khoảng 690k URL).

## 4. Môi trường

- Bạn (Claude Code) chạy trên máy Windows của tôi. Coi như máy này KHÔNG có GPU: ở đây chỉ viết code và kiểm thử trên mẫu nhỏ.
- Việc nặng (mã hoá BGE-M3, xếp hạng lại) chạy trên Kaggle hoặc Colab bằng GPU. Vì vậy code phải chạy được trên cả Windows lẫn Linux, không viết cứng đường dẫn: mọi đường dẫn lấy từ file cấu hình hoặc biến môi trường `MIR_DATA_DIR`, `MIR_WORK_DIR`.
- Ràng buộc của Kaggle cần thiết kế quanh nó: mỗi phiên có giới hạn thời gian (khoảng 12 giờ), GPU có hạn mức theo tuần, thư mục ghi `/kaggle/working` có giới hạn dung lượng (khoảng 20 GB), RAM khoảng 30 GB, dữ liệu vào ở `/kaggle/input` chỉ đọc. Các con số này là ước chừng: hãy kiểm tra thực tế khi chạy. Hệ quả: mọi job dài phải chia shard (`--shard i/N`), ghi kết quả theo từng shard, và chạy lại thì bỏ qua shard đã xong.
- Ba thành viên sẽ chạy song song trên ba tài khoản, mỗi người một nhóm shard.

## 5. Kiến trúc đã chốt (không thay đổi)

Giai đoạn 1 — tìm rộng, tối đa Recall. Đơn vị truy hồi là đoạn (chunk). Ba nhánh độc lập:

- Dense: vector dense của `BAAI/bge-m3`, tìm bằng FAISS.
- LSR: trọng số từ vựng (lexical weights) của `BAAI/bge-m3`, tìm bằng chỉ mục đảo.
- BM25: chỉ mục riêng, tách từ tiếng Việt bằng `pyvi`, tiếng Trung bằng `jieba`.

Gộp bằng RRF với k = 60. Phải chạy được cả 7 cấu hình: A Dense, B LSR, C BM25, D Dense+LSR, E Dense+BM25, F LSR+BM25, G Dense+LSR+BM25. Gộp có trọng số (α·Dense + β·LSR + γ·BM25) là tuỳ chọn thêm, làm sau RRF.

Giai đoạn 2 — chọn kỹ, tối đa Precision. Xếp hạng lại top-N ứng viên (mặc định N = 100) bằng multi-vector (ColBERT) của `BAAI/bge-m3`. Vector multi-vector KHÔNG lưu cho cả kho; chỉ mã hoá câu hỏi và N ứng viên tại thời điểm xếp hạng lại.

Lưu trữ: toàn bộ bằng file cục bộ (FAISS, numpy, scipy, parquet). Không dùng Milvus, Qdrant, Elasticsearch hay dịch vụ nào cần cài riêng.

Không làm trong bản này: tiếng Anh/PubMed, Query Expansion, fine-tuning, Clinical Core, reranker khác, mô hình tự chọn số lượng kết quả.

## 6. Cấu trúc dự án và định dạng file trung gian

Mã nguồn đặt trong gói `mir/`, mỗi bước một module có giao diện dòng lệnh (`python -m mir.<bước> --config configs/baseline.yaml ...`). Thêm `configs/`, `notebooks/` (notebook Kaggle chỉ gọi lại các lệnh trên), `tests/`, `results/`, `README.md`, `NOTES.md`.

Định dạng (giữ ổn định để ba người làm song song):

- Kho sạch `corpus_clean/part-*.parquet`: `doc_id` (str), `host`, `lang`, `title`, `text`.
- Đoạn `chunks/part-*.parquet`: `chunk_id` (= `{doc_id}_{thứ tự}`), `doc_id`, `lang`, `char_start`, `char_end`, `text`. Bất biến: `text` đúng bằng `corpus_clean.text[char_start:char_end]`.
- Vector `emb/shard-XXXX/`: `dense.npy` (float16, đã chuẩn hoá), `sparse.npz` (CSR, giá trị float16), `chunk_ids.parquet`.
- Kết quả truy hồi (run file) `runs/<tên>.parquet`: `qid`, `chunk_id`, `doc_id`, `rank`, `score`.
- Nhãn đánh giá (qrels) `qrels.parquet`: `qid`, `doc_id`, `chunk_text` (có thể rỗng).

## 7. Danh sách task

T0 — Khảo sát và dựng khung. Đo trên dữ liệu thật trong `out/`: số bài, phân bố theo tên miền và ngôn ngữ, phân bố độ dài, tỷ lệ bài trùng nội dung. Tạo cấu trúc dự án, `requirements.txt`, `configs/baseline.yaml`. Tạo tập mẫu cố định `sample/` (khoảng 2.000 bài lấy phân tầng theo tên miền, 50 câu hỏi) dùng cho mọi kiểm thử về sau.

T1 — Làm sạch. Với mỗi tên miền, tìm các dòng lặp lại (dòng xuất hiện ở từ 30% số bài trở lên, thống kê trên ít nhất 2.000 bài của tên miền đó khi có đủ) và loại chúng; bỏ bài còn dưới 100 ký tự sau khi làm sạch. Không gộp bài trùng nội dung (đáp án có thể trỏ tới bất kỳ `id` nào), chỉ báo cáo tỷ lệ. In ra danh sách dòng bị loại của 6 tên miền lớn nhất để tôi duyệt trước khi chạy toàn bộ.

T2 — Chia đoạn. Chia theo ranh giới dòng/câu, gộp các dòng liên tiếp tới ngưỡng độ dài (mặc định 500 ký tự cho tiếng Trung, 1.200 ký tự cho tiếng Việt; là tham số trong cấu hình), không chồng lấn ở bản đầu. Kiểm tra bất biến `char_start/char_end`. Báo cáo: tổng số đoạn trên mẫu, ước lượng cho toàn kho, phân bố độ dài tính bằng token của BGE-M3 và tỷ lệ đoạn vượt 512 token.

T3 — Mã hoá BGE-M3. Script `mir.encode` có `--shard i/N`, chạy tiếp được, xuất dense và sparse theo định dạng mục 6. Văn bản đưa vào model là tiêu đề + xuống dòng + đoạn; `chunk_text` khi nộp thì không kèm tiêu đề. Dùng fp16, `max_length` 512. Viết notebook Kaggle mẫu. Trước khi chạy toàn bộ: đo tốc độ trên khoảng 2.000 đoạn bằng GPU thật, rồi báo cho tôi thời gian và dung lượng ước tính cho toàn kho. DỪNG ở đây để tôi quyết định; nếu vượt hạn mức, đề xuất các phương án (giảm độ dài, tăng số shard, chạy tập con trước) kèm đánh đổi.

T4 — Chỉ mục. Dense: `IndexFlatIP` cho tập mẫu; cho toàn kho dùng IVF kết hợp lượng tử hoá vô hướng (SQ8), chọn `nlist`/`nprobe` bằng cách đo Recall@100 so với Flat trên một mẫu khoảng 100k đoạn và báo con số. LSR: chỉ mục đảo dựng từ `sparse.npz` (scipy CSC), điểm là tích vô hướng. BM25: dùng thư viện `bm25s`, tách từ theo ngôn ngữ của đoạn, câu hỏi tách bằng `pyvi`. Mọi chỉ mục phải nạp được trong khoảng 30 GB RAM; nếu không, chia theo shard và gộp kết quả.

T5 — Truy hồi giai đoạn 1 và gộp. Mỗi nhánh xuất run file độ sâu 1.000. `mir.fuse` tạo đủ 7 cấu hình A–G bằng RRF (k = 60), và tuỳ chọn gộp có trọng số (chuẩn hoá điểm min-max theo từng câu hỏi).

T6 — Xếp hạng lại giai đoạn 2. `mir.rerank` nhận một run file, lấy top-N, mã hoá câu hỏi và các đoạn ứng viên bằng multi-vector của BGE-M3, tính điểm tương tác muộn (MaxSim), xuất run file mới. Có `--shard` theo câu hỏi và chạy tiếp được.

T7 — Đánh giá. `mir.evaluate` nhận run file và qrels, tính Recall@10/50/100/500, nDCG@10, MRR@10, và Precision/Recall/F2 macro ở cả hai cấp theo đúng công thức mục 2. Kèm bảng tách theo ngôn ngữ của tài liệu đúng (tiếng Việt, tiếng Trung) để thấy lỗi nằm ở đâu. Vì BTC chưa công bố cách so khớp `chunk_text`, cho phép chọn quy tắc trong cấu hình: khớp chính xác sau khi chuẩn hoá khoảng trắng, chứa nhau, hoặc tỷ lệ trùng ký tự từ một ngưỡng. Xuất ba bảng vào `results/`: Thí nghiệm A (7 cấu hình × Recall@K), B (các cách gộp × Recall@100), C (có và không xếp hạng lại × nDCG@10, MRR@10). Kiểm thử bằng qrels giả. Hiện CHƯA có nhãn thật: DỪNG và hỏi tôi cách tạo tập đánh giá (tự gán nhãn, dùng LLM mở chấm, hay chỉ dựa vào bảng xếp hạng của BTC); không tự sinh nhãn rồi báo cáo như số liệu thật.

T8 — Tạo bài nộp. `mir.submit` nhận run file cuối: tài liệu lấy top `K_doc` (điểm tài liệu = thứ hạng của đoạn tốt nhất), đoạn lấy top `K_chunk`. Hai giá trị K nằm trong cấu hình; mặc định 10 và 10 chỉ là giá trị tạm, chưa kiểm chứng. Kiểm tra trước khi ghi: đủ 1.200 `id` kiểu số nguyên, `doc_id` là chuỗi và tồn tại trong `links_corpus`, mỗi `chunk_text` là chuỗi con nguyên văn của bài đã làm sạch, mức rỗng vẫn có mảng rỗng. Nén ZIP chứa đúng một file ở gốc.

T9 — Chạy toàn bộ và bàn giao. Viết `README.md` hướng dẫn chạy từng bước trên Kaggle cho ba người (ai chạy shard nào, tải dữ liệu lên và lấy kết quả về thế nào). Cập nhật `NOTES.md`: nguồn dữ liệu, model và phiên bản, mọi tham số, phần kho bị thiếu, các số đo đã có, và mục "Đề xuất sau baseline".

## 8. Tiêu chí chung cho mọi task

- Có kiểm thử chạy được trên `sample/` bằng CPU trong vài phút. Với kiểm thử cần model, cho phép dùng `BAAI/bge-m3` trên vài chục đoạn; không thay bằng model khác trong code chính.
- Mọi lệnh đều chạy lại an toàn: không ghi đè kết quả đã xong, không làm lại shard đã có.
- Cố định hạt giống ngẫu nhiên, ghi lại cấu hình đã dùng bên cạnh mỗi kết quả.
- Không cài thư viện ngoài `requirements.txt` mà không báo.

Bắt đầu bằng T0.
