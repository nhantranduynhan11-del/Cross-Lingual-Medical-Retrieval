# NOTES — ViBioMIR baseline v1

## Nguồn dữ liệu và model

- **Dữ liệu BTC:** `Data/query.parquet` (1.200 câu hỏi) và `Data/links_corpus.parquet` (4.394.718 URL, 97 tên miền).
- **Kho tài liệu:** do `crawl.py` thu thập. Bộ trích chữ phiên bản `EXTRACT_VERSION = 2` (3/10/2026) gồm:
  - 34 quy tắc riêng theo tên miền, phủ khoảng 96% số URL crawl được;
  - trafilatura cho các tên miền còn lại.
- **Model (dùng từ T3):** BAAI/bge-m3 trên Hugging Face.
  - Tokenizer đã dùng: snapshot `5617a9f61b028005a4858fdac845db406aefb181`.
  - Trọng số và các đầu `sparse_linear.pt`/`colbert_linear.pt` tải trên Kaggle. Chưa ghi phiên bản; xem log notebook T3.
- **Thư viện thêm:**
  - faiss-cpu 1.15, pyvi, jieba.
  - bm25s 0.3.12, chỉ để kiểm chứng.
  - tokenizers 0.22, transformers 5.15, torch 2.13 (bản CPU, cho kiểm thử).

## Phần kho không lấy được (không tìm cách vượt chặn)

| Tên miền | URL | Lý do |
|---|---:|---|
| familydoctor.com.cn (mọi tên miền con) | ~451k | HTTP 403 |
| zysjonline.com | 243k | Cloudflare đòi thử thách xác minh |
| bingli.iiyi.com, article.iiyi.com | 31k | HTTP 521, máy chủ gốc không phản hồi (3/10/2026) |
| pmc-ecm-healthblog.beta.pharmacity.io | 5,3k | HTTP 403 |
| baophutho.vn, baothanhhoa.vn | một phần bài | Nội dung bị mã hoá, giải bằng JS. Script không giải mã, các bài này ghi `empty`. |
| www.youlai.cn | 46k | Khoảng một nửa số trang treo. Chạy lại bằng `--retry-failed`. |

## Quyết định về trích chữ (crawl.py, phiên bản 2)

**Nguyên tắc chung**
- Lấy đầy đủ, đúng thứ tự phần nội dung chính của bài (tiêu đề câu hỏi, mô tả, câu trả lời, sapo, thân bài).
- Bỏ giao diện: menu, thanh bên, tin liên quan, hồ sơ bác sĩ, nút bấm, bản quyền.
- Đề thi không nói BTC trích nội dung bằng cách nào.

**Chuẩn hoá chung**
- Chuẩn hoá NFC. Một số báo Việt viết dấu dạng tổ hợp (NFD), như baodanang hay suckhoedoisong.
- Bỏ ký tự độ rộng 0. Đổi `　` thành dấu cách.
- Giải mã thực thể HTML bị mã hoá hai lần (120ask: `&#8226;`).
- Ô bảng cùng hàng cách nhau dấu cách.

**HTML hỏng**
- Trang dạng `<html></html><head>…` (baotayninh): lấy gốc nhiều chữ nhất.
- Luôn phân tích cả văn bản, không để lxml coi trang là đoạn HTML lẻ.

**Ngôn ngữ**
- So số chữ Hán với số chữ cái có dấu tiếng Việt.
- Trên 9.845 bài đã crawl, không tên miền nào còn bị gán lẫn hai ngôn ngữ.

**Theo từng tên miền**
- **cnkang** (bố cục cũ `#endText`): các đoạn trong HTML xếp ngược (`jbk11` … `jbk0`), đã sắp lại theo thứ tự đọc. Bỏ "核心提示" (tóm tắt bị cắt). *Rủi ro:* nếu bản BTC dùng để tạo `chunk_text` giữ thứ tự ngược thì so khớp nguyên văn có thể lệch. Cần xem lại ở T7.
- **ask.39.net:** bỏ hộp bách khoa bệnh tự chèn ("症状起因…", định nghĩa bệnh theo thẻ). Hộp này lặp ở mọi câu hỏi cùng thẻ bệnh. *Cần người duyệt xác nhận.* Nếu muốn giữ, sửa quy tắc rồi `--reextract` (cần `--save-raw`).
- **39.net (các kênh tin):** giữ "核心提示" khi nó mang nội dung riêng, khoảng 14% số trang. Bỏ khi trùng thân bài.
- **120ask, ask.39.net:** bỏ dòng câu hỏi lặp liền nhau (tiêu đề, mô tả, "补充说明：" giống hệt).
- **jbk.39.net:** chỉ lấy khối giới thiệu và thông tin cơ bản. Ghép nhãn với giá trị ("发病部位： 眼").
- **MediaWiki** (a-hospital, zhongyibaodian): bỏ hộp điều hướng và mục lục. Cắt các mục đuôi do mẫu trang chèn ("中药方专题", "参看"…).

## Số đo

**T0 (mẫu 10k URL shard 0, crawl.py bản cũ)**
- 9.845 bài: tiếng Trung 80%, tiếng Việt 20%.
- Độ dài trung vị 835 ký tự (tiếng Việt 3.288, tiếng Trung 694).
- Trùng nội dung trong mẫu 0,42%. Đây là chặn dưới, chưa đo được toàn kho.

**Kiểm chứng crawl.py v2 trên crawl_eval/** (995 trang, 85 tên miền)
- Tên miền không có quy tắc: 299/299 trang cho chữ giống hệt bản cũ, không hồi quy.
- Trích lại từ HTML gọn (`--save-raw`/`--reextract`) giống hệt trích từ HTML gốc: 995/995 trang.
- HTML gọn nén khoảng 11–13 KB/trang, tức khoảng 15 GB mỗi shard 1,23 triệu URL. Phần này chưa đo trên crawl thật quy mô lớn.

**Crawl lại 10k URL shard 0 bằng crawl.py v2 (4/10/2026)**
- Kết quả: 9.633 bài `ok`, 213 bài `empty`. Mọi bài đều `xv=2` và ở dạng NFC. Không tên miền nào lẫn ngôn ngữ.
- Tỷ lệ ngôn ngữ: tiếng Trung 79,6%, tiếng Việt 20,4%.
- Độ dài trung vị 597 ký tự (tiếng Việt 3.430, tiếng Trung 467). Bản cũ là 835, phần chênh chủ yếu do bỏ giao diện.
- Trùng nội dung: 0,27%.
- Có 2 bài cnkang chỉ có ảnh nhưng bị lấy toàn menu. Đã sửa sau đó bằng quy tắc `strict` (không tăng `EXTRACT_VERSION` vì chỉ khoảng 0,08% số bài cnkang). Hai bài này trong `out/` vẫn mang menu, và 1 bài nằm trong `sample/`.

**Dữ liệu đã crawl bằng bản cũ**
- 96% số URL thuộc tên miền có quy tắc mới, nên phải tải lại (`--redo-stale`).
- Shard 0 (mẫu): 9.557/9.958 bài.

**T1 (thống kê trên 9.633 bài trong out/, 4/10/2026)**
- Không nhóm nào có dòng lặp từ 30% số bài trở lên, vì quy tắc trích chữ của crawl.py đã bỏ giao diện. 120ask có "指导意见：" (32%), được giữ vì là tiêu đề mục.
- Mẫu 2.000 bài: 76 bài bị bỏ vì dưới 100 ký tự (39 bài 120ask, 10 bài ask.39.net…). Trên out/ có 449/9.633 bài dưới 100 ký tự (4,7%).
- Kết quả `--apply` trên out/ với force_drop đã duyệt: vào 9.633 bài, ra 9.184 bài (tiếng Trung 7.217, tiếng Việt 1.967). 449 bài bị bỏ vì dưới 100 ký tự, tất cả là tiếng Trung. Số ký tự giảm 0,3%. Có 25 bản sao thừa, không gộp.
- Dòng giao diện dưới ngưỡng đã đưa vào `force_drop` (đã duyệt 4/10/2026): nhathuoclongchau "Xem thêm:" 29,6%; khung bình luận laodong 16%; thanhnien "tin liên quan" 16%; suckhoedoisong "Xem thêm video…" 8,6%.

**T2 (sample/corpus_clean: 1.924 bài; token BGE-M3; max 400 / cha 480 / con 128; không chồng lấn)**

| Chiến lược | Số đoạn | Đoạn/bài | Token trung vị (p95) | `tiêu đề + đoạn` > 512 token | Ước lượng toàn kho (~3,41 triệu bài) |
|---|---:|---:|---|---:|---:|
| recursive | 4.464 | 2,32 | 333 (396) | 0% | ~7,9 triệu |
| structure (mặc định toàn kho) | 4.851 | 2,52 | 298 (394) | 0% | ~8,6 triệu |
| parent_child: đoạn con | 14.689 | 7,63 | 95 (126) | 0% | ~26 triệu |
| parent_child: đoạn cha | 4.160 | 2,16 | 347 (473) | — | ~7,4 triệu |

- Bất biến `char_start/char_end` sai 0 lần ở cả 3 chiến lược.
- Tốc độ chia đoạn trên CPU: 9.184 bài trong khoảng 5 giây, tức toàn kho khoảng 30 phút.
- Ước lượng số bài toàn kho = 3,70 triệu URL × 96,7% crawl được × 95,3% còn lại sau T1. Chưa đo.
- `n_tokens` đếm theo vị trí token trong cả bài. Mã hoá lại riêng từng đoạn có thể dư 1 token (lớn nhất 401/400, 129/128).

**T3 (số đo trên GPU: xem mục "Chạy thử đầu-cuối trên Kaggle" bên dưới)**
- Tự cài đặt BGE-M3 trên transformers, gồm dense (vector CLS chuẩn hoá) và lexical weights (`relu(sparse_linear)`, bỏ token đặc biệt, lấy max theo token id).
- Kiểm thử trên máy dùng model nhỏ trọng số ngẫu nhiên: phần sparse vector hoá cho kết quả giống hệt vòng lặp tham chiếu, và kết quả không phụ thuộc cách chia batch.
- Đối chiếu với FlagEmbedding bằng trọng số thật: ô (2) của `notebooks/e2e_test_kaggle.ipynb`, đã chạy ngày 05/10/2026, khớp.
- Đầu vào mỗi đoạn là `tiêu đề + "\n" + đoạn`, cắt ở max_length 512.
- Sparse lưu CSR float16 tự định dạng, vì scipy.sparse không hỗ trợ float16.
- Mỗi khối ghi dấu vân tay của bản chunks (`chunks_fp` = SHA-1 của tên file part và số đoạn mỗi part) vào `_DONE.json` (thêm ngày 05/10/2026). `mir.encode` và `mir.index` từ chối trộn các khối có dấu vân tay khác nhau. Lý do: số thứ tự khối chỉ có nghĩa khi cả 3 người mã hoá cùng một version `mir-data`.

**T4–T8 (code xong ngày 4/10/2026; kiểm thử đầu-cuối trên sample bằng model nhỏ trọng số ngẫu nhiên, CHƯA chạy với BGE-M3 thật)**

T4 — Dense
- FAISS: dùng `IndexFlatIP` khi dưới 200k đoạn, ngược lại dùng `IVF{nlist},SQ8` (tích vô hướng).
- `nlist`/`nprobe` chọn bằng `mir.index dense-tune` (Recall@100 so với Flat trên mẫu khoảng 100k đoạn, truy vấn là câu hỏi thật). Giá trị trong cấu hình (16384/64) là giá trị tạm, **chưa đo**.

T4 — LSR
- Chỉ mục đảo CSC theo shard, điểm là tích vô hướng.
- T5 có thể đọc thẳng `emb/*/sparse.npz` (chuyển CSC trong bộ nhớ từng shard) để khỏi lưu bản sao khoảng 8 GB, vì `/kaggle/working` chỉ khoảng 20 GB.

T4 — BM25: **khác đặc tả** — tự cài đặt bằng scipy thay vì dùng thẳng bm25s.
- Công thức là Lucene giống bm25s: `idf = ln(1 + (N − df + 0,5)/(df + 0,5))`, `tf/(tf + k1·(1 − b + b·dl/avgdl))`, k1 = 1,5, b = 0,75.
- Lý do: bm25s dựng một chỉ mục trong bộ nhớ, với khoảng 8,6 triệu đoạn sẽ vượt khoảng 30 GB RAM. Còn nếu chia shard bằng bm25s thì mỗi shard có IDF riêng, điểm giữa các shard không so được.
- Ở đây thống kê df/avgdl lấy trên toàn kho, rồi chia shard. `tests/test_t4.py` kiểm chứng điểm trùng bm25s (sai số < 1%, do lưu float16) khi kho chia 2 shard.
- Tách từ: tiếng Việt bằng pyvi, tiếng Trung bằng jieba, viết thường, bỏ token không có chữ/số. Câu hỏi tách bằng pyvi.
- Tốc độ trên máy cá nhân, 1 lõi: pyvi khoảng 340k ký tự/s, jieba khoảng 360k ký tự/s.

T5 — truy hồi và gộp
- Mỗi nhánh cho run độ sâu 1000.
- RRF k = 60 cho đủ 7 cấu hình A–G.
- Gộp có trọng số dùng điểm chuẩn hoá min-max theo câu hỏi, 3 bộ trọng số tạm.
- Câu hỏi được mã hoá **không** thêm tiền tố, còn đoạn được mã hoá kèm tiêu đề.

T6 — xếp hạng lại
- ColBERT của BGE-M3 với `colbert_linear` cho các token từ vị trí 1, chuẩn hoá L2. MaxSim chia cho số token câu hỏi, giống FlagEmbedding `colbert_score`.
- Chỉ mã hoá câu hỏi và top-100 ứng viên; mỗi ứng viên là `tiêu đề + "\n" + văn bản`.
- Chia việc theo câu hỏi (`--shard`), mỗi phần ghi một file và tự gộp khi đủ phần.
- Với parent_child (`rerank.unit: parent`), đoạn con được gom về đoạn cha theo rank tốt nhất, rồi xếp hạng lại và nộp bằng đoạn cha.

T7 — đánh giá
- Chỉ số: Recall@10/50/100/500, nDCG@10 và MRR@10 ở cấp tài liệu; Recall@K ở cấp đoạn; P/R/F2 macro ở cả hai cấp theo đúng tập bài nộp; bảng tách theo ngôn ngữ của tài liệu đúng.
- Khớp `chunk_text` chọn trong cấu hình: `exact`, `contains` hoặc `overlap` (mặc định, ngưỡng 0,8).
- Câu hỏi không có `chunk_text` đúng bị loại khỏi chỉ số cấp đoạn.
- Chỉ kiểm thử bằng **nhãn giả**. Chưa có số liệu thật nào.

T8 — bài nộp
- Kiểm tra trước khi ghi: đủ id câu hỏi, `doc_id` là chuỗi và có trong `links_corpus`, `chunk_text` là chuỗi con nguyên văn của bài đã làm sạch, mức rỗng vẫn có mảng rỗng.
- ZIP chứa đúng 1 file ở gốc.
- `k_doc` = `k_chunk` = 10 là giá trị tạm, **chưa kiểm chứng**.

Kiểm thử
- 58 kiểm thử chạy trên CPU trong khoảng 2 phút.
- `tests/test_pipeline.py` chạy T3→T8 trên sample thật (4.851 đoạn, 50 câu hỏi) bằng model nhỏ. Các bước đều chạy và bài nộp hợp lệ với `links_corpus` thật, nhưng **không có ý nghĩa về chất lượng**.
- Các lệnh dòng lệnh T4–T8 cũng đã chạy thử trên thư mục tạm, kể cả xếp hạng lại 2 shard rồi gộp.

**Chạy thử đầu-cuối trên Kaggle T4 ×2 (05/10/2026, `notebooks/e2e_test_kaggle.ipynb` bản e2e-v3)**
- Dữ liệu: tập mẫu 4.851 đoạn và 50 câu hỏi.
- T3→T8 chạy hết, không lỗi.
- Thời gian từng bước (phút, đã gồm tải model):

  | Bước | Phút |
  |---|---:|
  | Đo tốc độ | 0,7 |
  | T3 mã hoá | 0,8 |
  | BM25 | 0,2 |
  | Mã hoá câu hỏi | 0,3 |
  | T6 xếp hạng lại (5.000 ứng viên) | 1,5 |
  | Tạo bài nộp | 0,1 + 0,1 |
  | Các bước còn lại | ≈ 0 |

- Tốc độ mã hoá đo ở ô 3, trên 2 GPU, mỗi GPU 2.000 đoạn:

  | batch_tokens | Tốc độ | Bộ nhớ GPU tối đa |
  |---:|---:|---:|
  | 16.384 | 185 đoạn/s | 1,55 GB |
  | 65.536 (2 lần đo) | 161 / 157 đoạn/s | 2,78 GB |

  Giữ 16.384. Toàn kho khoảng 8,5 triệu đoạn ≈ 13–15 giờ trên một máy T4 ×2, tức ≈ 4,3–5 giờ mỗi người khi chia 3; dung lượng ≈ 22,8 GB (sparse khoảng 106 phần tử khác 0 mỗi đoạn).
- Đối chiếu với FlagEmbedding (ô 2), cùng BAAI/bge-m3 fp16, 32 đoạn — **khớp**:
  - dense: cos nhỏ nhất 0,99972;
  - sparse: cùng tập token 32/32, chênh trọng số lớn nhất 0,0000.
- Chạy lại với đủ 1.200 câu hỏi (`FULL_QUERIES = True`), vẫn trên 4.851 đoạn mẫu:

  | Bước | Phút |
  |---|---:|
  | T6 xếp hạng lại (120 nghìn ứng viên, 2 GPU) | 27,7 |
  | T5 truy hồi | 0,2 |
  | T5 gộp | 0,7 |
  | T3 mã hoá | 0,7 |

  File nộp đủ 1.200 câu và qua mọi bước kiểm tra. T6 phụ thuộc số câu hỏi × top-N chứ không phụ thuộc cỡ kho, nên trên toàn kho cũng mất khoảng 28 phút.

## Tổ chức code và chạy trên Kaggle (05/10/2026)

- Đường dẫn trong `configs/baseline.yaml` là đường dẫn tương đối, tính từ thư mục gốc repo. Trên Kaggle dùng biến môi trường `MIR_DATA_DIR`, `MIR_WORK_DIR`. Mọi module lấy thư mục chuẩn qua `cfg.path(...)` (`mir/config.py`).
- Notebook: ô (1) giống nhau ở mọi notebook (`mir/kaggle_nb.py`).
  - Luôn chép lại `mir-code` và in `VERSION.txt` do `tools/pack_kaggle.py` ghi.
  - Tự tìm dataset trong `/kaggle/input`.
  - Viết cấu hình làm việc `/kaggle/working/config.yaml` từ `OVERRIDES`; khoá viết sai thì báo lỗi.
  - Không tạo symlink trong `/kaggle/working`: đường dẫn tới dữ liệu đầu vào được ghi thẳng vào cấu hình.
  - T5 chạy nhánh BM25 bằng `--index` trỏ thẳng vào dataset `mir-bm25`.
- Các notebook đã chạy thử ngoài Kaggle trong thư mục giả lập `/kaggle` (`KAGGLE_ROOT`), lệnh GPU được bỏ qua:
  - T4 BM25 chạy hết trên kho 10k (22.849 đoạn): 32 giây.
  - T6 tạo được bài nộp hợp lệ từ run giả.
  - T3 chia phần và chép khối khi chạy lại đúng.
  - T5 truy hồi BM25 1.200 câu qua `--index`: 5 giây trên kho 10k.

  Chưa chạy lại trên Kaggle thật sau khi đổi.
- `mir.chunk` chia lại part có `chunks/<strategy>/part-X` cũ hơn `corpus_clean/part-X`, và cảnh báo part thừa. Lý do: trên máy người duyệt, `chunks/structure/part-00000` là của kho 10k; nếu chỉ so tên file, part này sẽ bị bỏ qua mà không báo khi chia lại kho đầy đủ.
- `results/clean_lines.json` (trước đây ở gốc repo) là bản đã duyệt trên 10k URL. Với kho đầy đủ phải chạy lại `--stats --force` và duyệt lại.

## Câu hỏi đang mở (cần người duyệt quyết định)

1. **Tập đánh giá (T7).** Đã giao cho một thành viên. Người đó tự gán nhãn trên 100 câu hỏi, kho đầy đủ, pool top-10 của 5 run, và tự viết công cụ. Xem `docs/t7_danh_gia.md`. Còn mở các câu ở mục 9 của tài liệu đó: ai đọc tài liệu tiếng Trung, quy mô sau khi gán thử, có dùng LLM mở chấm sơ bộ không. Chưa có nhãn thật; không tự sinh nhãn.
2. **Hộp bách khoa bệnh ở ask.39.net** đang bị bỏ khi trích (xem phần crawl phía trên).
3. **`k_doc`, `k_chunk`** cần chọn theo F2 khi đã có nhãn hoặc điểm từ leaderboard.
4. **`nlist`/`nprobe` cho toàn kho:** chạy ô (3) của notebook T5 trên kho thật để chọn.

## Đề xuất sau baseline

- Chọn `k_doc`/`k_chunk` theo từng câu hỏi bằng ngưỡng điểm, thay vì số cố định. Bài nộp chấm theo tập nên số lượng ảnh hưởng trực tiếp F2. Đặc tả hiện loại "mô hình tự chọn số lượng", nên để sau baseline.
- Thử chồng lấn (`chunk.overlap_tokens`) và parent_child trên sample, sau khi có nhãn.
