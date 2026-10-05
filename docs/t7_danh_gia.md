# T7 — Tập nhãn và đánh giá: giao việc

Người làm: **(điền tên)** · Người duyệt: **(điền tên)** · Bắt đầu: ngay, không cần chờ crawl xong.

## 0. Tóm tắt

BTC không công bố nhãn. Dataset trên Hugging Face chỉ có `query.parquet` và `links_corpus.parquet`. Muốn biết cấu hình
nào tốt hơn, hay mỗi câu nên trả về bao nhiêu tài liệu và bao nhiêu đoạn (`k_doc`, `k_chunk`), mà không tốn lượt nộp
leaderboard (10 lần/ngày), nhóm cần một **tập nhãn nội bộ**.

Việc gồm bốn phần:
1. Chọn **100 câu hỏi** cố định trong 1.200 câu.
2. Với mỗi câu, chấm mức liên quan của khoảng 20–40 tài liệu ứng viên do hệ thống trả về (gọi là "pool"), và trích
   đoạn nguyên văn chứa câu trả lời.
3. Viết công cụ hỗ trợ (đặc tả ở mục 6).
4. Chạy `mir.evaluate` để ra bảng so sánh và chọn `k_doc`, `k_chunk`.

Kết quả bàn giao liệt kê ở mục 8.

## 1. Hiểu đúng: gán nhãn cái gì

```
1.200 câu hỏi ──chọn 100 (cố định)──► mỗi câu: hợp top-10 tài liệu của 5 run ──► người chấm 0/1/2 + trích đoạn ──► qrels
                                      (Dense, LSR, BM25, G, G sau xếp hạng lại)
                                      ≈ 20–40 tài liệu/câu (chưa đo)
```

- **Chỉ gán nhãn một tập con câu hỏi, không gán nhãn kho.** Khi đo vẫn tìm trên **toàn kho**, tức chính chỉ mục dùng để
  nộp bài. Nếu đo trên kho nhỏ, số tài liệu nhiễu ít đi và điểm cao giả.
- **Không đọc cả kho.** Mỗi câu chỉ chấm pool: hợp các tài liệu đứng top-10 của vài run. Tài liệu ngoài pool coi như
  không liên quan. Đây là cách làm chuẩn của TREC ("pooling").
- **Nhãn không mất giá trị khi kho lớn lên.** Một tài liệu đã được chấm là liên quan hay không với một câu hỏi thì vẫn
  giữ nguyên khi kho có thêm bài. Gán nhãn trên phần kho đã crawl vẫn dùng được về sau; khi có run trên toàn kho,
  chỉ cần chấm thêm các tài liệu mới lọt vào pool.
- **Không dùng kho mẫu `sample/` để gán nhãn.** Đó là 2.000 bài chọn ngẫu nhiên, gần như không liên quan tới câu hỏi.

## 2. Vì sao thử nhiều lần vẫn rẻ

Phần nặng chỉ chạy **một lần** cho toàn kho: crawl, T3 (khoảng 13–15 giờ GPU), dựng chỉ mục, truy hồi 1.200 câu. Sau
đó mỗi lần thử chỉ dùng lại kết quả có sẵn:

| Thử gì | Chạy ở đâu | Thời gian | Ghi chú |
|---|---|---|---|
| Đổi cách gộp (RRF k, trọng số, cấu hình A–G) | CPU | vài phút (`mir.fuse` 0,7 phút cho 1.200 câu, đo trên kho mẫu) | không cần GPU |
| Đổi `k_doc`, `k_chunk`, ngưỡng | CPU | vài giây | chỉ chạy `mir.evaluate` |
| Xếp hạng lại riêng 100 câu có nhãn | GPU T4 ×2 | khoảng 2–3 phút (ngoại suy từ 27,7 phút cho 1.200 câu) | lọc run còn 100 câu, ghi ra tên khác (vd `runs/structure_G_eval.parquet`), rồi chạy `mir.rerank` trên run đó |
| Đổi cách chia đoạn hoặc cách mã hoá | GPU | khoảng 13–15 giờ nếu làm trên toàn kho | thử trước trên kho thu nhỏ (mục 9) |

## 3. Quy trình

### Giai đoạn 0 — chuẩn bị (làm ngay)

1. Đọc `docs/de_bai.md` (cách chấm), phần T7 trong `docs/NOTES.md`, và `mir/evaluate.py`.
2. Viết công cụ ở mục 6, kèm test.
3. Chọn 100 câu hỏi (lệnh `sample-queries`, mục 6). Đưa file vào git ngay. **Không đổi danh sách này sau khi đã bắt
   đầu gán nhãn.**
4. Hoàn thiện hướng dẫn gán nhãn từ bản nháp ở mục 4, lưu thành `docs/huong_dan_gan_nhan.md`.

### Giai đoạn 1 — gán thử 10 câu

Nguồn run cho đợt thử, chọn một trong hai:
- **(a) Run thật trên phần kho đã crawl.**
  1. Người làm T0–T2 chạy T1, T2 trên phần đã crawl rồi `py tools/pack_kaggle.py data`.
  2. Chạy `notebooks/e2e_test_kaggle.ipynb` với `CORPUS = 'full'`, `FULL_QUERIES = True`.
  3. Lấy thư mục `runs/` từ output.

  Run này có đủ 3 nhánh, kể cả tài liệu tiếng Trung.
- **(b) Chỉ BM25 trên máy cá nhân.** Chạy `py -m mir.index bm25-tokenize`, `bm25-build`, rồi
  `py -m mir.retrieve run --branch bm25`. Không cần GPU, nhưng gần như chỉ ra tài liệu tiếng Việt, vì câu hỏi tiếng
  Việt không trùng từ với bài tiếng Trung.

Các bước gán thử:
1. Gán 10 câu.
2. Đo thời gian chấm mỗi cặp (câu hỏi, tài liệu).
3. Hai người cùng gán **cùng 10 câu**, độc lập với nhau, rồi tính độ đồng thuận Cohen's κ trên nhãn nhị phân
   (mức ≥ 1 là liên quan).
4. Sửa hướng dẫn ở những chỗ hai người chấm khác nhau.

Điều kiện qua giai đoạn: κ ≥ 0,6. Báo người duyệt thời gian trung bình mỗi cặp. Con số này quyết định quy mô ở
giai đoạn 2, xem mục 9.

### Giai đoạn 2 — gán chính thức 100 câu (khi có dataset `mir-runs` của toàn kho)

1. Tạo pool từ 5 run: `structure_A` (Dense), `structure_B` (LSR), `structure_C` (BM25), `structure_G`,
   `structure_G_rr`. Lấy top-10 tài liệu mỗi run; bỏ các cặp đã chấm ở giai đoạn 1.
2. Gán theo từng đợt khoảng 10 câu, lưu file riêng cho mỗi người và mỗi đợt.
3. Gộp các đợt thành `eval/qrels.csv`.

### Giai đoạn 3 — đo và báo cáo

1. `py -m mir.evaluate --tables --qrels …` → `results/exp_A.md` (7 cấu hình × Recall@K), `exp_B.md` (các cách gộp
   × Recall@100), `exp_C.md` (có / không xếp hạng lại × nDCG@10, MRR@10, F2).
2. Vẽ đường **F2 theo K**: `k_doc`, `k_chunk` ∈ {1, 2, 3, 5, 7, 10, 15, 20}, trên run cuối. Đề xuất cặp K tốt nhất.
3. Xem bảng tách theo ngôn ngữ của tài liệu đúng (vi / zh) để biết lỗi nằm ở đâu.
4. Đối chiếu với leaderboard: nộp 1–2 bài với K đã chọn. Nếu thứ tự các cấu hình trên nhãn của mình ngược với
   leaderboard thì tin leaderboard, rồi xem lại hướng dẫn gán nhãn.

### Giai đoạn 4 — duy trì

Mỗi khi có cấu hình mới, đo tỷ lệ tài liệu đã chấm trong top-10 của nó (`judged@10`). Nếu dưới 0,9 thì chấm thêm các
cặp còn thiếu trước khi so sánh, vì cấu hình mới có thể bị thiệt chỉ vì tài liệu nó tìm được chưa ai chấm.

## 4. Hướng dẫn gán nhãn (bản nháp, hoàn thiện ở giai đoạn 1)

**Mức liên quan** của một tài liệu với một câu hỏi:

| Mức | Nghĩa |
|---|---|
| **2** | Trả lời trực tiếp câu hỏi, toàn bộ hoặc phần chính. |
| **1** | Có thông tin hữu ích cho câu hỏi nhưng chưa trả lời trọn, ví dụ chỉ một ý của câu hỏi nhiều ý, hoặc đúng bệnh nhưng khác khía cạnh được hỏi. |
| **0** | Không liên quan, hoặc chỉ trùng từ khoá. |

Quy tắc:
- Chấm theo nội dung, không quan tâm ngôn ngữ (tiếng Việt hay tiếng Trung) hay nguồn.
- Câu hỏi nhiều ý: trả lời được ít nhất một ý chính thì đạt mức ≥ 1.
- Bài trùng nội dung với một bài khác (khác id): chấm như nhau. BTC có thể coi bất kỳ id nào là đáp án.
- **Chấm mù:** bảng chấm không cho biết tài liệu đến từ run nào và đã được xáo thứ tự. Không xem thứ hạng.
- **Trích đoạn (`chunk_text`)** cho mọi tài liệu mức ≥ 1:
  - Chép **nguyên văn** đoạn ngắn nhất chứa câu trả lời, thường 1–3 câu, khoảng 300 ký tự trở xuống.
  - Phải chép từ văn bản đã làm sạch (`corpus_clean`) mà công cụ xuất ra, không chép từ trang web gốc.
  - Một tài liệu có thể có nhiều đoạn.
  - Lý do: `mir.evaluate` (`eval.match: overlap`) tính một đoạn hệ thống trả về là đúng khi nó phủ ít nhất 80% ký tự
    của đoạn nhãn. Đoạn nhãn càng ngắn và đúng trọng tâm thì phép đo càng có nghĩa.
- Không chắc chắn: chấm mức thấp hơn và ghi chú vào cột `note`.

## 5. Định dạng file (đưa vào git; `.gitignore` đang bỏ qua `*.parquet`, nên lưu nhãn bằng CSV UTF-8)

| File | Cột | Ghi chú |
|---|---|---|
| `eval/queries_eval.csv` | `qid, query, stratum` | 100 câu, cố định |
| `eval/judgments/<người>_<đợt>.csv` | `qid, doc_id, rel, chunk_text, annotator, seconds, note` | nhãn gốc từng người, từng đợt. Tài liệu có nhiều đoạn thì ghi nhiều dòng. |
| `eval/qrels.csv` | `qid, doc_id, rel, chunk_text` | bản gộp. Gồm cả các cặp mức 0, để biết cặp nào đã chấm. |
| `qrels.parquet` (sinh ra, không đưa vào git) | `qid, doc_id, chunk_text` | đầu vào hiện tại của `mir.evaluate`: chỉ các dòng mức ≥ 1 |

## 6. Đặc tả công cụ (người làm T7 tự viết)

Đề xuất đặt trong `mir/qrels.py`, dạng `py -m mir.qrels <lệnh>`, test ở `tests/test_qrels.py`. Phải giữ
`py -m pytest -q tests` qua hết. Không đổi định dạng file trung gian đã chốt (README mục 7).

| Lệnh | Vào | Ra | Yêu cầu |
|---|---|---|---|
| `sample-queries --n 100 --seed 42` | `Data/query.parquet` | `eval/queries_eval.csv` | Phân tầng theo độ dài câu hỏi (4 nhóm theo tứ phân vị số từ, mỗi nhóm 25 câu). Cùng seed thì ra cùng kết quả. |
| `pool --runs a.parquet,b.parquet … --depth 10 --batch 1` | các run, `corpus_clean/`, `chunks/structure/`, `eval/qrels.csv` (nếu có) | `eval/pool/<đợt>.csv` (+ `eval/pool/<đợt>_docs/<doc_id>.txt`) | Mỗi câu lấy hợp top-`depth` tài liệu của các run (`mir.runs.doc_ranking`). Bỏ các cặp đã có trong `qrels.csv`. Xáo thứ tự bằng seed cố định và không ghi tên run (chấm mù). Mỗi dòng gồm `qid, query, doc_id, url, lang, title`, đoạn tốt nhất của tài liệu trong các run, và các cột trống `rel, chunk_text, note`. Văn bản đầy đủ đã làm sạch của mỗi tài liệu ghi ra `<doc_id>.txt` để chép `chunk_text`. |
| `import eval/judgments/*.csv` | các file nhãn, `corpus_clean/` | `eval/qrels.csv`, `qrels.parquet` | `rel` phải thuộc {0, 1, 2}. `doc_id` phải có trong `corpus_clean`. `chunk_text` (nếu có) phải là **chuỗi con nguyên văn** của văn bản đã làm sạch; sai thì báo dòng lỗi và không ghi. Hai người chấm khác nhau một cặp thì liệt kê ra để thống nhất. `qrels.parquet` chỉ chứa các dòng mức ≥ 1. |
| `kappa a.csv b.csv` | hai file nhãn cùng câu hỏi | κ, bảng chéo, danh sách cặp lệch | Nhị phân theo mức ≥ 1; in thêm κ cho mức 2. |
| `sweep-k --run … --qrels …` | run, qrels | `results/t7_f2_by_k.md` | Tính F2 tài liệu và F2 đoạn với từng `k_doc`, `k_chunk`, dùng `mir.evaluate.evaluate`. |
| *(sửa `mir/evaluate.py`)* | | | `load_qrels` đọc được cả CSV có cột `rel` (chỉ giữ mức ≥ `eval.min_rel`, mặc định 1). In thêm `judged@10` (tỷ lệ tài liệu top-10 đã chấm) khi qrels có các dòng mức 0. |

Hàm có sẵn nên dùng lại:
- `mir.config.load`, `cfg.path("clean")`, `cfg.path("chunks", "structure")`, `cfg.path("runs")`.
- `mir.retrieve.load_queries(path)`.
- `mir.runs.read(path)`, `mir.runs.doc_ranking(df)` (thứ hạng tài liệu = thứ hạng đoạn tốt nhất).
- `mir.store.lookup(corpus_clean, "doc_id", ids, ["title", "text", "lang", "host"])`: chỉ đọc các dòng cần, không
  nạp cả kho.
- `mir.store.unit_rows(chunk_dir, chunk_ids)`: văn bản đoạn.
- `mir.evaluate.evaluate`, `mir.evaluate.match`, `mir.evaluate.f2`.
- URL gốc: `Data/links_corpus.parquet` (cột `id`, `url`).

Lấy dữ liệu về máy:
- Run: tải output của notebook T5 (dataset `mir-runs`), bằng giao diện web hoặc
  `kaggle datasets download <chủ>/mir-runs -p runs_full --unzip`.
- `corpus_clean/`, `chunks/`: lấy từ người làm T0–T2 hoặc từ dataset `mir-data`.

## 7. Quy mô và thời gian

Tính thô, **chưa đo**: 100 câu × 20–40 tài liệu = 2.000–4.000 cặp. Nếu mỗi cặp mất 20–40 giây thì tổng là 11–44 giờ
công. Giai đoạn 1 phải đo con số thật. Nếu quá sức một người thì báo người duyệt để chọn cách giảm (mục 9):
- giảm pool xuống top-5;
- bớt run;
- làm 50 câu trước;
- dùng LLM mở chấm sơ bộ.

## 8. Sản phẩm bàn giao và tiêu chí xong

1. `docs/huong_dan_gan_nhan.md`: hướng dẫn gán nhãn bản cuối.
2. `eval/queries_eval.csv`, `eval/judgments/*.csv`, `eval/qrels.csv`, tất cả trong git.
3. Công cụ ở mục 6 và test tương ứng; `py -m pytest -q tests` qua hết.
4. `results/t7_report.md`, gồm:
   - số câu, số cặp đã chấm, tỷ lệ cặp liên quan, thời gian trung bình mỗi cặp;
   - κ đo trên ít nhất 10 câu do hai người cùng chấm;
   - bảng A/B/C, đường F2 theo K, bảng theo ngôn ngữ;
   - đối chiếu với leaderboard.
5. Cập nhật `docs/NOTES.md`: cách tạo nhãn, các số ở trên, giới hạn của tập nhãn, và mọi model hoặc công cụ đã dùng
   khi gán nhãn (tên, phiên bản) để viết bài mô tả phương pháp.
6. `docs/leaderboard.md`: mỗi lần nộp một dòng (ngày, file, run, `k_doc`, `k_chunk`, F2 tài liệu, F2 đoạn, ghi chú).

Coi là xong khi:
- mọi cặp trong pool của 100 câu đã có nhãn;
- mọi `chunk_text` đã qua kiểm tra nguyên văn;
- mọi con số báo cáo ghi rõ nguồn nhãn ("nhãn người, N câu, κ = …").

**Không** báo số liệu từ nhãn chưa có người kiểm như số thật.

## 9. Câu hỏi còn mở (người duyệt quyết định)

1. **Đọc tài liệu tiếng Trung.** Khoảng 80% kho là tiếng Trung. Ai trong nhóm đọc được? Nếu dùng công cụ dịch:
   luật thi cấm LLM đóng (GPT, Gemini…). Phương án an toàn là dùng model mở ≤ 14B phát hành trước 1/6/2026 (ví dụ họ
   Qwen2.5 / Qwen3 cỡ ≤ 14B) chạy trên Kaggle để dịch hoặc tóm tắt sang tiếng Việt. Có được dùng Google Dịch không thì
   nhóm cần thống nhất, có thể hỏi BTC, và ghi lại vào NOTES.
2. **Quy mô** sau khi đo ở giai đoạn 1: 100 câu × top-10, hay thu nhỏ lại.
3. **LLM chấm sơ bộ** (tuỳ chọn), nếu chấm tay quá chậm:
   - model mở ≤ 14B chấm trước;
   - người duyệt mọi cặp LLM cho là liên quan, cộng một mẫu ngẫu nhiên các cặp còn lại;
   - báo cáo là "nhãn LLM + người kiểm", kèm độ khớp giữa LLM và người.

   Chỉ làm khi người duyệt đồng ý.
4. **Thử cách chia đoạn hoặc mã hoá khác** (sau baseline): dựng "kho thu nhỏ" gồm mọi tài liệu trong pool cộng khoảng
   200k tài liệu ngẫu nhiên làm nhiễu, chạy e2e với `CORPUS = 'full'` trên kho đó, rồi so tương đối giữa các phương án.
   Số tuyệt đối trên kho thu nhỏ sẽ cao hơn trên toàn kho.

## 10. Lưu ý

- Nhãn của BTC được tạo theo cách mình không biết, nên tập nhãn nội bộ chỉ là xấp xỉ. Dùng nó để **so sánh tương đối**
  và chọn tham số. Các quyết định lớn (K cuối, cấu hình nộp) phải được kiểm lại bằng leaderboard.
- Không chỉnh nhãn sau khi đã xem điểm của một cấu hình cụ thể.
- Chênh lệch vài điểm phần trăm giữa hai cấu hình trên 100 câu có thể chỉ là nhiễu. Xem thêm số câu mà cấu hình này
  thắng hoặc thua cấu hình kia.
