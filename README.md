# ViBioMIR baseline v1 — truy hồi tài liệu y khoa đa ngôn ngữ (AI GURU Stage 3)

Câu hỏi y khoa tiếng Việt → tài liệu và đoạn nội dung liên quan (tiếng Việt, tiếng Trung). Chấm bằng F2 macro ở hai cấp,
tài liệu và đoạn.

Repo gồm `crawl.py`, `mir/`, `configs/`, `notebooks/`. Dữ liệu và kết quả không nằm trong repo: dữ liệu BTC đặt vào `Data/`, các thư mục khác được tạo ra khi chạy lệnh.

```
crawl.py ─► out/ ─T0─► sample/ ─T1─► corpus_clean/ ─T2─► chunks/<strategy>/ ─T3 (GPU)─► emb/<strategy>/shard-XXXX/
                                                                 │                                  │
                                                     T4: BM25 (pyvi/jieba)          T4: FAISS IVF-SQ8 (dense), LSR (sparse)
                                                                 └──────────────► T5: 3 nhánh × độ sâu 1000 ─► RRF A–G
                                                                                        ─► T6: ColBERT top-100 ─► T8: bài nộp
                                                                                        ─► T7: đánh giá (cần nhãn)
```

## Cài đặt

```
py -m pip install -r requirements.txt
```

Đặt dữ liệu BTC vào `Data/query.parquet` và `Data/links_corpus.parquet` của thư mục dữ liệu. Đường dẫn lấy từ `configs/baseline.yaml` (`paths`), mặc định là `E:/Medical Retrieval`. Trên máy khác, sửa `paths.data_dir` và `paths.work_dir`, hoặc đặt hai biến môi trường `MIR_DATA_DIR` và `MIR_WORK_DIR` (ghi đè giá trị trong file); có thể đặt cả hai bằng thư mục repo. Riêng `crawl.py` nhận đường dẫn qua `--input` (mặc định `E:\Medical Retrieval\Data\links_corpus.parquet`). Mọi lệnh đều chạy lại an toàn: phần đã xong được bỏ qua, muốn làm lại thì thêm `--force`. Cấu hình đã dùng được ghi cạnh mỗi kết quả (`*.json`, `_config.json`, `_meta.json`, `_DONE.json`).

> Trên Windows, nếu `python` trỏ nhầm sang một bản Python khác (vd msys64) không có thư viện đã cài, dùng `py` thay cho `python`.

## Chia việc cho 3 người

| Bước | Ở đâu | Ai | Việc | Kết quả |
|---|---|---|---|---|
| 0. Crawl | máy cá nhân | cả 3 | `py crawl.py --input Data/links_corpus.parquet --out out --shard i/3 --save-raw --redo-stale --skip-hosts familydoctor.com.cn,zysjonline.com` (người i = 0, 1, 2) | `out/` của mỗi người |
| 1. T0–T2 | máy cá nhân | 1 người | gộp `out/`, khảo sát, làm sạch (duyệt), chia đoạn | dataset Kaggle **`mir-data`** |
| 2. T3 | Kaggle GPU T4 ×2 | cả 3 | `notebooks/t3_encode_kaggle.ipynb`, `MEMBER = 0/1/2` | dataset **`mir-emb-m0/m1/m2`** |
| 3. T4 BM25 | Kaggle CPU | 1 người | `notebooks/t4_bm25_cpu_kaggle.ipynb` | dataset **`mir-bm25`** |
| 4. T4+T5 | Kaggle GPU T4 | 1 người | `notebooks/t5_index_retrieve_kaggle.ipynb` | dataset **`mir-runs`** |
| 5. T6–T8 | Kaggle GPU T4 ×2 | 1 người | `notebooks/t6_rerank_submit_kaggle.ipynb` | `submissions/*.zip` → nộp lên leaderboard |

Bước 3 có thể làm song song với bước 2. Dataset **`mir-code`** gồm `mir/` và `configs/` của repo này; mọi notebook đều cần nó, nên cập nhật dataset này mỗi khi sửa code.

### Bước 0 — crawl (mỗi người một shard)
- Mỗi máy dùng thư mục `--out` riêng. Khi gộp về một máy, đặt mỗi `out/` thành một thư mục con, ví dụ `out/s0`, `out/s1`, `out/s2`. `mir/crawlio.py` đọc đệ quy mọi `part-*.jsonl.gz`, và nếu một id có nhiều bản thì lấy bản trích bằng quy tắc mới nhất (`xv`).
- `--save-raw` lưu HTML gọn, khoảng 12 KB/trang (ước tính khoảng 15 GB mỗi máy, chưa đo trên crawl lớn). Có nó thì khi sửa quy tắc trích chữ chỉ cần chạy `py crawl.py --out out --reextract stale`, không phải crawl lại.

### Bước 1 — T0, T1, T2 (máy cá nhân, CPU)
```
py -m mir.survey --config configs/baseline.yaml                    # results/t0_survey.md
py -m mir.make_sample --config configs/baseline.yaml --force       # sample/ (2.000 bài, 50 câu hỏi)
py -m mir.clean --config configs/baseline.yaml --stats --force     # clean_lines.json + results/t1_review.md → DUYỆT
py -m mir.clean --config configs/baseline.yaml --apply --force     # corpus_clean/
py -m mir.clean --config configs/baseline.yaml --apply --input sample/corpus_raw.parquet --out sample/corpus_clean --force
py -m mir.chunk --config configs/baseline.yaml                     # chunks/structure/ (toàn kho ~30 phút)
py -m mir.chunk --config configs/baseline.yaml --input sample/corpus_clean --out sample/chunks --strategy all
```
Tải lên Kaggle một dataset **`mir-data`** gồm:
- `Data/query.parquet` và `Data/links_corpus.parquet`
- `corpus_clean/`
- `chunks/structure/`
- `sample/corpus_clean/` và `sample/chunks/`

Có thể tải qua giao diện web hoặc dùng `kaggle datasets create -p <thư mục>`.

### Bước 2 — T3 mã hoá (3 tài khoản Kaggle, GPU T4 ×2)
Mở `notebooks/t3_encode_kaggle.ipynb` và chạy theo thứ tự:
1. Ô (1) và (2): chuẩn bị, rồi đối chiếu với FlagEmbedding.
2. Ô (3): đo tốc độ, và **gửi số đo cho người duyệt**.
3. Ô (4): đặt `RUN_FULL = True` và `MEMBER = m` rồi chạy.

Shard được chia như sau: người m chạy shard `2m/6` trên GPU 0 và `2m+1/6` trên GPU 1. Mỗi khối 100.000 đoạn ghi vào `emb/structure/shard-XXXX/` kèm `_DONE.json`. Nếu hết phiên 12 giờ thì chạy lại notebook, các khối đã xong sẽ được bỏ qua. Xong thì *Save Version* rồi tạo dataset `mir-emb-m{m}` từ output.

### Bước 3 — T4 BM25 (Kaggle CPU)
`notebooks/t4_bm25_cpu_kaggle.ipynb`: tách từ bằng pyvi (tiếng Việt) và jieba (tiếng Trung) theo từng part, rồi dựng ma trận BM25 với thống kê toàn kho. Kết quả tạo dataset `mir-bm25`.

### Bước 4 — T4 dense + T5 truy hồi (Kaggle, 1 tài khoản)
`notebooks/t5_index_retrieve_kaggle.ipynb` làm lần lượt:
1. Mã hoá 1.200 câu hỏi.
2. `dense-tune`: bảng Recall@100 so với Flat để chọn `nlist`/`nprobe`.
3. Dựng chỉ mục IVF-SQ8.
4. Truy hồi 3 nhánh (dense, LSR, BM25) với độ sâu 1000.
5. `mir.fuse`: tạo 7 cấu hình RRF A–G (k = 60) và các cấu hình gộp có trọng số.

Kết quả tạo dataset `mir-runs`.

### Bước 5 — T6 xếp hạng lại, T7 đánh giá, T8 nộp bài
`notebooks/t6_rerank_submit_kaggle.ipynb` làm lần lượt:
1. Xếp hạng lại top-100 bằng ColBERT trên 2 GPU.
2. `mir.evaluate`, chỉ chạy khi đã có nhãn.
3. `mir.submit`: kiểm tra đủ 1.200 câu, `doc_id` có trong `links_corpus`, `chunk_text` là chuỗi con nguyên văn của bài đã làm sạch; rồi nén ZIP chứa đúng 1 file.

## Lệnh từng bước

| Task | Lệnh |
|---|---|
| T3 | `python -m mir.encode --shard i/N --device cuda:0 [--bench 2000 --est-chunks N]` |
| T4 | `python -m mir.index dense\|dense-tune\|lsr\|bm25-tokenize\|bm25-build [--emb a,b,c] [--shard i/N]` |
| T5 | `python -m mir.retrieve qencode`; `python -m mir.retrieve run --branch all [--emb a,b,c]`; `python -m mir.fuse` |
| T6 | `python -m mir.rerank --run runs/structure_G.parquet --shard i/N --device cuda:0` (chạy lại khi đủ phần để gộp) |
| T7 | `python -m mir.evaluate --run runs/x.parquet --qrels q.parquet`; `python -m mir.evaluate --tables --qrels q.parquet` |
| T8 | `python -m mir.submit --run runs/structure_G_rr.parquet [--name sub1] [--k-doc 10 --k-chunk 10]` |

Đổi cách chia đoạn bằng `--strategy recursive|structure|parent_child`, mặc định lấy từ `chunk.strategy`. Với `parent_child` thì đặt `rerank.unit: parent` để tìm bằng đoạn con và xếp hạng lại, nộp bằng đoạn cha.

## Định dạng file trung gian

| Thư mục | Nội dung |
|---|---|
| `corpus_clean/part-*.parquet` | `doc_id` (str), `host`, `lang`, `title`, `text` |
| `chunks/<strategy>/part-*.parquet` | `chunk_id` (`{doc_id}_{k}`), `doc_id`, `lang`, `char_start`, `char_end`, `text`, `n_tokens` (+ `parent_id`). Bất biến: `text == corpus_clean.text[char_start:char_end]` |
| `chunks/parent_child/parents-*.parquet` | `parent_id` (`{doc_id}_p{k}`), `doc_id`, `lang`, `char_start`, `char_end`, `text`, `n_tokens` |
| `emb/<strategy>/shard-XXXX/` | `dense.npy` (float16, đã chuẩn hoá), `sparse.npz` (CSR float16, đọc bằng `mir.encode.load_sparse`), `chunk_ids.parquet`, `_DONE.json` |
| `index/<strategy>/` | `dense.faiss` + `dense_ids.parquet`; `bm25/shard-*.npz` (CSC) + `vocab.parquet` + `stats.json`; `lsr/` (tuỳ chọn) |
| `runs/<tên>.parquet` | `qid`, `chunk_id`, `doc_id`, `rank`, `score` (+ `<tên>.json` cấu hình) |
| `qrels.parquet` | `qid`, `doc_id`, `chunk_text` (có thể rỗng) |
| `submissions/<tên>.zip` | 1 file `<tên>.json` ở gốc |

## Mã nguồn

| File | Việc |
|---|---|
| `crawl.py` | crawl (đọc docstring đầu file) |
| `mir/config.py`, `crawlio.py` | cấu hình; đọc kết quả crawl |
| `mir/survey.py`, `make_sample.py` | T0 |
| `mir/clean.py` | T1 |
| `mir/chunk.py` | T2 |
| `mir/encode.py` | T3 (cả ColBERT cho T6) |
| `mir/index.py`, `bm25.py`, `sparse_index.py` | T4 |
| `mir/retrieve.py`, `fuse.py`, `runs.py`, `store.py` | T5 |
| `mir/rerank.py` | T6 |
| `mir/evaluate.py` | T7 |
| `mir/submit.py` | T8 |
