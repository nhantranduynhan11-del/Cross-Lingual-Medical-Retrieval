# mirret — pipeline truy hồi ViBioMIR (chia đoạn → 9 nhánh → RRF → rerank → bài nộp → đánh giá)

Đầu vào: kho do `mircrawl merge` tạo (`corpus_out/corpus/*.parquet`, văn bản có cấu trúc: `#` tiêu đề, `- ` danh sách,
`a | b` bảng, `---` ngăn hỏi/đáp), `data/query.parquet` (1.200 câu), `data/links_corpus_vi_en_zh.parquet`.
Mọi bước chạy được trên Kaggle bằng 6 notebook trong `notebooks/` (sinh bởi `tools/make_notebooks.py`).
Kế hoạch đánh giá khi không có nhãn của BTC: **[docs/EVAL_PLAN.md](docs/EVAL_PLAN.md)**.

## 1. Thiết kế

**A. Đoạn nộp = đoạn cha** (`mirret/chunk.py`): chia theo cấu trúc (tiêu đề, đoạn văn, danh sách, bảng, cặp hỏi/đáp),
mục tiêu ~300 token BGE-M3, tối thiểu 150, tối đa 400, **không chồng lấn**; bài ≤ 400 token giữ nguyên một đoạn. Khối quá
dài tách theo dòng → câu → mệnh đề → khoảng trắng. `text` là **chữ thuần nguyên văn** của bài (trùng khớp từng ký tự với
`to_plain()` của kho, có `char_start/char_end`) → dùng thẳng làm `chunk_text` khi nộp.
Đã kiểm tra trên 2.000 bài thật: 0 lỗi bất biến, trung vị 224–256 token, tối đa 400.

**B. Hai mức đơn vị lập chỉ mục:** đoạn cha (nộp) + **đoạn con** ≤ 128 token (tách theo câu, chỉ tạo cho đoạn cha
> 160 token). Tìm ở cả hai mức; kết quả đoạn con được dồn về đoạn cha (đoạn con tốt nhất đại diện), rồi gộp RRF hai mức
trong mỗi nhánh. Luôn **nộp đoạn cha**.

**C. Ngữ cảnh khi lập chỉ mục (không nộp):** `ctx` = "tiêu đề bài > đường dẫn mục" (≤ 220 ký tự; bỏ mục > 100 ký tự như
sapo). Đầu vào mã hoá/BM25 = `ctx + "\n" + text`; bài nộp chỉ có `text`.

**Stage 1 — 9 nhánh** (`mirret/retrieve.py`): câu hỏi {vi gốc, en, zh} × {Dense BGE-M3 (FAISS), LSR BGE-M3 (trọng số
từ vựng), BM25 (mỗi ngôn ngữ tài liệu một chỉ mục; vi: pyvi, zh: jieba)}. Gộp **RRF** k = 60 (`mirret/fuse.py`):
`rrf9` (9 nhánh), `rrf_vi`, `rrf_dense3`; hỗ trợ trọng số `ten*0.5`.

**Dịch câu hỏi** (`mirret/translate.py`): vi → en → zh bằng LLM 4-bit trên T4. Bản đang dùng: **Qwen2.5-14B-Instruct,
prompt "t2"** (`qtrans/qwen25_14b-t2`): luật dịch thuật ngữ / triệu chứng / phủ định / số liệu / tên thuốc, 3 ví dụ mẫu,
gợi ý thuật ngữ từ `configs/med_glossary_vi.tsv`; chạy lại trong phiên các câu lỗi. Đủ 1.200/1.200 câu (câu #71 sửa tay
trong `records-manual.jsonl`); chấm nửa ngẫu nhiên: ~2% lỗi nặng. Đã so với Qwen3-8B và Qwen3-14B (`translate compare`).

**Stage 2 — rerank** (`mirret/rerank.py`): top-N của run gộp, chấm bằng **ColBERT BGE-M3** (MaxSim) hoặc
**cross-encoder** `BAAI/bge-reranker-v2-m3`; câu hỏi cùng ngôn ngữ với đoạn (`match_lang`); phần sau hạng N giữ thứ tự cũ.
Lưu ý: F2 của BTC tính trên **tập** → rerank chỉ đổi điểm khi top_n > k_chunk.

**Baseline hiện tại (19/10/2026): `rrf9_w` + cross-encoder.** Chọn trên tập có nhãn 40 câu (nhãn LLM Sonnet,
chưa đối chiếu người; docs/EVAL_PLAN.md), kiểm tra cùng dấu ở 2 nửa câu hỏi (`tools/eval_split.py`):
- 9 nhánh đều giữ; en_lsr và en_bm25 không có đóng góp đo được khi bỏ khỏi RRF → **trọng số 0,5** (`rrf9_w`) tốt hơn rrf9
  ở doc/unit F2@5–20; bỏ hẳn (rrf7) làm giảm Recall@50.
- Stage 2: **cross-encoder** `bge-reranker-v2-m3` hơn rrf9 có ý nghĩa (doc F2@20 +0,09); ColBERT không hơn rrf9.
- Mỗi câu: 9 nhánh × 1.000 đoạn cha (~6.650 đoạn khác nhau) → RRF top-1.000 → rerank top-400 (401–1.000 giữ thứ tự RRF)
  → nộp 185 đoạn + 200 tài liệu (thứ hạng tài liệu = đoạn tốt nhất), đều nằm trong vùng đã rerank.

**Bài nộp** (`mirret/submit.py`): `relevant_docs` top-200 (doc_id **số nguyên**), `relevant_chunks` top-185 đoạn cha
(bỏ đoạn gần trùng trong cùng câu), file nén ≤ 99 MB (tự giảm k_chunk nếu vượt), `--verify-corpus` kiểm tra `chunk_text`
khớp chữ thuần của bài.

**Đánh giá:** `mirret/diag.py` (không nhãn, 1.200 câu), `mirret/judge.py` (chọn câu, pool, phiếu chấm ẩn nguồn, kappa),
`mirret/evaluate.py` (P / R trên pool / F2 cấp tài liệu và cấp đoạn với LCS ≥ 0,4 như BTC, giá trị biên, bảng cùng
ngân sách).

## 2. Chạy trên Kaggle

### Dataset

| Dataset | Tạo bằng | Nội dung |
|---|---|---|
| `mir2-code` | `python tools/pack_kaggle_ret.py code --user <tên>` → `build/mir2_code/` | `mirret/`, `mircrawl/`, `configs/retrieval.yaml`, `VERSION.txt` (zip) |
| `mir2-data` | `python tools/pack_kaggle_ret.py data --user <tên>` → `build/mir2_data/` | `data/*.parquet`, `corpus_out/corpus/` |
| `mir2-chunks` | output k1 | `chunks/{parents,children}/`, `index/bm25/` |
| `mir2-qtrans` | output k2 | `qtrans/<model>/records*.jsonl`, `compare_*/` |
| `mir2-emb-m0..m2` | output k3 (mỗi người một) | `emb/{parents,children}/<khối>/` |
| `mir2-runs` | output k4 | `runs/`, `qtrans/queries_*.parquet`, `qemb/`, `eval/diag/` |
| `mir2-rerank` | output k5 | `runs/rrf9_{colbert,cross}.parquet`, `submissions/*.zip` |
| `mir2-labels` | thư mục `eval/judge/` của k6 sau khi điền nhãn | phiếu đã chấm, `key.parquet`, `queries.csv` |

Đăng dataset: `kaggle datasets create -p build/mir2_code` (lần sau `kaggle datasets version -p build/mir2_code -m "…"`);
dữ liệu: thêm `--dir-mode zip`. Với output notebook: tab *Output* → *New Dataset*.

### Thứ tự notebook

| Notebook | Máy | Add Input | Việc |
|---|---|---|---|
| `k1_chunk_bm25` | CPU | code, data | chia đoạn A+B+C, báo cáo `results/chunk_report.md`, BM25 đoạn cha vi/en/zh |
| `k2_translate` | GPU T4 ×2 | code, data | dịch bằng `MODELS`, so sánh tự động + phiếu chấm mù (EVAL_PLAN mục 0) |
| `k3_encode` | GPU T4 ×2, mỗi người một phiên | code, data, chunks | mã hoá BGE-M3 (dense + LSR) đoạn cha + con; `MEMBER`, `MEMBERS`; chạy tiếp khi bị ngắt; mã hoá bổ sung bằng `SKIP_FROM` |
| `k4_index_retrieve` | GPU T4 | code, data, chunks, qtrans, mọi emb | `USE` = bản dịch; chỉ mục FAISS, mã hoá câu hỏi, 9 nhánh, RRF, chẩn đoán |
| `k5_rerank_submit` | GPU T4 ×2 | code, data, chunks, runs | rerank `METHODS` top-`TOP_N`, bài nộp cho run gộp và từng run rerank |
| `k6_eval` | CPU | code, data, chunks, runs, rerank (+ labels) | `MODE='pool'` → phiếu; `'extend'` → phiếu bổ sung; `'score'` → kappa, điểm, giá trị biên, bảng |

Ô (1) của mọi notebook in `VERSION.txt` của `mir2-code` — kiểm tra đúng bản trước khi chạy dài.
Mỗi notebook ghi thời gian từng bước và dung lượng output vào `results/timing_<tên>.json`.

**Kho đổi** (kho đã chốt bản merge 17/10 cho đến hết cuộc thi; ghi lại để tham khảo): tạo `mir2-data` bản mới → chạy lại k1 → k3 với `SKIP_FROM` = các dataset emb cũ
đầy đủ (chỉ mã hoá đoạn mới / đổi nội dung, theo cặp `unit_id` + hash) → k4 (nhận mọi `mir2-emb-*`).

## 3. Chạy trên máy (thử nhỏ)

```bash
pip install -r requirements-retrieval.txt
python -m mirret.chunk run --workers 4 && python -m mirret.chunk report
python -m mirret.index bm25 --level parents
python -m mirret.translate import-old --qexp <thư mục qexp cũ> --name old_qwen25_14b
python -m mirret.translate build --use old_qwen25_14b
python -m mirret.encode run --level parents --device cuda:0      # tương tự --level children
python -m mirret.index dense --level parents                     # tương tự children
python -m mirret.retrieve qencode --device cuda:0
python -m mirret.retrieve run
python -m mirret.fuse
python -m mirret.diag
python -m mirret.rerank --run runs/rrf9_w.parquet --method cross --top-n 400
python -m mirret.rerank --run runs/rrf9_w.parquet --method cross --qids queries.csv   # chỉ bộ nhãn → rrf9_w_cross_sub
python -m mirret.submit --run runs/rrf9_w_cross.parquet --verify-corpus
```

Đường dẫn: `configs/retrieval.yaml` (`paths`, các mục `*.out_dir`), ghi đè bằng biến môi trường `MIR_DATA_DIR`
(dữ liệu chỉ đọc) và `MIR_WORK_DIR` (kết quả), hoặc `--config <file>`.

## 4. Tham số chính (`configs/retrieval.yaml`)

| Khoá | Giá trị | Ý nghĩa |
|---|---|---|
| `chunk.parent.{target,min,max}` | 300 / 150 / 400 | độ dài đoạn cha (token BGE-M3) |
| `chunk.child.{max,min,min_parent}` | 128 / 24 / 160 | đoạn con; chỉ tạo cho đoạn cha > 160 token |
| `chunk.ctx.max_chars` | 220 | độ dài ngữ cảnh "tiêu đề > mục" |
| `encode.{max_length,batch_tokens}` | 512 / 16384 | mã hoá BGE-M3 fp16 |
| `index.dense.{flat_below,nlist,nprobe}` | 200000 / 16384 / 64 | FAISS: Flat khi nhỏ, IVF-SQ8 khi lớn |
| `bm25.{k1,b}` | 1,5 / 0,75 | BM25 Lucene, thống kê toàn cục |
| `translate.use` | `qwen25_14b` | bản dịch cho nhánh en/zh (chọn sau khi chấm) |
| `retrieve.{levels,depth}` | parents + children / 1000 | mức tìm, độ sâu mỗi nhánh |
| `fuse.rrf_k`, `fuse.configs` | 60 / **rrf9_w** (baseline), rrf9, rrf_vi, rrf_dense3 | gộp RRF |
| `rerank.{method,top_n,match_lang}` | **cross** / 400 / true | Stage 2 |
| `submit.{k_doc,k_chunk,max_zip_mb}` | 200 / 185 / 99 | bài nộp |
| `eval.{rel_threshold,lcs_threshold}` | 1 / 0,4 | ngưỡng nhãn liên quan; ngưỡng khớp đoạn như BTC |

## 5. Đã kiểm tra / chưa kiểm tra

- **Đã chạy trọn** trên máy, với kho thật thu nhỏ và **model ngẫu nhiên cỡ nhỏ** (thay BGE-M3, cross-encoder, Qwen):
  chia đoạn → BM25 → mã hoá 2 phần + chạy tiếp + mã hoá bổ sung → FAISS → 9 nhánh → RRF → rerank 2 phần gộp lại → chẩn
  đoán → pool / phiếu / nhãn giả lập (phiếu đổi tên kiểu Google Sheets, phiếu lưu bằng bảng mã Windows) / kappa / phân
  xử → điểm, giá trị biên, bảng → bài nộp hợp lệ (`--verify-corpus`). Cả 6 notebook chạy trọn trong thư mục `/kaggle`
  giả lập (`KAGGLE_ROOT`).
- **Đã chạy với model thật trên Kaggle, kho đầy đủ** (1.120.017 bài, merge 17/10): k1 → k6 với BGE-M3, bge-reranker-v2-m3,
  Qwen2.5-14B / Qwen3-8B / Qwen3-14B. Đo được: bge-reranker top-400 ~6 s/câu/GPU T4 (1.200 câu ~1 giờ trên 2 GPU).
  Điểm trên tập có nhãn 40 câu: `results/k6_score`, `results/score_rrf7` (không có trong repo, xem mục 6).
- **Chưa kiểm:** nhãn LLM chưa đối chiếu với người; các số thời gian khác trong notebook vẫn là ước tính.

## 6. Làm việc nhóm (20–31/10/2026)

- Lộ trình, phân việc A/B/C, paper nguồn từng task: https://claude.ai/artifact/KSmsvjrsipMogjb3B5SHBN (cần được chia sẻ
  quyền xem).
- **Nhánh:** `main` = baseline đã chốt (`rrf9_w` + cross). Mỗi người làm trên nhánh riêng (`a-reranker`, `b-q2d`,
  `c-e5`); chỉ merge vào `main` khi thay đổi thắng ở vòng chấm theo luật trong lộ trình.
- **Thử trên bộ nhãn trước, chạy đủ 1.200 câu sau:** danh sách câu có nhãn `queries.csv` (gửi qua Drive; nhãn do người
  phụ trách nhãn giữ). `mirret.rerank --qids queries.csv` hoặc notebook k5 `QIDS = 'queries.csv'` → `<run>_<method>_sub`.
  Module mới đọc danh sách câu bằng `runs.read_qids`.
- **Bàn giao vòng chấm:** gửi `runs/<tên>.parquet` + `.json` cho người phụ trách nhãn trước tối 22/10 (vòng 1) và 26/10
  (vòng 2); so với baseline bằng `tools/eval_split.py` trên thư mục eval nhận lại.
- **Code lên Kaggle:** mỗi người đóng gói nhánh của mình: `python tools/pack_kaggle_ret.py code --user <tên Kaggle>`;
  ô (1) của notebook in `VERSION.txt` để kiểm đúng bản.
- **Không commit dữ liệu** (`.gitignore`): kho, chunks, emb, runs, nhãn, bài nộp đi qua Kaggle Datasets hoặc Drive.
