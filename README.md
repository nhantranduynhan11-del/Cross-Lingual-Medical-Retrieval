# ViBioMIR baseline v1 — truy hồi tài liệu y khoa đa ngôn ngữ (AI GURU 2026, Stage 3)

Đầu vào là câu hỏi y khoa tiếng Việt. Hệ thống trả về tài liệu và đoạn nội dung liên quan, bằng tiếng Việt hoặc tiếng
Trung. Bài nộp được chấm bằng F2 macro ở hai cấp: tài liệu và đoạn. Đề bài: [`docs/de_bai.md`](docs/de_bai.md).

```
crawl.py ─► out/ ─T0─► sample/ ─T1─► corpus_clean/ ─T2─► chunks/structure/ ─T3 (GPU)─► emb/structure/shard-XXXX/
                                                            │                                │
                                                 T4: BM25 (pyvi/jieba)      T4: FAISS IVF-SQ8 (dense), LSR (sparse)
                                                            └────────► T5: 3 nhánh × độ sâu 1000 ─► RRF A–G
                                                                         ─► T6: ColBERT top-100 ─► T8: bài nộp (.zip)
                                                                         ─► T7: đánh giá (cần nhãn)
```

Tài liệu khác:
- [`docs/NOTES.md`](docs/NOTES.md): mọi quyết định, tham số và số đo.
- [`docs/t7_danh_gia.md`](docs/t7_danh_gia.md): việc T7 (tập nhãn và đánh giá).
- [`docs/HANDOFF.md`](docs/HANDOFF.md): trạng thái gần nhất.
- [`crawl_eval/README.md`](crawl_eval/README.md): kiểm chứng bộ trích chữ của `crawl.py`.

## Mục lục

1. [Repo có gì](#1-repo-có-gì)
2. [Cài đặt trên máy cá nhân](#2-cài-đặt-trên-máy-cá-nhân)
3. [Chạy trên máy cá nhân: crawl, T0–T2](#3-chạy-trên-máy-cá-nhân-crawl-t0t2)
4. [Chạy trên Kaggle: T3–T8](#4-chạy-trên-kaggle-t3t8)
5. [Chia việc cho 3 người](#5-chia-việc-cho-3-người)
6. [Lệnh từng bước](#6-lệnh-từng-bước)
7. [Định dạng file trung gian](#7-định-dạng-file-trung-gian)
8. [Lỗi thường gặp](#8-lỗi-thường-gặp)

## 1. Repo có gì

| Đường dẫn | Nội dung |
|---|---|
| `crawl.py` | Crawl nội dung từ danh sách URL của BTC. Đọc docstring đầu file để biết chi tiết. |
| `mir/` | Mã nguồn, mỗi bước một module có giao diện dòng lệnh: `py -m mir.<bước> --config configs/baseline.yaml …` |
| `configs/baseline.yaml` | Mọi tham số, có chú thích. |
| `notebooks/` | 5 notebook Kaggle. Các notebook chỉ gọi lại lệnh `mir`. |
| `tools/pack_kaggle.py` | Đóng gói dataset `mir-code` / `mir-data` để tải lên Kaggle. |
| `tests/` | Kiểm thử, chạy trên CPU. |
| `crawl_eval/` | Công cụ kiểm chứng bộ trích chữ của `crawl.py`. |
| `results/` | Báo cáo T0–T2 và `clean_lines.json` (danh sách dòng bị loại ở T1, đã duyệt). |
| `docs/` | Đề bài, đặc tả gốc (`BUILD_PROMPT.md`), `NOTES.md`, `HANDOFF.md`, `t7_danh_gia.md`. |

Dữ liệu không nằm trong git. Các thư mục sau được tạo khi chạy lệnh:
- `Data/`: dữ liệu BTC.
- `out/`: kết quả crawl.
- `sample/`: tập mẫu cố định.
- `corpus_clean/`, `chunks/`, `emb/`, `index/`, `qemb/`, `runs/`, `submissions/`.
- `kaggle/`: gói tải lên Kaggle.

Mọi lệnh đều chạy lại an toàn: phần đã xong được bỏ qua; muốn làm lại thì thêm `--force`. Cấu hình đã dùng được ghi
bên cạnh mỗi kết quả (`_meta.json`, `_config.json`, `_DONE.json`, `<run>.json`).

## 2. Cài đặt trên máy cá nhân

### 2.1. Yêu cầu

- Python 3.10 trở lên (máy người duyệt dùng 3.13) và Git. Không cần GPU.
- **Windows:** gõ `py` thay cho `python` trong mọi lệnh. Trên một số máy, `python` trỏ tới một bản Python khác (vd
  msys64) không có thư viện. **Linux/macOS:** gõ `python3`.
- Dung lượng ổ đĩa cho mỗi người crawl (ước tính, chưa đo trên crawl lớn):
  - phần kết quả crawl khoảng 1,5 GB;
  - phần HTML gọn `out/raw/` (khi bật `--save-raw`) khoảng 12–15 GB.

### 2.2. Lấy code và cài thư viện

```powershell
git clone <địa chỉ repo của nhóm> "Medical Retrieval"
cd "Medical Retrieval"
py -m pip install -r requirements.txt
```

`requirements.txt` cài bản torch CPU, đủ cho mọi việc trên máy cá nhân. Nếu muốn tách thư viện khỏi Python chung, tạo
môi trường ảo trước khi cài:
- PowerShell / cmd: `py -m venv .venv` rồi `.venv\Scripts\activate`
- Linux/macOS: `python3 -m venv .venv` rồi `source .venv/bin/activate`

### 2.3. Dữ liệu BTC

Tải `query.parquet` (1.200 câu hỏi) và `links_corpus.parquet` (4.394.718 URL) vào `Data/`. Hai file trên Hugging Face
trùng SHA-256 với bản nhóm đang dùng (kiểm ngày 05/10/2026).

```powershell
py -c "from huggingface_hub import hf_hub_download as d; [d('AIGuruTinix/ViBioMIR', f, repo_type='dataset', local_dir='Data') for f in ('query.parquet', 'links_corpus.parquet')]"
```

### 2.4. Kiểm tra cài đặt

```powershell
py -m pytest -q tests
```

Trên máy người duyệt: 65 bài qua trong khoảng 1–2 phút.
- Lần đầu chạy cần mạng để tải tokenizer BGE-M3 (khoảng 17 MB). Trọng số model không tải về máy.
- Một số bài tự bỏ qua nếu máy thiếu `sample/`, `Data/` hoặc `crawl_eval/html/`, vì các thư mục này không có trong git.

**Đường dẫn.** Mặc định mọi thứ nằm trong thư mục repo (mục `paths` của `configs/baseline.yaml`). Muốn đặt dữ liệu ở
ổ khác thì đặt hai biến môi trường:
- `MIR_DATA_DIR`: thư mục chứa `Data/` và `out/`;
- `MIR_WORK_DIR`: nơi ghi kết quả.

Ví dụ trên PowerShell: `$env:MIR_WORK_DIR = "D:\mir"`.

## 3. Chạy trên máy cá nhân: crawl, T0–T2

### 3.1. Crawl (cả 3 người, mỗi người một phần)

Người số `i` (0, 1, 2) chạy lệnh sau, thay `i` bằng số của mình:

```powershell
py crawl.py --out out --shard i/3 --save-raw --redo-stale --skip-hosts familydoctor.com.cn,zysjonline.com
```

Ý nghĩa các cờ:
- `--shard i/3`: chỉ lấy các URL có `id % 3 == i`.
- `--save-raw`: lưu HTML gọn vào `out/raw/`. Khi sửa quy tắc trích chữ, chỉ cần chạy
  `py crawl.py --out out --reextract stale`, không phải tải lại.
- `--redo-stale`: tải lại các bài đã trích bằng quy tắc cũ.
- `--skip-hosts`: bỏ hai tên miền chặn crawl (khoảng 690k URL). Danh sách tên miền không lấy được ghi ở
  `docs/NOTES.md`.

Trong khi crawl:
- Có thể dừng bất cứ lúc nào bằng Ctrl+C. Chạy lại **đúng lệnh cũ** để tiếp tục từ chỗ dừng.
- Theo dõi tiến độ ở `out/host_stats.tsv` (số bài theo tên miền) và `out/progress.tsv` (trạng thái từng URL).
- Chạy xong thì chạy thêm một lần với `--retry-failed` để thử lại các URL lỗi tạm thời (timeout, 429, 5xx).

### 3.2. Gộp kết quả crawl về một máy

Người làm T0–T2 gom `out/` của cả 3 người về máy mình:
- Mỗi người gửi các file `out/part-*.jsonl.gz` và `out/progress.tsv`, ví dụ nén lại rồi gửi qua Google Drive.
  Không cần gửi `raw/`; mỗi người giữ `raw/` trên máy mình để trích lại khi cần.
- Đặt phần của mỗi người vào một thư mục con: `out/s0/`, `out/s1/`, `out/s2/`. Phần crawl ngay trên máy này có thể
  để nguyên trong `out/`.

`mir/crawlio.py` đọc đệ quy mọi `part-*.jsonl.gz` trong `out/`. Nếu một id có nhiều bản, nó lấy bản trích bằng quy
tắc mới nhất (`xv`).

### 3.3. T0–T2: khảo sát, làm sạch, chia đoạn (1 người, CPU)

```powershell
py -m mir.survey --config configs/baseline.yaml                    # T0 → results/t0_survey.md
py -m mir.clean --config configs/baseline.yaml --stats --force     # T1 bước 1 → results/clean_lines.json, results/t1_review.md
```

`--force` thay danh sách cũ. `results/clean_lines.json` hiện có là bản đã duyệt trên 10k URL đầu tiên; với kho đầy đủ
phải tính lại và duyệt lại.

**Dừng lại và duyệt** `results/t1_review.md`: đây là các dòng giao diện lặp lại sẽ bị xoá ở 6 tên miền lớn nhất.
Muốn luôn bỏ hoặc luôn giữ một dòng thì sửa `clean.force_drop` / `clean.force_keep` trong `configs/baseline.yaml`, rồi
chạy lại lệnh trên.

Sau khi duyệt:

```powershell
py -m mir.clean --config configs/baseline.yaml --apply --force     # T1 bước 2 → corpus_clean/part-*.parquet (thay bản cũ)
py -m mir.chunk --config configs/baseline.yaml                     # T2 → chunks/structure/ (ước tính ~30 phút)
py tools/pack_kaggle.py data                                       # → kaggle/mir-data/, tải lên Kaggle (mục 4.3)
```

`mir.chunk` tự chia lại các part đã cũ hơn `corpus_clean`. Nếu nó cảnh báo có part thừa (không còn trong
`corpus_clean`) thì xoá các part đó trước khi đóng gói.

Lưu ý:
- **Không** chạy lại `mir.make_sample`. `sample/` là tập mẫu cố định (2.000 bài, 50 câu hỏi), dùng cho kiểm thử và
  notebook e2e. Lệnh đó còn nạp toàn bộ kho vào RAM.
- `sample/` không có trong git. Ai cần thì chép từ máy người duyệt, hoặc lấy trong dataset `mir-data`.
- T1 trên toàn kho chưa đo thời gian. T2 ước tính khoảng 30 phút (đo 5 giây cho 9.184 bài).

### 3.4. (Tuỳ chọn) T3–T8 trên máy có GPU NVIDIA

Thường các bước này chạy trên Kaggle (mục 4). Trên máy có GPU:
1. Cài torch bản CUDA theo hướng dẫn ở https://pytorch.org.
2. Chạy đúng các lệnh ở [mục 6](#6-lệnh-từng-bước) với `--device cuda:0`.

Trên CPU chỉ nên chạy với kho mẫu: mã hoá BGE-M3 cho toàn kho bằng CPU là không khả thi.

## 4. Chạy trên Kaggle: T3–T8

### 4.1. Cần biết về Kaggle

- **Dataset** là dữ liệu đầu vào, chỉ đọc. Notebook dùng dataset qua *Add Input*. Notebook tự tìm đường dẫn, nên tên
  thư mục bên trong dataset không quan trọng.
- **Output** là thư mục `/kaggle/working`, tối đa khoảng 20 GB.
  - Output chỉ được lưu khi bấm *Save Version → Save & Run All (Commit)*. Lệnh này chạy lại cả notebook trong phiên
    mới, chạy nền, nên không cần mở trình duyệt.
  - Từ tab *Output* của một version có thể tạo dataset mới (*New Dataset*) cho bước sau.
- Phiên GPU dài tối đa khoảng 12 giờ, và GPU có hạn mức theo tuần.
- Muốn bật GPU và Internet, tài khoản Kaggle phải xác minh số điện thoại.
- Dataset ở chế độ riêng tư phải được **chia sẻ** cho tài khoản của các thành viên khác thì họ mới *Add Input* được.
  Chia sẻ trong trang dataset → *Settings* → *Sharing*.

### 4.2. Các dataset dùng chung

| Dataset | Nội dung | Ai tạo | Tạo bằng |
|---|---|---|---|
| `mir-code` | `mir/`, `configs/`, `VERSION.txt` | người sửa code | `py tools/pack_kaggle.py code` |
| `mir-data` | `Data/`, `sample/`, `corpus_clean/`, `chunks/structure/`, `VERSION.txt` | người làm T0–T2 | `py tools/pack_kaggle.py data` |
| `mir-emb-m0`, `-m1`, `-m2` | `emb/structure/shard-XXXX/` | mỗi người một bộ (T3) | output của `t3_encode_kaggle` |
| `mir-bm25` | `index/structure/bm25/` | 1 người (T4) | output của `t4_bm25_cpu_kaggle` |
| `mir-runs` | `runs/`, `index/structure/dense.faiss` | 1 người (T5) | output của `t5_index_retrieve_kaggle` |

### 4.3. Tải lên và cập nhật `mir-code`, `mir-data`

1. Đóng gói:
   - `py tools/pack_kaggle.py code` → `kaggle/mir-code/`
   - `py tools/pack_kaggle.py data` → `kaggle/mir-data/`. Thêm `--no-full` nếu chỉ cần `Data/` và `sample/` cho
     notebook e2e.

   Lệnh in ra một dòng `VERSION`, ví dụ `05/10/2026 12:49 | git 26438d1 | mã 5517c92c8e`. Ghi lại dòng này.
2. Tải lên bằng trình duyệt:
   - Lần đầu: kaggle.com → *Datasets* → *New Dataset* → kéo thả toàn bộ nội dung thư mục `kaggle/mir-code/` →
     đặt tên `mir-code` → *Create*.
   - Các lần sau: mở trang dataset → *New Version* → tải lại toàn bộ.
   - Làm tương tự với `mir-data`.
3. Tải lên bằng Kaggle CLI (tuỳ chọn):
   1. `py -m pip install kaggle` và cài API token theo hướng dẫn của Kaggle.
   2. `py tools/pack_kaggle.py code --user <tên tài khoản Kaggle>` (ghi thêm `dataset-metadata.json`).
   3. `kaggle datasets create -p kaggle/mir-code --dir-mode zip` cho lần đầu, hoặc
      `kaggle datasets version -p kaggle/mir-code --dir-mode zip -m "ghi chú"` cho các lần sau.
   4. Mở trang dataset kiểm tra cây thư mục.

**Mỗi lần sửa code đều phải đóng gói lại và tạo New Version cho `mir-code`.** Notebook giữ phiên bản dataset lúc được
thêm vào. Nếu panel *Input* báo có bản mới thì bấm cập nhật. Ô (1) của mọi notebook in dòng `mir-code …` và
`mir-data …`; đối chiếu với dòng `VERSION` ở bước 1 để chắc chắn đang chạy đúng bản.

### 4.4. Mở notebook

1. kaggle.com → *Code* → *New Notebook* → *File* → *Import Notebook* → chọn file trong `notebooks/`. Khi file trong
   repo thay đổi thì import lại; ô (1) in `NB_VERSION` để đối chiếu.
2. Panel bên phải:
   - *Session options* → *Accelerator*: GPU T4 ×2, hoặc None với notebook CPU.
   - *Internet*: On.
   - *Add Input*: thêm các dataset ghi ở đầu notebook.
3. Chỉ sửa các biến **VIẾT HOA** ở đầu ô (1). Phần còn lại của ô (1) giống nhau ở mọi notebook:
   - chép `mir-code` mới nhất vào `/kaggle/working/code`;
   - tạo cấu hình làm việc `/kaggle/working/config.yaml` từ `configs/baseline.yaml` cộng với các giá trị trong
     `OVERRIDES`. Khoá viết sai sẽ báo lỗi ngay.

### 4.5. Từng notebook

| Notebook | Ai | Accelerator | Add Input | Biến cần đặt | Thời gian | Tạo dataset từ output |
|---|---|---|---|---|---|---|
| `e2e_test_kaggle` | bất kỳ | GPU T4 ×2 | `mir-code`, `mir-data` | `CORPUS`, `FULL_QUERIES` | ~5 phút (kho mẫu, 50 câu) | — |
| `t3_encode_kaggle` | cả 3 | GPU T4 ×2 | `mir-code`, `mir-data` | `MEMBER` = 0/1/2 | ~4,3–5 giờ/người (ước tính) | `mir-emb-m{MEMBER}` |
| `t4_bm25_cpu_kaggle` | 1 người | None (CPU) | `mir-code`, `mir-data` | — | ~1 giờ (ước tính) | `mir-bm25` |
| `t5_index_retrieve_kaggle` | 1 người | GPU T4 | `mir-code`, `mir-data`, `mir-bm25`, `mir-emb-m0..2` | `RUN_TUNE`, `NLIST`, `NPROBE` | chưa đo | `mir-runs` |
| `t6_rerank_submit_kaggle` | 1 người | GPU T4 ×2 | `mir-code`, `mir-data`, `mir-runs` | `RUN`, `K_DOC`, `K_CHUNK`, `QRELS` | ~30 phút | — (tải `submissions/*.zip`) |

- **e2e:** chạy cả đường ống trong một phiên.
  - `CORPUS = 'sample'`: kiểm tra trên kho mẫu.
  - `CORPUS = 'full'`: chạy trên phần kho đang có trong `mir-data`, ví dụ kho crawl dở, để có run thật cho đợt gán
    nhãn thử.
  - Nên chạy e2e mỗi khi sửa code, trước các bước tốn GPU.
- **T3:**
  - Mỗi người đặt `MEMBER` rồi *Save & Run All*. Người `m` mã hoá các khối `k` có `k % 6 ∈ {2m, 2m+1}`, trên 2 GPU.
  - Ba người phải dùng **cùng một version `mir-data`**: dòng `mir-data … dấu vân tay …` ở ô (1) phải giống nhau. Nếu
    lệch, T5 sẽ dừng với thông báo "bản chunks khác nhau".
  - Bị ngắt giữa chừng: *Add Input* thêm output của version bị ngắt rồi chạy lại. Các khối đã xong được chép sang,
    chỉ mã hoá phần còn thiếu.
  - Ô (4) in số khối đã xong trên tổng số khối của người đó.
- **T4 BM25:** chạy trên CPU nên không tốn hạn mức GPU; chạy song song với T3 được.
- **T5:**
  1. Chạy tương tác ô (1)–(3), đọc bảng Recall@100 của `dense-tune`.
  2. Đặt `NLIST`/`NPROBE` ở ô (4). Ghi bảng vào `docs/NOTES.md`.
  3. Đặt `RUN_TUNE = False` rồi *Save & Run All*.
  4. Nếu ô (1) hoặc lệnh dựng chỉ mục báo `[CẢNH BÁO] thiếu N khối emb` thì chưa đủ 3 dataset `mir-emb-m*`.
- **T6–T8:**
  - Xếp hạng lại `RUN` trên 2 GPU, đánh giá nếu đặt `QRELS`, rồi tạo `submissions/<RUN>_rr_d<K_DOC>_c<K_CHUNK>.zip`.
  - Bài nộp chỉ được ghi khi đã qua mọi bước kiểm tra: đủ 1.200 câu, `doc_id` có trong `links_corpus`, `chunk_text`
    là chuỗi con nguyên văn của bài đã làm sạch.

### 4.6. Lấy kết quả và nộp bài

1. Tải file `.zip` từ panel *Output* bên phải (phiên tương tác), hoặc từ tab *Output* của version.
2. Nộp ở http://leaderboard.aiguru.com.vn/ → *My Submissions*.
3. Ghi lại mỗi lần nộp: ngày, tên file, run, `k_doc`, `k_chunk`, điểm F2 hai cấp.

Giới hạn: tối đa 10 bài/ngày. Ở vòng riêng (Private Phase), mỗi người chỉ được nộp 5 bài tổng cộng.

## 5. Chia việc cho 3 người

| Bước | Ở đâu | Ai | Kết quả | Chặn bước |
|---|---|---|---|---|
| 0. Crawl | máy cá nhân | cả 3 (`--shard 0/3`, `1/3`, `2/3`) | `out/` của mỗi người | 1 |
| 1. T0–T2 | máy cá nhân | 1 người | dataset `mir-data` | 2, 3 |
| 2. T3 | Kaggle GPU T4 ×2 | cả 3 (`MEMBER = 0/1/2`) | `mir-emb-m0/m1/m2` | 4 |
| 3. T4 BM25 | Kaggle CPU | 1 người, song song với bước 2 | `mir-bm25` | 4 |
| 4. T4 dense + T5 | Kaggle GPU T4 | 1 người | `mir-runs` | 5 |
| 5. T6–T8 | Kaggle GPU T4 ×2 | 1 người | `submissions/*.zip` → leaderboard | — |
| T7. Tập nhãn, đánh giá | máy cá nhân (+ Kaggle) | 1 người, **bắt đầu ngay** | `eval/`, `results/exp_*.md` | chọn `k_doc`, `k_chunk` |

Chi tiết việc T7: [`docs/t7_danh_gia.md`](docs/t7_danh_gia.md).

## 6. Lệnh từng bước

Mọi lệnh nhận `--config configs/baseline.yaml` (mặc định). Trên Kaggle, notebook truyền `--config /kaggle/working/config.yaml`.

| Bước | Lệnh |
|---|---|
| T0 | `py -m mir.survey`; `py -m mir.make_sample` (đã tạo, không chạy lại) |
| T1 | `py -m mir.clean --stats` → duyệt → `py -m mir.clean --apply` |
| T2 | `py -m mir.chunk [--strategy recursive\|structure\|parent_child\|all]` |
| T3 | `py -m mir.encode --shard i/N --device cuda:0 [--bench 2000 --est-chunks N]` |
| T4 | `py -m mir.index dense\|dense-tune\|lsr\|bm25-tokenize\|bm25-build [--emb a,b,c] [--qemb …]` |
| T5 | `py -m mir.retrieve qencode --device cuda:0`; `py -m mir.retrieve run --branch all\|dense,lsr\|bm25 [--emb a,b,c] [--index …]`; `py -m mir.fuse` |
| T6 | `py -m mir.rerank --run runs/structure_G.parquet --shard i/N --device cuda:0`. Khi đủ mọi phần, chạy lại lệnh để gộp. |
| T7 | `py -m mir.evaluate --run runs/x.parquet --qrels q.parquet`; `py -m mir.evaluate --tables --qrels q.parquet` |
| T8 | `py -m mir.submit --run runs/structure_G_rr.parquet [--name sub1] [--k-doc 10 --k-chunk 10]` |
| Kaggle | `py tools/pack_kaggle.py code\|data [--no-full] [--user <tên>]` |

Cách chia đoạn chọn bằng `--strategy`, mặc định lấy từ `chunk.strategy`. Với `parent_child`, đặt `rerank.unit: parent`:
khi đó tìm bằng đoạn con, còn xếp hạng lại và nộp bằng đoạn cha.

## 7. Định dạng file trung gian

| Thư mục | Nội dung |
|---|---|
| `corpus_clean/part-*.parquet` | `doc_id` (str), `host`, `lang`, `title`, `text` |
| `chunks/<strategy>/part-*.parquet` | `chunk_id` (`{doc_id}_{k}`), `doc_id`, `lang`, `char_start`, `char_end`, `text`, `n_tokens` (+ `parent_id`). Bất biến: `text == corpus_clean.text[char_start:char_end]` |
| `chunks/parent_child/parents-*.parquet` | `parent_id` (`{doc_id}_p{k}`), `doc_id`, `lang`, `char_start`, `char_end`, `text`, `n_tokens` |
| `emb/<strategy>/shard-XXXX/` | `dense.npy` (float16, đã chuẩn hoá), `sparse.npz` (CSR float16, đọc bằng `mir.encode.load_sparse`), `chunk_ids.parquet`, `_DONE.json` (kèm dấu vân tay `chunks_fp` của bản chunks) |
| `index/<strategy>/` | `dense.faiss` + `dense_ids.parquet`; `bm25/shard-*.npz` (CSC) + `vocab.parquet` + `stats.json`; `lsr/` (tuỳ chọn) |
| `runs/<tên>.parquet` | `qid`, `chunk_id`, `doc_id`, `rank`, `score` (+ `<tên>.json` cấu hình) |
| `qrels.parquet` | `qid`, `doc_id`, `chunk_text` (có thể rỗng). Xem thêm định dạng nhãn ở `docs/t7_danh_gia.md`. |
| `submissions/<tên>.zip` | đúng 1 file `<tên>.json` ở gốc |

## 8. Lỗi thường gặp

| Hiện tượng | Cách xử lý |
|---|---|
| `ModuleNotFoundError` dù đã cài thư viện (Windows) | Lệnh đang dùng một bản Python khác. Gõ `py` thay cho `python`. |
| `UnicodeEncodeError` khi in tiếng Việt | PowerShell: `$env:PYTHONIOENCODING = "utf-8"`. |
| Notebook báo `Không tìm thấy '…' trong /kaggle/input` | Chưa *Add Input* dataset tương ứng. |
| Notebook chạy code cũ | So dòng `mir-code …` ở ô (1) với dòng `VERSION` lúc đóng gói. Tạo New Version cho `mir-code` và cập nhật version trong panel *Input*. |
| T3/T5 báo `bản chunks khác nhau` | Các khối emb được mã hoá từ các version `mir-data` khác nhau. Mã hoá lại phần bị lệch bằng đúng version chung. |
| `[CẢNH BÁO] thiếu N khối emb` | Thiếu dataset `mir-emb-m*`, hoặc có người chưa mã hoá xong. Xem ô (4) của notebook T3. |
| `… đã có (dùng --force …)` | Lệnh không ghi đè kết quả đã có. Đổi tên (`--name`) hoặc thêm `--force`. |
| Output Kaggle vượt ~20 GB | Xoá dữ liệu tạm trước khi lưu (T4 đã tự xoá `bm25/tf/`). |
