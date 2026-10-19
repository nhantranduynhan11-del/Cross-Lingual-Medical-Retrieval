# mircrawl — thu thập kho tài liệu ViBioMIR (bản danh sách URL mới)

Viết lại phần crawl, chạy chính trên **Kaggle CPU** (có thể chạy cả trên máy cá nhân). Đầu vào là danh sách mới
`data/links_corpus_vi_en_zh.parquet`. Bộ câu hỏi vẫn là `data/query.parquet` (1.200 câu).

| Nhóm | Số URL | Cách lấy |
|---|---:|---|
| Tiếng Việt | 714.469 | tải trang → trích nội dung chính (quy tắc riêng theo tên miền, còn lại dùng trafilatura) |
| Tiếng Trung | 240.000 | như trên |
| Tiếng Anh (PubMed) | 240.000 | **không tải HTML**: NCBI E-utilities `efetch` theo lô 200 PMID → tiêu đề + tóm tắt + metadata |

954.469 URL VI/ZH trùng hệt (cùng id, cùng URL) với danh sách cũ. 240.000 URL tiếng Anh đều có dạng
`https://pubmed.ncbi.nlm.nih.gov/<PMID>/`; PMID là `doc_id` khi nộp bài. Danh sách mới có 93 tên miền.

## 1. Chạy nhanh

```bash
pip install -r requirements-crawl.txt
python -m mircrawl probe --probe 3          # vài URL mỗi tên miền → bảng lỗi / chất lượng theo tên miền
python -m mircrawl crawl --shard 0/4 --time-budget 11.3
python -m mircrawl report --inputs out --links data/links_corpus_vi_en_zh.parquet --shard 0/4
python -m mircrawl merge --inputs out_shard0 out_shard1 out_shard2 out_shard3 --out corpus_out --links data/links_corpus_vi_en_zh.parquet
```

`python -m mircrawl <lệnh> -h` để xem mọi tham số.

## 2. Quy trình trên Kaggle

1. **Đóng gói** (máy cá nhân): `python tools/pack_kaggle_crawl.py --user <tên_kaggle>` → `build/kaggle_crawl/`
   gồm `mircrawl.zip` + `links_corpus_vi_en_zh.parquet`. Tải lên làm Kaggle Dataset (UI hoặc
   `kaggle datasets create -p build/kaggle_crawl`). Sửa code → `kaggle datasets version …`.
2. **Notebook** `notebooks/kaggle_crawl.ipynb`: *Add Input* dataset trên; *Settings* → **Internet: On**,
   Accelerator **None (CPU)**. Tuỳ chọn *Secrets*: `NCBI_API_KEY` (PubMed 10 yêu cầu/s thay vì 3), `NCBI_EMAIL`.
3. **Thăm dò** (`MODE = "probe"`, ~5–10 phút): xem tên miền nào bị chặn từ IP Kaggle (máy chủ ở Mỹ; kết quả có
   thể khác IP Việt Nam). Tên miền chặn Kaggle nhưng mở từ Việt Nam → crawl riêng trên máy cá nhân bằng
   `--hosts`, `merge` gộp được output từ mọi nơi.
4. **Crawl** (`MODE = "crawl"`): mỗi notebook một `SHARD` (`0/4` … `3/4`; N giữ nguyên ở mọi nơi). Bấm
   **Save Version → Save & Run All** (chạy nền). Script tự ngừng nhận URL mới trước `TIME_BUDGET_H` giờ và ghi
   xong mọi file.
5. **Chưa xong trong 1 phiên**: phiên mới của cùng notebook, *Add Input* → output phiên trước, chạy lại.
   Script bỏ qua URL đã xong, thử lại URL lỗi tạm thời (tối đa `--max-attempts` 3 lần qua mọi phiên) và **chép dữ
   liệu cũ sang output mới** → output phiên cuối luôn đủ dữ liệu của shard (chỉ cần Add Input bản mới nhất).
6. **Gộp** (`MODE = "merge"`): *Add Input* output cuối của mọi shard → `/kaggle/working/corpus_out/`.

Ước lượng (theo nhịp lịch sự đang đặt): nhóm chậm nhất là các kênh tin 39.net (~12,7k URL, 1 kết nối, nghỉ 2 s
vì gọi dày bị CAPTCHA) ≈ 10–11 giờ nếu một máy làm hết, và nhathuoclongchau (~80k URL, tối đa 4 kết nối vì lần
trước bị 429 hàng loạt). Với **4 shard** mỗi shard thường xong trong một phiên; 1 shard cần 2 phiên. PubMed
≈ 10–20 phút. Dung lượng đo được: ~1,6 KB/trang (bản ghi) + ~6,5 KB/trang (HTML gọn) → cả kho ≈ 8 GB, dưới giới
hạn 20 GB của `/kaggle/working` kể cả khi chỉ dùng một tài khoản (`--no-raw` nếu cần tiết kiệm).

## 3. Đầu ra

```
out/pages/<phiên>-00000.parquet   mọi URL đã thử (một id có thể có nhiều dòng: thử lại / trích lại)
out/raw/<phiên>-00000.parquet     HTML gọn (bỏ script/style; giữ JSON-LD, meta) — để reextract không cần mạng
out/runs/<phiên>.json             thống kê phiên
corpus_out/corpus/part-*.parquet  (merge) mỗi id một dòng, chỉ tài liệu dùng được
corpus_out/status.parquet         (merge) trạng thái cuối của mọi id
corpus_out/report.tsv             (merge) bảng theo tên miền: ok / ngắn / thông báo / đã gỡ / lỗi / chưa thử
```

Cột chính: `id, url, final_url, host, status, lang (vi|zh|en), doc_type, title, text, description, author,
published, modified, section, keywords, site_name, extra, extractor, xv, n_chars, text_hash, fetched_at`.
Sau merge thêm `dup_of, flags, attempts`.

- `status`: `ok` | `empty` (chữ quá ngắn, vẫn giữ tiêu đề) | `notice` (trang thông báo: 404 mềm, bài đã gỡ,
  chọn ngôn ngữ, trang xác minh) | `gone` (chuyển về trang chủ/đăng nhập) | `http_xxx` | `blocked_*` | `err_*` |
  `pm_missing` (PMID không còn) | `boilerplate` (merge: cùng nội dung lặp ở ≥20 trang một tên miền).
- `doc_type`: `qa` (hỏi đáp) | `knowledge` (bài kiến thức bệnh/triệu chứng/điều trị) | `book` (sách, cẩm nang,
  bách khoa) | `drug` (thuốc, dược liệu, bài thuốc) | `news` | `paper` (PubMed) | `other`.
- `published` / `modified`: luôn dạng `YYYY-MM-DD` (PubMed có thể chỉ `YYYY-MM`). Chỉ lấy từ nơi ghi rõ: quy tắc
  riêng, JSON-LD, thẻ meta, dòng ngày ở đầu bài ("Ngày đăng: …", "19/02/2014 08:24"). **Thiếu thì để trống,
  không đoán.** Tác giả cũng vậy (bỏ "Tác giả", "admin", tên tờ báo…).
- `extra` (JSON): trang hỏi đáp → `answers: [{author, author_title, time}]` theo thứ tự câu trả lời; PubMed →
  `pmid, journal, doi, pmcid, pub_types, mesh, language`; mọi trang → `canonical` nếu có.

### Định dạng cột `text`

Giữ cấu trúc để bước chia đoạn bám theo mục / cặp hỏi–đáp (mỗi dòng một khối, đúng thứ tự đọc):

```
## Triệu chứng          tiêu đề mục (h1–h6; đoạn ngắn toàn chữ đậm → "### ")
- Đau lưng dữ dội       mục danh sách
Loại | Tỷ lệ            hàng bảng (ô cách nhau " | ")
---                     ranh giới câu hỏi / từng câu trả lời (trang hỏi đáp)
Sỏi thận là …           đoạn văn
```

Dấu cấu trúc không thêm chữ nào của trang. `mircrawl.htmlx.to_plain(text)` bỏ chúng để lấy chữ thuần — dùng khi
tạo `chunk_text` (BTC đối chiếu chunk bằng LCS token sau chuẩn hoá NFKC/lowercase, ngưỡng 0,40 độ dài đoạn tham
chiếu). Trang hỏi đáp: dòng đầu là câu hỏi (`# …`), sau mỗi `---` là một câu trả lời, khớp thứ tự với
`extra.answers`.

Gợi ý cho bước chia đoạn: mục từ điển Đông y (zhongyibaodian, a-hospital, qihuangzhishu, zydcd) thường rất ngắn và
**tên mục chỉ có ở `title`** → ghép `title` vào đoạn khi lập chỉ mục. Tài liệu `dup_of` khác rỗng là bản trùng nội
dung: chỉ cần lập chỉ mục một bản nhưng khi trả kết quả nên trả cả nhóm id (nhãn BTC có thể trỏ vào id bất kỳ).

## 4. Xử lý theo các vấn đề của dữ liệu web

| Vấn đề | Cách xử lý |
|---|---|
| Trang gồm bài viết + menu, quảng cáo, nội dung liên quan | 34 quy tắc riêng (XPath vùng bài, bỏ phần giao diện) cho các tên miền lớn; còn lại trafilatura; HTML hỏng → lấy nhánh nhiều chữ nhất / vùng nhiều đoạn văn nhất |
| Trang chỉ là thông báo / chọn ngôn ngữ / xác minh | `fetch` nhận diện CAPTCHA, Cloudflare, trang đặt cookie bằng JS, chuyển về trang chủ; `classify.is_notice` nhận diện 404 mềm, bài đã gỡ, chọn ngôn ngữ (vi/zh/en) → `notice`; `merge` loại nội dung lặp ở ≥20 trang cùng tên miền |
| Tiêu đề / tác giả / ngày thiếu hoặc sai | Ưu tiên quy tắc riêng > JSON-LD > meta; bỏ hậu tố tên trang ("… \| Sức Khỏe Cộng Đồng"); merge bỏ tiêu đề lặp, tác giả mặc định và từ khoá SEO lặp ở ≥30% bài của một tên miền |
| Hỏi đáp: câu hỏi, câu trả lời, người trả lời | 120ask, ask.39.net: tách câu hỏi + từng câu trả lời (`---`), người trả lời / chức danh / thời gian vào `extra.answers` |
| Bài kiến thức, sách, cẩm nang | giữ tiêu đề mục, danh sách, bảng; MediaWiki: bỏ hộp điều hướng, mục lục, mục đuôi do mẫu chèn |
| Làm sạch nhưng không đổi nghĩa | chỉ chuẩn hoá NFC, khoảng trắng, ký tự độ rộng 0, thực thể HTML; không xoá dấu, từ phủ định, số, đơn vị |
| Bản trùng | `text_hash` → `dup_of` (giữ ánh xạ mọi id / URL gốc) |

## 5. Tên miền khó (đo từ IP Việt Nam, 12/10/2026 — chạy `probe` trên Kaggle để biết tình hình từ IP Kaggle)

| Tên miền | URL | Tình trạng |
|---|---:|---|
| www.familydoctor.com.cn (+ baidianfeng, qy, ask) | 26.882 | HTTP 403 |
| zysjonline.com | 6.591 | Cloudflare đòi thử thách → không vượt |
| pmc-ecm-healthblog.beta.pharmacity.io | 5.333 | HTTP 403 |
| baophutho.vn, baothanhhoa.vn | 2.070 | một phần bài bị mã hoá (giải bằng JS của trang) → `empty`, vẫn giữ tiêu đề + metadata |
| article.iiyi.com, bingli.iiyi.com | 734 | HTTP 521 / quá thời gian |
| thaythuocvietnam.vn | 2.927 | các bài thử đều chuyển về trang chủ (bài đã gỡ → `gone`) |
| www.youlai.cn | 4.139 | khoảng nửa số trang treo → chờ ngắn, phiên sau thử lại |

Script **không** vượt CAPTCHA / Cloudflare: bị chặn liên tiếp → tạm dừng nhóm tên miền 5, 10, 20 phút rồi bỏ nhóm
trong phiên; URL chưa thử để phiên sau. Bỏ hẳn một tên miền: `--skip-hosts familydoctor.com.cn,zysjonline.com`.

## 6. Kiến trúc

| Module | Việc |
|---|---|
| `fetch.py` | curl_cffi giả lập Chrome; cookie đặt bằng JS; HTTP/2 hỏng → HTTP/1.1; 429 + Retry-After; điều tốc thích ứng theo nhóm tên miền (`HostCtl`); quy tắc tốc độ riêng (`RATE_RULES`) |
| `htmlx.py` | giải mã bảng mã (GB18030 cho họ GB), phân tích HTML hỏng, làm gọn, vùng nội dung → văn bản có cấu trúc |
| `rules.py` | quy tắc trích riêng theo tên miền (vùng bài, phần bỏ, dòng bỏ, ngày, tác giả, chuyên mục, hỏi đáp) |
| `meta.py` | JSON-LD / OpenGraph / meta, chuẩn hoá ngày, tác giả, tiêu đề, breadcrumb |
| `classify.py` | ngôn ngữ (vi/zh/en), loại tài liệu, trang thông báo |
| `extract.py` | ghép tất cả cho một trang (chạy ở process con) |
| `pubmed.py` | E-utilities efetch theo lô, phân tích XML (bài báo + sách NCBI Bookshelf) |
| `store.py` | ghi parquet theo phần (an toàn khi bị ngắt), đọc trạng thái để chạy tiếp, chép dữ liệu phiên trước |
| `crawl.py` | vòng asyncio chính, process pool trích nội dung, ngân sách thời gian, báo cáo tiến độ |
| `merge.py` | gộp + kiểm tra chất lượng, báo cáo theo tên miền, trích lại từ `raw/` |

Đổi quy tắc trích: sửa `rules.py`, tăng `XV` trong `mircrawl/__init__.py`, chạy
`python -m mircrawl reextract --inputs out --out out` (không cần mạng) rồi `merge` (dòng `xv` mới thắng).

**Tận dụng từ codebase cũ** (`Cross-Lingual-Medical-Retrieval-master/crawl.py`): giải mã bảng mã, xử lý HTML hỏng,
các quy tắc trích riêng đã kiểm chứng, logic tải trang (cookie JS, HTTP/2, CAPTCHA, chuyển về trang chủ), quy tắc
tốc độ riêng (39.net, youlai), các dòng giao diện đã duyệt trong `corpus_clean/_meta.json` (force_drop). Kiểm
chứng: trên 4.358 trang HTML cũ (40 tên miền), chữ trích ra khớp bản cũ ở hầu hết tên miền (các chỗ khác đều là cải
thiện có chủ ý: bỏ form "症状自查", dòng điều hướng "详细>>", "查看更多…", tách đủ câu trả lời trùng nhau).
Trong mẫu HTML cũ, 246/300 trang baidu.com thực ra là trang "百度安全验证" bị lưu như nội dung — bản mới đánh dấu
`notice`. Ngưỡng "rỗng" hạ
từ 50 xuống 10 ký tự cho tiếng Trung để giữ các mục từ điển ngắn nhưng có nội dung.
