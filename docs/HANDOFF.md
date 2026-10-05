# Bàn giao phiên làm việc — ViBioMIR baseline v1 (cập nhật 05/10/2026, sau phiên dọn dẹp)

Đọc theo thứ tự khi mở phiên mới:
1. `docs/HANDOFF.md` (file này).
2. `README.md`: cách chạy trên máy cá nhân và trên Kaggle.
3. `docs/NOTES.md`: mọi quyết định và số đo.
4. `docs/BUILD_PROMPT.md`: đặc tả gốc và quy tắc làm việc.
5. `docs/t7_danh_gia.md`: việc T7, đã giao cho một thành viên.

## 1. Trạng thái

| Phần | Trạng thái |
|---|---|
| Crawl | Đang làm, 3 máy, mỗi máy một shard. Máy người duyệt (shard 0/3): 107.724 URL đã ghi trạng thái (104.330 `ok`). Lần ghi cuối lúc 05:27 ngày 05/10; lúc bàn giao không có tiến trình crawl nào chạy. Máy này **không có `out/raw/`**, tức crawl chưa bật `--save-raw`. |
| T0–T2 | Xong trên 10k URL. `results/clean_lines.json` là bản đã duyệt trên 10k. Kho đầy đủ: chạy lại `--stats --force`, duyệt, `--apply --force`, rồi `mir.chunk`. |
| T3 | Đã kiểm trên Kaggle: khớp FlagEmbedding, 185 đoạn/s trên T4 ×2. Toàn kho ước tính 13–15 giờ GPU, khoảng 5 giờ mỗi người. |
| T4–T6, T8 | Code xong. Chạy được đầu-cuối trên Kaggle với kho mẫu. T6 cho 1.200 câu: 27,7 phút. |
| T7 | Code đánh giá xong. **Chưa có nhãn thật.** Đã giao cho một thành viên; người đó tự viết công cụ theo `docs/t7_danh_gia.md`. |
| Kiểm thử | `py -m pytest -q tests`: 65 bài qua, khoảng 1–2 phút. |

## 2. Phiên 05/10 (dọn dẹp) đã làm

- **Cấu trúc:**
  - `BUILD_PROMPT.md`, `NOTES.md`, `HANDOFF.md` chuyển vào `docs/`.
  - File đề bài đổi tên thành `docs/de_bai.md`.
  - `clean_lines.json` chuyển vào `results/` (`clean.lines_file`).
- **`mir/`:**
  - Đường dẫn thư mục chuẩn gom vào `cfg.path(...)`; đường dẫn trong cấu hình là đường dẫn tương đối theo repo, thay cho `E:/...`.
  - `store.unit_rows` dùng chung cho evaluate/submit/rerank.
  - Bỏ code thừa; sửa docstring sai.
  - Dấu vân tay `chunks_fp` trong `_DONE.json` của T3.
  - `mir.chunk` chia lại part đã cũ hơn `corpus_clean`.
  - `mir.rerank` báo lỗi rõ khi run có câu hỏi không nằm trong file câu hỏi.
  - CLI và định dạng file **không đổi**.
- **Notebook:**
  - 5 notebook dùng chung ô (1) (`mir/kaggle_nb.py`): chép lại code, in `VERSION.txt`, tự tìm dataset, ghi đè cấu hình bằng dict, không symlink.
  - e2e có thêm `CORPUS = 'full'` để chạy trên phần kho đã crawl.
  - T3 tự chép các khối đã xong khi chạy lại.
  - Bản cũ (e2e-v3, đã chạy được trên Kaggle) còn ở `kaggle/e2e_test_kaggle.ipynb`.
- **`tools/pack_kaggle.py`:** đóng gói `mir-code` / `mir-data` kèm `VERSION.txt`. Lần đóng gói cuối được ghi ở mục 5.
- **`crawl.py`:**
  - Mặc định `--input` là `Data/links_corpus.parquet` cạnh `crawl.py`.
  - Sửa lỗi `py crawl.py --help`: dấu `%` trong chuỗi trợ giúp làm argparse báo lỗi.
  - Logic crawl và trích chữ không đổi.
- **Tài liệu:** README viết lại (máy cá nhân, Kaggle, chia việc, lỗi thường gặp); thêm `docs/t7_danh_gia.md`.

## 3. Việc còn mở (cần người duyệt quyết định)

1. **T7:** các câu ở mục 9 của `docs/t7_danh_gia.md`: ai đọc tài liệu tiếng Trung, quy mô, có dùng LLM mở chấm sơ bộ không.
2. **Hộp bách khoa bệnh ở ask.39.net** đang bị bỏ khi trích. Chờ xác nhận giữ hay bỏ.
3. **`k_doc` = `k_chunk` = 10** là giá trị tạm.
4. **`nlist`/`nprobe`** cho toàn kho chưa đo. Chạy ô (3) của notebook T5.
5. **`--save-raw` trên máy shard 0:** chưa bật. Nếu sau này sửa quy tắc trích chữ, shard này phải tải lại (`--redo-stale`) thay vì trích lại từ HTML đã lưu.

## 4. Việc tiếp theo

1. Tạo New Version cho `mir-code` bằng `kaggle/mir-code/` (đã đóng gói) và import lại các notebook. Chạy
   `e2e_test_kaggle` (`CORPUS='sample'`) một lần để xác nhận các notebook mới chạy được trên Kaggle thật.
2. Crawl xong 3 shard → `--retry-failed` → gộp `out/` về một máy (README mục 3.2).
3. T1, T2 trên toàn kho (README mục 3.3) → `py tools/pack_kaggle.py data` → New Version `mir-data`.
4. T3 trên 3 tài khoản (`MEMBER = 0/1/2`), song song là T4 BM25. Sau đó T5, rồi T6–T8.
5. T7 chạy song song từ bây giờ (`docs/t7_danh_gia.md`).

## 5. Git / Kaggle

- Repo có một commit "Baseline v1". Người dùng thường chỉ push `crawl.py mir configs notebooks requirements.txt README.md .gitignore`.
- **Chưa commit:** mọi thay đổi của phiên 05/10, cộng các thay đổi chưa commit từ trước:
  - `configs/baseline.yaml`: `batch_tokens` 16384;
  - `mir/encode.py`: SDPA, đo bộ nhớ đúng GPU.
- Đề xuất đưa thêm vào git: `tests/`, `docs/`, `tools/`, `results/` (báo cáo nhỏ, có `clean_lines.json` đã duyệt), `crawl_eval/*.py` và `crawl_eval/README.md` (`crawl_eval/html/` đã nằm trong .gitignore).
- `kaggle/` (trong .gitignore) do `tools/pack_kaggle.py` tạo lại:
  - `kaggle/mir-code`: `mir/`, `configs/`, `VERSION.txt`. Lần đóng gói cuối:
    `05/10/2026 13:03 | git 26438d1 + sửa chưa commit | mã 717bdc8891`. Nếu commit trước khi tải lên thì đóng gói lại
    để dòng git đúng; mã nội dung không đổi;
  - `kaggle/mir-data`: `Data/`, `sample/`, kho 10k (`corpus_clean/`, `chunks/structure/`, 22.849 đoạn, dấu vân tay `d24d48bca444`).

  Các file được tạo bằng hard link, không tốn thêm dung lượng.

## 6. Lưu ý môi trường (Windows của người dùng)

- **Python:** dùng `py`, không dùng `python`. `python` trỏ tới bản msys64 không có thư viện. Python 3.13.
- **Mã hoá console:** console dùng cp1252, nên khi in tiếng Việt cần `PYTHONIOENCODING=utf-8`. Các module `mir` đã tự đặt utf-8 cho stdout.
- **Sửa code có dấu `\`:** sửa qua heredoc trong Bash có thể làm hỏng chuỗi `"\n"`. Dùng công cụ Edit/Write.
- **Tokenizer** BGE-M3 đã có trong cache HF. Trọng số model **không** tải về máy. Không có GPU: việc nặng chạy trên Kaggle T4 ×2.
- **Chạy thử notebook ngoài Kaggle:** đặt `KAGGLE_ROOT` tới thư mục giả lập có `input/mir-code`, `input/mir-data`, `working/`. Các ô cần GPU sẽ lỗi.

## 7. Cách làm việc người dùng yêu cầu

- Trả lời bằng tiếng Việt.
- Hỏi trước khi bắt đầu mỗi task.
- Báo cáo gồm: file đã tạo/sửa, lệnh chạy lại, số đo thật, điểm chưa chắc. Số chưa đo thì ghi "chưa đo".
- Không đổi kiến trúc đã chốt ở `docs/BUILD_PROMPT.md` mục 5. Đề xuất khác ghi vào `docs/NOTES.md`, mục "Đề xuất sau baseline".
- Chỉ dùng model mở ≤ 14B, phát hành trước 1/6/2026. Không gọi API model đóng. Không tự sinh nhãn rồi báo như số thật.
