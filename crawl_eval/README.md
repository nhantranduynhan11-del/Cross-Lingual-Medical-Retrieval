# crawl_eval — kiểm chứng bộ trích chữ của crawl.py

Bộ HTML mẫu và công cụ để viết, sửa và kiểm tra quy tắc trích chữ theo tên miền (`SITE_RULES` trong `crawl.py`).

## Dữ liệu

- `html/<tên miền>__<id>.html.gz`: HTML gốc, khoảng 1.000 trang thuộc 85 tên miền.
  - Mỗi tên miền lấy 6–19 trang, rải theo các kiểu đường dẫn khác nhau, cả trang cũ lẫn trang mới.
  - Tạo bằng `fetch_eval.py`. Dùng lại cho mọi lần kiểm tra mà không cần gọi mạng.
- `index.jsonl`: id, url, tên miền, trạng thái tải.
- `crawl_v1.py`: bản `crawl.py` trước ngày 3/10/2026, dùng làm mốc so sánh.

## Công cụ

| Lệnh | Việc |
|---|---|
| `py crawl_eval/fetch_eval.py [--hosts a.com]` | Tải thêm trang mẫu, bỏ qua trang đã có. |
| `py crawl_eval/report.py [--hosts a.com]` | Trích chữ toàn bộ, viết `report/<tên miền>.txt` gồm dòng đầu/cuối và các dòng lặp ≥ 50% số trang. |
| `py crawl_eval/eval_rules.py [--hosts a.com]` | Mỗi quy tắc so với trafilatura: số trang không khớp, số ký tự "mất" (có thể mất nội dung) và "thêm". |
| `py crawl_eval/induce.py --hosts a.com` | Gợi ý vùng nội dung chính (XPath) cho một tên miền mới. |
| `py crawl_eval/dom.py <id> [--root XPATH]` | In cây HTML kèm chữ để viết XPath. |
| `py crawl_eval/cmp_rule.py <host> --keep XPATH ...` | Thử một quy tắc chưa đưa vào `crawl.py`. |
| `py crawl_eval/check_raw.py` | Trích từ HTML gọn (`--save-raw`) phải giống hệt trích từ HTML gốc. |
| `py crawl_eval/compare_versions.py crawl_eval/crawl_v1.py` | So với bản cũ. Tên miền không có quy tắc phải không đổi. |

## Khi sửa hoặc thêm quy tắc

1. Viết hoặc sửa quy tắc trong `SITE_RULES` của `crawl.py`.
2. Tăng `EXTRACT_VERSION` lên 1. Đặt `"since"` của các quy tắc vừa đổi bằng số mới.
3. Chạy lần lượt và đọc kết quả:
   - `eval_rules.py --hosts <tên miền>`
   - `check_raw.py`
   - `compare_versions.py`
   - `py -m pytest -q tests`
4. Cập nhật dữ liệu đã crawl:
   - Máy có `out/raw/` (đã chạy `--save-raw`): `python crawl.py --out out --reextract stale`. Không cần mạng.
   - Máy không có `out/raw/`: `python crawl.py --out out --shard i/3 --redo-stale ...`. Phải tải lại.
