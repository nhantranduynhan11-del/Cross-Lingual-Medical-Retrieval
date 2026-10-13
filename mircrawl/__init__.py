"""mircrawl — thu thập và trích xuất kho tài liệu ViBioMIR.

Tiếng Việt / tiếng Trung: tải trang theo URL của BTC rồi trích phần nội dung chính.
Tiếng Anh: URL PubMed → lấy tiêu đề + tóm tắt qua NCBI E-utilities (theo lô 200 PMID), không tải trang HTML.
"""
__version__ = "1.0.0"

# Phiên bản bộ trích xuất, ghi vào cột "xv" của mỗi bản ghi.
# Đổi quy tắc trích → tăng số này; `python -m mircrawl reextract` trích lại từ HTML đã lưu (không cần mạng).
XV = 1
