"""python -m mircrawl <lệnh> …   (xem README_CRAWL.md)

  probe     thăm dò: vài URL mỗi tên miền → bảng lỗi / chất lượng trích theo tên miền (chạy đầu tiên trên Kaggle)
  crawl     crawl một shard; chạy lại = chạy tiếp; dừng êm trước --time-budget giờ
  report    thống kê theo tên miền từ các thư mục output
  merge     gộp mọi shard / phiên → corpus/*.parquet + status.parquet + report.tsv
  reextract trích lại từ HTML đã lưu (raw/) khi đổi quy tắc, không cần mạng
"""
import argparse
import os
import sys

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
if sys.platform == "win32":
    for _s in (sys.stdout, sys.stderr):
        try:
            _s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

DEFAULT_LINKS = os.path.join("data", "links_corpus_vi_en_zh.parquet")


def _crawl_args(p, probe=False):
    p.add_argument("--links", default=DEFAULT_LINKS, help="danh sách URL của BTC (.parquet cột id,url hoặc .jsonl)")
    p.add_argument("--out", default="out_probe" if probe else "out")
    p.add_argument("--shard", default="0/1", help="i/N: chỉ lấy id %% N == i (mỗi máy / notebook một shard)")
    p.add_argument("--resume", nargs="*", default=[],
                   help="thư mục output của các phiên trước (vd /kaggle/input/…): bỏ qua id đã xong, chép file sang --out")
    p.add_argument("--no-carry", action="store_true", help="không chép file của phiên trước sang --out")
    p.add_argument("--time-budget", type=float, default=0, help="số giờ tối đa của phiên (0 = không giới hạn)")
    p.add_argument("--grace", type=float, default=240, help="ngừng nhận URL mới trước hạn chót bấy nhiêu giây")
    p.add_argument("--max-attempts", type=int, default=3, help="số lần thử tối đa (qua mọi phiên) cho URL lỗi tạm thời")
    p.add_argument("--limit", type=int, default=0, help="chỉ làm N URL ngẫu nhiên (chạy thử)")
    p.add_argument("--hosts", default="", help="chỉ các tên miền này (phẩy ngăn cách, khớp cả tên miền con)")
    p.add_argument("--skip-hosts", default="", help="bỏ các tên miền này")
    p.add_argument("--probe", type=int, default=5 if probe else 0, help="số URL mỗi tên miền (×20 cho PubMed)")
    p.add_argument("--concurrency", type=int, default=128, help="tổng số kết nối đồng thời")
    p.add_argument("--per-host", type=int, default=4, help="kết nối tối đa / tên miền nhỏ")
    p.add_argument("--per-host-big", type=int, default=8, help="kết nối tối đa / tên miền lớn")
    p.add_argument("--big-host", type=int, default=5000, help="tên miền có từ bấy nhiêu URL (trong shard) là 'lớn'")
    p.add_argument("--delay", type=float, default=0.2, help="nghỉ (giây) giữa 2 lần gọi của mỗi kết nối")
    p.add_argument("--slow-scale", type=float, default=1.0, help="nhân nhịp nghỉ của các quy tắc tốc độ riêng")
    p.add_argument("--timeout", type=float, default=25)
    p.add_argument("--retries", type=int, default=2)
    p.add_argument("--fail-threshold", type=int, default=12, help="điểm lỗi liên tiếp để tạm dừng một nhóm tên miền")
    p.add_argument("--cooldown", type=float, default=300, help="giây tạm dừng lần đầu (gấp đôi mỗi lần sau)")
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1),
                   help="số process trích nội dung (mặc định: số lõi - 1)")
    p.add_argument("--no-raw", dest="save_raw", action="store_false",
                   help="không lưu HTML gọn vào raw/ (tiết kiệm ~10 KB/trang nhưng không reextract được)")
    p.add_argument("--min-chars", type=int, default=30, help="dưới số ký tự này → 'empty' (vi/en)")
    p.add_argument("--min-chars-zh", type=int, default=10, help="dưới số ký tự này → 'empty' (zh)")
    p.add_argument("--part-rows", type=int, default=5000, help="số dòng mỗi file parquet")
    p.add_argument("--flush-secs", type=float, default=600, help="ghi file ít nhất mỗi bấy nhiêu giây")
    p.add_argument("--report-secs", type=float, default=60)
    p.add_argument("--pubmed-batch", type=int, default=200, help="số PMID mỗi yêu cầu efetch")
    p.add_argument("--pubmed-rps", type=float, default=0, help="yêu cầu/giây tới NCBI (0 = tự chọn theo API key)")
    p.add_argument("--ncbi-email", default=os.environ.get("NCBI_EMAIL", ""),
                   help="email liên hệ gửi kèm yêu cầu NCBI (khuyến nghị; hoặc biến môi trường NCBI_EMAIL)")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="mircrawl", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    _crawl_args(sub.add_parser("crawl", help="crawl một shard"))
    _crawl_args(sub.add_parser("probe", help="thăm dò vài URL mỗi tên miền"), probe=True)
    p = sub.add_parser("report", help="thống kê theo tên miền")
    p.add_argument("--inputs", nargs="+", required=True)
    p.add_argument("--links", default=None, help="danh sách URL gốc (để tính URL chưa thử)")
    p.add_argument("--shard", default=None, help="i/N khi --links là cả danh sách nhưng inputs chỉ là một shard")
    p.add_argument("--out", default=None, help="ghi report.tsv vào thư mục này")
    p = sub.add_parser("merge", help="gộp mọi shard / phiên thành corpus")
    p.add_argument("--inputs", nargs="+", required=True)
    p.add_argument("--out", default="corpus_out")
    p.add_argument("--links", default=None)
    p.add_argument("--no-short", action="store_true", help="không đưa tài liệu 'empty' (chỉ có tiêu đề) vào corpus")
    p.add_argument("--boiler-min", type=int, default=20)
    p = sub.add_parser("reextract", help="trích lại từ raw/ (không cần mạng)")
    p.add_argument("--inputs", nargs="+", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--hosts", default="")
    p.add_argument("--workers", type=int, default=None)
    p.add_argument("--min-chars", type=int, default=30)
    p.add_argument("--min-chars-zh", type=int, default=10)
    a = ap.parse_args(argv)

    if a.cmd in ("crawl", "probe"):
        from .crawl import run
        run(a)
        if a.cmd == "probe":
            from .merge import report
            report([a.out], out=a.out)
            print(f"\nXem từng trang: {a.out}/pages/*.parquet (cột title, text, published, author, section…)")
    elif a.cmd == "report":
        from .merge import report
        report(a.inputs, a.links, a.shard, a.out)
    elif a.cmd == "merge":
        from .merge import merge
        merge(a.inputs, a.out, a.links, include_short=not a.no_short, boiler_min=a.boiler_min)
    elif a.cmd == "reextract":
        from .merge import reextract
        reextract(a.inputs, a.out, a.hosts, a.workers, a.min_chars, a.min_chars_zh)


if __name__ == "__main__":
    main()
