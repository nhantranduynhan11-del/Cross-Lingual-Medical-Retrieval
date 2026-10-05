"""Kiểm tra --save-raw/--reextract: trích chữ từ HTML gọn (strip_html) phải giống hệt trích từ HTML gốc.

    py crawl_eval/check_raw.py
"""
import gzip
import os
import sys
from concurrent.futures import ProcessPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import crawl  # noqa: E402


def one(f):
    host = f[:-8].split("__")[0]
    body = gzip.open(os.path.join(HERE, "html", f)).read()
    t1, x1, l1, raw = crawl.process_page(body, host, want_raw=True)
    t2, x2, l2, _ = crawl.process_page(raw, host)
    return f, (t1, x1, l1) == (t2, x2, l2), len(body), len(raw.encode("utf-8")), len(gzip.compress(raw.encode("utf-8"))), x1, x2


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    files = sorted(os.listdir(os.path.join(HERE, "html")))
    with ProcessPoolExecutor(6) as ex:
        res = list(ex.map(one, files, chunksize=4))
    bad = [r for r in res if not r[1]]
    print(f"{len(res)} trang: {len(res) - len(bad)} giống hệt, {len(bad)} khác")
    for f, _, _, _, _, x1, x2 in bad[:15]:
        i = next((k for k, (a, b) in enumerate(zip(x1, x2)) if a != b), min(len(x1), len(x2)))
        print(f"  {f}: len {len(x1)} vs {len(x2)} | gốc …{x1[max(0, i - 30):i + 50]!r} | gọn …{x2[max(0, i - 30):i + 50]!r}")
    gz = sum(r[4] for r in res) / len(res)
    print(f"HTML gọn nén gzip: trung bình {gz / 1024:.1f} KB/trang (HTML gốc chưa nén {sum(r[2] for r in res) / len(res) / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
