"""So bộ trích chữ của crawl.py hiện tại với một phiên bản cũ trên toàn bộ crawl_eval/html/.

    py crawl_eval/compare_versions.py <crawl_cu.py>

In thống kê giống/khác theo nhóm (tên miền có quy tắc / không có quy tắc) và liệt kê mọi trang KHÔNG có quy tắc
mà chữ thay đổi — đó là chỗ cần xem (thay đổi ngoài ý muốn ở bộ trích chung).
"""
import collections
import gzip
import importlib.util
import os
import sys
from concurrent.futures import ProcessPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
_OLD = {}


def _load(path):
    if path not in _OLD:
        src = open(path, encoding="utf-8").read().replace("\nimport aiohttp\n", "\n")   # bản cũ import aiohttp ở đầu file
        m = importlib.util.module_from_spec(importlib.util.spec_from_loader("crawl_old", loader=None))
        exec(compile(src, path, "exec"), m.__dict__)
        _OLD[path] = m
    return _OLD[path]


def one(args):
    f, old_path = args
    import crawl
    host = f[:-8].split("__")[0]
    b = gzip.open(os.path.join(HERE, "html", f)).read()
    return host, f, _load(old_path).extract(b, host), crawl.extract(b, host), crawl.find_rule(host) is not None


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    old_path = os.path.abspath(sys.argv[1])
    files = sorted(os.listdir(os.path.join(HERE, "html")))
    with ProcessPoolExecutor(6) as ex:
        res = list(ex.map(one, [(f, old_path) for f in files], chunksize=4))
    st = collections.defaultdict(collections.Counter)
    for h, f, a, c, r in res:
        k = "có quy tắc" if r else "không quy tắc"
        st[k]["trang"] += 1
        st[k]["chữ giống"] += a[1] == c[1]
        st[k]["tiêu đề giống"] += a[0] == c[0]
        st[k]["ngôn ngữ giống"] += a[2] == c[2]
    for k, v in st.items():
        print(k, dict(v))
    print("\nTrang KHÔNG có quy tắc nhưng chữ thay đổi:")
    for h, f, a, c, r in res:
        if not r and a[1] != c[1]:
            A, B = a[1], c[1]
            i = next((k for k, (x, y) in enumerate(zip(A, B)) if x != y), min(len(A), len(B)))
            print(f"  {f}: {len(A)} -> {len(B)} | cũ …{A[max(0, i - 25):i + 45]!r} | mới …{B[max(0, i - 25):i + 45]!r}")
    print("\nNgôn ngữ thay đổi:")
    for h, f, a, c, r in res:
        if a[2] != c[2]:
            print(f"  {f}: {a[2]} -> {c[2]} | {c[1][:80]!r}")


if __name__ == "__main__":
    main()
