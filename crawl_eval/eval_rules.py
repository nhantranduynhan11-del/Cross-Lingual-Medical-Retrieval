"""Kiểm mọi quy tắc SITE_RULES của crawl.py trên crawl_eval/html/, so với trafilatura.

    py crawl_eval/eval_rules.py [--hosts a.com,b.vn] [--show 3]

Mỗi tên miền có quy tắc: số trang, số trang quy tắc không khớp (rơi về trafilatura),
"mất" = ký tự trong các dòng trafilatura không tìm thấy trong kết quả quy tắc (có thể mất nội dung → phải xem),
"thêm" = ký tự trong các dòng quy tắc không có trong trafilatura (nội dung trafilatura bỏ sót, hoặc giao diện → phải xem).
So sánh sau khi bỏ khoảng trắng và ký hiệu trình bày ("- ", "| "...), nên khác biệt định dạng không bị tính.
"""
import argparse
import collections
import gzip
import os
import re
import sys
from concurrent.futures import ProcessPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import crawl  # noqa: E402

_STRIP = re.compile(r"[\s|*#>\-–•·]+")


def n(s):
    return _STRIP.sub("", s)


def one(args):
    path, host = args
    import lxml.html
    import trafilatura
    src = crawl._XML_DECL.sub("", crawl.decode_html(gzip.open(path).read()), count=1)
    try:
        doc = lxml.html.document_fromstring(src)
    except Exception:
        return host, os.path.basename(path), None
    tr = crawl._lines(trafilatura.extract(src, include_comments=False, include_tables=True,
                                          favor_recall=True, deduplicate=False))
    rule = crawl.find_rule(host)
    try:
        _, ru = crawl._site_extract(doc, rule)
    except Exception as e:
        return host, os.path.basename(path), ("ERR", str(e))
    final = crawl.extract(gzip.open(path).read(), host)[1]
    matched = len(ru) >= rule.get("min_chars", 30)
    if not matched:
        return host, os.path.basename(path), ("NOMATCH", len(tr), len(final))
    nr, nt = n(ru), n(tr)
    sh_r, sh_t = shingles(nr), shingles(nt)
    miss = [l for l in tr.split("\n") if len(n(l)) >= 4 and n(l) not in nr and cover(n(l), sh_r) < 0.8]
    extra = [l for l in ru.split("\n") if len(n(l)) >= 4 and n(l) not in nt and cover(n(l), sh_t) < 0.8]
    return host, os.path.basename(path), ("OK", len(tr), len(ru), miss, extra)


K = 8


def shingles(s):
    return {s[i:i + K] for i in range(max(1, len(s) - K + 1))}


def cover(line, sh):
    """Tỷ lệ cụm K ký tự của dòng có mặt ở bên kia (khác cách ngắt dòng/khoảng trắng không bị tính là mất)."""
    if len(line) < K:
        return 1.0 if line in "".join(sh) else 0.0
    s = shingles(line)
    return len(s & sh) / len(s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hosts", default="")
    ap.add_argument("--show", type=int, default=3)
    ap.add_argument("--cpu", type=int, default=6)
    a = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    only = [h.strip() for h in a.hosts.split(",") if h.strip()]
    items = []
    for f in sorted(os.listdir(os.path.join(HERE, "html"))):
        host = f[:-8].split("__")[0]
        if crawl.find_rule(host) is None:
            continue
        if only and not any(host == x or host.endswith("." + x) for x in only):
            continue
        items.append((os.path.join(HERE, "html", f), host))
    with ProcessPoolExecutor(a.cpu) as ex:
        res = list(ex.map(one, items, chunksize=2))
    by = collections.defaultdict(list)
    for h, f, r in res:
        by[h].append((f, r))
    for h in sorted(by, key=lambda h: -len(by[h])):
        rows = by[h]
        nomatch = [f for f, r in rows if r and r[0] == "NOMATCH"]
        errs = [(f, r) for f, r in rows if r and r[0] == "ERR"]
        ok = [(f, r) for f, r in rows if r and r[0] == "OK"]
        miss = sum(sum(len(l) for l in r[3]) for _, r in ok)
        extra = sum(sum(len(l) for l in r[4]) for _, r in ok)
        tr = sum(r[1] for _, r in ok) or 1
        print(f"\n### {h}: {len(rows)} trang | không khớp {len(nomatch)} | lỗi {len(errs)} | "
              f"mất {miss} ({100 * miss / tr:.1f}%) | thêm {extra} ({100 * extra / tr:.1f}%)")
        for f in nomatch:
            print(f"   KHÔNG KHỚP: {f}")
        for f, r in errs:
            print(f"   LỖI: {f} {r[1][:200]}")
        ml = sorted(((l, f) for f, r in ok for l in r[3]), key=lambda x: -len(x[0]))[:a.show]
        el = sorted(((l, f) for f, r in ok for l in r[4]), key=lambda x: -len(x[0]))[:a.show]
        for l, f in ml:
            print(f"   - mất  [{f.split('__')[1][:-8]}] {l[:150]}")
        for l, f in el:
            print(f"   + thêm [{f.split('__')[1][:-8]}] {l[:150]}")


if __name__ == "__main__":
    main()
