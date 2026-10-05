"""Tìm vùng chứa nội dung chính của từng tên miền (gợi ý để viết SITE_RULES trong crawl.py).

Với mỗi trang: lấy các câu dài trong kết quả trafilatura (gần như chắc chắn là nội dung bài), tìm thẻ nhỏ nhất
chứa >= 90% số câu đó, rồi đi lên tới thẻ gần nhất có id/class → "chữ ký" vùng nội dung.
Tên miền mà một chữ ký phủ phần lớn số trang → ứng viên tốt cho quy tắc keep.

    py crawl_eval/induce.py [--hosts a.com,b.vn]
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

_N = re.compile(r"\s+")


def norm(s):
    return _N.sub("", s or "")


def signature(el):
    """XPath ngắn cho thẻ: ưu tiên id, rồi class (token đầu), đi lên tối đa 4 cấp."""
    cur = el
    for _ in range(5):
        if cur is None or not isinstance(cur.tag, str):
            break
        if cur.get("id") and not re.search(r"\d{3,}", cur.get("id")):
            return f'//{cur.tag}[@id="{cur.get("id")}"]'
        cls = (cur.get("class") or "").split()
        if cls:
            return f'//{cur.tag}[contains(concat(" ",normalize-space(@class)," ")," {cls[0]} ")]'
        if cur.tag in ("article", "main"):
            return f"//{cur.tag}"
        cur = cur.getparent()
    return "?"


def analyse(path_host):
    path, host = path_host
    import lxml.html
    import trafilatura
    body = gzip.open(path).read()
    src = crawl._XML_DECL.sub("", crawl.decode_html(body), count=1)
    try:
        doc = lxml.html.document_fromstring(src)
    except Exception:
        return host, None
    ref = crawl._lines(trafilatura.extract(src, include_comments=False, include_tables=True,
                                           favor_recall=True, deduplicate=False))
    longs = sorted({norm(l) for l in ref.split("\n") if len(l) >= 25}, key=len, reverse=True)[:12]
    if len(longs) < 2:
        return host, None
    for bad in doc.xpath("//script|//style|//noscript"):
        if bad.getparent() is not None:
            bad.drop_tree()
    best = None
    for el in doc.iter():
        if not isinstance(el.tag, str) or el.tag in ("html", "body"):
            continue
        t = norm(el.text_content())
        if len(t) < sum(len(x) for x in longs) * 0.9:
            continue
        hit = sum(1 for x in longs if x in t)
        if hit >= 0.9 * len(longs) and (best is None or len(t) < best[1]):
            best = (el, len(t))
    if best is None:
        return host, None
    el = best[0]
    return host, (signature(el), round(best[1] / max(1, len(norm(ref))), 2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hosts", default="")
    ap.add_argument("--cpu", type=int, default=4)
    a = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    only = [h.strip() for h in a.hosts.split(",") if h.strip()]
    items = []
    for f in sorted(os.listdir(os.path.join(HERE, "html"))):
        host = f[:-8].split("__")[0]
        if only and not any(host == x or host.endswith("." + x) for x in only):
            continue
        items.append((os.path.join(HERE, "html", f), host))
    with ProcessPoolExecutor(a.cpu) as ex:
        res = list(ex.map(analyse, items, chunksize=4))
    by = collections.defaultdict(list)
    for h, r in res:
        by[h].append(r)
    for h in sorted(by, key=lambda h: -len(by[h])):
        rs = by[h]
        ok = [r for r in rs if r]
        c = collections.Counter(r[0] for r in ok)
        ratio = collections.defaultdict(list)
        for s, q in ok:
            ratio[s].append(q)
        print(f"{h}  ({len(rs)} trang, {len(rs) - len(ok)} không xác định)")
        for s, n in c.most_common(4):
            rr = sorted(ratio[s])
            print(f"    {n:3d}x  {s}   (độ dài vùng / trafilatura: trung vị {rr[len(rr) // 2]})")


if __name__ == "__main__":
    main()
