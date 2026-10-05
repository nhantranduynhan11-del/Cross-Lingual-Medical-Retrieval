"""In cây HTML (thẻ, id, class, chữ đầu) của một trang trong crawl_eval/html/ — để viết XPath cho SITE_RULES.

    py crawl_eval/dom.py <id> [--root XPATH] [--depth 8] [--min 20]
--min: chỉ in nhánh có ít nhất N ký tự chữ (ẩn các nhánh rỗng / icon).
"""
import argparse
import gzip
import glob
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import crawl  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("id")
    ap.add_argument("--root", default="//body")
    ap.add_argument("--depth", type=int, default=8)
    ap.add_argument("--min", type=int, default=20)
    a = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    import lxml.html
    f = glob.glob(os.path.join(HERE, "html", f"*__{a.id}.html.gz"))[0]
    doc = lxml.html.document_fromstring(crawl._XML_DECL.sub("", crawl.decode_html(gzip.open(f).read()), count=1))
    for bad in doc.xpath("//script|//style|//noscript"):
        if bad.getparent() is not None:
            bad.drop_tree()

    def show(el, d):
        if not isinstance(el.tag, str) or d > a.depth:
            return
        n = len(re.sub(r"\s+", "", el.text_content() or ""))
        if n < a.min:
            return
        own = re.sub(r"\s+", " ", (el.text or "")).strip()[:50]
        attrs = " ".join(f'{k}="{v[:40]}"' for k, v in el.attrib.items() if k in ("id", "class"))
        print(f"{'  ' * d}<{el.tag} {attrs}> [{n}] {own}")
        for c in el:
            show(c, d + 1)

    for r in doc.xpath(a.root):
        show(r, 0)


if __name__ == "__main__":
    main()
