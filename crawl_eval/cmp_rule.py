"""So kết quả của một quy tắc keep/drop (thử nghiệm) với trafilatura trên các trang của một tên miền.

    py crawl_eval/cmp_rule.py <host> --keep XPATH [--keep XPATH2 ...] [--drop XPATH] [--first]
In mỗi trang: độ dài 2 bên, các dòng CHỈ có trong trafilatura (quy tắc có thể làm mất nội dung)
và CHỈ có trong quy tắc (quy tắc lấy thêm — nội dung trafilatura bỏ sót, hoặc giao diện).
"""
import argparse
import gzip
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import crawl  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("host")
    ap.add_argument("--keep", action="append", default=[])
    ap.add_argument("--drop", default=None)
    ap.add_argument("--first", action="store_true")
    ap.add_argument("--stop-re", default=None)
    ap.add_argument("--cut-re", default=None)
    ap.add_argument("--show", type=int, default=6, help="số dòng khác biệt in ra mỗi bên")
    a = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    import lxml.html
    import trafilatura
    rule = {"suffix": a.host, "keep": a.keep, "drop": a.drop, "title": None, "stop": set(), "first": a.first,
            "stop_re": a.stop_re, "cut_re": a.cut_re}
    files = sorted(f for f in os.listdir(os.path.join(HERE, "html")) if f.split("__")[0] == a.host)
    tot = {"miss": 0, "extra": 0, "nomatch": 0}
    for f in files:
        src = crawl._XML_DECL.sub("", crawl.decode_html(gzip.open(os.path.join(HERE, "html", f)).read()), count=1)
        doc = lxml.html.document_fromstring(src)
        tr = crawl._lines(trafilatura.extract(src, include_comments=False, include_tables=True,
                                              favor_recall=True, deduplicate=False))
        _, ru = crawl._site_extract(doc, rule)
        trs, rus = tr.split("\n"), ru.split("\n")
        S_ru, S_tr = set(rus), set(trs)
        miss = [l for l in trs if l not in S_ru and l not in ru]
        extra = [l for l in rus if l not in S_tr and l not in tr]
        print(f"\n--- {f[:-8]}  trafilatura {len(tr)} | quy tắc {len(ru)}" + ("  [KHÔNG KHỚP]" if not ru else ""))
        if not ru:
            tot["nomatch"] += 1
            continue
        tot["miss"] += sum(map(len, miss))
        tot["extra"] += sum(map(len, extra))
        for l in miss[:a.show]:
            print(f"   - chỉ trafilatura: {l[:140]}")
        if len(miss) > a.show:
            print(f"   - ... (+{len(miss) - a.show} dòng)")
        for l in extra[:a.show]:
            print(f"   + chỉ quy tắc:     {l[:140]}")
        if len(extra) > a.show:
            print(f"   + ... (+{len(extra) - a.show} dòng)")
    print(f"\nTỔNG: {len(files)} trang, không khớp {tot['nomatch']}, ký tự chỉ-trafilatura {tot['miss']}, "
          f"ký tự chỉ-quy-tắc {tot['extra']}")


if __name__ == "__main__":
    main()
