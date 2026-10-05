"""Chạy bộ trích chữ của crawl.py trên crawl_eval/html/ và viết báo cáo theo tên miền.

    py crawl_eval/report.py                       # -> crawl_eval/report/summary.tsv, report/<host>.txt
    py crawl_eval/report.py --hosts cnkang.com    # chỉ vài tên miền, in ra màn hình
    py crawl_eval/report.py --compare old.py      # so với một phiên bản crawl.py khác

Dấu hiệu cần xem: dòng lặp ở >= 50% số trang của một tên miền (thường là giao diện),
tỷ lệ dòng rất ngắn cao (chữ bị tách vụn), bài quá ngắn, tiêu đề trùng nhau giữa các trang.
"""
import argparse
import collections
import gzip
import importlib.util
import json
import os
import re
import sys
from concurrent.futures import ProcessPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


_MOD = {}


def _extract(args):
    path, host, mod_path = args
    if mod_path not in _MOD:
        _MOD[mod_path] = load_module(mod_path, "crawl_" + str(len(_MOD)))
    body = gzip.open(path).read()
    try:
        return _MOD[mod_path].extract(body, host)
    except Exception as e:
        return ("", f"[LỖI TRÍCH] {type(e).__name__}: {e}", "?")


def run(mod_path, items, cpu):
    with ProcessPoolExecutor(cpu) as ex:
        return list(ex.map(_extract, [(p, h, mod_path) for p, h, _ in items], chunksize=4))


def host_report(host, rows):
    """rows: [(id, url, title, text, lang)] → (dòng tóm tắt, nội dung chi tiết)."""
    n = len(rows)
    df = collections.Counter()
    for r in rows:
        df.update(set(r[3].split("\n")))
    rep = [(l, c) for l, c in df.most_common() if c >= max(2, 0.5 * n)]
    lens = sorted(len(r[3]) for r in rows)
    lines = [l for r in rows for l in r[3].split("\n") if l]
    short = sum(1 for l in lines if len(l) <= 4) / max(1, len(lines))
    tdup = n - len({r[2] for r in rows})
    langs = collections.Counter(r[4] for r in rows)
    summ = {"host": host, "n": n, "len_p50": lens[n // 2] if n else 0, "len_min": lens[0] if n else 0,
            "n_lt100": sum(1 for x in lens if x < 100), "short_line%": round(100 * short, 1),
            "rep_lines": len(rep), "title_dup": tdup, "langs": dict(langs)}
    out = [f"===== {host}  ({n} trang)  {json.dumps(summ, ensure_ascii=False)}",
           "Dòng lặp ở >=50% số trang:"] + [f"   [{c}/{n}] {l[:120]}" for l, c in rep[:40]]
    for i, u, t, x, lg in rows:
        ls = x.split("\n")
        out += ["", f"--- {i} {u}", f"    title: {t!r}  | {len(x)} ký tự, {len(ls)} dòng, lang={lg}"]
        out += [f"    | {l[:150]}" for l in ls[:8]]
        if len(ls) > 14:
            out.append("    | ...")
        out += [f"    | {l[:150]}" for l in ls[max(8, len(ls) - 6):]]
    return summ, "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--crawl", default=os.path.join(ROOT, "crawl.py"))
    ap.add_argument("--compare", default="", help="crawl.py phiên bản khác để so sánh")
    ap.add_argument("--hosts", default="")
    ap.add_argument("--cpu", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    a = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    idx = {}
    for line in open(os.path.join(HERE, "index.jsonl"), encoding="utf-8"):
        r = json.loads(line)
        idx[r["id"]] = r
    only = [h.strip().lower() for h in a.hosts.split(",") if h.strip()]
    items = []
    for f in sorted(os.listdir(os.path.join(HERE, "html"))):
        host, i = f[:-8].split("__")
        i = int(i)
        if only and not any(host == x or host.endswith("." + x) for x in only):
            continue
        items.append((os.path.join(HERE, "html", f), host, i))
    res = run(os.path.abspath(a.crawl), items, a.cpu)
    old = run(os.path.abspath(a.compare), items, a.cpu) if a.compare else None
    by = collections.defaultdict(list)
    for k, ((p, h, i), (t, x, lg)) in enumerate(zip(items, res)):
        by[h].append((i, idx.get(i, {}).get("url", ""), t, x, lg))
    os.makedirs(os.path.join(HERE, "report"), exist_ok=True)
    summs = []
    for h in sorted(by, key=lambda h: -len(by[h])):
        s, txt = host_report(h, sorted(by[h]))
        summs.append(s)
        if only:
            print(txt + "\n")
        with open(os.path.join(HERE, "report", re.sub(r"[^A-Za-z0-9.\-]", "_", h) + ".txt"), "w", encoding="utf-8") as f:
            f.write(txt + "\n")
    cols = list(summs[0].keys())
    with open(os.path.join(HERE, "report", "summary.tsv"), "w", encoding="utf-8") as f:
        f.write("\t".join(cols) + "\n")
        for s in summs:
            f.write("\t".join(str(s[c]) for c in cols) + "\n")
    if not only:
        for s in summs:
            print("\t".join(str(s[c]) for c in cols))
    if old is not None:
        diff = [(it, o, n) for it, o, n in zip(items, old, res) if o != n]
        print(f"\nSo với {a.compare}: {len(items) - len(diff)} giống, {len(diff)} khác")
        cnt = collections.Counter(it[1] for it, _, _ in diff)
        for h, c in cnt.most_common():
            print(f"   {h}: {c}")


if __name__ == "__main__":
    main()
