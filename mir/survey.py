"""T0: khảo sát dữ liệu crawl. python -m mir.survey --config configs/baseline.yaml"""
import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from urllib.parse import urlparse

import numpy as np
import pyarrow.parquet as pq

from . import config
from .crawlio import iter_docs, part_files, read_progress


def survey(crawl_path, links_path):
    n = 0
    hosts, langs = Counter(), Counter()
    host_lang = defaultdict(Counter)
    lens = []
    lens_by_lang = defaultdict(list)
    hashes = Counter()
    for d in iter_docs(crawl_path):                       # mỗi id đúng một lần (bản trích mới nhất)
        n += 1
        t = d.get("text", "")
        hosts[d["host"]] += 1
        langs[d["lang"]] += 1
        host_lang[d["host"]][d["lang"]] += 1
        lens.append(len(t))
        lens_by_lang[d["lang"]].append(len(t))
        hashes[hashlib.md5(t.encode("utf-8")).hexdigest()] += 1
    dup_extra = sum(c - 1 for c in hashes.values() if c > 1)

    def q(a):
        if not a:
            return {}
        a = np.asarray(a)
        return {"mean": float(a.mean()), **{f"p{p}": float(np.percentile(a, p)) for p in (5, 25, 50, 75, 95, 99)},
                "max": int(a.max())}

    links = pq.read_table(links_path).to_pydict()
    link_hosts = Counter(urlparse(u).netloc.lower() for u in links["url"])
    prog = read_progress(crawl_path)
    return {
        "n_part_files": len(part_files(crawl_path)),
        "n_docs": n,
        "lang": dict(langs),
        "hosts": dict(hosts.most_common()),
        "host_lang": {h: dict(c) for h, c in host_lang.items()},
        "len_chars": q(lens),
        "len_chars_by_lang": {k: q(v) for k, v in lens_by_lang.items()},
        "dup_extra_copies": dup_extra,
        "dup_ratio": (dup_extra / n) if n else 0.0,
        "progress_status": dict(Counter(prog.values()).most_common()),
        "n_progress": len(prog),
        "links_total": len(links["id"]),
        "links_by_host_top10": dict(link_hosts.most_common(10)),
        "links_n_hosts": len(link_hosts),
    }


def to_markdown(s):
    L = ["# T0 — Khảo sát dữ liệu crawl", ""]
    L.append(f"- File part: {s['n_part_files']}; bài đọc được: **{s['n_docs']:,}**; "
             f"URL đã ghi trạng thái: {s['n_progress']:,} / {s['links_total']:,} trong kho")
    if s["n_docs"]:
        tot = s["n_docs"]
        L.append("- Ngôn ngữ: " + ", ".join(f"{k} {v:,} ({100*v/tot:.1f}%)" for k, v in s["lang"].items()))
        L.append(f"- Bài trùng nội dung (bản sao thừa): {s['dup_extra_copies']:,} ({100*s['dup_ratio']:.2f}%)")
        L.append("- Độ dài (ký tự): " + ", ".join(f"{k}={v:,.0f}" for k, v in s["len_chars"].items()))
        for lg, qq in s["len_chars_by_lang"].items():
            L.append(f"  - {lg}: " + ", ".join(f"{k}={v:,.0f}" for k, v in qq.items()))
    L += ["", "## Trạng thái crawl", ""] + [f"- {k}: {v:,}" for k, v in s["progress_status"].items()]
    L += ["", "## Theo tên miền", "", "| tên miền | bài | tỷ lệ | ngôn ngữ |", "|---|---:|---:|---|"]
    for h, c in s["hosts"].items():
        L.append(f"| {h} | {c:,} | {100*c/max(1, s['n_docs']):.1f}% | {s['host_lang'][h]} |")
    L += ["", f"Kho có {s['links_n_hosts']} tên miền; 10 tên miền đầu theo URL: {s['links_by_host_top10']}"]
    return "\n".join(L) + "\n"


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/baseline.yaml")
    a = ap.parse_args()
    cfg = config.load(a.config)
    s = survey(cfg.crawl_path, cfg.links_path)
    out = cfg.results()
    (out / "t0_survey.json").write_text(json.dumps(s, ensure_ascii=False, indent=1), encoding="utf-8")
    md = to_markdown(s)
    (out / "t0_survey.md").write_text(md, encoding="utf-8")
    print(md)


if __name__ == "__main__":
    main()
