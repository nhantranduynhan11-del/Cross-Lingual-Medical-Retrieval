"""T0: tạo tập mẫu cố định sample/ (phân tầng theo tên miền). Không ghi đè nếu đã có, trừ khi --force."""
import argparse
import sys
import json
import random
from collections import defaultdict

import pyarrow as pa
import pyarrow.parquet as pq

from . import config
from .crawlio import iter_docs


def allocate(sizes, n_total, min_per):
    """Chia n_total cho từng tầng: tối thiểu min_per (nếu tầng đủ bài), phần còn lại theo tỷ lệ."""
    alloc = {h: min(s, min_per) for h, s in sizes.items()}
    left = n_total - sum(alloc.values())
    if left > 0:
        room = {h: sizes[h] - alloc[h] for h in sizes}
        tot = sum(room.values())
        if tot > 0:
            for h in sizes:
                alloc[h] += min(room[h], int(left * room[h] / tot))
            left = n_total - sum(alloc.values())      # phần dư do làm tròn
            for h in sorted(sizes, key=lambda x: -sizes[x]):
                if left <= 0:
                    break
                add = min(left, sizes[h] - alloc[h])
                alloc[h] += add
                left -= add
    return alloc


def build(cfg, force=False):
    sc = cfg["sample"]
    out = cfg.work_dir / sc["dir"]
    docs_p, q_p = out / "corpus_raw.parquet", out / "query.parquet"
    if docs_p.exists() and not force:
        print(f"{docs_p} đã tồn tại → bỏ qua (dùng --force để tạo lại)")
        return
    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(cfg["seed"])
    by_host, seen = defaultdict(list), set()
    for d in iter_docs(cfg.crawl_path):
        if d["id"] in seen:
            continue
        seen.add(d["id"])
        by_host[d["host"]].append(d)
    sizes = {h: len(v) for h, v in by_host.items()}
    alloc = allocate(sizes, sc["n_docs"], sc["min_per_host"])
    chosen = []
    for h in sorted(by_host):
        v = sorted(by_host[h], key=lambda d: d["id"])
        chosen += rng.sample(v, alloc[h])
    chosen.sort(key=lambda d: d["id"])
    cols = ["id", "url", "host", "lang", "title", "text"]
    pq.write_table(pa.table({c: [d[c] for d in chosen] for c in cols}), docs_p)

    qt = pq.read_table(cfg.queries_path).to_pydict()
    idx = sorted(rng.sample(range(len(qt["id"])), sc["n_queries"]))
    pq.write_table(pa.table({"id": [qt["id"][i] for i in idx], "query": [qt["query"][i] for i in idx]}), q_p)

    (out / "sample_meta.json").write_text(json.dumps({
        "seed": cfg["seed"], "n_docs": len(chosen), "n_queries": len(idx),
        "docs_available_at_sampling": len(seen), "per_host": {h: alloc[h] for h in sorted(alloc)},
        "config": dict(sc)}, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"Mẫu: {len(chosen)} bài ({len(alloc)} tên miền) từ {len(seen)} bài có sẵn, {len(idx)} câu hỏi → {out}")
    if len(seen) < sc["n_docs"]:
        print(f"[LƯU Ý] kho crawl mới có {len(seen)} bài < {sc['n_docs']}: mẫu lấy toàn bộ; nên tạo lại khi crawl nhiều hơn.")


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/baseline.yaml")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    build(config.load(a.config), a.force)


if __name__ == "__main__":
    main()
