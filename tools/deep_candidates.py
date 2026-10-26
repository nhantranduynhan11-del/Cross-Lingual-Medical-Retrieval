"""Ứng viên sâu: tài liệu các nhánh tìm ra nhưng nằm NGOÀI tập ứng viên đã rerank (top-400 đoạn của lin07) → run để
rerank (k5), mỗi tài liệu một đoạn cha (đoạn có RRF cao nhất giữa các nhánh).

Lý do (26/10, LB): probe_pool_tail R 0,0273 (reranker đã giữ ~94% đáp án trong tập ứng viên) nhưng probe_branches1000
(1.000 tài liệu/câu ngoài tập, xếp RRF thứ hạng tài liệu của 9 nhánh) R 0,2263, P 0,0184 → fusion tuyến tính + cắt
top-400 bỏ rơi tài liệu chỉ một nhánh tìm ra. probe_branches100 P 0,0439 < F2/5 ≈ 0,06 → không thêm thẳng được, phải
rerank. Thứ tự tài liệu giống hệt pool_probe (RRF k=60 trên thứ hạng tài liệu của từng nhánh, top --source-depth đoạn).
--n-main tài liệu không phải PubMed (đo được trên LB public) + --n-pubmed tài liệu PubMed (cho bản nộp private).

  python tools/deep_candidates.py --pool results/lin07s/runs/lin07.parquet \
      --sources results/mir2-runs/runs/{vi,en,zh}_{dense,lsr,bm25}.parquet --name deep1000 --out results/deep/runs
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.dataset as ds

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mirret import config, runs  # noqa: E402


def read_units(path, depth):
    """(qid, doc_id, k, rank) với k = số thứ tự đoạn cha (unit_id '<doc>_p<k>') — số nguyên cho máy ít RAM."""
    t = ds.dataset(str(path)).to_table(columns=["qid", "unit_id", "doc_id", "rank"],
                                       filter=ds.field("rank") <= depth).to_pandas()
    k = t["unit_id"].str.rsplit("_p", n=1).str[1].astype("int32")
    bad = (t["unit_id"] != t["doc_id"].astype(str) + "_p" + k.astype(str)).sum()
    if bad:
        raise SystemExit(f"{path}: {bad} unit_id không theo dạng <doc>_p<k>")
    return pd.DataFrame({"qid": t["qid"].astype("int32"), "doc_id": t["doc_id"].astype("int64"), "k": k,
                         "rank": t["rank"].astype("int32")})


def main():
    config.stdout_utf8()
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--pool", required=True, help="run gộp (lin07): tập ứng viên = tài liệu trong top --pool-depth đoạn")
    ap.add_argument("--pool-depth", type=int, default=400)
    ap.add_argument("--sources", nargs="+", required=True)
    ap.add_argument("--source-depth", type=int, default=1000)
    ap.add_argument("--n-main", type=int, default=1000, help="số tài liệu không phải PubMed mỗi câu")
    ap.add_argument("--n-pubmed", type=int, default=200, help="số tài liệu PubMed mỗi câu (bản private)")
    ap.add_argument("--rrf-k", type=int, default=60)
    ap.add_argument("--name", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    cfg = config.load(a.config)
    pm = runs.pubmed_ids(cfg)
    pool = read_units(a.pool, a.pool_depth)[["qid", "doc_id"]].drop_duplicates()
    pool["in_pool"] = True
    doc_s, unit_s = [], []
    for p in a.sources:
        u = read_units(p, a.source_depth)
        d = u.sort_values(["qid", "rank"]).drop_duplicates(["qid", "doc_id"])   # thứ hạng tài liệu trong nhánh:
        d["pm"] = d["doc_id"].isin(pm)                                           # bài thường đánh số trong các bài
        r_np = d[~d["pm"]].groupby("qid").cumcount() + 1                         # không phải PubMed (giống hệt
        r_all = d.groupby("qid").cumcount() + 1                                  # pool_probe), bài PubMed trong mọi bài;
        d["s"] = 1.0 / (a.rrf_k + r_np.reindex(d.index).fillna(r_all))          # tính TRƯỚC khi bỏ tập ứng viên
        d = d.merge(pool, on=["qid", "doc_id"], how="left")
        d = d[d["in_pool"].isna()]
        doc_s.append(d[["qid", "doc_id", "s"]])
        u = u.merge(pool, on=["qid", "doc_id"], how="left")
        u = u[u["in_pool"].isna()]
        unit_s.append(u.assign(s=1.0 / (a.rrf_k + u["rank"]))[["qid", "doc_id", "k", "s"]])
        print(f"  {Path(p).name}: {len(d) / max(1, d['qid'].nunique()):.0f} tài liệu mới/câu")
    docs = pd.concat(doc_s).groupby(["qid", "doc_id"], as_index=False)["s"].sum()
    units = pd.concat(unit_s).groupby(["qid", "doc_id", "k"], as_index=False)["s"].sum()
    best = units.sort_values(["qid", "doc_id", "s"], ascending=[True, True, False]).drop_duplicates(["qid", "doc_id"])
    docs = docs.merge(best[["qid", "doc_id", "k"]], on=["qid", "doc_id"])
    docs["pm"] = docs["doc_id"].isin(pm)
    docs = docs.sort_values(["qid", "s"], ascending=[True, False])
    keep = pd.concat([docs[~docs["pm"]].groupby("qid").head(a.n_main), docs[docs["pm"]].groupby("qid").head(a.n_pubmed)])
    keep = keep.sort_values(["qid", "s"], ascending=[True, False])
    keep["rank"] = keep.groupby("qid").cumcount() + 1
    run = pd.DataFrame({"qid": keep["qid"].astype("int64"),
                        "unit_id": keep["doc_id"].astype(str) + "_p" + keep["k"].astype(str),
                        "doc_id": keep["doc_id"].astype("int64"), "rank": keep["rank"].astype("int64"),
                        "score": keep["s"].astype("float64")})
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    runs.write(run, out / f"{a.name}.parquet", {"deep": "tài liệu ngoài tập ứng viên, mỗi tài liệu 1 đoạn cha",
                                                "pool": str(a.pool), "pool_depth": a.pool_depth,
                                                "sources": [str(p) for p in a.sources],
                                                "source_depth": a.source_depth, "n_main": a.n_main,
                                                "n_pubmed": a.n_pubmed, "rrf_k": a.rrf_k})
    n = keep.groupby("qid").agg(tot=("doc_id", "size"), pm=("pm", "sum"))
    print(f"→ {out / (a.name + '.parquet')}: {len(run)} cặp, {run['qid'].nunique()} câu; mỗi câu TB {n['tot'].mean():.0f} "
          f"(PubMed {n['pm'].mean():.0f}), p10 {np.percentile(n['tot'], 10):.0f}")


if __name__ == "__main__":
    main()
