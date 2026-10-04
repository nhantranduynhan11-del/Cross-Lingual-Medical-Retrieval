"""T5 (phần 2): gộp các nhánh.

RRF: điểm(đoạn) = Σ_nhánh 1 / (k + rank_nhánh(đoạn)), k = fuse.rrf_k (60). Đủ 7 cấu hình A–G (fuse.configs).
Có trọng số (tuỳ chọn): α·Dense + β·LSR + γ·BM25, mỗi nhánh chuẩn hoá điểm min-max theo từng câu hỏi; đoạn không có
trong nhánh nào thì điểm nhánh đó = 0.

  python -m mir.fuse --config ... [--strategy structure]
     đọc runs/<strategy>_{dense,lsr,bm25}.parquet → runs/<strategy>_{A..G}.parquet, runs/<strategy>_W<a>-<b>-<c>.parquet
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

from . import config, runs

BRANCHES = ("dense", "lsr", "bm25")


def rrf(branch_runs, k=60, depth=1000):
    parts = []
    for df in branch_runs:
        x = df[["qid", "chunk_id", "doc_id", "rank"]].copy()
        x["score"] = 1.0 / (k + x["rank"])
        parts.append(x[["qid", "chunk_id", "doc_id", "score"]])
    allr = pd.concat(parts, ignore_index=True)
    best_rank = pd.concat([df[["qid", "chunk_id", "rank"]] for df in branch_runs]).groupby(["qid", "chunk_id"])["rank"].min()
    g = allr.groupby(["qid", "chunk_id", "doc_id"], as_index=False)["score"].sum()
    g = g.merge(best_rank.rename("rank").reset_index(), on=["qid", "chunk_id"])     # hoà điểm: ưu tiên rank tốt nhất
    return runs.rerank_by_score(g, depth)


def weighted(branch_runs, weights, depth=1000):
    parts = []
    for df, w in zip(branch_runs, weights):
        x = df[["qid", "chunk_id", "doc_id", "rank", "score"]].copy()
        g = x.groupby("qid")["score"]
        lo, hi = g.transform("min"), g.transform("max")
        x["score"] = w * ((x["score"] - lo) / (hi - lo).where(hi > lo, 1.0)).where(hi > lo, 1.0)
        parts.append(x)
    allr = pd.concat(parts, ignore_index=True)
    g = allr.groupby(["qid", "chunk_id", "doc_id"], as_index=False).agg(score=("score", "sum"), rank=("rank", "min"))
    return runs.rerank_by_score(g, depth)


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/baseline.yaml")
    ap.add_argument("--strategy", default=None)
    ap.add_argument("--runs", default=None, help="thư mục runs (mặc định <work>/runs)")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    cfg = config.load(a.config)
    fc, depth = cfg["fuse"], cfg["retrieve"]["depth"]
    strategy = a.strategy or cfg["chunk"]["strategy"]
    rdir = Path(a.runs) if a.runs else cfg.work_dir / cfg["retrieve"]["out_dir"]
    have = {b: runs.read(rdir / f"{strategy}_{b}.parquet") for b in BRANCHES if (rdir / f"{strategy}_{b}.parquet").exists()}
    print(f"nhánh có sẵn: {sorted(have)}")
    for name, brs in fc["configs"].items():
        if not all(b in have for b in brs):
            print(f"  {name}: thiếu nhánh {[b for b in brs if b not in have]} → bỏ qua")
            continue
        dst = rdir / f"{strategy}_{name}.parquet"
        if dst.exists() and not a.force:
            continue
        runs.write(rrf([have[b] for b in brs], fc["rrf_k"], depth), dst,
                   {"fusion": "rrf", "k": fc["rrf_k"], "branches": brs, "strategy": strategy}, force=a.force)
        print(f"  → {dst}")
    if all(b in have for b in BRANCHES):
        for w in fc.get("weighted") or []:
            dst = rdir / f"{strategy}_W{'-'.join(f'{x:g}' for x in w)}.parquet"
            if dst.exists() and not a.force:
                continue
            runs.write(weighted([have[b] for b in BRANCHES], w, depth), dst,
                       {"fusion": "weighted_minmax", "weights": dict(zip(BRANCHES, w)), "strategy": strategy},
                       force=a.force)
            print(f"  → {dst}")


if __name__ == "__main__":
    main()
