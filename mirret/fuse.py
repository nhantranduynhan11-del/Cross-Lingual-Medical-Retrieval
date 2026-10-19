"""Gộp run bằng RRF: điểm(đơn vị) = Σ_run w_run / (k + rank_run(đơn vị)), k = fuse.rrf_k (60).

  python -m mirret.fuse                         # mọi cấu hình trong fuse.configs (vd rrf9 = 9 nhánh)
  python -m mirret.fuse --only rrf9
  python -m mirret.fuse --runs vi_dense,zh_dense --name thu_nghiem
Run thiếu (vd chưa có bản dịch) → bỏ qua cấu hình đó và báo.
"""
import argparse

import pandas as pd

from . import config, runs


def parse_item(item):
    """"bm25_zh*0.5" → ("bm25_zh", 0.5)."""
    name, _, w = str(item).partition("*")
    return name.strip(), float(w) if w else 1.0


def rrf(run_list, k=60, depth=1000, weights=None):
    parts = []
    for df, w in zip(run_list, weights or [1.0] * len(run_list)):
        x = df[["qid", "unit_id", "doc_id", "rank"]].copy()
        x["score"] = w / (k + x["rank"])
        parts.append(x)
    allr = pd.concat(parts, ignore_index=True)
    g = allr.groupby(["qid", "unit_id", "doc_id"], as_index=False).agg(score=("score", "sum"), rank=("rank", "min"))
    return runs.rerank_by_score(g, depth)                 # bằng điểm: ưu tiên thứ hạng tốt nhất ở một nhánh


def load_runs(rdir, names, depth=None):
    have = {}
    for n in names:
        p = rdir / f"{n}.parquet"
        if p.exists():
            have[n] = runs.read(p, depth)
    return have


def main():
    config.stdout_utf8()
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--only", default=None)
    ap.add_argument("--runs", default=None, help="danh sách run để gộp (tên file trong runs/, không đuôi)")
    ap.add_argument("--name", default=None)
    a = ap.parse_args()
    cfg = config.load(a.config)
    fc = cfg["fuse"]
    rdir = cfg.path("runs")
    if a.runs:
        configs = {a.name or "fused": a.runs.split(",")}
    else:
        configs = {n: v for n, v in fc["configs"].items() if not a.only or n in a.only.split(",")}
        unknown = [n for n in (a.only or "").split(",") if n and n not in fc["configs"]]
        if unknown:                                           # vd mir2-code cũ chưa có rrf9_w trong retrieval.yaml
            raise SystemExit(f"fuse.configs không có {unknown} (có: {list(fc['configs'])}) — kiểm tra bản mir2-code")
    for name, items in configs.items():
        items = [parse_item(x) for x in items]
        have = load_runs(rdir, [r for r, _ in items], fc["depth"])
        miss = [r for r, _ in items if r not in have]
        if miss:
            print(f"  {name}: thiếu run {miss} → bỏ qua")
            if a.only or a.runs:                              # gộp theo yêu cầu cụ thể mà thiếu run → báo lỗi
                raise SystemExit(f"thiếu run {miss} trong {rdir}")
            continue
        df = rrf([have[r] for r, _ in items], fc["rrf_k"], fc["depth"], [w for _, w in items])
        runs.write(df, rdir / f"{name}.parquet", {"fusion": "rrf", "k": fc["rrf_k"], "runs": [r for r, _ in items],
                                                  "weights": [w for _, w in items]})
        print(f"  → {rdir / (name + '.parquet')} ({df['qid'].nunique()} câu hỏi)")


if __name__ == "__main__":
    main()
