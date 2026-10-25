"""Gộp run bằng RRF: điểm(đơn vị) = Σ_run w_run / (k + rank_run(đơn vị)), k = fuse.rrf_k (60).

  python -m mirret.fuse                         # mọi cấu hình trong fuse.configs (vd rrf9 = 9 nhánh)
  python -m mirret.fuse --only rrf9
  python -m mirret.fuse --runs vi_dense,zh_dense --name thu_nghiem
Run thiếu (vd chưa có bản dịch) → bỏ qua cấu hình đó và báo.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from . import config, runs


def parse_item(item):
    """"bm25_zh*0.5" → ("bm25_zh", 0.5)."""
    name, _, w = str(item).partition("*")
    return name.strip(), float(w) if w else 1.0


def minmax_norm(df):
    x = df.copy()
    min_s = x.groupby("qid")["score"].transform("min")
    max_s = x.groupby("qid")["score"].transform("max")
    diff = max_s - min_s
    x["norm"] = np.where(diff > 1e-9, (x["score"] - min_s) / (diff + 1e-9), 1.0)
    return x


def linear_fuse(dense_runs, sparse_runs, alpha=0.7, depth=1000):
    """Gộp tuyến tính min-max lin07/lin12:
    Score = alpha * Score_dense + (1 - alpha) * Score_sparse
    """
    dense_parts = []
    w_d_total = sum(w for _, w in dense_runs) if dense_runs else 1.0
    for df, w in dense_runs:
        n = minmax_norm(df)
        n["w_score"] = n["norm"] * w
        dense_parts.append(n[["qid", "unit_id", "doc_id", "w_score", "rank"]])
    if dense_parts:
        d_df = pd.concat(dense_parts, ignore_index=True).groupby(["qid", "unit_id", "doc_id"], as_index=False).agg(
            dense_sum=("w_score", "sum"), best_rank=("rank", "min")
        )
        d_df["dense_score"] = d_df["dense_sum"] / w_d_total
    else:
        d_df = pd.DataFrame(columns=["qid", "unit_id", "doc_id", "dense_score", "best_rank"])

    sparse_parts = []
    w_s_total = sum(w for _, w in sparse_runs) if sparse_runs else 1.0
    for df, w in sparse_runs:
        n = minmax_norm(df)
        n["w_score"] = n["norm"] * w
        sparse_parts.append(n[["qid", "unit_id", "doc_id", "w_score", "rank"]])
    if sparse_parts:
        s_df = pd.concat(sparse_parts, ignore_index=True).groupby(["qid", "unit_id", "doc_id"], as_index=False).agg(
            sparse_sum=("w_score", "sum"), best_rank=("rank", "min")
        )
        s_df["sparse_score"] = s_df["sparse_sum"] / w_s_total
    else:
        s_df = pd.DataFrame(columns=["qid", "unit_id", "doc_id", "sparse_score", "best_rank"])

    if d_df.empty and s_df.empty:
        return pd.DataFrame(columns=["qid", "unit_id", "doc_id", "score", "rank"])

    merged = pd.merge(d_df[["qid", "unit_id", "doc_id", "dense_score", "best_rank"]],
                      s_df[["qid", "unit_id", "doc_id", "sparse_score", "best_rank"]],
                      on=["qid", "unit_id", "doc_id"], how="outer", suffixes=("_d", "_s"))
    merged["dense_score"] = merged["dense_score"].fillna(0.0)
    merged["sparse_score"] = merged["sparse_score"].fillna(0.0)
    merged["rank"] = merged[["best_rank_d", "best_rank_s"]].min(axis=1).fillna(9999).astype(int)
    merged["score"] = alpha * merged["dense_score"] + (1.0 - alpha) * merged["sparse_score"]

    merged = merged.sort_values(["qid", "score", "rank"], ascending=[True, False, True])
    merged["rank"] = merged.groupby("qid").cumcount() + 1
    return merged[merged["rank"] <= depth][["qid", "unit_id", "doc_id", "score", "rank"]].reset_index(drop=True)


def rrf(run_list, k=60, depth=1000, weights=None):
    parts = []
    for df, w in zip(run_list, weights or [1.0] * len(run_list)):
        x = df[["qid", "unit_id", "doc_id", "rank"]].copy()
        x["score"] = w / (k + x["rank"])
        parts.append(x)
    allr = pd.concat(parts, ignore_index=True)
    g = allr.groupby(["qid", "unit_id", "doc_id"], as_index=False).agg(score=("score", "sum"), rank=("rank", "min"))
    return runs.rerank_by_score(g, depth)                 # bằng điểm: ưu tiên thứ hạng tốt nhất ở một nhánh


def load_runs(rdirs, names, depth=None):
    if not isinstance(rdirs, (list, tuple)):
        rdirs = [rdirs]
    rdirs = [Path(d) for d in rdirs if d]
    have = {}
    for n in names:
        found = False
        candidates = [f"{n}.parquet"]
        if n.startswith("e5_"):
            lang = n[3:]
            candidates.extend([f"e5_{lang}.parquet", f"{lang}_e5.parquet"])
        elif n.endswith("_e5"):
            lang = n.split("_")[0]
            candidates.extend([f"{lang}_e5.parquet", f"e5_{lang}.parquet"])

        for d in rdirs:
            if not d.exists():
                continue
            for cand in candidates:
                p = d / cand
                if p.exists():
                    have[n] = runs.read(p, depth)
                    found = True
                    break
            if found:
                break
    return have


def main():
    import numpy as np
    config.stdout_utf8()
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--only", default=None)
    ap.add_argument("--runs", default=None, help="danh sách run để gộp (tên file trong runs/, không đuôi)")
    ap.add_argument("--run-dirs", default=None, help="các thư mục chứa runs (phẩy ngăn cách)")
    ap.add_argument("--out", "--out-dir", default=None, dest="out_dir", help="Thư mục ghi kết quả runs đã fuse")
    ap.add_argument("--name", default=None)
    ap.add_argument("--alpha", type=float, default=0.7, help="Trọng số alpha cho linear fusion (mặc định 0.7 dense)")
    a = ap.parse_args()
    cfg = config.load(a.config)
    fc = cfg["fuse"]
    rdir = Path(a.out_dir) if a.out_dir else cfg.path("runs")
    rdir.mkdir(parents=True, exist_ok=True)
    search_dirs = [rdir]
    if a.run_dirs:
        for x in a.run_dirs.split(","):
            if x.strip():
                p = Path(x.strip())
                if p not in search_dirs:
                    search_dirs.append(p)
    for extra in (cfg.work_dir / "runs", cfg.work_dir / "runs_e5"):
        if extra not in search_dirs:
            search_dirs.append(extra)

    if a.runs:
        configs = {a.name or "fused": a.runs.split(",")}
    else:
        configs = {n: v for n, v in fc["configs"].items() if not a.only or n in a.only.split(",")}
    for name, spec in configs.items():
        is_linear = name.startswith("lin") or (isinstance(spec, dict) and "dense" in spec)
        if isinstance(spec, dict):
            dense_items = [parse_item(x) for x in spec.get("dense", [])]
            sparse_items = [parse_item(x) for x in spec.get("sparse", [])]
            alpha = spec.get("alpha", a.alpha)
            req_names = [r for r, _ in dense_items + sparse_items]
        else:
            items = [parse_item(x) for x in spec]
            req_names = [r for r, _ in items]
            alpha = a.alpha
            if "0" in name:
                try:
                    import re
                    m = re.search(r"lin(\d+)", name)
                    if m:
                        alpha = float("0." + m.group(1))
                except Exception:
                    pass
            dense_items = [x for x in items if "dense" in x[0] or "e5" in x[0]]
            sparse_items = [x for x in items if not ("dense" in x[0] or "e5" in x[0])]

        have = load_runs(search_dirs, req_names, fc["depth"])
        miss = [r for r in req_names if r not in have]
        if miss:
            print(f"  {name}: thiếu run {miss} → bỏ qua")
            continue

        if is_linear:
            d_runs = [(have[r], w) for r, w in dense_items]
            s_runs = [(have[r], w) for r, w in sparse_items]
            df = linear_fuse(d_runs, s_runs, alpha=alpha, depth=fc["depth"])
            meta = {"fusion": "linear", "alpha": alpha, "dense_runs": [r for r, _ in dense_items],
                    "sparse_runs": [r for r, _ in sparse_items]}
        else:
            items = [parse_item(x) for x in spec]
            df = rrf([have[r] for r, _ in items], fc["rrf_k"], fc["depth"], [w for _, w in items])
            meta = {"fusion": "rrf", "k": fc["rrf_k"], "runs": [r for r, _ in items],
                    "weights": [w for _, w in items]}

        runs.write(df, rdir / f"{name}.parquet", meta)
        print(f"  → {rdir / (name + '.parquet')} ({df['qid'].nunique()} câu hỏi) [{meta.get('fusion')}]")


if __name__ == "__main__":
    main()
