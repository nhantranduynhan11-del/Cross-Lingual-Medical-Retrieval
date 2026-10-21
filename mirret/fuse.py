"""Gộp run bằng RRF: điểm(đơn vị) = Σ_run w_run / (k + rank_run(đơn vị)), k = fuse.rrf_k (60).

  python -m mirret.fuse                         # mọi cấu hình trong fuse.configs (vd rrf9 = 9 nhánh)
  python -m mirret.fuse --only rrf9
  python -m mirret.fuse --runs vi_dense,zh_dense --name thu_nghiem
  python -m mirret.fuse --runs rrf9_w_cross,q2d9_cross --name rrf9w_q2d9_cross --union
        hợp các run ĐÃ rerank bằng CÙNG reranker (điểm so được): top-N đã rerank của mọi run xếp theo điểm reranker,
        rồi phần đuôi theo RRF — tương đương rerank tập ứng viên lớn hơn (hợp các top-N), không cần chạy lại GPU.
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


def union_rerank(run_list, top_n=400, depth=1000, k=60):
    """Run đã rerank bằng cùng một reranker → một run: đoạn trong top_n của ít nhất một run xếp theo điểm reranker (có ở
    nhiều run: trung bình — chỉ khác nhau do nhiễu fp16), sau đó các đoạn còn lại theo RRF của thứ hạng gốc."""
    head = pd.concat([df[df["rank"] <= top_n] for df in run_list], ignore_index=True)
    head = head.groupby(["qid", "unit_id", "doc_id"], as_index=False).agg(score=("score", "mean"), rank=("rank", "min"))
    tail = rrf([df[df["rank"] > top_n] for df in run_list], k, None)
    tail = tail[~tail.set_index(["qid", "unit_id"]).index.isin(head.set_index(["qid", "unit_id"]).index)].copy()
    tail["score"] = -100000.0 - tail["rank"]              # luôn sau phần đã rerank, giữ thứ tự RRF
    return runs.rerank_by_score(pd.concat([head, tail], ignore_index=True), depth)


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
    ap.add_argument("--union", action="store_true", help="--runs là các run đã rerank bằng cùng reranker → union_rerank")
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
        lst = [have[r] for r, _ in items]
        # trộn nhánh biến thể chạy trên bộ nhãn (vd 40 câu) với nhánh baseline (1.200 câu) → chỉ gộp các câu của nhánh
        # nhỏ; run phủ gần đủ (vd thiếu vài câu dịch) không bị lọc như trước
        sizes = [x["qid"].nunique() for x in lst]
        small = [set(x["qid"].unique()) for x, n in zip(lst, sizes) if n * 2 < max(sizes)]
        if small:
            keep = set.intersection(*small)
            print(f"  {name}: chỉ gộp {len(keep)} câu hỏi của các nhánh chạy trên tập con")
            lst = [x[x["qid"].isin(keep)] for x in lst]
        if a.union:
            df = union_rerank(lst, cfg["rerank"]["top_n"], fc["depth"], fc["rrf_k"])
            meta = {"fusion": "union_rerank", "top_n": cfg["rerank"]["top_n"], "runs": [r for r, _ in items]}
        else:
            df = rrf(lst, fc["rrf_k"], fc["depth"], [w for _, w in items])
            meta = {"fusion": "rrf", "k": fc["rrf_k"], "runs": [r for r, _ in items], "weights": [w for _, w in items]}
        runs.write(df, rdir / f"{name}.parquet", meta)
        print(f"  → {rdir / (name + '.parquet')} ({df['qid'].nunique()} câu hỏi)")


if __name__ == "__main__":
    main()
