"""T6: xếp hạng lại top-N bằng multi-vector (ColBERT) của BGE-M3 — điểm tương tác muộn MaxSim như FlagEmbedding:
  điểm(q, d) = (1/|q|) Σ_i max_j <q_i, d_j>   (q_i, d_j: vector ColBERT đã chuẩn hoá, bỏ <s>)
Vector ColBERT KHÔNG lưu cho cả kho: chỉ mã hoá câu hỏi và N ứng viên khi xếp hạng lại.
Đầu vào ứng viên: tiêu đề + "\\n" + văn bản (như T3). rerank.unit = parent: đoạn con (parent_child) được gom về đoạn cha
(giữ thứ hạng tốt nhất), xếp hạng lại và trả về đoạn cha (chunk_id = parent_id).

  python -m mir.rerank --run runs/structure_G.parquet --device cuda:0 [--shard i/N]
     → runs/structure_G_rr/part-i-of-N.parquet ; khi đủ mọi phần: runs/structure_G_rr.parquet
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import config, runs, store


def maxsim(q, d):
    """q [m, H], d [n, H] (float16/32) → điểm ColBERT."""
    if len(q) == 0 or len(d) == 0:
        return 0.0
    s = q.astype(np.float32) @ d.astype(np.float32).T
    return float(s.max(axis=1).sum() / len(q))


def to_parent(df, chunk_dir):
    """Run đoạn con → run đoạn cha (giữ rank tốt nhất)."""
    m = store.lookup(chunk_dir, "chunk_id", df["chunk_id"].unique().tolist(), ["parent_id"])
    df = df.copy()
    df["chunk_id"] = df["chunk_id"].map(lambda c: m[c]["parent_id"])
    df = df.sort_values(["qid", "rank"]).drop_duplicates(["qid", "chunk_id"], keep="first")
    df["rank"] = df.groupby("qid").cumcount() + 1
    return df.reset_index(drop=True)


def candidate_texts(ids, chunk_dir, clean_dir):
    """{id đoạn hoặc đoạn cha: tiêu đề + "\\n" + văn bản} — cùng dạng đầu vào như khi mã hoá ở T3."""
    rows = store.unit_rows(chunk_dir, ids, ["doc_id", "text"])
    titles = store.lookup(clean_dir, "doc_id", list({r["doc_id"] for r in rows.values()}), ["title"])
    return {k: ((titles.get(r["doc_id"], {}).get("title") or "").strip() + "\n" + r["text"]) for k, r in rows.items()}


def rerank_run(enc, df, qtext, texts, rc, log=print):
    """df: run (đã cắt top_n); qtext: {qid: câu hỏi}; texts: {chunk_id: văn bản ứng viên} → run mới (điểm ColBERT)."""
    out = []
    qids = sorted(df["qid"].unique())
    qv = dict(zip(qids, enc.colbert([qtext[q] for q in qids], rc["query_max_length"], rc["batch_tokens"])))
    t0 = time.time()
    for n, (q, g) in enumerate(df.groupby("qid", sort=True)):
        ids = [c for c in g["chunk_id"] if c in texts]
        dv = enc.colbert([texts[c] for c in ids], rc["doc_max_length"], rc["batch_tokens"])
        sc = [maxsim(qv[q], d) for d in dv]
        out.append(pd.DataFrame({"qid": q, "chunk_id": ids, "doc_id": [runs.doc_of(c) for c in ids],
                                 "rank": g.set_index("chunk_id").loc[ids, "rank"].values, "score": sc}))
        if (n + 1) % 50 == 0:
            log(f"  {n + 1}/{len(qids)} câu hỏi, {(time.time() - t0) / (n + 1):.2f}s/câu")
    return runs.rerank_by_score(pd.concat(out, ignore_index=True)) if out else df.iloc[:0]


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/baseline.yaml")
    ap.add_argument("--run", required=True, help="run file cần xếp hạng lại")
    ap.add_argument("--strategy", default=None)
    ap.add_argument("--queries", default=None)
    ap.add_argument("--chunks", default=None, help="mặc định <work>/chunks/<strategy>")
    ap.add_argument("--clean", default=None, help="mặc định <work>/corpus_clean")
    ap.add_argument("--unit", default=None, help="chunk | parent (mặc định rerank.unit)")
    ap.add_argument("--shard", default="0/1", help="i/N theo câu hỏi")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    cfg = config.load(a.config)
    rc, ec = cfg["rerank"], cfg["encode"]
    strategy = a.strategy or cfg["chunk"]["strategy"]
    unit = a.unit or rc["unit"]
    chunk_dir = cfg.path("chunks", strategy, a.chunks)
    clean_dir = cfg.path("clean", override=a.clean)
    run_path = Path(a.run)
    part_dir = run_path.with_name(run_path.stem + "_rr")
    final = run_path.with_name(run_path.stem + "_rr.parquet")
    si, sn = (int(x) for x in a.shard.split("/"))
    part = part_dir / f"part-{si}-of-{sn}.parquet"
    if final.exists() and not a.force:
        print(f"{final} đã có → bỏ qua")
        return
    if not part.exists() or a.force:
        from .encode import M3Encoder
        from .retrieve import load_queries
        qids, qtexts = load_queries(cfg.path("queries", override=a.queries))
        df = runs.read(run_path)
        missing = set(df["qid"]) - set(qids)
        if missing:
            raise SystemExit(f"{len(missing)} câu hỏi trong run không có trong file câu hỏi (vd {sorted(missing)[:5]}). "
                             "Muốn xếp hạng lại một phần câu hỏi: lọc run rồi ghi ra tên khác, vd runs/<run>_eval.parquet.")
        if unit == "parent":
            df = to_parent(df, chunk_dir)
        mine = [q for k, q in enumerate(sorted(df["qid"].unique())) if k % sn == si]
        df = df[df["qid"].isin(mine) & (df["rank"] <= rc["top_n"])]
        texts = candidate_texts(df["chunk_id"].unique().tolist(), chunk_dir, clean_dir)
        enc = M3Encoder.load(ec["model"], a.device, ec["fp16"], ec["max_length"], colbert=True)
        res = rerank_run(enc, df, dict(zip(qids, qtexts)), texts, rc)
        runs.write(res, part, {"run": str(run_path), "unit": unit, "top_n": rc["top_n"], "shard": a.shard,
                               "rerank": rc, "model": ec["model"]}, force=True)
        print(f"→ {part}")
    parts = sorted(part_dir.glob(f"part-*-of-{sn}.parquet"))
    if len(parts) == sn:
        res = pd.concat([runs.read(p) for p in parts], ignore_index=True)
        runs.write(res, final, {"run": str(run_path), "unit": unit, "top_n": rc["top_n"], "parts": sn}, force=a.force)
        print(f"đủ {sn} phần → {final}")
    else:
        print(f"đã có {len(parts)}/{sn} phần; chạy các shard còn lại rồi chạy lại lệnh này để gộp")


if __name__ == "__main__":
    main()
