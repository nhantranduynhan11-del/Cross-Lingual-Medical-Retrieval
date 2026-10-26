"""Ghép ứng viên sâu (deep_candidates → bge → Qwen3i) vào tập ứng viên đã rerank (top-400 của lin07) → run Qwen và run
ens gộp + danh sách tài liệu autok (bản có PubMed và bản bỏ PubMed). Chạy trên máy, không GPU.

Cách gộp (cố định trước khi nộp, không dò tham số):
  - run Qwen gộp: đơn vị top-400 của lin07 + đơn vị sâu, xếp theo P(yes) (cùng model, cùng câu hỏi → so được), phần đuôi
    của lin07 giữ sau với điểm -(100000 + hạng) như rerank.with_tail;
  - run ens gộp: (z_bge + z_logit(qwen)) / 2, chuẩn hoá z bằng trung bình / độ lệch của 400 đơn vị gốc của câu đó (giống
    ensemble.py trên top-400) → đơn vị gốc giữ đúng thứ tự lin07_ensi; đơn vị sâu đặt vào cùng thang;
  - tài liệu: autok như mix_submission (p tài liệu = max P(yes) của các đơn vị đã chấm; k ∈ [--k-min, --k-doc] tối đa
    5·Σp / (4·Σp_all + k)); bản _np = bỏ PubMed sau khi cắt (như final1200_cbud200_nopm).
Không có --deep-qwen → chỉ tập gốc: phải tái tạo đúng final1200_cbud200 (kiểm tra cách làm).

  python tools/merge_deep.py --deep-qwen results/deep/runs/deep180q_pub_qwen3i180.parquet \
      results/deep/runs/deep180q_hid_qwen3i180.parquet --name deep180 --out results/deep
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mirret import config, runs  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mix_submission import auto_k  # noqa: E402


def logit(p):
    p = p.astype(np.float64).clip(1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def main():
    config.stdout_utf8()
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--base", default="results/lin07s/runs/lin07.parquet", help="run gộp tầng 1 (phần đuôi)")
    ap.add_argument("--base-qwen", default="results/full1200/runs/lin07_qwen3i_1200.parquet")
    ap.add_argument("--base-bge", default="results/lin07s/runs/lin07_cross.parquet")
    ap.add_argument("--deep-qwen", nargs="*", default=[], help="run Qwen3i của ứng viên sâu (có thể nhiều phần)")
    ap.add_argument("--deep-bge", default="results/deep1000/runs/deep1000_cross1200.parquet")
    ap.add_argument("--top-n", type=int, default=400)
    ap.add_argument("--k-doc", type=int, default=400)
    ap.add_argument("--k-min", type=int, default=100)
    ap.add_argument("--name", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    cfg = config.load(a.config)
    pm = runs.pubmed_ids(cfg)
    bq = runs.read(a.base_qwen, depth=a.top_n)[["qid", "unit_id", "doc_id", "score"]].rename(columns={"score": "qwen"})
    bb = runs.read(a.base_bge, depth=a.top_n)[["qid", "unit_id", "score"]].rename(columns={"score": "bge"})
    pool = bq.merge(bb, on=["qid", "unit_id"], how="left")
    if pool["bge"].isna().any():
        raise SystemExit(f"thiếu điểm bge cho {int(pool['bge'].isna().sum())} đơn vị gốc")
    pool["deep"] = False
    parts = [pool]
    if a.deep_qwen:
        dq = pd.concat([runs.read(p) for p in a.deep_qwen], ignore_index=True)
        dq = dq[(dq["score"] >= 0) & (dq["score"] <= 1)][["qid", "unit_id", "doc_id", "score"]].rename(columns={"score": "qwen"})
        db = runs.read(a.deep_bge)[["qid", "unit_id", "score"]].rename(columns={"score": "bge"})
        deep = dq.merge(db, on=["qid", "unit_id"], how="left")
        if deep["bge"].isna().any():
            raise SystemExit(f"thiếu điểm bge cho {int(deep['bge'].isna().sum())} đơn vị sâu")
        dup = deep.merge(pool[["qid", "doc_id"]].drop_duplicates(), on=["qid", "doc_id"]).shape[0]
        if dup:
            raise SystemExit(f"{dup} đơn vị sâu thuộc tài liệu đã có trong tập gốc")
        deep["deep"] = True
        parts.append(deep)
        print(f"đơn vị sâu: {len(deep)} ({deep['qid'].nunique()} câu, TB {len(deep) / deep['qid'].nunique():.0f}/câu)")
    df = pd.concat(parts, ignore_index=True)
    df["ql"] = logit(df["qwen"])
    st = df[~df["deep"]].groupby("qid").agg(mb=("bge", "mean"), sb=("bge", "std"), mq=("ql", "mean"), sq=("ql", "std"))
    df = df.join(st, on="qid")
    zb = ((df["bge"] - df["mb"]) / df["sb"].replace(0, np.nan)).fillna(0.0)
    zq = ((df["ql"] - df["mq"]) / df["sq"].replace(0, np.nan)).fillna(0.0)
    df["ens"] = (zb + zq) / 2
    base = runs.read(a.base)
    head = set(zip(df["qid"], df["unit_id"]))
    tail = base[[k not in head for k in zip(base["qid"], base["unit_id"])]][["qid", "unit_id", "doc_id", "rank"]]
    out = Path(a.out)
    (out / "runs").mkdir(parents=True, exist_ok=True)
    for col, nm in (("qwen", f"{a.name}_qwen3i"), ("ens", f"{a.name}_ensi")):
        h = df[["qid", "unit_id", "doc_id", col]].rename(columns={col: "score"})
        h = h.sort_values(["qid", "score"], ascending=[True, False])
        h["rank"] = h.groupby("qid").cumcount() + 1
        n_head = h.groupby("qid")["rank"].max()
        t = tail.sort_values(["qid", "rank"]).copy()
        t["rank"] = t.groupby("qid").cumcount() + 1 + t["qid"].map(n_head).fillna(0).astype(int)
        t["score"] = -100000.0 - t["rank"]
        r = pd.concat([h, t], ignore_index=True)[["qid", "unit_id", "doc_id", "rank", "score"]]
        r = r.sort_values(["qid", "rank"]).reset_index(drop=True)
        runs.write(r, out / "runs" / f"{nm}.parquet", {"merge_deep": col, "base_qwen": a.base_qwen, "base_bge": a.base_bge,
                                                       "deep_qwen": a.deep_qwen, "deep_bge": a.deep_bge if a.deep_qwen else None})
        print(f"→ {out / 'runs' / (nm + '.parquet')}")
    # tài liệu: thứ hạng theo run Qwen gộp (như mix_submission doc_lists), autok trên P(yes) của mọi đơn vị đã chấm
    rq = runs.read(out / "runs" / f"{a.name}_qwen3i.parquet")
    dr = runs.doc_ranking(rq[rq["rank"] <= a.k_doc * 3], "max", a.top_n).sort_values(["qid", "drank"])
    docs = {int(q): [int(d) for d in g["doc_id"]][:a.k_doc] for q, g in dr.groupby("qid")}
    # mọi đơn vị đã chấm (kể cả hạng > 400) có p; phần đuôi điểm âm → p = 0 sau clip
    docs = auto_k(docs, out / "runs" / f"{a.name}_qwen3i.parquet", a.k_min, a.k_doc, top_n=10 ** 9)
    sets = {f"{a.name}_docs_pm": docs, f"{a.name}_docs_np": {q: [d for d in v if d not in pm] for q, v in docs.items()}}
    deep_docs = set(zip(df.loc[df["deep"], "qid"], df.loc[df["deep"], "doc_id"]))
    for nm, dd in sets.items():
        (out / f"{nm}.json").write_text(json.dumps([{"id": q, "relevant_docs": v} for q, v in sorted(dd.items())]),
                                        encoding="utf-8")
        n = np.mean([len(v) for v in dd.values()])
        nd = np.mean([sum((q, d) in deep_docs for d in v) for q, v in dd.items()])
        print(f"→ {out / (nm + '.json')}: TB {n:.1f} tài liệu/câu, trong đó sâu {nd:.1f}")


if __name__ == "__main__":
    main()
