"""Dùng model fine-tune CHỈ cho bài PubMed (tiếng Anh): trong mỗi câu, đoạn vi/zh giữ nguyên điểm của run gốc; các đoạn
PubMed giữ đúng TẬP giá trị điểm gốc của chúng nhưng gán lại theo thứ tự của model ft (hoán vị trong nhóm PubMed) → thang
P(yes) (autok, phân bổ khúc) và z-score (ens) không đổi, chỉ đổi bài PubMed nào đứng trên.

Lý do (LB 29/10): Qwen/bge fine-tune trên câu tổng hợp thua ở vi/zh (FINAL 0,2373 < 0,2453) vì thiên về trang hỏi-đáp
tiếng Trung ngắn; phần PubMed không đo được trên public (đáp án public không có PubMed) → chỉ áp cho PubMed, kiểm ở private.

  python tools/en_reorder.py --base results/lin07m/runs/lin07m_qwen3i.parquet results/lin07m/runs/lin07m_ensi.parquet \
      --ft results/lin07m/runs/lin07m_qwen3iftpm.parquet results/lin07m/runs/lin07m_crossftpm.parquet \
      --out results/lin07m/runs --suffix _enft
  --ft nhiều run → thứ tự ft = TB z-score (P(yes) đổi sang logit trước) của các run.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mirret import config, runs  # noqa: E402


def ft_order(paths):
    parts = []
    for p in paths:
        r = runs.read(p)
        r = r[r["score"] > -1e4][["qid", "unit_id", "score"]].copy()
        s = r["score"].astype(float)
        if s.between(0, 1).all():                             # P(yes) → logit (dồn sát 0/1, z trực tiếp mất phân biệt)
            s = np.log(s.clip(1e-6, 1 - 1e-6) / (1 - s.clip(1e-6, 1 - 1e-6)))
        g = s.groupby(r["qid"])
        r["z"] = ((s - g.transform("mean")) / g.transform("std").replace(0, np.nan)).fillna(0.0)
        parts.append(r[["qid", "unit_id", "z"]])
    return pd.concat(parts).groupby(["qid", "unit_id"], as_index=False)["z"].mean()


def reorder(base, ft, pm):
    head = base[base["score"] > -1e4].copy()
    tail = base[base["score"] <= -1e4]
    head["pm"] = head["doc_id"].isin(pm)
    head = head.merge(ft, on=["qid", "unit_id"], how="left")
    n_changed = 0
    out = []
    for q, g in head.groupby("qid", sort=False):
        g = g.copy()
        sel = g["pm"] & g["z"].notna()
        if sel.sum() > 1:
            vals = np.sort(g.loc[sel, "score"].to_numpy())[::-1]           # điểm gốc của nhóm PubMed, giảm dần
            order = g.loc[sel].sort_values("z", ascending=False).index     # thứ tự theo model ft
            n_changed += int((g.loc[order, "score"].to_numpy() != vals).sum())
            g.loc[order, "score"] = vals
        out.append(g)
    head = pd.concat(out, ignore_index=True).drop(columns=["pm", "z"])
    head = head.sort_values(["qid", "score"], ascending=[True, False])
    head["rank"] = head.groupby("qid").cumcount() + 1
    n_head = head.groupby("qid")["rank"].max()
    tail = tail.sort_values(["qid", "rank"]).copy()
    tail["rank"] = tail.groupby("qid").cumcount() + 1 + tail["qid"].map(n_head).fillna(0).astype(int)
    tail["score"] = -100000.0 - tail["rank"]
    return pd.concat([head, tail], ignore_index=True)[["qid", "unit_id", "doc_id", "rank", "score"]], n_changed


def main():
    config.stdout_utf8()
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--base", nargs="+", required=True, help="run gốc đã rerank (Qwen P(yes), ens …)")
    ap.add_argument("--ft", nargs="+", required=True, help="run của model ft chỉ trên đoạn PubMed (k5 PUBMED_ONLY)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--suffix", default="_enft")
    a = ap.parse_args()
    cfg = config.load(a.config)
    pm = runs.pubmed_ids(cfg)
    ft = ft_order(a.ft)
    print(f"điểm ft: {len(ft)} đoạn PubMed, {ft['qid'].nunique()} câu")
    for p in a.base:
        b = runs.read(p)
        r, n = reorder(b, ft, pm)
        dst = Path(a.out) / f"{Path(p).stem}{a.suffix}.parquet"
        runs.write(r, dst, {"from": str(p), "ft_pubmed_order": [str(x) for x in a.ft]})
        print(f"→ {dst}: đổi điểm {n} đoạn PubMed")


if __name__ == "__main__":
    main()
