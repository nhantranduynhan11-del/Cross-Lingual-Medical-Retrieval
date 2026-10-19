"""So run với baseline trên HAI NỬA câu hỏi có nhãn (chia ngẫu nhiên, cố định seed) từ eval/perq_<run>.csv.

Dùng khi chọn cấu hình (trọng số RRF, bỏ nhánh, top_n…) trên chính tập có nhãn: chỉ chấp nhận thay đổi khi chênh lệch
cùng dấu ở cả hai nửa (docs/EVAL_PLAN.md mục 7). Không thay cho kiểm định trong `mirret.evaluate score`.

  python tools/eval_split.py --eval results/k6_score/eval --baseline rrf9 --runs rrf7,rrf9_w --metrics doc_F2@20,doc_R@50
"""
import argparse
import random
from pathlib import Path

import numpy as np
import pandas as pd


def boot(d, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    b = rng.choice(d, size=(n, len(d)), replace=True).mean(axis=1)
    return float(d.mean()), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", required=True)
    ap.add_argument("--baseline", default="rrf9")
    ap.add_argument("--runs", required=True)
    ap.add_argument("--metrics", default="doc_F2@5,doc_F2@20,unit_F2@20,doc_R@50")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    ev = Path(a.eval)
    base = pd.read_csv(ev / f"perq_{a.baseline}.csv")
    qids = sorted(base["qid"].unique())
    random.Random(a.seed).shuffle(qids)
    halves = {"nửa 1": set(qids[: len(qids) // 2]), "nửa 2": set(qids[len(qids) // 2:]), "tất cả": set(qids)}
    for run in a.runs.split(","):
        df = pd.read_csv(ev / f"perq_{run}.csv")
        print(f"\n## {run} − {a.baseline}  ({len(qids)} câu; KTC 95% bootstrap)")
        for m in a.metrics.split(","):
            if m not in df.columns:
                print(f"  {m}: không có cột")
                continue
            mm = pd.merge(df[["qid", m]], base[["qid", m]], on="qid", suffixes=("_r", "_b")).dropna()
            cells, signs = [], []
            for h, qs in halves.items():
                d = (mm[mm.qid.isin(qs)][f"{m}_r"] - mm[mm.qid.isin(qs)][f"{m}_b"]).to_numpy()
                mean, lo, hi = boot(d)
                cells.append(f"{h} {mean:+.3f} [{lo:+.3f}, {hi:+.3f}]")
                if h != "tất cả":
                    signs.append(np.sign(round(mean, 4)))
            same = "cùng dấu ở 2 nửa" if signs[0] == signs[1] and signs[0] != 0 else "KHÔNG cùng dấu → chưa kết luận"
            print(f"  {m}: " + " | ".join(cells) + f"  → {same}")


if __name__ == "__main__":
    main()
