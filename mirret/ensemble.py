"""A2: kết hợp điểm reranker đã lưu (CPU, không chạy lại GPU) trên top-N của run Stage 1.

Mỗi câu hỏi, trên đúng tập top-N của run gốc, chuẩn hoá z-score từng nguồn điểm:
  bge   : logit của cross-encoder (giữ nguyên)
  qwen3 : P(yes) → logit log(p / (1 − p)) trước khi chuẩn hoá (xác suất dồn sát 1, z-score trực tiếp mất phân biệt)
  rrf   : điểm RRF của run gốc
Run cố định (không dò λ để tránh khớp vào bộ nhãn):
  <run>_ens  = (z_bge + z_qwen3) / 2                       (cần --bge và --qwen)
  <run>_ip08 = 0,8·z_rr + 0,2·z_rrf,  <run>_ip06 = 0,6·z_rr + 0,4·z_rrf
               z_rr = bge (mặc định) hoặc qwen3 (--ip-with qwen)
Phần đuôi (hạng > N) giữ thứ tự run gốc như rerank.py. Chỉ dùng các câu có trong mọi run đầu vào.

  python -m mirret.ensemble --base runs/rrf9_w.parquet --bge runs/rrf9_w_cross.parquet --qwen runs/rrf9_w_q3.parquet
     → runs/rrf9_w_ens.parquet, rrf9_w_ip08.parquet, rrf9_w_ip06.parquet
  python -m mirret.ensemble --base runs/rrf9_w.parquet --qwen runs/rrf9_w_qwen3i_sub.parquet --ip-with qwen --suffix q3i_sub
     → runs/rrf9_w_ip08q3i_sub.parquet, rrf9_w_ip06q3i_sub.parquet   (qwen3 trộn với RRF, không cần bge)
"""
import argparse
from pathlib import Path

import numpy as np

from . import config, runs
from .rerank import with_tail

LAMBDAS = {"ip08": 0.8, "ip06": 0.6}


def zscore(df, col):
    g = df.groupby("qid")[col]
    sd = g.transform("std").replace(0, np.nan)
    return ((df[col] - g.transform("mean")) / sd).fillna(0.0)


def scores(path, top_n, qids, name):
    r = runs.read(path, depth=top_n, qids=qids)
    return r[["qid", "unit_id", "score"]].rename(columns={"score": name})


def main():
    config.stdout_utf8()
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="run Stage 1 (vd runs/rrf9_w.parquet)")
    ap.add_argument("--bge", default=None, help="run đã rerank bằng bge (cross) trên cùng run gốc")
    ap.add_argument("--qwen", default=None, help="run đã rerank bằng qwen3/qwen3i")
    ap.add_argument("--ip-with", default="bge", choices=["bge", "qwen"], help="reranker trộn với RRF ở ip08/ip06")
    ap.add_argument("--top-n", type=int, default=400)
    ap.add_argument("--out", default=None, help="thư mục ghi (mặc định: cạnh run gốc)")
    ap.add_argument("--suffix", default="", help="thêm vào tên run, vd _sub")
    a = ap.parse_args()
    ip_src = a.bge if a.ip_with == "bge" else a.qwen
    if not ip_src:
        raise SystemExit(f"--ip-with {a.ip_with} cần --{a.ip_with}")

    base_path = Path(a.base)
    srcs = [p for p in (a.bge, a.qwen) if p]
    qids = set.intersection(*[set(runs.read(p)["qid"].unique()) for p in srcs])
    base = runs.read(base_path, qids=qids)
    df = base[base["rank"] <= a.top_n][["qid", "unit_id", "doc_id", "rank", "score"]].rename(columns={"score": "rrf"})
    cols = []
    if a.bge:
        df = df.merge(scores(a.bge, a.top_n, qids, "bge"), on=["qid", "unit_id"], how="left")
        cols.append("bge")
    if a.qwen:
        df = df.merge(scores(a.qwen, a.top_n, qids, "qwen"), on=["qid", "unit_id"], how="left")
        cols.append("qwen")
    miss = df[cols].isna().sum()
    if miss.any():
        raise SystemExit(f"thiếu điểm cho {miss.to_dict()} cặp: các run rerank phải cùng run gốc và cùng top-N {a.top_n}")

    df["z_rrf"] = zscore(df, "rrf")
    if a.bge:
        df["z_bge"] = zscore(df, "bge")
    if a.qwen:
        p = df["qwen"].astype(np.float64).clip(1e-6, 1 - 1e-6)
        df["qlogit"] = np.log(p / (1 - p))
        df["z_qwen"] = zscore(df, "qlogit")
    z_rr = df["z_bge"] if a.ip_with == "bge" else df["z_qwen"]
    out = {n: (l * z_rr + (1 - l) * df["z_rrf"], f"{l}*z_{a.ip_with} + {1 - l:.1f}*z_rrf") for n, l in LAMBDAS.items()}
    if a.bge and a.qwen:
        out["ens"] = ((df["z_bge"] + df["z_qwen"]) / 2, "(z_bge + z_logit_qwen) / 2")

    out_dir = Path(a.out) if a.out else base_path.parent
    print(f"{len(qids)} câu hỏi, {len(df)} cặp (top-{a.top_n})")
    for name, (sc, formula) in out.items():
        r = runs.rerank_by_score(df[["qid", "unit_id", "doc_id", "rank"]].assign(score=sc.values))
        tag = f"{base_path.stem}_{name}{a.suffix}"
        runs.write(with_tail(r, base, a.top_n), out_dir / f"{tag}.parquet",
                   {"base": str(base_path), "bge": a.bge, "qwen": a.qwen, "top_n": a.top_n, "method": name,
                    "formula": formula, "qids": len(qids)})
        print(f"→ {out_dir / (tag + '.parquet')}")


if __name__ == "__main__":
    main()
