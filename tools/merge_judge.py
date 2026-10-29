"""Ghép nhãn giám khảo LLM (mirret.llmjudge, k10) vào baseline lin07 → danh sách tài liệu + run cho span_chunks. Chạy trên
máy, không GPU. Cách ghép cố định trước khi nộp (không dò trên đáp án):

  - điểm tài liệu (các tài liệu đã chấm, hạng ens 1..hi): s = (1 − w)·z(ens) + w·z(logit P_judge), z chuẩn hoá theo câu
    trên chính các tài liệu đã chấm; ens = điểm đoạn tốt nhất của tài liệu trong lin07_ensi; P_judge = P(2) + ½P(1);
  - xác suất cho autok + phân bổ đoạn (--prob): mean = (P_judge + P(yes) Qwen3-Reranker) / 2 (mặc định), judge, qwen;
  - phần đuôi chưa chấm (hạng ens hi+1 … --k-max, như autok gốc 100–400): nối sau các tài liệu đã chấm theo thứ tự ens,
    P_judge ước bằng trung bình P_judge của 50 hạng chấm cuối của câu đó;
  - tài liệu: xếp theo s, autok k ∈ [--k-min, --k-max] tối đa 5·Σp / (4·Σp_all + k) (như mix_submission.auto_k); bản _np
    bỏ PubMed sau khi cắt (như final1200_cbud200_nopm);
  - câu chưa chấm (chạy thử một phần câu): giữ nguyên danh sách tài liệu của --base-docs và run gốc.
Đầu ra (--out): <name>_docs_pm.json, <name>_docs_np.json, runs/<name>_order.parquet (run thứ tự cho --order),
runs/<name>_probs.parquet (P mỗi tài liệu cho --probs) → tools/span_chunks.py → (chạy thử) tools/splice_sub.py.

  python tools/merge_judge.py --labels results/judge_pub_a --name judge --out results/judge \
      --base-docs results/mix_submit/final1200_vi2300_pm.json
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
    p = np.asarray(p, dtype=np.float64).clip(1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def zq(df, col):
    g = df.groupby("qid")[col]
    return ((df[col] - g.transform("mean")) / g.transform("std").replace(0, np.nan)).fillna(0.0)


def load_labels(paths):
    files = []
    for p in map(Path, paths):
        files += sorted(p.rglob("labels-*.parquet")) if p.is_dir() else [p]
    if not files:
        raise SystemExit(f"không thấy labels-*.parquet trong {paths}")
    lab = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    lab["qid"], lab["doc_id"] = lab["qid"].astype(int), lab["doc_id"].astype(int)
    dup = lab.duplicated(["qid", "doc_id"]).sum()
    if dup:
        raise SystemExit(f"{dup} cặp (câu, tài liệu) bị chấm hai lần — trùng phần giữa các tài khoản?")
    print(f"nhãn: {len(files)} file, {len(lab)} cặp, {lab['qid'].nunique()} câu, model {lab['model'].unique().tolist()}; "
          f"không đọc được {int(lab['prel'].isna().sum())}")
    return lab


def main():
    config.stdout_utf8()
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--labels", nargs="+", required=True, help="thư mục output k10 (tìm labels-*.parquet) hoặc file")
    ap.add_argument("--ens", default="results/full1200/runs/lin07_ensi_1200.parquet")
    ap.add_argument("--qwen", default="results/full1200/runs/lin07_qwen3i_1200.parquet")
    ap.add_argument("--base-docs", default=None, help="bài nộp .json: danh sách tài liệu cho câu chưa chấm")
    ap.add_argument("--w", type=float, default=0.5, help="trọng số giám khảo trong điểm xếp tài liệu (1 = chỉ giám khảo)")
    ap.add_argument("--prob", default="mean", choices=["mean", "judge", "qwen"])
    ap.add_argument("--k-min", type=int, default=100)
    ap.add_argument("--k-max", type=int, default=400, help="> hạng chấm cuối: nối phần đuôi chưa chấm (xem trên)")
    ap.add_argument("--top-n", type=int, default=400)
    ap.add_argument("--name", required=True)
    ap.add_argument("--out", default="results/judge")
    a = ap.parse_args()
    cfg = config.load(a.config)
    pm = runs.pubmed_ids(cfg)
    lab = load_labels(a.labels)
    jq = set(lab["qid"])
    hi = int(lab["drank"].max())
    ens = runs.read(a.ens, depth=1000)
    qw = runs.read(a.qwen, depth=a.top_n)
    pq_ = qw[(qw["score"] >= 0) & (qw["score"] <= 1)].groupby(["qid", "doc_id"])["score"].max().rename("pqwen")
    dr = runs.doc_ranking(ens[ens["qid"].isin(jq)], "max", a.top_n)
    dr = dr[dr["rank"] <= a.top_n][["qid", "doc_id", "drank", "score"]]       # tài liệu có đoạn trong phần đã rerank
    d = dr.merge(lab[["qid", "doc_id", "prel", "p2"]], on=["qid", "doc_id"], how="inner").join(pq_, on=["qid", "doc_id"])
    miss = dr[dr["drank"] <= hi].shape[0] - len(d)
    if miss:
        print(f"  cảnh báo: {miss} tài liệu hạng ≤ {hi} chưa có nhãn (bỏ khỏi danh sách)")
    d["prel"] = d["prel"].fillna(d["pqwen"])                  # cặp giám khảo không đọc được → dùng P(yes)
    d["pqwen"] = d["pqwen"].fillna(0.0)
    d["jl"] = logit(d["prel"])
    d["s"] = (1 - a.w) * zq(d, "score") + a.w * zq(d, "jl")
    d["p"] = {"mean": (d["prel"] + d["pqwen"]) / 2, "judge": d["prel"], "qwen": d["pqwen"]}[a.prob]
    # phần đuôi chưa chấm (hạng ens > hi): nối sau theo thứ tự ens, P_judge ước bằng TB P_judge của 50 hạng chấm cuối
    tp = d[d["drank"] > hi - 50].groupby("qid")["prel"].mean().rename("prel")
    t = dr[(dr["drank"] > hi) & (dr["drank"] <= a.k_max)].join(tp, on="qid").join(pq_, on=["qid", "doc_id"])
    t["pqwen"] = t["pqwen"].fillna(0.0)
    t["p"] = {"mean": (t["prel"] + t["pqwen"]) / 2, "judge": t["prel"], "qwen": t["pqwen"]}[a.prob]
    t["s"] = -1e9 - t["drank"]
    print(f"phần đuôi chưa chấm (hạng {hi + 1}–{a.k_max}): TB {len(t) / max(len(jq), 1):.0f} tài liệu/câu, "
          f"P_judge ước TB {tp.mean():.3f}")
    d = pd.concat([d, t], ignore_index=True)
    d = d.sort_values(["qid", "s", "drank"], ascending=[True, False, True]).reset_index(drop=True)
    d["jrank"] = d.groupby("qid").cumcount() + 1
    sp = d[d["drank"] <= hi][["qid", "doc_id", "drank", "jrank"]]
    print(f"Spearman(hạng ens, hạng mới) TB theo câu: "
          f"{sp.groupby('qid').apply(lambda g: g['drank'].corr(g['jrank'], method='spearman')).mean():.3f}; "
          f"Σp / câu trung vị {d.groupby('qid')['p'].sum().median():.1f} (prel {d.groupby('qid')['prel'].sum().median():.1f}, "
          f"P(yes) {d.groupby('qid')['pqwen'].sum().median():.1f})")
    out = Path(a.out)
    (out / "runs").mkdir(parents=True, exist_ok=True)
    # run P: một dòng / tài liệu đã chấm (đơn vị tốt nhất), câu chưa chấm giữ run Qwen gốc
    best = runs.doc_ranking(ens[ens["qid"].isin(jq)], "max", a.top_n)[["qid", "doc_id", "unit_id"]]
    pr = d.merge(best, on=["qid", "doc_id"])[["qid", "unit_id", "doc_id", "jrank", "p"]]
    pr = pr.rename(columns={"jrank": "rank", "p": "score"})
    pr = pd.concat([pr, qw[~qw["qid"].isin(jq)][["qid", "unit_id", "doc_id", "rank", "score"]]], ignore_index=True)
    runs.write(pr.sort_values(["qid", "rank"]).reset_index(drop=True), out / "runs" / f"{a.name}_probs.parquet",
               {"merge_judge": "probs", "prob": a.prob, "labels": a.labels})
    # run thứ tự: đơn vị của tài liệu đã chấm theo s (trong tài liệu giữ thứ tự ens), rồi phần còn lại theo ens
    e = ens[ens["qid"].isin(jq)].merge(d[["qid", "doc_id", "jrank"]], on=["qid", "doc_id"], how="left")
    e["jrank"] = e["jrank"].fillna(10 ** 6)
    e = e.sort_values(["qid", "jrank", "rank"])
    e["rank"] = e.groupby("qid").cumcount() + 1
    e["score"] = -e["rank"].astype(float)
    od = pd.concat([e[["qid", "unit_id", "doc_id", "rank", "score"]],
                    ens[~ens["qid"].isin(jq)][["qid", "unit_id", "doc_id", "rank", "score"]]], ignore_index=True)
    runs.write(od.sort_values(["qid", "rank"]).reset_index(drop=True), out / "runs" / f"{a.name}_order.parquet",
               {"merge_judge": "order", "w": a.w, "ens": a.ens, "labels": a.labels})
    # tài liệu
    docs = {int(q): [int(x) for x in g["doc_id"]] for q, g in d.groupby("qid")}
    docs = auto_k(docs, out / "runs" / f"{a.name}_probs.parquet", a.k_min, a.k_max, top_n=10 ** 9)
    base = {}
    if a.base_docs:
        base = {int(s["id"]): s["relevant_docs"] for s in json.loads(Path(a.base_docs).read_text("utf-8"))}
    if len(jq) < 1200 and not base:
        raise SystemExit("chỉ một phần câu có nhãn → cần --base-docs cho các câu còn lại")
    allq = sorted(set(base) | set(docs))
    full = {q: docs.get(q, base.get(q, [])) for q in allq}
    sets = {f"{a.name}_docs_pm": full,
            f"{a.name}_docs_np": {q: ([x for x in v if x not in pm] if q in jq else v) for q, v in full.items()}}
    for nm, dd in sets.items():
        (out / f"{nm}.json").write_text(json.dumps([{"id": q, "relevant_docs": v} for q, v in sorted(dd.items())]),
                                        encoding="utf-8")
        kj = [len(v) for q, v in dd.items() if q in jq]
        print(f"→ {out / (nm + '.json')}: câu đã chấm TB {np.mean(kj):.1f} tài liệu (trung vị {np.median(kj):.0f})")
    if base:
        ov = [len(set(docs[q]) & set(base.get(q, []))) / max(len(docs[q]), 1) for q in docs]
        nb = [len(base.get(q, [])) for q in docs]
        print(f"so với --base-docs trên câu đã chấm: trùng {np.mean(ov):.1%} danh sách mới, base TB {np.mean(nb):.1f} tài liệu")
    (out / f"{a.name}_meta.json").write_text(json.dumps(vars(a), indent=1, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
