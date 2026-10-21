"""Đánh giá trên tập có nhãn thủ công (eval/judge/qrels.parquet) — KHÔNG giả định có nhãn của BTC.

Định nghĩa (macro trên các câu hỏi có nhãn; R_q = tập liên quan TRONG POOL của câu q, nhãn >= eval.rel_threshold):
  Cấp tài liệu (thứ hạng tài liệu = đoạn tốt nhất; tài liệu liên quan nếu có ÍT NHẤT một đoạn đã chấm >= ngưỡng):
    P@K  = |topK ∩ R_q| / |topK|            (mục chưa chấm tính là KHÔNG liên quan → cận dưới)
    Pj@K = |topK ∩ R_q| / |topK ∩ đã chấm|  (chỉ trên mục đã chấm → "condensed", cận trên thô)
    R@K  = |topK ∩ R_q| / |R_q|             (Recall TRÊN POOL — không phải Recall trên toàn kho)
    F2@K = 5·P·R / (4·P + R)
    judged@K = |topK ∩ đã chấm| / |topK|    (thấp → kết quả không tin cậy: chạy `mirret.judge extend`)
  Cấp đoạn (như máy chấm BTC: cùng câu hỏi, cùng doc_id, LCS token / |đoạn đúng| >= 0,4 sau chuẩn hoá NFKC + chữ thường):
    một đoạn trả về là ĐÚNG nếu khớp ít nhất một đoạn liên quan đã chấm; R = tỷ lệ đoạn liên quan được ít nhất một đoạn
    trong topK khớp. Đoạn trả về chưa chấm nhưng khớp đoạn liên quan đã chấm vẫn được tính (đơn vị khác nhau).
  nDCG@10 (nhãn 0/1/2) ở cấp đoạn.

  python -m mirret.evaluate score    [--runs a,b,…] [--baseline rrf9] [--ks 10,50,100,185]
  python -m mirret.evaluate marginal [--k 50]          # đóng góp biên của từng nhánh (union + bỏ-một-nhánh khỏi RRF)
  python -m mirret.evaluate table --ks 20              # bảng tổng hợp ở K = số cuối của --ks → eval/table_k20.md
                                                       # (9 nhánh, union cùng ngân sách, RRF, rerank)
"""
import argparse
import re
import unicodedata

import numpy as np
import pandas as pd

from . import config, fuse, runs
from .diag import BRANCHES, boot_ci, fmt

_WTOK = re.compile(r"[㐀-鿿豈-﫿]|[^\W_]+", re.U)


def toks(s):
    s = unicodedata.normalize("NFKC", s or "").lower()
    return _WTOK.findall(s)


def lcs_len(a, b):
    if not a or not b:
        return 0
    if len(a) < len(b):
        a, b = b, a
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0]
        for j, y in enumerate(b, 1):
            cur.append(prev[j - 1] + 1 if x == y else (prev[j] if prev[j] > cur[j - 1] else cur[j - 1]))
        prev = cur
    return prev[-1]


def overlap(pred_toks, gold_toks):
    """LCS / |gold| — cận trên rẻ (số token gold xuất hiện trong pred) để bỏ qua phép tính LCS khi chắc chắn < ngưỡng."""
    return lcs_len(pred_toks, gold_toks) / max(1, len(gold_toks))


def f2(p, r):
    return 5 * p * r / (4 * p + r) if (p + r) > 0 else 0.0


class Judged:
    def __init__(self, qrels, thr, lcs_thr, unit_text):
        self.thr, self.lcs_thr = thr, lcs_thr
        self.q = qrels
        self.judged_units = {q: dict(zip(g["unit_id"], g["grade"])) for q, g in qrels.groupby("qid")}
        doc_grade = qrels.groupby(["qid", "doc_id"])["grade"].max()
        self.judged_docs = {q: g.droplevel(0).to_dict() for q, g in doc_grade.groupby(level=0)}
        rel = qrels[qrels["grade"] >= thr]
        self.rel_units = {q: list(zip(g["unit_id"], g["doc_id"], g["text"].map(toks))) for q, g in rel.groupby("qid")}
        self.unit_text = unit_text
        self._tok_cache = {}

    def qids(self):
        return sorted(self.judged_units)

    def ttoks(self, u):
        if u not in self._tok_cache:
            self._tok_cache[u] = toks(self.unit_text.get(u, ""))
        return self._tok_cache[u]

    def unit_hits(self, q, unit, doc):
        """Chỉ số các đoạn liên quan (của câu q) mà đoạn trả về khớp (cùng doc, LCS >= ngưỡng; trùng id = khớp)."""
        out = []
        for i, (gu, gd, gt) in enumerate(self.rel_units.get(q, [])):
            if gd != doc:
                continue
            if gu == unit:
                out.append(i)
                continue
            pt = self.ttoks(unit)
            if sum(1 for t in gt if t in set(pt)) / max(1, len(gt)) < self.lcs_thr:
                continue
            if overlap(pt, gt) >= self.lcs_thr:
                out.append(i)
        return out


def score_run(df, J, ks, doc_agg="max", norm_depth=400):
    """→ DataFrame theo câu hỏi với các chỉ số ở từng K. Danh sách tài liệu = runs.doc_ranking(doc_agg) như bài nộp."""
    by_q = {q: g.sort_values("rank") for q, g in df.groupby("qid")}
    dr = runs.doc_ranking(df, doc_agg, norm_depth).sort_values(["qid", "drank"])
    docs_q = {q: list(g["doc_id"]) for q, g in dr.groupby("qid")}
    rows = []
    for q in J.qids():
        g = by_q.get(q)
        units = list(zip(g["unit_id"], g["doc_id"])) if g is not None else []
        docs = docs_q.get(q, [])
        jd = J.judged_docs.get(q, {})
        Rd = {d for d, gr in jd.items() if gr >= J.thr}
        ju = J.judged_units.get(q, {})
        n_rel_u = len(J.rel_units.get(q, []))
        r = {"qid": q, "n_rel_docs": len(Rd), "n_rel_units": n_rel_u}
        hits = {}
        for u, d in units[:max(ks)]:
            hits[u] = J.unit_hits(q, u, d) if n_rel_u else []
        for k in ks:
            tk = docs[:k]
            tp = len(set(tk) & Rd)
            judged = sum(1 for d in tk if d in jd)
            p = tp / len(tk) if tk else 0.0
            pj = tp / judged if judged else np.nan
            rr = tp / len(Rd) if Rd else np.nan
            r.update({f"doc_P@{k}": p, f"doc_Pj@{k}": pj, f"doc_R@{k}": rr,
                      f"doc_F2@{k}": f2(p, rr) if Rd else np.nan, f"doc_judged@{k}": judged / len(tk) if tk else np.nan})
            tu = units[:k]
            ok = [u for u, _ in tu if hits.get(u)]
            found = {i for u, _ in tu for i in hits.get(u, [])}
            up = len(ok) / len(tu) if tu else 0.0
            ur = len(found) / n_rel_u if n_rel_u else np.nan
            r.update({f"unit_P@{k}": up, f"unit_R@{k}": ur, f"unit_F2@{k}": f2(up, ur) if n_rel_u else np.nan,
                      f"unit_judged@{k}": (sum(1 for u, _ in tu if u in ju) / len(tu)) if tu else np.nan})
        gains = [ju.get(u, 0) for u, _ in units[:10]]
        ideal = sorted([gr for gr in ju.values()], reverse=True)[:10]
        dcg = sum(gn / np.log2(i + 2) for i, gn in enumerate(gains))
        idcg = sum(gn / np.log2(i + 2) for i, gn in enumerate(ideal))
        r["nDCG@10"] = dcg / idcg if idcg else np.nan
        rows.append(r)
    return pd.DataFrame(rows)


def paired(a, b, col, n=2000, seed=0):
    """Chênh lệch trung bình (a − b) theo câu hỏi: KTC 95% bootstrap + kiểm định hoán vị dấu (hai phía)."""
    m = pd.merge(a[["qid", col]], b[["qid", col]], on="qid").dropna()
    d = (m[f"{col}_x"] - m[f"{col}_y"]).to_numpy()
    if len(d) == 0:
        return np.nan, np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed)
    bs = rng.choice(d, size=(n, len(d)), replace=True).mean(axis=1)
    flips = rng.choice([-1, 1], size=(n, len(d))) * d
    p = float((np.abs(flips.mean(axis=1)) >= abs(d.mean()) - 1e-12).mean())
    return float(d.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5)), p


def load(cfg, run_names, depth):
    qrels = pd.read_parquet(cfg.path("eval") / "judge" / "qrels.parquet")
    qids = set(qrels["qid"])
    rdir = cfg.path("runs")
    have = {}
    for n in run_names:
        p = rdir / f"{n}.parquet"
        if p.exists():
            have[n] = runs.read(p, depth=depth, qids=qids)
    from .rerank import unit_rows
    need = set(qrels["unit_id"]) | set().union(*[set(df["unit_id"]) for df in have.values()] or [set()])
    text = {u: r[2] for u, r in unit_rows(cfg, list(need)).items()}
    return qrels, have, text


def summarize(per_q, ks, nb):
    out = {}
    for c in per_q.columns:
        if c in ("qid", "n_rel_docs", "n_rel_units"):
            continue
        out[c] = boot_ci(per_q[c].to_numpy(dtype=float), nb)
    return out


def doc_depth(cfg, ks):
    """Số đoạn đọc vào: đủ sâu để có k_doc tài liệu như submit (max(k_doc·3, k_chunk + 100)), không chỉ max(ks) đoạn."""
    sc = cfg["submit"]
    return max(max(ks), sc["k_doc"] * 3, sc["k_chunk"] + 100)


def cmd_score(cfg, names, baseline, ks, doc_aggs=("max",), log=print):
    """doc_aggs khác 'max' → thêm dòng <run>~<agg> (chỉ đổi thứ hạng tài liệu; chỉ số đoạn giống run gốc)."""
    ec = cfg["eval"]
    qrels, have, text = load(cfg, names, doc_depth(cfg, ks))
    J = Judged(qrels, ec["rel_threshold"], ec["lcs_threshold"], text)
    nd = cfg["rerank"]["top_n"]
    res = {(n if agg == "max" else f"{n}~{agg}"): score_run(df, J, ks, agg, nd)
           for n, df in have.items() for agg in doc_aggs}
    out = cfg.path("eval")
    summ = {n: summarize(pq_, ks, ec["bootstrap"]) for n, pq_ in res.items()}
    for n, pq_ in res.items():
        pq_.to_csv(out / f"perq_{n}.csv", index=False)
    L = ["# Đánh giá trên tập có nhãn thủ công", "",
         f"{len(J.qids())} câu hỏi có nhãn; liên quan = nhãn >= {ec['rel_threshold']}. Recall ở đây là **Recall trên "
         "pool** (chỉ tính tài liệu đã được chấm), KHÔNG phải Recall trên toàn kho. Khoảng: KTC 95% bootstrap. "
         "P = mục chưa chấm tính là sai (cận dưới); Pj = chỉ trên mục đã chấm (cận trên thô).", "",
         "So sánh công bằng giữa các nhánh: dùng K nhỏ (≤ độ sâu pool của mỗi nhánh, mặc định 10) — ở đó judged ≈ 1. "
         "Ở K lớn (185/200, giống bài nộp) run gộp/rerank đóng góp sâu hơn vào pool nên được chấm nhiều hơn nhánh đơn: "
         "chỉ dùng R@K và xem judged@K trước khi kết luận."]
    for k in ks:
        L += ["", f"## K = {k}", "",
              "| run | judged doc | doc P | doc Pj | doc R | doc F2 | judged đoạn | unit P | unit R | unit F2 |",
              "|---|---:|---|---:|---|---|---:|---|---|---|"]
        for n, s in summ.items():
            L.append(f"| {n} | {s[f'doc_judged@{k}'][0]:.2f} | {fmt(*s[f'doc_P@{k}'])} | {s[f'doc_Pj@{k}'][0]:.3f} | "
                     f"{fmt(*s[f'doc_R@{k}'])} | {fmt(*s[f'doc_F2@{k}'])} | {s[f'unit_judged@{k}'][0]:.2f} | "
                     f"{fmt(*s[f'unit_P@{k}'])} | {fmt(*s[f'unit_R@{k}'])} | {fmt(*s[f'unit_F2@{k}'])} |")
    L += ["", "## nDCG@10 (cấp đoạn, nhãn 0/1/2)", "", "| run | nDCG@10 |", "|---|---|"]
    L += [f"| {n} | {fmt(*s['nDCG@10'])} |" for n, s in summ.items()]
    if baseline in res:
        for k in sorted({ks[0], ks[-1]}):
            L += ["", f"## So sánh cặp với {baseline}, K = {k} (chênh lệch TB theo câu hỏi, KTC 95%, p hoán vị)", "",
                  "| run | Δ doc F2 | p | Δ unit F2 | p | Δ doc R | p |", "|---|---|---:|---|---:|---|---:|"]
            for n, pq_ in res.items():
                if n == baseline:
                    continue
                a = paired(pq_, res[baseline], f"doc_F2@{k}")
                b = paired(pq_, res[baseline], f"unit_F2@{k}")
                c = paired(pq_, res[baseline], f"doc_R@{k}")
                L.append(f"| {n} | {fmt(*a[:3])} | {a[3]:.3f} | {fmt(*b[:3])} | {b[3]:.3f} | {fmt(*c[:3])} | "
                         f"{c[3]:.3f} |")
    L += ["", "Đọc kết quả: judged@K < 0,8 → nhiều mục của run chưa được chấm, P/F2 bị đánh giá THẤP hơn thực tế "
          "(chạy `mirret.judge extend --runs <run>:<K>` rồi chấm phần bổ sung). Chênh lệch có KTC chứa 0 → chưa đủ bằng "
          "chứng để nói run nào tốt hơn. Nhiều phép so sánh cùng lúc → p nhỏ đơn lẻ (vd 0,03) chưa phải bằng chứng mạnh."]
    md = "\n".join(L) + "\n"
    (out / "score.md").write_text(md, encoding="utf-8")
    log(md)
    return res


def cmd_marginal(cfg, k, log=print):
    """Union lần lượt từng nhánh: tài liệu MỚI vs tài liệu ĐÚNG MỚI; tài liệu đúng chỉ một nhánh tìm ra;
    bỏ từng nhánh khỏi RRF: thay đổi doc R@k và doc F2@k (có KTC)."""
    ec = cfg["eval"]
    names = [b for b in BRANCHES if (cfg.path("runs") / f"{b}.parquet").exists()]
    qrels, have, text = load(cfg, names, doc_depth(cfg, [max(k, 200)]))
    J = Judged(qrels, ec["rel_threshold"], ec["lcs_threshold"], text)
    tops = {n: runs.topk_lists(df, k, "doc") for n, df in have.items()}
    rel = {q: {d for d, g in J.judged_docs.get(q, {}).items() if g >= J.thr} for q in J.qids()}
    judged = {q: set(J.judged_docs.get(q, {})) for q in J.qids()}

    def step_table(order):
        cur = {q: set() for q in J.qids()}
        rows = []
        for n in order:
            new = {q: set(tops[n].get(q, [])) - cur[q] for q in J.qids()}
            rows.append({"branch": n,
                         "new_docs": np.mean([len(new[q]) for q in J.qids()]),
                         "new_judged": np.mean([len(new[q] & judged[q]) for q in J.qids()]),
                         "new_relevant": np.mean([len(new[q] & rel[q]) for q in J.qids()]),
                         "union_recall_pool": np.nanmean([len((cur[q] | new[q]) & rel[q]) / len(rel[q]) if rel[q] else np.nan
                                                          for q in J.qids()])})
            for q in J.qids():
                cur[q] |= new[q]
        return rows

    greedy, left, cur = [], list(names), {q: set() for q in J.qids()}
    while left:
        best = max(left, key=lambda n: np.mean([len((set(tops[n].get(q, [])) - cur[q]) & rel[q]) for q in J.qids()]))
        greedy.append(best)
        left.remove(best)
        for q in J.qids():
            cur[q] |= set(tops[best].get(q, []))
    uniq = {n: np.mean([len((set(tops[n].get(q, [])) - set().union(*[set(tops[m].get(q, [])) for m in names if m != n]))
                            & rel[q]) for q in J.qids()]) for n in names}
    L = ["# Đóng góp biên của từng nhánh (tập có nhãn)", "",
         f"Top-{k} tài liệu mỗi nhánh. \"Mới\" = chưa có trong union các nhánh trước; \"đúng mới\" = mới VÀ được chấm "
         "liên quan. Chỉ \"đúng mới\" nói về Recall; \"mới\" nhiều mà \"đúng mới\" ít = nhánh thêm nhiễu.", ""]
    for title, order in (("Thứ tự cố định", names), ("Thứ tự tham lam theo số tài liệu ĐÚNG mới", greedy)):
        L += [f"## {title}", "", "| bước | nhánh | mới TB | mới đã chấm TB | ĐÚNG mới TB | Recall pool của union |",
              "|---:|---|---:|---:|---:|---:|"]
        for i, r in enumerate(step_table(order), 1):
            L.append(f"| {i} | {r['branch']} | {r['new_docs']:.1f} | {r['new_judged']:.1f} | {r['new_relevant']:.2f} | "
                     f"{r['union_recall_pool']:.3f} |")
        L.append("")
    L += ["Tài liệu ĐÚNG chỉ một nhánh tìm ra (TB/câu): " +
          ", ".join(f"{n} {v:.2f}" for n, v in sorted(uniq.items(), key=lambda x: -x[1])), ""]
    full = fuse.rrf([have[n] for n in names], cfg["fuse"]["rrf_k"], 1000)
    base = score_run(full, J, [k])
    L += [f"## Bỏ từng nhánh khỏi RRF ({len(names)} nhánh), top-{k}", "",
          f"| bỏ nhánh | Δ doc R@{k} | p | Δ doc F2@{k} | p | Δ unit F2@{k} | p |", "|---|---|---:|---|---:|---|---:|"]
    for n in names:
        part = score_run(fuse.rrf([have[m] for m in names if m != n], cfg["fuse"]["rrf_k"], 1000), J, [k])
        a = paired(part, base, f"doc_R@{k}")
        b = paired(part, base, f"doc_F2@{k}")
        c = paired(part, base, f"unit_F2@{k}")
        L.append(f"| {n} | {fmt(*a[:3])} | {a[3]:.3f} | {fmt(*b[:3])} | {b[3]:.3f} | {fmt(*c[:3])} | {c[3]:.3f} |")
    L += ["", "Δ âm có KTC không chứa 0 → bỏ nhánh làm GIẢM chất lượng (nhánh có giá trị). Δ dương có KTC không chứa 0 → "
          "nhánh đang làm hại RRF (bỏ hoặc giảm trọng số). Còn lại: chưa đủ bằng chứng — giữ nguyên, ghi là giả thuyết."]
    md = "\n".join(L) + "\n"
    (cfg.path("eval") / "marginal.md").write_text(md, encoding="utf-8")
    log(md)


def cmd_table(cfg, ks, log=print):
    """Bảng báo cáo: 9 nhánh, union (tập, cùng ngân sách), các RRF, các reranker — cùng tập câu hỏi có nhãn."""
    ec = cfg["eval"]
    rdir = cfg.path("runs")
    names = [b for b in BRANCHES if (rdir / f"{b}.parquet").exists()]
    others = sorted(p.stem for p in rdir.glob("*.parquet") if p.stem not in names and "@" not in p.stem)
    qrels, have, text = load(cfg, names + others, max(ks))
    J = Judged(qrels, ec["rel_threshold"], ec["lcs_threshold"], text)
    k = ks[-1]
    rows = []
    for n, df in have.items():
        s = summarize(score_run(df, J, [k]), [k], ec["bootstrap"])
        kind = "nhánh" if n in names else ("xếp hạng lại" if n.endswith(("_colbert", "_cross")) else "gộp")
        rows.append((kind, n, s))
    # union cùng ngân sách: xen kẽ (round-robin) hạng 1 của mọi nhánh, rồi hạng 2… tới khi đủ đúng K đoạn → so được với
    # RRF top-K (cùng số mục nộp)
    rr = []
    for q in J.qids():
        lists = [runs.topk_lists(have[n][have[n]["qid"] == q], k, "unit").get(q, []) for n in names]
        seen, order = set(), []
        for i in range(k):
            for L_ in lists:
                if len(order) < k and i < len(L_) and L_[i] not in seen:
                    seen.add(L_[i])
                    order.append(L_[i])
        rr += [(q, u, runs.doc_of(u), r + 1, -r) for r, u in enumerate(order)]
    if rr:
        U = pd.DataFrame(rr, columns=["qid", "unit_id", "doc_id", "rank", "score"])
        rows.append(("union", f"union{len(names)} (xen kẽ, đủ K)", summarize(score_run(U, J, [k]), [k], ec["bootstrap"])))
    L = ["# Bảng tổng hợp (tập có nhãn)", "",
         f"Cùng {len(J.qids())} câu hỏi, cùng ngân sách K = {k} (đoạn cha / tài liệu). Recall = Recall trên pool.", "",
         f"| loại | run | judged@{k} | doc P | doc R | doc F2 | unit P | unit R | unit F2 | nDCG@10 |",
         "|---|---|---:|---|---|---|---|---|---|---|"]
    for kind, n, s in rows:
        L.append(f"| {kind} | {n} | {s[f'doc_judged@{k}'][0]:.2f} | {fmt(*s[f'doc_P@{k}'])} | {fmt(*s[f'doc_R@{k}'])} | "
                 f"{fmt(*s[f'doc_F2@{k}'])} | {fmt(*s[f'unit_P@{k}'])} | {fmt(*s[f'unit_R@{k}'])} | "
                 f"{fmt(*s[f'unit_F2@{k}'])} | {fmt(*s['nDCG@10'])} |")
    md = "\n".join(L) + "\n"
    (cfg.path("eval") / f"table_k{k}.md").write_text(md, encoding="utf-8")
    log(md)


def main():
    config.stdout_utf8()
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["score", "marginal", "table"])
    ap.add_argument("--config", default=None)
    ap.add_argument("--runs", default=None)
    ap.add_argument("--baseline", default="rrf9")
    ap.add_argument("--ks", default=None)
    ap.add_argument("--k", type=int, default=50)
    ap.add_argument("--doc-agg", default="max", help="score: cách gộp đoạn → tài liệu, vd 'max,sum2,sum3' (runs.DOC_AGGS)")
    a = ap.parse_args()
    cfg = config.load(a.config)
    ks = [int(x) for x in a.ks.split(",")] if a.ks else cfg["eval"]["ks"]
    if a.cmd == "score":
        names = a.runs.split(",") if a.runs else BRANCHES + list(cfg["fuse"]["configs"])
        cmd_score(cfg, names, a.baseline, ks, a.doc_agg.split(","))
    elif a.cmd == "marginal":
        cmd_marginal(cfg, a.k)
    else:
        cmd_table(cfg, ks)


if __name__ == "__main__":
    main()
