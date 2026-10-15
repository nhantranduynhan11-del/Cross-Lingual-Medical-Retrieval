"""Chẩn đoán KHÔNG cần nhãn trên toàn bộ 1.200 câu hỏi: độ trùng giữa các nhánh, đóng góp vào union, thành phần
và độ nhạy của RRF, thay đổi sau xếp hạng lại.

CÁC CHỈ SỐ NÀY ĐO ĐA DẠNG / ẢNH HƯỞNG, KHÔNG ĐO RECALL. Một nhánh thêm nhiều tài liệu "mới" có thể chỉ thêm nhiễu;
hai nhánh trùng nhau nhiều có thể cùng đúng hoặc cùng sai. Muốn biết nhánh nào tìm được tài liệu ĐÚNG phải dùng tập có
nhãn (mirret.judge + mirret.evaluate).

  python -m mirret.diag [--runs vi_dense,…] [--fused rrf9] [--reranked rrf9_colbert] [--ks 10,50,100,200]
     → eval/diag/diag.md + các bảng CSV
"""
import argparse
import itertools
import random

import numpy as np
import pandas as pd

from . import config, fuse, runs

BRANCHES = [f"{lg}_{m}" for lg in ("vi", "en", "zh") for m in ("dense", "lsr", "bm25")]


def rbo(a, b, p=0.9):
    """Rank-biased overlap (Webber 2010), dạng cắt ngắn tới độ sâu min(|a|,|b|) — nặng ký ở đầu danh sách."""
    k = min(len(a), len(b))
    if k == 0:
        return np.nan
    sa, sb, s = set(), set(), 0.0
    for d in range(1, k + 1):
        sa.add(a[d - 1])
        sb.add(b[d - 1])
        s += p ** (d - 1) * len(sa & sb) / d
    return (1 - p) * s


def boot_ci(x, n=1000, seed=0):
    x = np.asarray([v for v in x if not np.isnan(v)], dtype=float)
    if len(x) == 0:
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed)
    m = rng.choice(x, size=(n, len(x)), replace=True).mean(axis=1)
    return float(x.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def fmt(m, lo, hi, nd=3):
    return "-" if np.isnan(m) else f"{m:.{nd}f} [{lo:.{nd}f}, {hi:.{nd}f}]"


def doc_meta(cfg):
    """{doc_id: lang} từ chunks/parents (nhẹ: chỉ 2 cột)."""
    import pyarrow.dataset as ds
    t = ds.dataset(str(cfg.path("chunks") / "parents")).to_table(columns=["doc_id", "lang"]).to_pandas()
    return dict(zip(t["doc_id"], t["lang"]))


def pairwise(lists, names, k, n_boot):
    rows = []
    for a, b in itertools.combinations(names, 2):
        J, O, R = [], [], []
        for q in set(lists[a]) & set(lists[b]):
            A, B = lists[a][q][:k], lists[b][q][:k]
            if not A or not B:
                continue
            sa, sb = set(A), set(B)
            J.append(len(sa & sb) / len(sa | sb))
            O.append(len(sa & sb) / min(len(sa), len(sb)))
            R.append(rbo(A, B))
        rows.append({"a": a, "b": b, "k": k, "n_q": len(J), "jaccard": boot_ci(J, n_boot),
                     "overlap_coef": boot_ci(O, n_boot), "rbo": boot_ci(R, n_boot)})
    return rows


def union_growth(lists, order, k):
    """Thêm lần lượt từng nhánh vào union: trung bình |union| và số tài liệu MỚI mỗi bước (theo câu hỏi)."""
    qs = sorted(set().union(*[set(lists[n]) for n in order]))
    cur = {q: set() for q in qs}
    out = []
    for n in order:
        new = [len(set(lists[n].get(q, [])[:k]) - cur[q]) for q in qs]
        for q in qs:
            cur[q] |= set(lists[n].get(q, [])[:k])
        out.append({"branch": n, "new_docs_mean": float(np.mean(new)), "union_mean": float(np.mean([len(cur[q]) for q in qs]))})
    return out


def greedy_order(lists, names, k):
    order, left = [], list(names)
    qs = sorted(set().union(*[set(lists[n]) for n in names]))
    cur = {q: set() for q in qs}
    while left:
        best = max(left, key=lambda n: np.mean([len(set(lists[n].get(q, [])[:k]) - cur[q]) for q in qs]))
        order.append(best)
        left.remove(best)
        for q in qs:
            cur[q] |= set(lists[best].get(q, [])[:k])
    return order


def exclusive(lists, names, k):
    """Trung bình số tài liệu chỉ MỘT nhánh có trong top-k (không nhánh nào khác có trong top-k của nó)."""
    out = {}
    for n in names:
        v = []
        for q, L in lists[n].items():
            others = set().union(*[set(lists[m].get(q, [])[:k]) for m in names if m != n])
            v.append(len(set(L[:k]) - others))
        out[n] = float(np.mean(v)) if v else np.nan
    return out


def lang_profile(lists, names, k, lang_of):
    rows = []
    for n in names:
        c = {"vi": 0, "zh": 0, "en": 0, "?": 0}
        tot = 0
        for q, L in lists[n].items():
            for d in L[:k]:
                c[lang_of.get(d, "?") if lang_of.get(d) in c else "?"] += 1
                tot += 1
        rows.append({"run": n, **{lg: c[lg] / max(1, tot) for lg in c}, "n_q": len(lists[n]),
                     "mean_len": tot / max(1, len(lists[n]))})
    return rows


def composition(fused_lists, branch_lists, names, k, depth):
    """Với mỗi đơn vị trong top-k của run gộp: bao nhiêu nhánh có nó trong top-depth; tỷ lệ đơn vị chỉ một nhánh có."""
    support, sole = [], {n: 0 for n in names}
    share = {n: [] for n in names}
    for q, L in fused_lists.items():
        sets = {n: set(branch_lists[n].get(q, [])[:depth]) for n in names}
        top = L[:k]
        for u in top:
            s = [n for n in names if u in sets[n]]
            support.append(len(s))
            if len(s) == 1:
                sole[s[0]] += 1
        for n in names:
            share[n].append(len(set(top) & sets[n]) / max(1, len(top)))
    support = np.array(support)
    return {"support_hist": {i: float((support == i).mean()) for i in range(0, len(names) + 1)},
            "sole_share": {n: sole[n] / max(1, len(support)) for n in names},
            "branch_share": {n: float(np.mean(share[n])) for n in names}}


def loo(branch_runs, names, rrf_k, depth, k, n_boot):
    """Bỏ từng nhánh khỏi RRF: top-k thay đổi bao nhiêu (Jaccard, RBO với RRF đủ nhánh) — mức ẢNH HƯỞNG, không phải giá trị."""
    full = runs.topk_lists(fuse.rrf([branch_runs[n] for n in names], rrf_k, depth), k, "unit")
    rows = []
    for n in names:
        rest = [m for m in names if m != n]
        part = runs.topk_lists(fuse.rrf([branch_runs[m] for m in rest], rrf_k, depth), k, "unit")
        J = [len(set(full[q]) & set(part.get(q, []))) / max(1, len(set(full[q]) | set(part.get(q, [])))) for q in full]
        R = [rbo(full[q], part.get(q, [])) for q in full]
        rows.append({"removed": n, "jaccard_vs_full": boot_ci(J, n_boot), "rbo_vs_full": boot_ci(R, n_boot)})
    return rows


def rrf_k_sensitivity(branch_runs, names, depth, k):
    base = runs.topk_lists(fuse.rrf([branch_runs[n] for n in names], 60, depth), k, "unit")
    out = {}
    for kk in (10, 30, 100):
        alt = runs.topk_lists(fuse.rrf([branch_runs[n] for n in names], kk, depth), k, "unit")
        out[kk] = float(np.mean([len(set(base[q]) & set(alt.get(q, []))) / max(1, len(base[q])) for q in base]))
    return out


def rerank_change(before, after, k):
    J, tau = [], []
    from scipy.stats import kendalltau
    for q, A in before.items():
        B = after.get(q, [])
        if not B:
            continue
        J.append(len(set(A[:k]) & set(B[:k])) / max(1, len(set(A[:k]) | set(B[:k]))))
        common = [u for u in A[:k] if u in set(B[:k])]
        if len(common) >= 3:
            pa = {u: i for i, u in enumerate(A)}
            pb = {u: i for i, u in enumerate(B)}
            t = kendalltau([pa[u] for u in common], [pb[u] for u in common]).statistic
            tau.append(t)
    return boot_ci(J), boot_ci(tau)


def main():
    config.stdout_utf8()
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--runs", default=",".join(BRANCHES))
    ap.add_argument("--fused", default="rrf9")
    ap.add_argument("--reranked", default="", help="run sau xếp hạng lại (phẩy ngăn cách), so với --fused")
    ap.add_argument("--ks", default="10,50,100,200")
    ap.add_argument("--depth", type=int, default=200, help="độ sâu mỗi nhánh dùng cho RRF/thành phần")
    ap.add_argument("--sample", type=int, default=0, help="chỉ N câu hỏi ngẫu nhiên (nhanh hơn)")
    a = ap.parse_args()
    cfg = config.load(a.config)
    rdir = cfg.path("runs")
    out = cfg.path("eval") / "diag"
    out.mkdir(parents=True, exist_ok=True)
    ks = [int(x) for x in a.ks.split(",")]
    kmax = max(ks + [a.depth])
    nb = cfg["eval"]["bootstrap"]
    names = [n for n in a.runs.split(",") if (rdir / f"{n}.parquet").exists()]
    missing = [n for n in a.runs.split(",") if n not in names]
    br = {n: runs.read(rdir / f"{n}.parquet", depth=kmax) for n in names}
    if a.sample:
        qs = sorted(set().union(*[set(df["qid"]) for df in br.values()]))
        keep = set(random.Random(cfg["seed"]).sample(qs, min(a.sample, len(qs))))
        br = {n: df[df["qid"].isin(keep)] for n, df in br.items()}
    docs = {n: runs.topk_lists(df, kmax, "doc") for n, df in br.items()}
    units = {n: runs.topk_lists(df, kmax, "unit") for n, df in br.items()}
    lang_of = doc_meta(cfg)
    L = ["# Chẩn đoán không cần nhãn (diag)", "",
         "> Các chỉ số dưới đây đo **độ đa dạng / mức ảnh hưởng** giữa các nhánh trên toàn bộ câu hỏi. "
         "**Không phải Recall**: tài liệu \"mới\" hay \"chỉ một nhánh có\" chưa chắc đã đúng. "
         "Kết luận về Recall/Precision chỉ lấy từ tập có nhãn thủ công (mirret.evaluate).", "",
         f"Nhánh có run: {', '.join(names)}" + (f" · THIẾU: {', '.join(missing)}" if missing else ""),
         "Khoảng trong ngoặc: KTC 95% bootstrap theo câu hỏi.", ""]
    # 1. phủ & ngôn ngữ
    L += ["## 1. Độ phủ và ngôn ngữ tài liệu trong top-k (cấp tài liệu)", "",
          "| run | số câu có kết quả | số tài liệu TB (top-" + str(ks[-1]) + ") | vi | zh | en |", "|---|---:|---:|---:|---:|---:|"]
    for r in lang_profile(docs, names, ks[-1], lang_of):
        L.append(f"| {r['run']} | {r['n_q']} | {r['mean_len']:.0f} | {r['vi']:.0%} | {r['zh']:.0%} | {r['en']:.0%} |")
    # 2. trùng cặp
    for k in ks[:3]:
        rows = pairwise(docs, names, k, nb)
        pd.DataFrame([{**{x: r[x] for x in ("a", "b", "k", "n_q")}, "jaccard": r["jaccard"][0],
                       "overlap_coef": r["overlap_coef"][0], "rbo": r["rbo"][0]} for r in rows]
                     ).to_csv(out / f"pairwise_doc_k{k}.csv", index=False)
        M = pd.DataFrame(index=names, columns=names, dtype=float)
        for r in rows:
            M.loc[r["a"], r["b"]] = M.loc[r["b"], r["a"]] = r["overlap_coef"][0]
        L += ["", f"## 2. Tỷ lệ tài liệu chung giữa hai nhánh, top-{k} (overlap coefficient, cấp tài liệu)", "",
              "| | " + " | ".join(names) + " |", "|---|" + "---:|" * len(names)]
        for n in names:
            L.append(f"| {n} | " + " | ".join("—" if n == m else f"{M.loc[n, m]:.2f}" for m in names) + " |")
    # 3. union
    k = ks[1] if len(ks) > 1 else ks[0]
    g = greedy_order(docs, names, k)
    L += ["", f"## 3. Union các nhánh (top-{k} tài liệu mỗi nhánh)", "",
          "Thứ tự cố định (vi → en → zh) và thứ tự tham lam (mỗi bước chọn nhánh thêm nhiều tài liệu MỚI nhất). "
          "\"Mới\" ≠ \"đúng\".", "", "| bước | nhánh (cố định) | tài liệu mới TB | |union| TB | nhánh (tham lam) | mới TB | |union| TB |",
          "|---:|---|---:|---:|---|---:|---:|"]
    for i, (x, y) in enumerate(zip(union_growth(docs, names, k), union_growth(docs, g, k)), 1):
        L.append(f"| {i} | {x['branch']} | {x['new_docs_mean']:.1f} | {x['union_mean']:.1f} | {y['branch']} | "
                 f"{y['new_docs_mean']:.1f} | {y['union_mean']:.1f} |")
    ex = exclusive(docs, names, k)
    L += ["", f"Tài liệu chỉ một nhánh có trong top-{k} (TB/câu): " +
          ", ".join(f"{n} {v:.1f}" for n, v in sorted(ex.items(), key=lambda x: -x[1]))]
    # 4. thành phần & độ nhạy của RRF
    if names:
        fused_name = a.fused
        fp = rdir / f"{fused_name}.parquet"
        fused = runs.read(fp, depth=kmax) if fp.exists() else fuse.rrf([br[n] for n in names], cfg["fuse"]["rrf_k"],
                                                                         a.depth)
        if a.sample:
            fused = fused[fused["qid"].isin(set().union(*[set(df["qid"]) for df in br.values()]))]
        fl = runs.topk_lists(fused, kmax, "unit")
        kc = ks[1] if len(ks) > 1 else ks[0]
        c = composition(fl, units, names, kc, a.depth)
        L += ["", f"## 4. Thành phần của {fused_name} (top-{kc} đoạn, nhánh xét tới top-{a.depth})", "",
              "Số nhánh cùng có một đoạn: " + ", ".join(f"{i} nhánh {v:.0%}" for i, v in c["support_hist"].items() if v),
              "", "| nhánh | tỷ lệ top-k của run gộp có trong nhánh | tỷ lệ đoạn CHỈ nhánh này đưa vào |", "|---|---:|---:|"]
        for n in names:
            L.append(f"| {n} | {c['branch_share'][n]:.0%} | {c['sole_share'][n]:.1%} |")
        lo = loo(br, names, cfg["fuse"]["rrf_k"], a.depth, kc, nb)
        L += ["", f"## 5. Bỏ từng nhánh khỏi RRF (top-{kc}): mức thay đổi so với RRF đủ nhánh", "",
              "Thay đổi lớn = nhánh có ẢNH HƯỞNG lớn lên danh sách cuối — chưa nói là tốt hay xấu.", "",
              "| bỏ nhánh | Jaccard với RRF đủ | RBO với RRF đủ |", "|---|---|---|"]
        for r in lo:
            L.append(f"| {r['removed']} | {fmt(*r['jaccard_vs_full'])} | {fmt(*r['rbo_vs_full'])} |")
        sens = rrf_k_sensitivity(br, names, a.depth, kc)
        L += ["", f"Độ ổn định theo hằng số k của RRF (tỷ lệ top-{kc} giữ nguyên so với k=60): " +
              ", ".join(f"k={kk}: {v:.0%}" for kk, v in sens.items())]
        for rr in [x for x in a.reranked.split(",") if x]:
            p = rdir / f"{rr}.parquet"
            if p.exists():
                after = runs.topk_lists(runs.read(p, depth=kmax), kmax, "unit")
                (j, jl, jh), (t, tl, th) = rerank_change(fl, after, kc)
                L += ["", f"Xếp hạng lại {rr} so với {fused_name}, top-{kc}: Jaccard {fmt(j, jl, jh)}, "
                      f"Kendall τ trên đoạn chung {fmt(t, tl, th)}"]
    L += ["", "## Đọc kết quả thế nào", "",
          "- Hai nhánh trùng nhiều (overlap > 0,7) → gần như thừa nhau về TẬP kết quả; giữ cả hai chỉ có lợi nếu thứ hạng "
          "của chúng bổ sung nhau (xem mục 5) và tập có nhãn xác nhận.",
          "- Nhánh thêm nhiều tài liệu mới / có nhiều tài liệu riêng → ứng viên để kiểm tra trên tập có nhãn: có thể là "
          "Recall bổ sung (vd nhánh zh mở ra tài liệu tiếng Trung) hoặc chỉ là nhiễu.",
          "- Nhánh ảnh hưởng mạnh tới RRF (mục 5) mà tập có nhãn cho thấy độ chính xác thấp → giảm trọng số hoặc bỏ.",
          "- Các quyết định giữ/bỏ chỉ dựa trên `mirret.evaluate` (tập có nhãn) — xem docs/EVAL_PLAN.md."]
    md = "\n".join(L) + "\n"
    (out / "diag.md").write_text(md, encoding="utf-8")
    print(md)


if __name__ == "__main__":
    main()
