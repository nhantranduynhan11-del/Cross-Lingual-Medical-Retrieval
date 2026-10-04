"""T7: đánh giá run file với qrels (qid, doc_id, chunk_text — chunk_text có thể rỗng).

Chỉ số (macro trên các câu hỏi có nhãn):
  - Cấp tài liệu (thứ hạng tài liệu = thứ hạng đoạn tốt nhất): Recall@K, nDCG@10, MRR@10.
  - Cấp đoạn: Recall@K = tỷ lệ đoạn đúng được ít nhất một đoạn trong top-K (cùng doc_id) khớp nội dung.
  - P/R/F2 theo TẬP như bài nộp (mục 2): tài liệu = top k_doc tài liệu, đoạn = top k_chunk đoạn;
    F2 = 5PR/(4P+R). Cấp đoạn chỉ tính trên câu hỏi có ít nhất một chunk_text không rỗng.
  - Bảng tách theo ngôn ngữ của tài liệu đúng (vi / zh): Recall@K chỉ tính trên tài liệu đúng của ngôn ngữ đó.
Khớp chunk_text (eval.match): exact (sau khi chuẩn hoá khoảng trắng) | contains (chứa nhau)
  | overlap (tỷ lệ ký tự của đoạn đúng nằm trong đoạn trả về >= eval.overlap_threshold).

  python -m mir.evaluate --run runs/structure_G.parquet [--qrels qrels.parquet]
  python -m mir.evaluate --tables [--strategy structure]     # results/exp_A.md, exp_B.md, exp_C.md
"""
import argparse
import difflib
import json
import math
import re
import sys
from pathlib import Path

import pandas as pd

from . import config, runs, store

_WS = re.compile(r"\s+")


def norm(s):
    return _WS.sub(" ", s or "").strip()


def match(pred, gold, rule="overlap", thr=0.8):
    p, g = norm(pred), norm(gold)
    if not p or not g:
        return False
    if rule == "exact":
        return p == g
    if rule == "contains":
        return g in p or p in g
    if g in p:
        return True
    sm = difflib.SequenceMatcher(None, g, p, autojunk=False)
    return sum(b.size for b in sm.get_matching_blocks()) / len(g) >= thr


def f2(p, r):
    return 5 * p * r / (4 * p + r) if (p + r) > 0 else 0.0


def ndcg(ranked, rel, k=10):
    dcg = sum(1 / math.log2(i + 2) for i, d in enumerate(ranked[:k]) if d in rel)
    idcg = sum(1 / math.log2(i + 2) for i in range(min(k, len(rel))))
    return dcg / idcg if idcg else 0.0


def mrr(ranked, rel, k=10):
    return next((1 / (i + 1) for i, d in enumerate(ranked[:k]) if d in rel), 0.0)


def evaluate(run, qrels, texts, doc_lang, ec, k_doc, k_chunk):
    """run: DataFrame run; qrels: DataFrame; texts: {chunk_id: văn bản (đoạn hoặc đoạn cha)} cho các đoạn trong top;
    doc_lang: {doc_id: 'vi'|'zh'}. → (tổng hợp dict, bảng theo câu hỏi DataFrame)."""
    ks = ec["ks"]
    rule, thr = ec["match"], ec["overlap_threshold"]
    gold_docs = qrels.groupby("qid")["doc_id"].apply(set).to_dict()
    gq = qrels[qrels["chunk_text"].fillna("").str.strip() != ""]
    gold_chunks = {q: list(zip(g["doc_id"], g["chunk_text"])) for q, g in gq.groupby("qid")}
    run = run.sort_values(["qid", "rank"])
    by_q = {q: g for q, g in run.groupby("qid")}
    rows = []
    for q, gd in gold_docs.items():
        g = by_q.get(q, run.iloc[:0])
        chunks = list(zip(g["chunk_id"], g["doc_id"]))
        docs = list(dict.fromkeys(d for _, d in chunks))
        r = {"qid": q, "n_gold_docs": len(gd)}
        for k in ks:
            r[f"doc_R@{k}"] = len(set(docs[:k]) & gd) / len(gd)
            for lg in ("vi", "zh"):
                gl = {d for d in gd if doc_lang.get(d) == lg}
                r[f"doc_R@{k}_{lg}"] = len(set(docs[:k]) & gl) / len(gl) if gl else None
        r["nDCG@10"], r["MRR@10"] = ndcg(docs, gd), mrr(docs, gd)
        sd = set(docs[:k_doc])
        p = len(sd & gd) / len(sd) if sd else 0.0
        rr = len(sd & gd) / len(gd)
        r["doc_P"], r["doc_R"], r["doc_F2"] = p, rr, f2(p, rr)
        gc = gold_chunks.get(q)
        if gc:
            def hit(c, d):
                return [i for i, (gdid, gt) in enumerate(gc) if gdid == d and match(texts.get(c, ""), gt, rule, thr)]
            hits = {c: hit(c, d) for c, d in chunks[:max(max(ks), k_chunk)]}
            for k in ks:
                found = {i for c, _ in chunks[:k] for i in hits.get(c, [])}
                r[f"chunk_R@{k}"] = len(found) / len(gc)
            sel = chunks[:k_chunk]
            tp = sum(1 for c, _ in sel if hits.get(c))
            found = {i for c, _ in sel for i in hits.get(c, [])}
            cp = tp / len(sel) if sel else 0.0
            cr = len(found) / len(gc)
            r["chunk_P"], r["chunk_R"], r["chunk_F2"] = cp, cr, f2(cp, cr)
        rows.append(r)
    per_q = pd.DataFrame(rows)
    summ = {"n_queries": len(per_q), "n_queries_with_chunks": int(per_q["chunk_F2"].notna().sum())
            if "chunk_F2" in per_q else 0}
    for c in per_q.columns:
        if c not in ("qid", "n_gold_docs"):
            v = per_q[c].dropna()
            summ[c] = round(float(v.mean()), 4) if len(v) else None
    summ |= {"match": rule, "overlap_threshold": thr, "k_doc": k_doc, "k_chunk": k_chunk}
    return summ, per_q


def load_context(cfg, run, strategy, unit=None):
    """Văn bản các đoạn trong top của run (để khớp chunk_text) + ngôn ngữ tài liệu."""
    ks = cfg["eval"]["ks"]
    kmax = max(max(ks), cfg["submit"]["k_chunk"])
    top = run[run["rank"] <= kmax]
    ids = top["chunk_id"].unique().tolist()
    chunk_dir = cfg.work_dir / cfg["chunk"]["out_dir"] / strategy
    is_parent = unit == "parent" or any(c.rsplit("_", 1)[1].startswith("p") for c in ids[:50])   # id đoạn cha: doc_pK
    if is_parent:
        t = store.lookup(chunk_dir, "parent_id", ids, ["text"], prefix="parents-")
    else:
        t = store.lookup(chunk_dir, "chunk_id", ids, ["text"])
    texts = {k: v["text"] for k, v in t.items()}
    return texts


def doc_langs(cfg, doc_ids):
    m = store.lookup(cfg.work_dir / cfg["clean"]["out_dir"], "doc_id", list(doc_ids), ["lang"])
    return {k: v["lang"] for k, v in m.items()}


def load_qrels(path):
    q = pd.read_parquet(path)
    q["qid"] = q["qid"].astype(int)
    q["doc_id"] = q["doc_id"].astype(str)
    if "chunk_text" not in q:
        q["chunk_text"] = ""
    return q


def eval_run(cfg, run_path, qrels, strategy):
    run = runs.read(run_path)
    texts = load_context(cfg, run, strategy)
    langs = doc_langs(cfg, set(qrels["doc_id"]))
    return evaluate(run, qrels, texts, langs, cfg["eval"], cfg["submit"]["k_doc"], cfg["submit"]["k_chunk"])


def tables(cfg, rdir, qrels, strategy):
    """Thí nghiệm A (7 cấu hình × Recall@K), B (cách gộp × Recall@100), C (có/không xếp hạng lại × nDCG@10, MRR@10)."""
    ks = cfg["eval"]["ks"]
    res, out = {}, {}

    def get(name):
        p = rdir / f"{name}.parquet"
        if p.exists() and name not in res:
            res[name] = eval_run(cfg, p, qrels, strategy)[0]
        return res.get(name)

    A = ["| Cấu hình | nhánh | " + " | ".join(f"doc R@{k}" for k in ks) + " | " + " | ".join(f"chunk R@{k}" for k in ks) + " |",
         "|---|---|" + "---:|" * (2 * len(ks))]
    for name, brs in cfg["fuse"]["configs"].items():
        s = get(f"{strategy}_{name}")
        if s:
            A.append(f"| {name} | {'+'.join(brs)} | " + " | ".join(f"{s.get(f'doc_R@{k}') or 0:.4f}" for k in ks) + " | "
                     + " | ".join(f"{s.get(f'chunk_R@{k}') or 0:.4f}" for k in ks) + " |")
    B = ["| Cách gộp | doc R@100 | chunk R@100 |", "|---|---:|---:|"]
    for p in sorted(rdir.glob(f"{strategy}_*.parquet")):
        n = p.stem[len(strategy) + 1:]
        if n == "G" or n.startswith("W"):
            s = get(p.stem)
            B.append(f"| {'RRF k=' + str(cfg['fuse']['rrf_k']) if n == 'G' else 'có trọng số ' + n[1:]} | "
                     f"{s.get('doc_R@100') or 0:.4f} | {s.get('chunk_R@100') or 0:.4f} |")
    C = ["| Run | xếp hạng lại | nDCG@10 | MRR@10 | doc F2 | chunk F2 |", "|---|---|---:|---:|---:|---:|"]
    for p in sorted(rdir.glob(f"{strategy}_*_rr.parquet")):
        base = p.stem[:-3]
        for nm, rr in ((base, "không"), (p.stem, "có")):
            s = get(nm)
            if s:
                C.append(f"| {base} | {rr} | {s['nDCG@10']:.4f} | {s['MRR@10']:.4f} | {s['doc_F2']:.4f} | "
                         f"{s.get('chunk_F2') or 0:.4f} |")
    head = (f"qrels: {len(set(qrels['qid']))} câu hỏi; khớp đoạn: {cfg['eval']['match']}"
            f" (ngưỡng {cfg['eval']['overlap_threshold']}); k_doc={cfg['submit']['k_doc']}, k_chunk={cfg['submit']['k_chunk']}.")
    for name, body in (("A", A), ("B", B), ("C", C)):
        out[name] = f"# Thí nghiệm {name} ({strategy})\n\n{head}\n\n" + "\n".join(body) + "\n"
    return out, res


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/baseline.yaml")
    ap.add_argument("--run", default=None)
    ap.add_argument("--qrels", default=None, help="mặc định eval.qrels")
    ap.add_argument("--strategy", default=None)
    ap.add_argument("--runs", default=None)
    ap.add_argument("--tables", action="store_true")
    a = ap.parse_args()
    cfg = config.load(a.config)
    qpath = a.qrels or cfg["eval"]["qrels"]
    if not qpath:
        raise SystemExit("Chưa có qrels (eval.qrels = null). Cần quyết định cách tạo tập đánh giá trước — xem NOTES.md.")
    qrels = load_qrels(qpath)
    strategy = a.strategy or cfg["chunk"]["strategy"]
    res_dir = cfg.work_dir / "results"
    res_dir.mkdir(parents=True, exist_ok=True)
    if a.tables:
        rdir = Path(a.runs) if a.runs else cfg.work_dir / cfg["retrieve"]["out_dir"]
        out, res = tables(cfg, rdir, qrels, strategy)
        for n, md in out.items():
            (res_dir / f"exp_{n}.md").write_text(md, encoding="utf-8")
            print(md)
        (res_dir / f"exp_{strategy}_all.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
        return
    summ, per_q = eval_run(cfg, a.run, qrels, strategy)
    name = Path(a.run).stem
    (res_dir / f"eval_{name}.json").write_text(json.dumps(summ, ensure_ascii=False, indent=1), encoding="utf-8")
    per_q.to_csv(res_dir / f"eval_{name}_per_query.csv", index=False)
    print(json.dumps(summ, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
