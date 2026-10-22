"""Ghép bài nộp: danh sách TÀI LIỆU từ một hệ, danh sách ĐOẠN từ hệ khác (vd tài liệu từ lin07q_cross, đoạn từ lin07_cross).

Tài liệu chỉ cần doc_id → lấy từ file run (.parquet, thứ hạng tài liệu như submit: runs.doc_ranking) hoặc từ bài nộp
(.json). Đoạn cần nguyên văn → lấy từ bài nộp .json có sẵn. Kiểm tra như submit (đủ câu, id nguyên, doc_id có trong
links) rồi ghi <out_dir>/<name>.json + .zip. Chạy trên máy, không cần mir2-chunks.

  python tools/mix_submission.py --docs results/lin07q/runs/lin07q_cross.parquet \
      --chunks results/lin07s/submissions/lin07_cross.json --name mix_lin07qD_lin07C --out results/mix_submit
Cắt số tài liệu THEO TỪNG CÂU (--auto-probs run_qwen3.parquet): p(tài liệu) = max P(yes) của các đoạn của nó trong top-N
đã rerank (ngoài top-N = 0); theo thứ tự tài liệu của --docs, chọn k ∈ [--k-min, k_doc] tối đa kỳ vọng F2 ≈ 5·Σ_{i≤k} p_i /
(4·Σ_all p + k). Không có tham số dò; câu không có trong run xác suất giữ k_doc.
"""
import argparse
import json
import sys
from pathlib import Path

import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mirret import config, runs, submit  # noqa: E402


def doc_lists(path, k_doc, agg="max", norm_depth=400):
    p = Path(path)
    if p.suffix == ".json":
        return {int(s["id"]): [int(d) for d in s["relevant_docs"]][:k_doc] for s in json.loads(p.read_text("utf-8"))}
    run = runs.read(p, depth=k_doc * 3)
    dr = runs.doc_ranking(run, agg, norm_depth).sort_values(["qid", "drank"])
    return {int(q): [int(d) for d in g["doc_id"]][:k_doc] for q, g in dr.groupby("qid")}


def auto_k(docs, prob_run, k_min, k_max, top_n=400):
    import numpy as np
    r = runs.read(prob_run)
    r = r.assign(p=r["score"].where(r["rank"] <= top_n, 0.0).clip(0, 1))
    pdoc = r.groupby(["qid", "doc_id"])["p"].max()
    tot = pdoc.groupby(level=0).sum()
    out, ks = {}, []
    for q, ds in docs.items():
        if q not in tot.index:
            out[q] = ds
            continue
        pq_ = pdoc.loc[q]
        p = np.array([pq_.get(d, 0.0) for d in ds])
        k = np.arange(1, len(p) + 1)
        ef2 = 5 * np.cumsum(p) / (4 * tot.loc[q] + k)
        lo, hi = min(k_min, len(p)), min(k_max, len(p))
        kq = int(lo + np.argmax(ef2[lo - 1:hi]))
        out[q] = ds[:kq]
        ks.append(kq)
    if ks:
        ks = np.array(ks)
        print(f"  cắt theo câu ({len(ks)} câu có xác suất): k trung vị {int(np.median(ks))}, ={k_min}: {(ks == k_min).mean():.0%}, "
              f"={k_max}: {(ks == k_max).mean():.0%}")
    return out


def main():
    config.stdout_utf8()
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--docs", required=True, help="run .parquet hoặc bài nộp .json cho danh sách tài liệu")
    ap.add_argument("--chunks", required=True, help="bài nộp .json cho danh sách đoạn (giữ nguyên)")
    ap.add_argument("--name", required=True)
    ap.add_argument("--out", default="results/mix_submit")
    ap.add_argument("--k-doc", type=int, default=None)
    ap.add_argument("--k-chunk", type=int, default=None, help="cắt danh sách đoạn còn k đoạn đầu (mặc định giữ nguyên)")
    ap.add_argument("--doc-agg", default="max", choices=list(runs.DOC_AGGS))
    ap.add_argument("--auto-probs", default=None, help="run có điểm P(yes) (qwen3/qwen3i) → cắt số tài liệu theo từng câu")
    ap.add_argument("--k-min", type=int, default=100, help="--auto-probs: số tài liệu tối thiểu mỗi câu")
    a = ap.parse_args()
    cfg = config.load(a.config)
    k_doc = a.k_doc or cfg["submit"]["k_doc"]
    docs = doc_lists(a.docs, k_doc, a.doc_agg, cfg["rerank"]["top_n"])
    if a.auto_probs:
        docs = auto_k(docs, a.auto_probs, a.k_min, k_doc, cfg["rerank"]["top_n"])
    chunks = {int(s["id"]): s["relevant_chunks"][:a.k_chunk] for s in json.loads(Path(a.chunks).read_text("utf-8"))}
    qids = [int(x) for x in pq.read_table(cfg.path("queries"), columns=["id"]).column("id").to_pylist()]
    sub = [{"id": q, "relevant_docs": docs.get(q, []), "relevant_chunks": chunks.get(q, [])} for q in qids]
    link_ids = set(pq.read_table(cfg.path("links"), columns=["id"]).column("id").to_pylist())
    errs = submit.validate(sub, qids, link_ids)
    if errs:
        print(f"LỖI ({len(errs)}), KHÔNG ghi bài nộp:\n  " + "\n  ".join(errs[:30]))
        raise SystemExit(1)
    js, zp = submit.write(sub, a.out, a.name)
    meta = {"docs_from": a.docs, "chunks_from": a.chunks, "k_doc": k_doc, "k_chunk": a.k_chunk, "doc_agg": a.doc_agg,
            "auto_probs": a.auto_probs, "k_min": a.k_min if a.auto_probs else None,
            "queries": len(sub), "chunks": sum(len(s["relevant_chunks"]) for s in sub), "zip_mb": round(zp.stat().st_size / 1e6, 1)}
    (Path(a.out) / f"{a.name}_meta.json").write_text(json.dumps(meta, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"Hợp lệ: {len(sub)} câu, tài liệu từ {Path(a.docs).name}, đoạn từ {Path(a.chunks).name} → {zp} ({meta['zip_mb']} MB)")


if __name__ == "__main__":
    main()
