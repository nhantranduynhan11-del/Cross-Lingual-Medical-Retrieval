"""Tạo bài nộp từ run cuối (đoạn cha).

  relevant_docs  : top k_doc tài liệu (thứ hạng tài liệu = đoạn tốt nhất), doc_id là SỐ NGUYÊN (id của links_corpus;
                   dự án cũ: ghi dạng chuỗi → máy chấm coi là PMID → 0 điểm)
  relevant_chunks: top k_chunk đoạn cha {"doc_id", "chunk_text"}; chunk_text = chữ thuần nguyên văn của đoạn (không kèm
                   ngữ cảnh). Đoạn gần trùng (>= 0,8, BTC gộp) trong cùng câu hỏi bị bỏ để nhường chỗ.
Kiểm tra: đủ 1.200 câu, id nguyên, doc_id có trong links, chunk_text khớp chữ thuần của bài (--verify-corpus).
File nén > submit.max_zip_mb → giảm k_chunk 5% mỗi lần tới khi vừa (báo lại).

  python -m mirret.submit --run runs/rrf9_colbert.parquet [--name bai1] [--k-doc 200] [--k-chunk 185] [--verify-corpus]
"""
import argparse
import json
import zipfile
from pathlib import Path

import pyarrow.parquet as pq

from . import config, runs
from .rerank import unit_rows


def _norm(s):
    return " ".join((s or "").lower().split())


import numpy as np


def build(run, qids, k_doc, k_chunk, texts, auto_k=False, min_k=100, max_k=400, r_total=10.0):
    by_q = {q: g.sort_values("rank") for q, g in run.groupby("qid")}
    sub = []
    chosen_ks = []
    for q in qids:
        g = by_q.get(q)
        docs, chunks, seen = [], [], set()
        if g is not None:
            cur_k_doc = k_doc
            if auto_k and len(g) >= min_k:
                scores = g["score"].values.astype(float)
                if (scores >= 0.0).all() and (scores <= 1.0).all():
                    probs = scores
                else:
                    probs = 1.0 / (1.0 + np.exp(-np.clip(scores, -10.0, 10.0)))
                cum_probs = np.cumsum(probs)
                ks = np.arange(1, len(probs) + 1)
                exp_f2 = (5.0 * cum_probs) / (4.0 * r_total + ks)
                valid_mask = (ks >= min_k) & (ks <= min(max_k, len(ks)))
                if np.any(valid_mask):
                    valid_ks = ks[valid_mask]
                    valid_f2 = exp_f2[valid_mask]
                    cur_k_doc = int(valid_ks[np.argmax(valid_f2)])
            chosen_ks.append(cur_k_doc)

            docs = list(dict.fromkeys(int(d) for d in g["doc_id"]))[:cur_k_doc]
            for u, d in zip(g["unit_id"], g["doc_id"]):
                if len(chunks) >= k_chunk:
                    break
                t = texts.get(u)
                if not t:
                    continue
                key = (int(d), _norm(t)[:300])
                if key in seen:
                    continue
                seen.add(key)
                chunks.append({"doc_id": int(d), "chunk_text": t})
        sub.append({"id": int(q), "relevant_docs": docs, "relevant_chunks": chunks})
    if auto_k and chosen_ks:
        print(f"  [Auto-K] K tài liệu trung bình: {np.mean(chosen_ks):.1f} (min {min(chosen_ks)}, max {max(chosen_ks)})")
    return sub


def validate(sub, qids, link_ids, plains=None):
    errs = []
    got = [s["id"] for s in sub]
    if sorted(got) != sorted(qids) or len(set(got)) != len(got):
        errs.append(f"id câu hỏi không khớp: có {len(set(got))}, cần {len(qids)}")
    for s in sub:
        if not isinstance(s["id"], int) or isinstance(s["id"], bool):
            errs.append(f"id {s['id']!r} không phải số nguyên")
        for d in s["relevant_docs"]:
            if not isinstance(d, int) or d not in link_ids:
                errs.append(f"câu {s['id']}: doc_id {d!r} không có trong links")
        for c in s["relevant_chunks"]:
            d, t = c["doc_id"], c["chunk_text"]
            if not isinstance(d, int) or d not in link_ids:
                errs.append(f"câu {s['id']}: chunk doc_id {d!r} không có trong links")
            elif not t:
                errs.append(f"câu {s['id']}: chunk_text rỗng")
            elif plains is not None and d in plains and t not in plains[d]:
                errs.append(f"câu {s['id']}: chunk_text không phải chuỗi con của bài {d}")
    return errs


def write(sub, out_dir, name):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    js, zp = out_dir / f"{name}.json", out_dir / f"{name}.zip"
    with js.open("w", encoding="utf-8") as f:
        json.dump(sub, f, ensure_ascii=False)
    with zipfile.ZipFile(zp, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        z.write(js, arcname=js.name)                       # đúng một file ở gốc
    return js, zp


def corpus_plains(cfg, doc_ids):
    import pyarrow.dataset as ds
    from .chunk import parse
    t = ds.dataset(str(cfg.path("corpus"))).to_table(columns=["id", "text"],
                                                     filter=ds.field("id").isin(list(doc_ids))).to_pydict()
    return {int(i): parse(x or "")[0] for i, x in zip(t["id"], t["text"])}


def main():
    config.stdout_utf8()
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--run", required=True)
    ap.add_argument("--name", default=None)
    ap.add_argument("--k-doc", type=int, default=None)
    ap.add_argument("--k-chunk", type=int, default=None)
    ap.add_argument("--auto-k", action="store_true", default=None, help="Tự động tính K tài liệu theo kỳ vọng E[F2]")
    ap.add_argument("--min-k", type=int, default=None)
    ap.add_argument("--max-k", type=int, default=None)
    ap.add_argument("--r-total", type=float, default=None)
    ap.add_argument("--verify-corpus", action="store_true", help="đối chiếu chunk_text với chữ thuần của bài (chậm hơn)")
    a = ap.parse_args()
    cfg = config.load(a.config)
    sc = cfg["submit"]
    k_doc, k_chunk = a.k_doc or sc["k_doc"], a.k_chunk or sc["k_chunk"]
    auto_k = a.auto_k if a.auto_k is not None else sc.get("auto_k", False)
    min_k = a.min_k or sc.get("auto_k_min", 100)
    max_k = a.max_k or sc.get("auto_k_max", 400)
    r_total = a.r_total or sc.get("auto_k_r", 10.0)

    qids = [int(x) for x in pq.read_table(cfg.path("queries"), columns=["id"]).column("id").to_pylist()]
    run = runs.read(a.run, depth=max((max_k if auto_k else k_doc) * 3, k_chunk + 100))
    ids = run[run["rank"] <= k_chunk + 100]["unit_id"].unique().tolist()
    texts = {u: r[2] for u, r in unit_rows(cfg, ids).items()}
    link_ids = set(pq.read_table(cfg.path("links"), columns=["id"]).column("id").to_pylist())
    name = a.name or Path(a.run).stem
    out_dir = cfg.path("submissions")
    while True:
        sub = build(run, qids, k_doc, k_chunk, texts, auto_k=auto_k, min_k=min_k, max_k=max_k, r_total=r_total)
        plains = corpus_plains(cfg, {c["doc_id"] for s in sub for c in s["relevant_chunks"]}) if a.verify_corpus else None
        errs = validate(sub, qids, link_ids, plains)
        if errs:
            print(f"LỖI ({len(errs)}), KHÔNG ghi bài nộp:\n  " + "\n  ".join(errs[:30]))
            raise SystemExit(1)
        js, zp = write(sub, out_dir, name)
        mb = zp.stat().st_size / 1e6
        if mb <= sc["max_zip_mb"] or k_chunk <= 10:
            break
        k_new = int(k_chunk * 0.95)
        print(f"  {zp.name}: {mb:.1f} MB > {sc['max_zip_mb']} MB → giảm k_chunk {k_chunk} → {k_new}")
        k_chunk = k_new
    n_chunks = sum(len(s["relevant_chunks"]) for s in sub)
    (out_dir / f"{name}_meta.json").write_text(json.dumps(
        {"run": a.run, "k_doc": k_doc, "k_chunk": k_chunk, "auto_k": auto_k, "zip_mb": round(mb, 1), "queries": len(sub),
         "chunks": n_chunks}, indent=1), encoding="utf-8")
    print(f"Hợp lệ: {len(sub)} câu hỏi, {n_chunks:,} đoạn, k_doc={k_doc} (auto_k={auto_k}), k_chunk={k_chunk} → {zp} ({mb:.1f} MB)")


if __name__ == "__main__":
    main()
